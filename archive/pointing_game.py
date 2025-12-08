#!/usr/bin/env python
"""
Pointing Game Evaluation for CAM Methods - Distributed Version

This script evaluates various Class Activation Mapping (CAM) methods using the 
Pointing Game metric on VOC and COCO datasets.

Usage:
    Single GPU:  python compare_cams_multilabel.py --dataset voc --max_samples 100
    Multi GPU:   torchrun --nproc_per_node=4 compare_cams_multilabel.py --dataset voc --max_samples 100
"""

import os
import sys
import argparse
import json
import random
from datetime import datetime
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import cv2
from tqdm import tqdm

# SpikingJelly imports
from spikingjelly.activation_based import neuron, surrogate, functional, layer

# Project imports
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from utils.args_utils import setup_config_dictionary
from src.utils.data_utils import get_datasets
from src.utils.model_utils import get_model
from src.utils.save_utils import load_model

# CAM imports
from src.cams.gradcam import SNNGradCAM
from src.cams.ablation_cam import AblationCAM
from src.cams.mc_ablation_cam import MCAblationCAM
from src.cams.sam import SpikeActivationMap


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed Pointing Game Evaluation for CAM Methods")
    parser.add_argument("--dataset", type=str, default="voc", choices=["voc", "coco"],
                        help="Dataset to evaluate on")
    parser.add_argument("--max_samples", type=int, default=100,
                        help="Maximum number of samples to process")
    parser.add_argument("--n_time_bins", type=int, default=8,
                        help="Number of time bins for SNN")
    parser.add_argument("--k_trials", type=int, default=20,
                        help="Number of trials for MC Ablation CAM")
    parser.add_argument("--channel_batch_size", type=int, default=32,
                        help="Channel batch size for ablation methods")
    parser.add_argument("--output_dir", type=str, default="./cam_evaluation_results",
                        help="Directory to save results")
    parser.add_argument("--min_correct_classes", type=int, default=1,
                        help="Minimum number of correctly classified classes required")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--save_all_visualizations", action="store_true", default=True,
                        help="Save visualizations for all processed samples")
    return parser.parse_args()


def setup_distributed():
    """Initialize distributed training if available."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        
        return rank, world_size, local_rank, True
    else:
        # Single GPU mode
        return 0, 1, 0, False


def cleanup_distributed(is_distributed):
    """Cleanup distributed training."""
    if is_distributed:
        dist.destroy_process_group()


def is_main_process(rank):
    """Check if this is the main process."""
    return rank == 0


def get_conv_layers(m):
    return [x for x in m.modules() if isinstance(x, layer.Conv2d)]


def get_lif_layers(m):
    return [x for x in m.modules() if isinstance(x, (neuron.LIFNode, neuron.IFNode))]


@contextmanager
def grad_cam_context(model):
    """Context manager for GradCAM computation."""
    prev_training = model.training
    for p in model.parameters():
        p.requires_grad = True
    model.train()
    for m in model.modules():
        if isinstance(m, (layer.BatchNorm1d, layer.BatchNorm2d, layer.BatchNorm3d, 
                          layer.Dropout, layer.Dropout2d)):
            m.eval()
    try:
        yield
    finally:
        model.zero_grad()
        for p in model.parameters():
            p.requires_grad = False
        model.train(prev_training)
        functional.reset_net(model)


class CAMEvaluator:
    """Class to handle CAM generation and evaluation."""
    
    def __init__(self, model, device, k_trials=20, channel_batch_size=32):
        self.model = model
        self.device = device
        self.k_trials = k_trials
        self.channel_batch_size = channel_batch_size
        
    def generate_grad_cam(self, events, class_idx, target_layer_idx):
        with grad_cam_context(self.model):
            target = get_conv_layers(self.model)[target_layer_idx]
            grad_cam = SNNGradCAM(self.model, target)
            return grad_cam.generate(events, class_idx=class_idx).detach()

    def generate_ablation_cam(self, events, class_idx, target_layer_idx):
        functional.reset_net(self.model)
        target = get_conv_layers(self.model)[target_layer_idx]
        ab_cam = AblationCAM(self.model, target, channel_batch_size=self.channel_batch_size)
        return ab_cam.generate(events, class_idx=class_idx).detach()

    def generate_mc_ablation_cam(self, events, class_idx, target_layer_idx):
        functional.reset_net(self.model)
        target = get_lif_layers(self.model)[target_layer_idx]
        mc_cam = MCAblationCAM(self.model, target, channel_batch_size=self.channel_batch_size, 
                               k_trials=self.k_trials, use_post_hook=False)
        return mc_cam.generate(events, class_idx=class_idx).detach()

    def generate_sam(self, events, target_layer_idx):
        functional.reset_net(self.model)
        target = get_lif_layers(self.model)[target_layer_idx]
        sam = SpikeActivationMap(self.model, target)
        return sam.generate(events).detach()

    def generate_all_cams(self, events, class_idx, conv_idx, lif_idx):
        """Generate CAMs using multiple methods."""
        cams = {}
        cams['GradCAM'] = self.generate_grad_cam(events, class_idx, conv_idx).cpu()
        cams['AblationCAM'] = self.generate_ablation_cam(events, class_idx, conv_idx).cpu()
        cams['MC_Ablation'] = self.generate_mc_ablation_cam(events, class_idx, lif_idx).cpu()
        cams['SAM'] = self.generate_sam(events, lif_idx).cpu()
        return cams


def calculate_pointing_game(cam, bboxes):
    """Check if the maximum point of the CAM lies inside any bounding box."""
    if isinstance(cam, torch.Tensor):
        cam = cam.numpy()
    
    y_max, x_max = np.unravel_index(cam.argmax(), cam.shape)
    
    hit = False
    for box in bboxes:
        xmin, ymin, xmax, ymax = box
        if xmin <= x_max <= xmax and ymin <= y_max <= ymax:
            hit = True
            break
    
    return hit, (x_max, y_max)


def get_scaled_bboxes(target, orig_size, new_size, dataset_choice, map_det_label_to_idx):
    """Extract and scale bboxes to model input size."""
    w_orig, h_orig = orig_size
    w_new, h_new = new_size
    scale_x = w_new / w_orig
    scale_y = h_new / h_orig
    
    boxes_by_class = defaultdict(list)
    
    if dataset_choice == 'voc':
        objects = target['annotation']['object']
        if not isinstance(objects, list):
            objects = [objects]
        
        for obj in objects:
            name = obj['name']
            idx = map_det_label_to_idx(name)
            if idx == -1:
                continue
            
            bnd = obj['bndbox']
            x1 = float(bnd['xmin']) * scale_x
            y1 = float(bnd['ymin']) * scale_y
            x2 = float(bnd['xmax']) * scale_x
            y2 = float(bnd['ymax']) * scale_y
            boxes_by_class[idx].append([x1, y1, x2, y2])
            
    elif dataset_choice == 'coco':
        for obj in target:
            cat_id = obj['category_id']
            idx = map_det_label_to_idx(cat_id)
            if idx == -1:
                continue
            
            x, y, w, h = obj['bbox']
            x1 = x * scale_x
            y1 = y * scale_y
            x2 = (x + w) * scale_x
            y2 = (y + h) * scale_y
            boxes_by_class[idx].append([x1, y1, x2, y2])
    
    return boxes_by_class


def denormalize(tensor):
    """Un-normalize ImageNet tensors for display."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.permute(1, 2, 0).numpy()
    img = std * img + mean
    return np.clip(img, 0, 1)


def save_visualization(sample, methods, output_path):
    """Save visualization for a single sample."""
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods + 1, figsize=(4 * (n_methods + 1), 4))
    
    # Original image with GT boxes
    ax_orig = axes[0]
    img_np = denormalize(sample['img'])
    ax_orig.imshow(img_np)
    ax_orig.set_title(f"GT: {sample['class_name']}\n{sample['layer_info']}")
    ax_orig.axis('off')
    
    for box in sample['bboxes']:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, 
                                   edgecolor='g', facecolor='none')
        ax_orig.add_patch(rect)
    
    # CAMs
    for j, method in enumerate(methods):
        ax = axes[j + 1]
        cam_data = sample['cams'][method]
        heatmap = cam_data['map']
        hit = cam_data['hit']
        max_pt = cam_data['max_pt']
        
        ax.imshow(img_np)
        ax.imshow(heatmap, cmap='jet', alpha=0.5)
        ax.plot(max_pt[0], max_pt[1], 'rx', markersize=10, markeredgewidth=2)
        
        for box in sample['bboxes']:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, 
                                       edgecolor='w', facecolor='none', linestyle='--')
            ax.add_patch(rect)
        
        status = "HIT" if hit else "MISS"
        color = "green" if hit else "red"
        ax.set_title(f"{method}\n{status}", color=color, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def gather_results(local_results, world_size, is_distributed):
    """Gather results from all processes."""
    if not is_distributed or world_size == 1:
        return local_results
    
    # Serialize results to JSON string
    results_str = json.dumps(local_results)
    
    # Gather all results
    all_results_str = [None] * world_size
    dist.all_gather_object(all_results_str, results_str)
    
    # Merge results
    merged_results = defaultdict(lambda: defaultdict(lambda: {'hits': 0, 'total': 0}))
    for result_str in all_results_str:
        result = json.loads(result_str)
        for layer_key, method_res in result.items():
            for method, res in method_res.items():
                merged_results[layer_key][method]['hits'] += res['hits']
                merged_results[layer_key][method]['total'] += res['total']
    
    return dict(merged_results)


def main():
    args = parse_args()
    
    # Setup distributed
    rank, world_size, local_rank, is_distributed = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    if is_main_process(rank):
        print(f"Running on {world_size} GPU(s)")
        print(f"Device: {device}")
        print(f"Arguments: {args}")
    
    # Set seeds
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)
    
    # Setup output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{args.dataset}_{timestamp}"
    images_dir = output_dir / "visualizations"
    
    if is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        print(f"Results will be saved to: {output_dir}")
    
    # Synchronize before continuing
    if is_distributed:
        dist.barrier()
    
    # Configuration
    config = setup_config_dictionary()
    INPUT_SIZE = (224, 224)
    
    if args.dataset == 'voc':
        config["DATASET_NAME"] = "voc_multilabel"
        DETECTION_DATASET_NAME = "voc_detection"
        config["MODEL_NAME"] = "spiking_vgg16_bn"
        NUM_CLASSES = 20
    else:  # coco
        config["DATASET_NAME"] = "coco_multilabel"
        DETECTION_DATASET_NAME = "coco"
        config["MODEL_NAME"] = "spiking_vgg16_bn"
        NUM_CLASSES = 80
    
    config["N_TIME_BINS"] = args.n_time_bins
    
    if is_main_process(rank):
        print(f"Configured for {args.dataset.upper()} dataset with model {config['MODEL_NAME']}")
    
    # Load model
    spiking_neuron = neuron.IFNode
    model = get_model(
        model_name=config["MODEL_NAME"],
        num_classes=NUM_CLASSES,
        in_channels=3,
        spiking_neuron=spiking_neuron,
        pretrained=False,
        detach_reset=False,
        surrogate_function=surrogate.ATan(),
    )
    
    functional.set_step_mode(model, 'm')
    functional.set_backend(model, "torch", instance=spiking_neuron)
    model.to(device)
    
    load_model(model, config["MODEL_SAVE_DIR"], config["DATASET_NAME"])
    model.eval()
    
    if is_main_process(rank):
        convs = get_conv_layers(model)
        lifs = get_lif_layers(model)
        print(f"Model loaded with {len(convs)} Conv layers and {len(lifs)} Spiking layers.")
    
    # Load datasets
    if is_main_process(rank):
        print(f"Loading {config['DATASET_NAME']} for class mappings...")
    
    ds_cls, _ = get_datasets(config["DATASET_NAME"], n_time_bins=1)
    
    if args.dataset == 'voc':
        class_names = ds_cls.classes
        class_to_idx = ds_cls.class_to_idx
        def map_det_label_to_idx(label_name):
            return class_to_idx.get(label_name, -1)
    else:  # coco
        class_names = ds_cls.classes
        cat_id_to_idx = ds_cls.cat_id_to_idx
        def map_det_label_to_idx(cat_id):
            return cat_id_to_idx.get(cat_id, -1)
    
    if is_main_process(rank):
        print(f"Loading {DETECTION_DATASET_NAME} for evaluation...")
    
    _, detection_ds_test = get_datasets(DETECTION_DATASET_NAME, n_time_bins=1)
    
    if is_main_process(rank):
        print(f"Detection dataset loaded with {len(detection_ds_test)} samples.")
    
    # Setup evaluator
    evaluator = CAMEvaluator(model, device, k_trials=args.k_trials, 
                             channel_batch_size=args.channel_batch_size)
    
    # Layer pairs to evaluate
    requested_layer_pairs = [
        (6, 6),
        (9, 9),
        (12, 12),
    ]
    
    # Distribute samples across processes
    all_indices = list(range(len(detection_ds_test)))
    random.Random(args.seed).shuffle(all_indices)  # Same shuffle across all processes
    
    # Each process handles a subset
    indices_per_process = len(all_indices) // world_size
    start_idx = rank * indices_per_process
    end_idx = start_idx + indices_per_process if rank < world_size - 1 else len(all_indices)
    local_indices = all_indices[start_idx:end_idx]
    
    # Adjust max samples per process
    max_samples_per_process = args.max_samples // world_size
    if rank < args.max_samples % world_size:
        max_samples_per_process += 1
    
    if is_main_process(rank):
        print(f"Starting Pointing Game evaluation...")
        print(f"Each process will handle up to {max_samples_per_process} samples")
    
    # Results storage
    results = defaultdict(lambda: defaultdict(lambda: {'hits': 0, 'total': 0}))
    visual_samples = []
    
    # Main evaluation loop
    sample_count = 0
    pbar = tqdm(local_indices, desc=f"GPU {rank} - Evaluating", 
                disable=not is_main_process(rank), total=len(local_indices))
    
    for i in pbar:
        if sample_count >= max_samples_per_process:
            break
        
        img_tensor, target = detection_ds_test[i]
        
        # Get original size
        if args.dataset == 'voc':
            orig_w = int(target['annotation']['size']['width'])
            orig_h = int(target['annotation']['size']['height'])
        else:  # coco
            orig_w, orig_h = 640, 400
        
        # Prepare input
        input_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
        input_tensor = input_tensor.repeat(config["N_TIME_BINS"], 1, 1, 1, 1)
        input_tensor = input_tensor.to(device)
        
        # Model prediction
        functional.reset_net(model)
        with torch.no_grad():
            outputs = model(input_tensor)
            time_sum = outputs.sum(dim=0)
            probs = torch.sigmoid(time_sum)
        
        # Identify correct classes
        gt_boxes_map = get_scaled_bboxes(target, (orig_w, orig_h), INPUT_SIZE, 
                                          args.dataset, map_det_label_to_idx)
        gt_classes = set(gt_boxes_map.keys())
        pred_classes = set(torch.where(probs[0] > 0.5)[0].cpu().numpy())
        correct_classes = gt_classes.intersection(pred_classes)
        
        if len(correct_classes) < args.min_correct_classes:
            continue
        
        for cls_idx in correct_classes:
            bboxes = gt_boxes_map[cls_idx]
            
            for (c_idx, l_idx) in requested_layer_pairs:
                layer_key = f"C{c_idx}_L{l_idx}"
                cams = evaluator.generate_all_cams(input_tensor, cls_idx, c_idx, l_idx)
                
                sample_result = {
                    'img': img_tensor.cpu(),
                    'class_name': class_names[cls_idx],
                    'class_idx': cls_idx,
                    'bboxes': bboxes,
                    'cams': {},
                    'layer_info': layer_key,
                    'sample_idx': i
                }
                
                for method_name, cam_data in cams.items():
                    cam_agg = cam_data.mean(dim=0).numpy()
                    
                    if cam_agg.shape != INPUT_SIZE:
                        cam_agg = cv2.resize(cam_agg, INPUT_SIZE)
                    
                    hit, max_pt = calculate_pointing_game(cam_agg, bboxes)
                    
                    results[layer_key][method_name]['hits'] += int(hit)
                    results[layer_key][method_name]['total'] += 1
                    
                    sample_result['cams'][method_name] = {
                        'map': cam_agg,
                        'hit': hit,
                        'max_pt': max_pt
                    }
                
                visual_samples.append(sample_result)
        
        sample_count += 1
        pbar.set_postfix({'processed': sample_count})
    
    # Convert defaultdict to regular dict for serialization
    results_dict = {k: dict(v) for k, v in results.items()}
    
    # Gather results from all processes
    if is_distributed:
        dist.barrier()
    
    all_results = gather_results(results_dict, world_size, is_distributed)
    
    # Save visualizations (each process saves its own)
    if args.save_all_visualizations:
        methods = ['GradCAM', 'AblationCAM', 'MC_Ablation', 'SAM']
        for idx, sample in enumerate(visual_samples):
            filename = f"gpu{rank}_sample{sample['sample_idx']}_class{sample['class_idx']}_{sample['layer_info']}.png"
            save_visualization(sample, methods, images_dir / filename)
    
    if is_distributed:
        dist.barrier()
    
    # Main process saves statistics
    if is_main_process(rank):
        # Create results DataFrame
        data = []
        for layer_key, method_res in all_results.items():
            for method, res in method_res.items():
                acc = res['hits'] / res['total'] if res['total'] > 0 else 0
                data.append({
                    'Layer': layer_key,
                    'Method': method,
                    'Pointing_Game_Acc': acc,
                    'Pointing_Game_Acc_Pct': f"{acc:.2%}",
                    'Hits': res['hits'],
                    'Total': res['total']
                })
        
        df = pd.DataFrame(data)
        df = df.sort_values(['Layer', 'Method'])
        
        # Save CSV
        csv_path = output_dir / "pointing_game_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
        
        # Save JSON
        json_path = output_dir / "pointing_game_results.json"
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Raw results saved to: {json_path}")
        
        # Save config
        config_path = output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        print(f"Config saved to: {config_path}")
        
        # Print results
        print("\n" + "="*60)
        print("Pointing Game Results (Correctly Classified Samples Only):")
        print("="*60)
        print(df.to_string(index=False))
        
        # Create summary plot
        fig, ax = plt.subplots(figsize=(12, 6))
        pivot_df = df.pivot(index='Layer', columns='Method', values='Pointing_Game_Acc')
        pivot_df.plot(kind='bar', ax=ax)
        ax.set_ylabel('Pointing Game Accuracy')
        ax.set_title('CAM Method Comparison - Pointing Game')
        ax.legend(title='Method')
        ax.set_ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / "summary_plot.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nSummary plot saved to: {output_dir / 'summary_plot.png'}")
        print(f"Visualizations saved to: {images_dir}")
        print(f"\nTotal visualizations saved: {len(list(images_dir.glob('*.png')))}")
    
    # Cleanup
    cleanup_distributed(is_distributed)
    
    if is_main_process(rank):
        print("\nEvaluation complete!")


if __name__ == "__main__":
    main()