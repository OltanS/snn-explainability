import os
import sys
import json
import random
from datetime import datetime
from collections import defaultdict
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

from src.utils.args_utils import parse_args
from src.utils.data_utils import get_datasets
from src.utils.model_utils import get_model, get_conv_layers, get_lif_layers
from src.utils.save_utils import load_model
from src.utils.distributed_utils import ddp_setup, ddp_cleanup

# CAM imports
from src.cams.generation_utils import (
    generate_grad_cam, 
    generate_ablation_cam, 
    generate_mc_ablation_cam, 
    generate_sam, 
    generate_noise_grad_cam,
    generate_mc_ablation_cam_transient,
)

# ============================================================================
# Distributed Utilities
# ============================================================================
def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0

def is_main_process():
    return get_rank() == 0

# ============================================================================
# Model Utilities
# ============================================================================
def get_valid_layer_pairs(model, requested_indices):
    """
    Filter layer pairs to only include valid indices for the model.
    Assumes we want the i-th Conv layer paired with the i-th LIF layer.
    """
    n_conv = len(get_conv_layers(model))
    n_lif = len(get_lif_layers(model))
    
    valid = []
    for idx in requested_indices:
        if idx < n_conv and idx < n_lif:
            valid.append((idx, idx))
            
    if is_main_process():
        print(f"Layer pairs: Requested indices {requested_indices}, Valid pairs {valid} "
              f"(Model: {n_conv} Conv, {n_lif} LIF)")
    return valid

# ============================================================================
# CAM Generators
# ============================================================================
def generate_random_cam(events):
    T, _, _, H, W = events.shape
    return torch.rand((T, H, W), device=events.device)

def generate_all_cams(model, events, class_idx, conv_idx, lif_idx, args):
    """Generate all CAM types for a given input and class."""
    cams = {}
    if 'grad' in args.cam_methods:
        cams['grad'] = generate_grad_cam(events, class_idx, conv_idx, model)
    if 'ablation' in args.cam_methods:
        cams['ablation'] = generate_ablation_cam(events, class_idx, conv_idx, model, batch_size=args.channel_batch_size)
    if 'mc_ablation' in args.cam_methods:
        cams['mc_ablation'], _ = generate_mc_ablation_cam(events, class_idx, lif_idx, model, k_trials=args.k_trials, batch_size=args.channel_batch_size)
    if 'sam' in args.cam_methods:
        cams['sam'] = generate_sam(events, lif_idx, model)
    if 'noise_grad_surrogate' in args.cam_methods:
        cams['noise_grad_surrogate'] = generate_noise_grad_cam(events, class_idx, conv_idx, model, k_trials=args.k_trials, batch_size=args.channel_batch_size, noise_type="surrogate")
    if 'noise_grad_uniform' in args.cam_methods:
        cams['noise_grad_uniform'] = generate_noise_grad_cam(events, class_idx, conv_idx, model, k_trials=args.k_trials, batch_size=args.channel_batch_size, noise_type="uniform")
    if 'noise_grad_gaussian' in args.cam_methods:
        cams['noise_grad_gaussian'] = generate_noise_grad_cam(events, class_idx, conv_idx, model, k_trials=args.k_trials, batch_size=args.channel_batch_size, noise_type="gaussian")
    if 'mc_ablation_transient_surrogate' in args.cam_methods:
        cams['mc_ablation_transient_surrogate'], _ = generate_mc_ablation_cam_transient(events, class_idx, lif_idx, model, k_trials=args.k_trials, batch_size=args.channel_batch_size, noise_type="surrogate")
    if 'mc_ablation_transient_uniform' in args.cam_methods:
        cams['mc_ablation_transient_uniform'], _ = generate_mc_ablation_cam_transient(events, class_idx, lif_idx, model, k_trials=args.k_trials, batch_size=args.channel_batch_size, noise_type="uniform")
    if 'random' in args.cam_methods:
        cams['random'] = generate_random_cam(events)
    
    # Move all to CPU to save GPU memory
    for k, v in cams.items():
        cams[k] = v.detach().cpu()
        
    return cams

# ============================================================================
# Pointing Game Logic
# ============================================================================
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


def save_all_layers_visualization(sample_data, methods, output_path):
    """
    Save visualization grid: Rows (Layers) x Columns (Classes -> [Input + Methods]).
    sample_data: dict containing 'img', 'layers': {layer_key: {class_idx: {'class_name', 'bboxes', 'cams': {method: data}}}}
    """
    img_np = denormalize(sample_data['img'])
    
    layers = sorted(list(sample_data['layers'].keys()))
    if not layers: return
    
    # Assume consistent classes across layers
    first_layer = sample_data['layers'][layers[0]]
    classes = sorted(list(first_layer.keys()))
    if not classes: return
    
    n_rows = len(layers)
    n_subcols = len(methods) + 1 # Input (with GT) + methods
    n_cols = len(classes) * n_subcols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)
    
    for r, layer_key in enumerate(layers):
        for c_grp, class_idx in enumerate(classes):
            col_base = c_grp * n_subcols
            class_data = sample_data['layers'][layer_key][class_idx]
            bboxes = class_data['bboxes']
            class_name = class_data['class_name']
            
            # 1. Plot Input with GT boxes
            ax_inp = axes[r, col_base]
            ax_inp.imshow(img_np)
            
            for box in bboxes:
                x1, y1, x2, y2 = box
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, 
                                           edgecolor='g', facecolor='none')
                ax_inp.add_patch(rect)
            
            # Set titles/labels
            if r == 0:
                ax_inp.set_title(f"Class: {class_name}\nGT")
            
            if col_base == 0: # First column of the row
                ax_inp.set_ylabel(layer_key, rotation=90, size='large')
            
            ax_inp.axis('off')
            
            # 2. Plot Methods
            for m_idx, method in enumerate(methods):
                if method not in class_data['cams']:
                    continue
                    
                cam_data = class_data['cams'][method]
                heatmap = cam_data['map']
                hit = cam_data['hit']
                max_pt = cam_data['max_pt']
                
                ax = axes[r, col_base + 1 + m_idx]
                ax.imshow(img_np)
                ax.imshow(heatmap, cmap='jet', alpha=0.5)
                ax.plot(max_pt[0], max_pt[1], 'rx', markersize=10, markeredgewidth=2)
                
                for box in bboxes:
                    x1, y1, x2, y2 = box
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, 
                                               edgecolor='w', facecolor='none', linestyle='--')
                    ax.add_patch(rect)
                
                status = "HIT" if hit else "MISS"
                color = "green" if hit else "red"
                
                if r == 0:
                    ax.set_title(f"{method}\n{status}", color=color, fontweight='bold')
                else:
                    ax.set_title(f"{status}", color=color, fontweight='bold')
                
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
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    is_distributed = world_size > 1

    if is_distributed:
        ddp_setup(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if is_main_process():
        print(f"Running on {world_size} GPU(s)")
        print(f"Device: {device}")
        print(f"Arguments: {args}")
    
    # Set seeds
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)
    
    # Setup output directories
    output_dir = Path("./img") / f"{args.dataset_name}_pointing_game"
    
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Results will be saved to: {output_dir}")
    
    # Synchronize before continuing
    if is_distributed:
        dist.barrier()
    
    # Configuration
    INPUT_SIZE = (224, 224)
    
    if 'voc' in args.dataset_name:
        dataset_choice = 'voc'
        DETECTION_DATASET_NAME = "voc_detection"
        NUM_CLASSES = 20
    elif 'coco' in args.dataset_name:
        dataset_choice = 'coco'
        DETECTION_DATASET_NAME = "coco"
        NUM_CLASSES = 80
    else:
        if is_main_process():
            print(f"Warning: Dataset {args.dataset_name} not recognized as VOC or COCO. Defaulting to VOC logic.")
        dataset_choice = 'voc'
        DETECTION_DATASET_NAME = "voc_detection"
        NUM_CLASSES = 20
    
    if is_main_process():
        print(f"Configured for {dataset_choice.upper()} dataset with model {args.model_name}")
    
    # Load model
    spiking_neuron = neuron.IFNode
    model = get_model(
        model_name=args.model_name,
        num_classes=NUM_CLASSES,
        in_channels=3,
        spiking_neuron=spiking_neuron,
        pretrained=False,
        readout=args.readout,
        detach_reset=False,
        surrogate_function=surrogate.ATan(),
    )
    
    functional.set_step_mode(model, 'm')
    functional.set_backend(model, "torch", instance=spiking_neuron)
    model.to(device)
    
    load_model(model, args.model_save_dir, args.dataset_name)
    model.eval()
    
    if is_main_process():
        convs = get_conv_layers(model)
        lifs = get_lif_layers(model)
        print(f"Model loaded with {len(convs)} Conv layers and {len(lifs)} Spiking layers.")
    
    # Load datasets
    if is_main_process():
        print(f"Loading {args.dataset_name} for class mappings...")
    
    ds_cls, _ = get_datasets(args.dataset_name, args.data_dir, n_time_bins=1)
    
    if dataset_choice == 'voc':
        class_names = ds_cls.classes
        class_to_idx = ds_cls.class_to_idx
        def map_det_label_to_idx(label_name):
            return class_to_idx.get(label_name, -1)
    else:  # coco
        class_names = ds_cls.classes
        cat_id_to_idx = ds_cls.cat_id_to_idx
        def map_det_label_to_idx(cat_id):
            return cat_id_to_idx.get(cat_id, -1)
    
    if is_main_process():
        print(f"Loading {DETECTION_DATASET_NAME} for evaluation...")
    
    _, detection_ds_test = get_datasets(DETECTION_DATASET_NAME, args.data_dir, n_time_bins=1)
    
    if is_main_process():
        print(f"Detection dataset loaded with {len(detection_ds_test)} samples.")
    
    # Layer pairs to evaluate
    layer_pairs = get_valid_layer_pairs(model, args.target_layers)
    
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
    
    if is_main_process():
        print(f"Starting Pointing Game evaluation...")
        print(f"Each process will handle up to {max_samples_per_process} samples")
    
    # Results storage
    results = defaultdict(lambda: defaultdict(lambda: {'hits': 0, 'total': 0}))    

    # Main evaluation loop
    sample_count = 0
    vis_count = 0
    min_correct_classes = 2 # Default
    
    pbar = tqdm(local_indices, desc=f"GPU {rank} - Evaluating", 
                disable=not is_main_process(), total=max_samples_per_process)
    
    for i in pbar:
        if sample_count >= max_samples_per_process:
            break
        
        img_tensor, target = detection_ds_test[i]
        
        # Get original size
        if dataset_choice == 'voc':
            orig_w = int(target['annotation']['size']['width'])
            orig_h = int(target['annotation']['size']['height'])
        else:  # coco
            orig_w, orig_h = 640, 400
        
        # Prepare input
        input_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
        input_tensor = input_tensor.repeat(args.n_time_bins, 1, 1, 1, 1)
        input_tensor = input_tensor.to(device)
        
        # Model prediction
        functional.reset_net(model)
        with torch.no_grad():
            outputs = model(input_tensor)
            time_sum = outputs.sum(dim=0)
            probs = torch.sigmoid(time_sum)
        
        # Identify correct classes
        gt_boxes_map = get_scaled_bboxes(target, (orig_w, orig_h), INPUT_SIZE, 
                                          dataset_choice, map_det_label_to_idx)
        gt_classes = set(gt_boxes_map.keys())
        pred_classes = set(torch.where(probs[0] > 0.5)[0].cpu().numpy())
        correct_classes = gt_classes.intersection(pred_classes)
        
        if len(correct_classes) < min_correct_classes:
            continue
        
        # Prepare visualization structure for this sample if needed
        should_visualize = vis_count < args.num_visualizations
        if should_visualize:
            sample_data = {
                'img': img_tensor.cpu(),
                'sample_idx': i,
                'layers': defaultdict(dict)
            }

        for cls_idx in correct_classes:
            bboxes = gt_boxes_map[cls_idx]
            
            for (c_idx, l_idx) in layer_pairs:
                layer_key = f"C{c_idx}_L{l_idx}"
                cams = generate_all_cams(model, input_tensor, cls_idx, c_idx, l_idx, args)
                
                # Store data for visualization
                if should_visualize:
                    sample_data['layers'][layer_key][cls_idx] = {
                        'class_name': class_names[cls_idx],
                        'bboxes': bboxes,
                        'cams': {}
                    }

                for method_name, cam_tensor in cams.items():
                    cam_agg = cam_tensor.mean(dim=0).numpy()
                    
                    if cam_agg.shape != INPUT_SIZE:
                        cam_agg = cv2.resize(cam_agg, INPUT_SIZE)
                    
                    hit, max_pt = calculate_pointing_game(cam_agg, bboxes)
                    
                    results[layer_key][method_name]['hits'] += int(hit)
                    results[layer_key][method_name]['total'] += 1
                    
                    if should_visualize:
                        sample_data['layers'][layer_key][cls_idx]['cams'][method_name] = {
                            'map': cam_agg,
                            'hit': hit,
                            'max_pt': max_pt
                        }
        
        if should_visualize:
            filename = f"gpu{rank}_sample{i}_pointing_game.png"
            save_all_layers_visualization(sample_data, args.cam_methods, output_dir / filename)
            vis_count += 1
        
        sample_count += 1
        pbar.set_postfix({'processed': sample_count})
    
    # Convert defaultdict to regular dict for serialization
    results_dict = {k: dict(v) for k, v in results.items()}
    
    # Gather results from all processes
    if is_distributed:
        dist.barrier()
    
    all_results = gather_results(results_dict, world_size, is_distributed)
    
    if is_distributed:
        dist.barrier()
    
    # Main process saves statistics
    if is_main_process():
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
        print(f"Visualizations saved to: {output_dir}")
    
    # Cleanup
    if is_distributed:
        ddp_cleanup()

    if is_main_process():
        print("\nEvaluation complete.")


if __name__ == "__main__":
    main()