import os
import gc
from tqdm.auto import tqdm
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from spikingjelly.activation_based import neuron, surrogate, functional, layer

# Local imports
from src.utils.args_utils import parse_args
from src.utils.data_utils import get_dataloaders
from src.utils.model_utils import get_model, get_conv_layers, get_lif_layers
from src.utils.save_utils import load_model
from src.cams.generation_utils import (
    generate_grad_cam, 
    generate_ablation_cam, 
    generate_mc_ablation_cam, 
    generate_sam, 
    generate_noise_grad_cam,
)
from src.utils.distributed_utils import ddp_setup, ddp_cleanup
from src.noisy_node import noise_injection

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
    if 'random' in args.cam_methods:
        cams['random'] = generate_random_cam(events)
    return cams

# ============================================================================
# Scoring Functions
# ============================================================================
def get_scores(model, inputs, class_idx, is_multilabel=False):
    """Compute activation and confidence scores for a class."""
    functional.reset_net(model)
    with torch.no_grad():
        out = model(inputs)
    logits = out.sum(dim=0)[0]
    
    if is_multilabel:
        conf = torch.sigmoid(logits)
    else:
        conf = F.softmax(logits, dim=0)
    
    return logits[class_idx].item(), conf[class_idx].item()

def get_mc_scores(model, inputs, class_idx, k_trials, is_multilabel=False):
    """Compute Monte Carlo smoothed scores by injecting noise into LIF layers."""
    act_sum, conf_sum = 0.0, 0.0
    
    with noise_injection(model, noise_type="surrogate"):
        with torch.no_grad():
            for _ in range(k_trials):
                functional.reset_net(model)
                out = model(inputs)
                logits = out.sum(dim=0)[0]
                
                if is_multilabel:
                    conf = torch.sigmoid(logits)
                else:
                    conf = F.softmax(logits, dim=0)
                
                act_sum += logits[class_idx].item()
                conf_sum += conf[class_idx].item()
    
    functional.reset_net(model)
    
    return act_sum / k_trials, conf_sum / k_trials

# ============================================================================
# Masking & Metrics
# ============================================================================
def mask_events(events, cam, keep_percent):
    """Mask events keeping top 'keep_percent' of pixels per timestep."""
    T, H, W = cam.shape
    cam_flat = cam.view(T, -1)
    num_keep = max(1, int(H * W * keep_percent))
    
    _, indices = torch.topk(cam_flat, num_keep, dim=1)
    mask_flat = torch.zeros_like(cam_flat)
    mask_flat.scatter_(1, indices, 1.0)
    
    return events * mask_flat.view(T, 1, 1, H, W)

def compute_cam_variance(cam):
    """Compute spatial variance averaged over time."""
    if isinstance(cam, torch.Tensor):
        cam = cam.detach().cpu().numpy()
    return float(cam.var(axis=(1, 2)).mean())

def compute_metrics(base_scores, masked_scores):
    """Compute drop and increase metrics from base and masked scores."""
    base_act, base_conf = base_scores
    masked_act, masked_conf = masked_scores
    
    return {
        'drop_conf': max(0, base_conf - masked_conf),
        'drop_act': max(0, base_act - masked_act),
        'increase_conf': 1.0 if masked_conf > base_conf else 0.0,
        'increase_act': 1.0 if masked_act > base_act else 0.0,
        'masked_conf': masked_conf,
        'masked_act': masked_act,
        'base_conf': base_conf,
        'base_act': base_act,
    }

# ============================================================================
# Visualization
# ============================================================================
def save_all_layers_visualization(events, all_layer_cams, save_dir, sample_idx):
    """
    Save a visualization grid: 
    Rows (Layers) x Columns (Classes -> [Input + Methods]).
    all_layer_cams: dict {layer_label: {class_idx: {method_name: cam_tensor}}}
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Sum over time & channels for input visualization
    input_img = events.sum(dim=(0, 1, 2)).cpu().numpy()
    
    layers = sorted(list(all_layer_cams.keys()))
    if not layers: return
    
    # Assume consistent classes and methods across layers
    first_layer = all_layer_cams[layers[0]]
    classes = sorted(list(first_layer.keys()))
    if not classes: return
    
    methods = list(first_layer[classes[0]].keys())
    
    n_rows = len(layers)
    n_subcols = len(methods) + 1 # Input + methods
    n_cols = len(classes) * n_subcols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows), squeeze=False)
    
    for r, layer_label in enumerate(layers):
        for c_grp, class_idx in enumerate(classes):
            # Base column index for this class group
            col_base = c_grp * n_subcols
            
            cams = all_layer_cams[layer_label][class_idx]
            
            # 1. Plot Input
            ax_inp = axes[r, col_base]
            ax_inp.imshow(input_img, cmap='hot', interpolation='nearest')
            
            # Set titles/labels
            if r == 0:
                ax_inp.set_title(f"Class {class_idx}\nInput")
            
            if col_base == 0: # First column of the row
                ax_inp.set_ylabel(layer_label, rotation=90, size='large')
            
            ax_inp.set_xticks([])
            ax_inp.set_yticks([])
            
            # 2. Plot Methods
            for m_idx, method in enumerate(methods):
                cam = cams[method]
                # Aggregate over time
                cam_map = cam.sum(dim=0).cpu().numpy()
                cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min() + 1e-8)
                
                ax = axes[r, col_base + 1 + m_idx]
                ax.imshow(input_img, cmap='gray', alpha=0.5)
                ax.imshow(cam_map, cmap='jet', alpha=0.5)
                
                if r == 0:
                    ax.set_title(method)
                
                ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/sample_{sample_idx}_all_layers.svg", dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================================
# Label Handling
# ============================================================================
def get_target_classes(labels, is_multilabel):
    """Extract target class indices from labels."""
    if is_multilabel:
        # labels is (B, num_classes) multi-hot tensor
        return torch.where(labels[0] > 0.5)[0].tolist()
    else:
        return [labels[0].item()]

def check_prediction_correct(model, events, target_classes, is_multilabel, threshold=0.5):
    """Check if model correctly predicts the target class(es)."""
    functional.reset_net(model)
    with torch.no_grad():
        out = model(events)
    logits = out.sum(dim=0)[0]
    
    if is_multilabel:
        probs = torch.sigmoid(logits)
        predicted = (probs > threshold).nonzero(as_tuple=True)[0].tolist()
        # Check if all target classes are predicted
        return all(tc in predicted for tc in target_classes)
    else:
        pred_class = logits.argmax().item()
        return pred_class == target_classes[0]

# ============================================================================
# Main Processing
# ============================================================================
def process_sample(model, events, target_classes, layer_pairs, args, is_multilabel, collect_cams=False):
    """Process a single sample and return results for all classes and layer pairs."""
    results = []
    variances = []
    cams_to_save = {}
    
    for class_idx in target_classes:
        base_scores = get_scores(model, events, class_idx, is_multilabel)
        base_mc_scores = get_mc_scores(model, events, class_idx, args.k_trials, is_multilabel)
        
        for conv_idx, lif_idx in layer_pairs:
            layer_label = f"L{conv_idx}"
            cams = generate_all_cams(model, events, class_idx, conv_idx, lif_idx, args)
            
            if collect_cams:
                if layer_label not in cams_to_save:
                    cams_to_save[layer_label] = {}
                cams_to_save[layer_label][class_idx] = cams
            
            for method in args.cam_methods:
                if method not in cams: continue
                cam = cams[method]
                
                variances.append({
                    'Layer': layer_label,
                    'Method': method,
                    'Class': class_idx,
                    'Variance': compute_cam_variance(cam),
                })
                
                masked_input = mask_events(events, cam, args.top_percent)
                masked_scores = get_scores(model, masked_input, class_idx, is_multilabel)
                masked_mc_scores = get_mc_scores(model, masked_input, class_idx, args.k_trials, is_multilabel)
                
                std_metrics = compute_metrics(base_scores, masked_scores)
                mc_metrics = compute_metrics(base_mc_scores, masked_mc_scores)
                
                results.append({
                    'Layer': layer_label,
                    'Method': method,
                    'Class': class_idx,
                    'Masking %': args.top_percent * 100,
                    'Confidence Drop': std_metrics['drop_conf'],
                    'Activation Drop': std_metrics['drop_act'],
                    'Confidence Increase Rate': std_metrics['increase_conf'],
                    'Activation Increase Rate': std_metrics['increase_act'],
                    'Masked Confidence': std_metrics['masked_conf'],
                    'Masked Activation': std_metrics['masked_act'],
                    'Base Confidence': std_metrics['base_conf'],
                    'Base Activation': std_metrics['base_act'],
                    'MC Confidence Drop': mc_metrics['drop_conf'],
                    'MC Activation Drop': mc_metrics['drop_act'],
                    'MC Confidence Increase Rate': mc_metrics['increase_conf'],
                    'MC Activation Increase Rate': mc_metrics['increase_act'],
                    'MC Masked Confidence': mc_metrics['masked_conf'],
                    'MC Masked Activation': mc_metrics['masked_act'],
                    'MC Base Confidence': mc_metrics['base_conf'],
                    'MC Base Activation': mc_metrics['base_act'],
                })
            
            if not collect_cams:
                del cams
    
    return results, variances, cams_to_save

def main():
    args = parse_args()

    # --- Distributed Setup ---
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))

    is_distributed = world_size > 1

    if is_distributed:
        ddp_setup(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        print(f"DDP initialized: Rank {rank} (Local {local_rank})")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Non-distributed mode on {device}")

    # --- Configuration ---
    is_multilabel = args.dataset_name.endswith("_composite") or args.dataset_name.endswith("_multilabel")
    
    if is_main_process():
        print(f"Seed: {torch.seed()}")
        print(f"Dataset: {args.dataset_name} (multilabel: {is_multilabel})")
        print(f"Model: {args.model_name}")

    # --- Data Loading ---
    train_loader, test_loader = get_dataloaders(
        args.dataset_name,
        args.data_dir,
        batch_size=1,
        n_time_bins=args.n_time_bins,
        on_ddp=is_distributed
    )

    sample_events, sample_labels = next(iter(train_loader))
    is_single_image = sample_events.dim() == 4
    C = sample_events.shape[1] if is_single_image else sample_events.shape[2]
    
    # Get number of classes
    ds = train_loader.dataset
    if hasattr(ds, 'classes'):
        num_classes = len(ds.classes)
    elif hasattr(ds, 'num_classes'):
        num_classes = ds.num_classes
    elif hasattr(ds, 'base_dataset'):
        num_classes = len(ds.base_dataset.classes)
    else:
        num_classes = len(ds.dataset.classes)

    # --- Model Setup ---
    spiking_neuron = neuron.IFNode if is_single_image else neuron.LIFNode
    
    if is_main_process():
        print(f"Spiking neuron: {spiking_neuron.__name__}")

    model_params = {
        "model_name": args.model_name,
        "num_classes": num_classes,
        "in_channels": C,
        "spiking_neuron": spiking_neuron,
        "pretrained": False,
        "detach_reset": False,
        "surrogate_function": surrogate.ATan(),
    }
    if not is_single_image:
        model_params["tau"] = args.tau

    model = get_model(**model_params)
    functional.set_step_mode(model, 'm')
    functional.set_backend(model, "torch", instance=spiking_neuron)
    model.to(device)
    
    load_model(model, args.model_save_dir, args.dataset_name)
    model.eval()
    
    if is_main_process():
        print("Model loaded.")

    # --- Evaluation Configuration ---
    handled_samples = args.max_samples
    if is_distributed:
        handled_samples = handled_samples // dist.get_world_size()

    layer_pairs = get_valid_layer_pairs(model, args.target_layers)
    
    # --- Processing Loop ---
    local_results = []
    local_variances = []
    sample_count = 0
    vis_count = 0
    vis_dir = f"img/{args.dataset_name}"

    pbar = tqdm(total=handled_samples, desc="Processing") if is_main_process() else None

    for events, labels in test_loader:
        if sample_count >= handled_samples:
            break
        
        if events.dim() == 4:
            events = events.unsqueeze(1).repeat(1, args.n_time_bins, 1, 1, 1)
        events = events.transpose(0, 1).to(device)
        labels = labels.to(device)
        
        target_classes = get_target_classes(labels, is_multilabel)
        
        if not target_classes:
            # if pbar: pbar.update(1)
            continue

        # if len(target_classes) != 2:
        #     if pbar: pbar.update(1)
        #     continue
        
        if not check_prediction_correct(model, events, target_classes, is_multilabel):
            # if pbar: pbar.update(1)
            continue
        
        should_visualize = is_main_process() and vis_count < args.num_visualizations

        results, variances, cams_to_save = process_sample(
            model, events, target_classes, layer_pairs, args, is_multilabel,
            collect_cams=should_visualize
        )
        
        local_results.extend(results)
        local_variances.extend(variances)
        
        if should_visualize and cams_to_save:
            save_all_layers_visualization(
                events, cams_to_save, vis_dir, sample_count
            )
            vis_count += 1
        
        sample_count += 1
        if pbar: pbar.update(1)
        
        gc.collect()
        torch.cuda.empty_cache()

    if pbar: pbar.close()
    print(f"Rank {get_rank()}: Processed {sample_count} samples.")

    # --- Aggregate Results ---
    if is_distributed:
        all_results = [None] * dist.get_world_size()
        all_variances = [None] * dist.get_world_size()
        
        dist.all_gather_object(all_results, local_results)
        dist.all_gather_object(all_variances, local_variances)
        
        if is_main_process():
            final_results = [r for sublist in all_results for r in sublist]
            final_variances = [v for sublist in all_variances for v in sublist]
    else:
        final_results = local_results
        final_variances = local_variances

    # --- Save & Print Results ---
    if is_main_process():
        df = pd.DataFrame(final_results)
        df_var = pd.DataFrame(final_variances)
        
        dataset_name = args.dataset_name
        df.to_csv(f"metrics/cam_faithfulness_metrics_{dataset_name}.csv", index=False)
        df_var.to_csv(f"metrics/cam_spatial_variances_{dataset_name}.csv", index=False)
        
        print(f"\nSaved results to metrics/cam_faithfulness_metrics_{dataset_name}.csv")
        print(f"Saved variances to metrics/cam_spatial_variances_{dataset_name}.csv")
        print(f"Saved {vis_count} visualizations to {vis_dir}/")
        
        print(f"\n{'='*60}")
        print(f"Summary: Metrics @ {args.top_percent * 100:.0f}% Kept")
        print(f"{'='*60}")
        
        metrics_to_print = [
            ('Variance', 'Spatial Variance', df_var, True),
            ('Confidence Drop', 'Confidence Drop', df, False),
            ('Activation Drop', 'Activation Drop', df, False),
            ('Confidence Increase Rate', 'Confidence Increase Rate', df, True),
            ('MC Confidence Drop', 'MC Confidence Drop', df, False),
            ('MC Activation Drop', 'MC Activation Drop', df, False),
            ('MC Confidence Increase Rate', 'MC Confidence Increase Rate', df, True),
            # ('MC Masked Confidence', 'MC Masked Confidence', df, True),
            # ('MC Masked Activation', 'MC Masked Activation', df, True),
            # ('MC Base Confidence', 'MC Base Confidence', df, True),
            # ('MC Base Activation', 'MC Base Activation', df, True),
        ]
        
        for col, name, data, higher_better in metrics_to_print:
            direction = "(Higher is better)" if higher_better else "(Lower is better)"
            print(f"\n{name} {direction}:")
            print(data.groupby(['Layer', 'Method'])[col].mean().unstack())

if __name__ == "__main__":
    main()