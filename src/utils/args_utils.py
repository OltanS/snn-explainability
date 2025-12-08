import argparse
import os

from utils.model_utils import AVAILABLE_MODELS
from utils.data_utils import AVAILABLE_DATASETS

def parse_args():
    parser = argparse.ArgumentParser(description="Spiking Neural Network Training and Evaluation")

    # Locations
    parser.add_argument("--model_save_dir", type=str, default="./models_save", help="Directory to save models")
    parser.add_argument("--data_dir", type=str, default="./data", help="Root directory for datasets")
    
    # Model & Dataset
    parser.add_argument("--model_name", type=str, default="simple_snn_no_vote_no_last_spiking", choices=AVAILABLE_MODELS, help="Name of the model architecture")
    parser.add_argument("--dataset_name", type=str, default="dvs_gesture", choices=AVAILABLE_DATASETS, help="Name of the dataset")
    # Training Hyperparameters
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--n_time_bins", type=int, default=16, help="Number of time steps (T)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--neuron_type", type=str, default="lif", choices=["lif", "if"], help="Type of spiking neuron model")
    parser.add_argument("--tau", type=float, default=2.0, help="Membrane time constant tau")
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "ce", "bce"], help="Loss function")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing for cross-entropy loss")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"], help="Optimizer")

    # Distributed / System
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--save_every", type=int, default=25, help="Save checkpoint every N epochs")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # CAM / Evaluation specific
    parser.add_argument("--cam_methods", nargs="+", default=['grad', 'ablation', 'mc_ablation', 'sam', 'random'], help="CAM methods to evaluate")
    parser.add_argument("--max_samples", type=int, default=200, help="Max samples to process in CAM comparison")
    parser.add_argument("--top_percent", type=float, default=0.2, help="Percentage of pixels to keep for masking")
    parser.add_argument("--target_layers", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="Indices of layers to evaluate (same index used for Conv and (L)IF)")
    parser.add_argument("--k_trials", type=int, default=20, help="Number of trials for MC Ablation")
    parser.add_argument("--channel_batch_size", type=int, default=16, help="Batch size for channel ablation")
    parser.add_argument("--num_visualizations", type=int, default=20, help="Number of CAM visualizations to save")

    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    return args