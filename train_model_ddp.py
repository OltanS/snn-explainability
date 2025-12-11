import os
import random
from functools import partial
import numpy as np


import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from spikingjelly.activation_based import neuron, surrogate, functional

from src.utils.args_utils import parse_args
from src.utils.save_utils import save_model, load_model
from src.utils.distributed_utils import ddp_setup, ddp_cleanup, ddp_model
from src.utils.data_utils import get_dataloaders
from src.utils.model_utils import get_model
from src.utils.train_utils import run_epoch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    args = parse_args()
    
    # DDP Setup
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))

    is_distributed = world_size > 1
    if is_distributed:
        ddp_setup(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)

    if rank == 0:
        print(f"Training on DATASET={args.dataset_name} MODEL={args.model_name}")
        print(f"Params: LR={args.lr}, WD={args.weight_decay}, TAU={args.tau}, BS={args.batch_size}")


    train_loader, test_loader = get_dataloaders(
        args.dataset_name, 
        args.data_dir,
        args.batch_size,
        args.n_time_bins,
        on_ddp=is_distributed,
        num_workers=args.num_workers
    )

    # Infer number of channels/classes
    event_batches, labels = next(iter(train_loader))
    num_channels = event_batches.shape[-3]

    if labels.dim() == 2:
        num_classes = labels.size(1)
    else:
        ds = train_loader.dataset
        if hasattr(ds, "classes"):
            num_classes = len(ds.classes)
        elif hasattr(ds, "dataset") and hasattr(ds.dataset, "classes"):
            num_classes = len(ds.dataset.classes)
        else:
            raise ValueError("Could not determine number of classes from dataset.")

    neuron_choice = neuron.LIFNode if args.neuron_type == "lif" else neuron.IFNode
    neuron_arguments = {
        "tau": args.tau,
        "detach_reset": True,
        "surrogate_function": surrogate.ATan()
    }

    if neuron_choice == neuron.IFNode:
        neuron_arguments.pop("tau")

    model_arguments = {
        "num_classes": num_classes,
        "in_channels": num_channels,
        "spiking_neuron": partial(neuron_choice, **neuron_arguments),
        "pretrained": False,
    }

    model = get_model(args.model_name, **model_arguments)
    functional.set_step_mode(model, 'm')
    functional.set_backend(model, "cupy", instance=neuron_choice)

    # Loss selection
    if args.loss == "mse":
        criterion = nn.MSELoss()
    elif args.loss == "ce":
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    elif args.loss == "bce":
        criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer selection
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    start_epoch = 0
    if args.resume:
        start_epoch, _ = load_model(model, args.resume, dataset_name=args.dataset_name, optimizer=optimizer, device=device)
    
    warmup_epochs = 5
    total_epochs = args.epochs
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-5),
        ],
        milestones=[warmup_epochs],
        last_epoch=start_epoch - 1
    )

    if is_distributed:
        model = ddp_model(model, local_rank)
    else:
        model.to(device)
    

    # Explicit accumulation steps logic
    target_batch_size = 32
    accumulation_steps = max(1, target_batch_size // args.batch_size)
    
    if rank == 0:
        print(f"Total Epochs: {args.epochs}, Accumulation Steps: {accumulation_steps}")
        print(f"Model name : {args.model_name}\n Model architecture:\n{model}")

    for epoch in range(start_epoch, args.epochs):
        if is_distributed and hasattr(train_loader, "sampler") and train_loader.sampler is not None:
            train_loader.sampler.set_epoch(epoch)
        
        run_epoch(model, train_loader, local_rank, epoch, optimizer, criterion, mode="TRAIN", n_time_bins=args.n_time_bins, accumulation_steps=accumulation_steps)
        avg_test_loss, _ = run_epoch(model, test_loader, local_rank, epoch, optimizer, criterion, mode="TEST", n_time_bins=args.n_time_bins)
        scheduler.step()
        
        if (epoch + 1) % args.save_every == 0:
            if rank == 0:
                save_model(model, args.model_save_dir, optimizer, epoch, avg_test_loss, dataset_name=args.dataset_name)
                print(f"Model checkpoint saved at epoch {epoch+1}")

if __name__ == "__main__":
    try:
        torch.hub.set_dir("./torch_cache")
        main()
    finally:
        if torch.distributed.is_initialized():
            ddp_cleanup()