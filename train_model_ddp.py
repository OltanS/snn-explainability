import warnings
import os
warnings.filterwarnings("ignore", message=".*video decoding and encoding capabilities.*deprecated.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*pts_unit", category=UserWarning)

import torch
import torch.nn as nn
import snntorch.functional as SF

from src.utils.config_utils import setup_config_dictionary
from src.utils.save_utils import save_model, load_model
from src.utils.distributed_utils import ddp_setup, ddp_cleanup, ddp_model
from src.utils.data_utils import get_dataloaders
from src.utils.model_utils import get_model
from src.utils.train_utils import run_epoch

def main():
    CONTINUE_FROM_CHECKPOINT = False
    '''
    This must always be run with DDP enabled, even if on a single node, via torchrun.
    ''' 
    config = setup_config_dictionary()
    ddp_setup(config["LOCAL_RANK"])
    device = torch.device(f"cuda:{config['LOCAL_RANK']}")

    # Warn on unconventional/unknown normalization choices per model
    model = config.get("MODEL_NAME", "").lower()
    norm  = config.get("NORM_TYPE", "").lower()

    print(f"Training on DATASET={config.get('DATASET_NAME')} MODEL_NAME={model} with NORM_TYPE={norm}\nlearning rate={config.get('LEARNING_RATE')}, weight decay={config.get('WEIGHT_DECAY')}, beta={config.get('BETA')}")

    conventional_norms = {
        "vgg11": {"batchnorm", "bntt"},
        "resnet": {"batchnorm", "bntt"},
    }

    if model in conventional_norms and norm and norm not in conventional_norms[model]:
        conv = ", ".join(sorted(conventional_norms[model]))
        warnings.warn(
            f"NORM_TYPE='{norm}' is uncommon for MODEL_NAME='{model}'. This might cause performance issues, proceed with caution.\
            Conventional choices: {conv}.",
            RuntimeWarning,
        )

    train_loader, test_loader = get_dataloaders(
        config["DATASET_NAME"], 
        config["BATCH_SIZE"], 
        config["ON_DDP"]
    )

    # Infer number of channels from the first batch: (B, T, C, H, W)
    event_batches, _ = next(iter(train_loader))
    num_channels = event_batches.shape[2]

    model = get_model(
        config["MODEL_NAME"],
        num_classes=len(train_loader.dataset.classes) if hasattr(train_loader.dataset, "classes") else len(train_loader.dataset.dataset.classes),
        num_channels=num_channels,
        time_steps=config["N_TIME_BINS"],
        beta=config["BETA"],
        norm_type=config["NORM_TYPE"]
    )



    criterion = nn.CrossEntropyLoss()
    # criterion = SF.ce_max_membrane_loss()
    
    # optimizer = torch.optim.SGD(model.parameters(), lr=config["LEARNING_RATE"], weight_decay=config["WEIGHT_DECAY"], momentum=0.9)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["LEARNING_RATE"], weight_decay=config["WEIGHT_DECAY"])
    # milestones = [m for m in sorted({int(config["EPOCHS"] * f) for f in (0.5, 0.7, 0.9)}) if 0 < m < config["EPOCHS"]]
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer,
    #     milestones=milestones,
    #     gamma=0.1         # decay by factor of 10
    # )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["EPOCHS"],
        eta_min=1e-6
    )
    start_epoch = 0

    if CONTINUE_FROM_CHECKPOINT:
        start_epoch, _ = load_model(model, config["MODEL_SAVE_DIR"], optimizer=optimizer, dataset_name=config["DATASET_NAME"], device=device)

    model = ddp_model(model, config["LOCAL_RANK"])
    print(f"Total Epochs to run: {config['EPOCHS']}")
    for epoch in range(start_epoch, config["EPOCHS"]):
        avg_epoch_train_loss = run_epoch(model, train_loader, config["LOCAL_RANK"], epoch, optimizer, criterion, mode="TRAIN")
        avg_epoch_test_loss = run_epoch(model, test_loader, config["LOCAL_RANK"], epoch, optimizer, criterion, mode="TEST")
        scheduler.step()
        if (epoch + 1) % config["SAVE_EVERY_N_EPOCHS"] == 0:
            if config["GLOBAL_RANK"] == 0:
                save_model(model, config["MODEL_SAVE_DIR"], optimizer, epoch, avg_epoch_test_loss, dataset_name=config["DATASET_NAME"], norm_type=config["NORM_TYPE"])
    ddp_cleanup()


if __name__ == "__main__":
    main()