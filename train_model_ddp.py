import torch
import torch.nn as nn
import snntorch.functional as SF

from src.utils.config_utils import setup_config_dictionary
from src.utils.save_utils import save_model
from src.utils.distributed_utils import ddp_setup, ddp_cleanup, ddp_model
from src.utils.data_utils import get_dataloaders
from src.utils.model_utils import get_model
from src.utils.train_utils import run_epoch

def main():
    '''
    This must always be run with DDP enabled, even if on a single node, via torchrun.
    '''
    config = setup_config_dictionary()
    ddp_setup(config["LOCAL_RANK"])

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
        num_classes=len(train_loader.dataset.classes),
        num_channels=num_channels,
        time_steps=config["N_TIME_BINS"],
        beta=config["BETA"],
        enable_bntt=config["ENABLE_BNTT"],
    )

    #criterion = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
    criterion = SF.ce_max_membrane_loss()
    model = ddp_model(model, config["LOCAL_RANK"])
    
    optimizer = torch.optim.SGD(model.parameters(), lr=config["LEARNING_RATE"], momentum=0.9, weight_decay=config["WEIGHT_DECAY"])
    milestones = [
        int(0.5 * config["EPOCHS"]),
        int(0.7 * config["EPOCHS"]),
        int(0.9 * config["EPOCHS"]),
    ]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=milestones,
        gamma=0.1
    )

    print(f"Total Epochs to run: {config['EPOCHS']}")
    for epoch in range(config["EPOCHS"]):
        avg_epoch_train_loss = run_epoch(model, train_loader, config["LOCAL_RANK"], epoch, optimizer, criterion, mode="TRAIN")
        avg_epoch_test_loss = run_epoch(model, test_loader, config["LOCAL_RANK"], epoch, optimizer, criterion, mode="TEST")
        scheduler.step()
        if (epoch + 1) % config["SAVE_EVERY_N_EPOCHS"] == 0:
            if config["GLOBAL_RANK"] == 0:
                save_model(model, config["MODEL_SAVE_DIR"], optimizer, epoch, avg_epoch_test_loss, dataset_name=config["DATASET_NAME"])
    ddp_cleanup()


if __name__ == "__main__":
    main()