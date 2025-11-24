import warnings

warnings.filterwarnings("ignore", message=".*video decoding and encoding capabilities.*deprecated.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*pts_unit", category=UserWarning)

from networkx import config
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from spikingjelly.activation_based.encoding import PoissonEncoder

from spikingjelly.activation_based import neuron, surrogate, functional

from src.utils.config_utils import setup_config_dictionary
from src.utils.save_utils import save_model, load_model
from src.utils.distributed_utils import ddp_setup, ddp_cleanup, ddp_model
from src.utils.data_utils import get_dataloaders
from src.utils.model_utils import get_model
from src.utils.train_utils import run_epoch

'''
This must always be run with DDP enabled, even if on a single node, via torchrun.
''' 
def main():
    CONTINUE_FROM_CHECKPOINT = False
    
    
    config = setup_config_dictionary()
    ddp_setup(config["LOCAL_RANK"])
    device = torch.device(f"cuda:{config['LOCAL_RANK']}")

    # Warn on unconventional/unknown normalization choices per model
    model_name = config.get("MODEL_NAME", "").lower()

    print(f"Training on DATASET={config.get('DATASET_NAME')} MODEL_NAME={model_name},\nlearning rate={config.get('LEARNING_RATE')}, weight decay={config.get('WEIGHT_DECAY')}, tau={config.get('TAU')}")

    train_loader, test_loader = get_dataloaders(
        config["DATASET_NAME"], 
        config["BATCH_SIZE"],
        config["N_TIME_BINS"],
        config["ON_DDP"]
    )

    # Infer number of channels from the first batch
    event_batches, labels = next(iter(train_loader))
    num_channels = event_batches.shape[-3]
    print(f"Event batches shape: {event_batches.shape}\nLabels shape: {labels.shape}")
    if labels.dim() == 2:
        num_classes = labels.size(1)
    else:
        num_classes = len(train_loader.dataset.classes) if hasattr(train_loader.dataset, "classes") else len(train_loader.dataset.dataset.classes)

    model = get_model(
        config["MODEL_NAME"],
        num_classes=num_classes,
        in_channels=num_channels,
        spiking_neuron=neuron.LIFNode,
        pretrained=False,
        # tau=config["TAU"],
        detach_reset=True,
        surrogate_function=surrogate.ATan()
    )
    functional.set_step_mode(model, 'm')
    functional.set_backend(model, "cupy", instance=neuron.LIFNode)


    # criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["LEARNING_RATE"], weight_decay=config["WEIGHT_DECAY"])
    # optimizer = torch.optim.SGD(model.parameters(), lr=config["LEARNING_RATE"], momentum=0.9, weight_decay=config["WEIGHT_DECAY"])
    start_epoch = 0
    if CONTINUE_FROM_CHECKPOINT:
        start_epoch, _ = load_model(model, config["MODEL_SAVE_DIR"], optimizer=optimizer, dataset_name=config["DATASET_NAME"], device=device)
    
    warmup_epochs = 5
    total_epochs = config["EPOCHS"]
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-5),
        ],
        milestones=[warmup_epochs],
        last_epoch=start_epoch - 1
    )
    encoder = None
    # encoder = PoissonEncoder(step_mode='m')


    model = ddp_model(model, config["LOCAL_RANK"])
    
    print(f"Total Epochs to run: {config['EPOCHS']}")
    for epoch in range(start_epoch, config["EPOCHS"]):
        if config["ON_DDP"] and hasattr(train_loader, "sampler") and train_loader.sampler is not None:
            train_loader.sampler.set_epoch(epoch)
        
        accumulation_steps = 32 // config["BATCH_SIZE"] if config["BATCH_SIZE"] < 32 else 1
        avg_epoch_train_loss, avg_epoch_train_accuracy = run_epoch(model, train_loader, config["LOCAL_RANK"], epoch, optimizer, criterion, mode="TRAIN", encoder=encoder, n_time_bins=config["N_TIME_BINS"], accumulation_steps=accumulation_steps)
        avg_epoch_test_loss, avg_epoch_test_accuracy = run_epoch(model, test_loader, config["LOCAL_RANK"], epoch, optimizer, criterion, mode="TEST", encoder=encoder, n_time_bins=config["N_TIME_BINS"])
        scheduler.step()
        
        if (epoch + 1) % config["SAVE_EVERY_N_EPOCHS"] == 0:
            if config["GLOBAL_RANK"] == 0:
                save_model(model, config["MODEL_SAVE_DIR"], optimizer, epoch, avg_epoch_test_loss, dataset_name=config["DATASET_NAME"])
                print(f"Model checkpoint saved at epoch {epoch+1} to {config['MODEL_SAVE_DIR']}")

if __name__ == "__main__":
    try:
        torch.hub.set_dir("./torch_cache")
        main()
    finally:
        ddp_cleanup()