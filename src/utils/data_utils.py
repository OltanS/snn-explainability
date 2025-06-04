import os

import tonic
import tonic.transforms as T
import snntorch.spikegen as spikegen

import torch
from torchvision import datasets
from torchvision.transforms import v2

from src.utils.config_utils import setup_config_dictionary

def get_dataset_transforms(dataset_name):
    config = setup_config_dictionary()
    if dataset_name == "tiny_imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        return v2.Compose([
            v2.Resize(64),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean, std),
            lambda x : spikegen.rate(x, num_steps=config["N_TIME_BINS"]),
        ])
    
    elif dataset_name == "dvs_gesture":
        sensor_size = tonic.datasets.DVSGesture.sensor_size
        return T.Compose([
            T.Denoise(filter_time=10000),
            T.ToFrame(sensor_size=sensor_size, n_time_bins=config["N_TIME_BINS"]),
            v2.ToDtype(torch.float32, scale=True),
        ])
    
    else:
        raise ValueError(f"Dataset {dataset_name} is not implemented.")

def get_datasets(dataset_name):
    config = setup_config_dictionary()
    transforms = get_dataset_transforms(dataset_name)
    if dataset_name == "tiny_imagenet":
        DATA_DIR = config["DATA_SAVE_DIR"] + "/tiny-imagenet-200"
        TRAIN_DIR = os.path.join(DATA_DIR, "train")
        VAL_DIR = os.path.join(DATA_DIR, "val")
        VAL_IMG_DIR = os.path.join(VAL_DIR, "images")
        VAL_ANN = os.path.join(VAL_DIR, "val_annotations.txt")

        train_ds = datasets.ImageFolder(TRAIN_DIR, transform=transforms)
        test_ds = datasets.ImageFolder(VAL_DIR, transform=transforms)

    elif dataset_name == "dvs_gesture":
        train_ds = tonic.datasets.DVSGesture(save_to=config["DATA_SAVE_DIR"], transform=transforms, train=True)
        test_ds = tonic.datasets.DVSGesture(save_to=config["DATA_SAVE_DIR"], transform=transforms, train=False)

    else:
        raise ValueError(f"Dataset {dataset_name} is not implemented.")
        
    return train_ds, test_ds

def get_dataloaders(dataset_name, batch_size, on_ddp=False):
    '''
    Dataloaders always load data of shape batch_size x time_bins x num_polarities x height x width
    '''
    train_ds, test_ds = get_datasets(dataset_name)
    
    if on_ddp:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_ds)
        test_sampler = DistributedSampler(test_ds)
    else:
        train_sampler = None
        test_sampler = None
    
    # PadTensors actually doesn't do anything since we use a fixed N_TIME_BINS in the transforms.
    train_loader = torch.utils.data.DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=(train_sampler is None), 
        collate_fn=tonic.collation.PadTensors(), 
        num_workers=2, 
        pin_memory=False, 
        sampler=train_sampler
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_ds, 
        batch_size=batch_size, 
        shuffle=(test_sampler is None), 
        collate_fn=tonic.collation.PadTensors(), 
        num_workers=2, 
        pin_memory=False, 
        sampler=test_sampler
    )
    
    return train_loader, test_loader 