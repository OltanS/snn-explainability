# %%
import os
import shutil
from tqdm import tqdm

import tonic
import tonic.transforms as T
from snntorch import spikegen

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

from models.spiking_vgg11 import SpikingVGG11
from src.config_utils import setup_config_dictionary
from src.save_utils import save_model
from src.distributed_utils import ddp_setup, ddp_cleanup, ddp_model, training_optimisations

from torch.utils.data.distributed import DistributedSampler

batch_size = 8
epochs = 30
NUM_STEPS = 30

config = setup_config_dictionary()
ddp_setup(config["LOCAL_RANK"])
device = config["LOCAL_RANK"]

DATA_DIR = config["DATA_SAVE_DIR"] + "/tiny-imagenet-200"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
VAL_IMG_DIR = os.path.join(VAL_DIR, "images")
VAL_ANN = os.path.join(VAL_DIR, "val_annotations.txt")

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

tf = v2.Compose([
    v2.Resize(64),
    v2.RandomHorizontalFlip(),
    v2.ToTensor(),
    v2.Normalize(mean, std),
])

train_ds = datasets.ImageFolder(TRAIN_DIR, transform=tf)
val_ds   = datasets.ImageFolder(VAL_DIR,   transform=tf)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=4, sampler=DistributedSampler(train_ds))
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, sampler=DistributedSampler(val_ds))

model = SpikingVGG11(num_classes=len(train_ds.classes), time_steps=NUM_STEPS).to(device)
model = ddp_model(model, config["LOCAL_RANK"])
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    model.train()
    epoch_train_loss = 0
    epoch_test_loss = 0
    pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", disable=config["LOCAL_RANK"] != 0)
    for event_batches, labels in pbar:
        event_batches = event_batches.to(device)
        labels = labels.to(device)
        event_batches = spikegen.rate(event_batches, num_steps=NUM_STEPS)  # Convert images to spikes, # T, B, 3, 64 ,64
        event_batches = event_batches.permute(1, 0, 2, 3, 4)  # B, T, C, H, W

        optimizer.zero_grad()
        out_spikes, _ = model(event_batches)
        out_spikes = out_spikes.sum(dim=1) # sum over time
        out_classes = F.softmax(out_spikes, dim=1)
        loss = criterion(out_spikes.view(-1, len(train_ds.classes)), labels.view(-1))
        loss.backward()
        epoch_train_loss += loss.item()
        optimizer.step()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    avg_epoch_train_loss = epoch_train_loss / len(train_loader)
    if config["LOCAL_RANK"] == 0:
        print(f"Epoch {epoch}, Train Loss: {avg_epoch_train_loss}")

save_model(model, config["MODEL_SAVE_DIR"], optimizer, epoch, avg_epoch_train_loss, dataset_name="tinyimagenet", model_name="spiking_vgg11")

ddp_cleanup()


