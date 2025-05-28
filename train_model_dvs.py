
import tonic
import tonic.transforms as T

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm

from models.basicsnn import BasicSNN
from src.config_utils import setup_config_dictionary
from src.save_utils import save_model
from src.distributed_utils import ddp_setup, ddp_cleanup, ddp_model, training_optimisations

batch_size = 64
epochs = 15

config = setup_config_dictionary()
ddp_setup(config["LOCAL_RANK"])

device = config["LOCAL_RANK"]

gesture_map = {
    0: "hand_clapping",
    1: "right_hand_wave",
    2: "left_hand_wave",
    3: "right_hand_clockwise",
    4: "right_hand_counter_clockwise",
    5: "left_hand_clockwise",
    6: "left_hand_counter_clockwise",
    7: "forearm_roll_forward",
    8: "forearm_roll_backward",
    9: "drums",
    10: "guitar",
    #11: "random_other_gestures", # -> not implemented here
}

sensor_size = tonic.datasets.DVSGesture.sensor_size

transform = T.Compose([
    # T.Denoise(filter_time=10000),
    T.ToFrame(sensor_size=sensor_size, time_window=500 * 10 ** 3)
])

train_dataset = tonic.datasets.DVSGesture(save_to=config["DATA_SAVE_DIR"], transform=transform, train=True)
# frames shape is (time_bins, num_polarities, height, width)
# we must pad the frames to the same size
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=tonic.collation.PadTensors(), num_workers=0, pin_memory=False, sampler=DistributedSampler(train_dataset))
# dataloader will return (batch_size, time_bins, num_polarities, height, width)
firing_threshold = 1.0
model = BasicSNN(num_classes=len(gesture_map), num_channels=2, threshold=firing_threshold).to(device)
model = ddp_model(model, config["LOCAL_RANK"])

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    model.train()
    epoch_train_loss = 0
    epoch_test_loss = 0
    pbar  = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}", unit="batch", disable=config["LOCAL_RANK"] != 0)
    for event_batches, labels in pbar:
        event_batches = event_batches.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        out_spikes, _ = model(event_batches)
        out_spikes = out_spikes.sum(dim=1) # sum over time
        out_classes = F.softmax(out_spikes, dim=1)
        loss = criterion(out_spikes.view(-1, len(gesture_map)), labels.view(-1))
        loss.backward()
        epoch_train_loss += loss.item()
        optimizer.step()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        
    
    avg_epoch_train_loss = epoch_train_loss / len(train_loader)
    print(f"Epoch {epoch + 1}, Train Loss: {avg_epoch_train_loss}")

if config["GLOBAL_RANK"] == 0:
    save_model(model, config["MODEL_SAVE_DIR"], optimizer, epoch, avg_epoch_train_loss, dataset_name="dvsgesture", model_name="basicsnn")


