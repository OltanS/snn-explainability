from tqdm import tqdm
import contextlib

import torch
import spikingjelly.activation_based.functional as functional

def apply_criterion(criterion, output, labels):
    outputs = output.mean(dim=0)
    if isinstance(criterion, torch.nn.CrossEntropyLoss):
        loss = criterion(outputs, labels)
    
    elif isinstance(criterion, torch.nn.MSELoss):
        num_classes = output.size(-1)
        labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
        loss = criterion(outputs, labels_one_hot)
    else:
        raise ValueError(f"Unsupported criterion type: {type(criterion)}.")

    preds = outputs.argmax(dim=-1)
    acc = (preds == labels).float().mean()

    return loss, acc


def run_epoch(model, loader, local_rank, epoch, optimizer, criterion, mode, encoder=None, n_time_bins=None):
    model.train() if mode == "TRAIN" else model.eval()
    epoch_loss = 0
    epoch_corrects = 0
    samples_seen = 0

    context = torch.no_grad() if mode == "TEST" else contextlib.nullcontext()
    device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'

    with context:
        pbar  = tqdm(loader, total=len(loader), desc=f"Epoch {epoch + 1}", unit="batch", disable=local_rank != 0)
        for event_batches, labels in pbar:
            event_batches = event_batches.to(device)
            # no time dimension
            if event_batches.dim() == 4:
                # Add a time dimension
                event_batches = event_batches.unsqueeze(1)
                # copy along time dimension
                event_batches = event_batches.repeat(1, n_time_bins, 1, 1, 1)

            if encoder:
                event_batches = encoder(event_batches)
            # from B T C H W to T B C H W
            event_batches = event_batches.transpose(0, 1)

            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(event_batches)

            # print(f"event_batches shape after processing: {event_batches.shape}")
            # print(f"labels shape: {labels.shape}")
            # print(f"output shape: {output.shape}")
            # print("outputs sum sample:", output.sum(dim=0)[5])
            # print("matching label sample:", labels[5])
            loss, batch_acc_t = apply_criterion(criterion, output, labels)
            batch_acc = float(batch_acc_t.item())
            epoch_loss += loss.item()

            batch_size = labels.size(0)
            epoch_corrects += batch_acc * batch_size
            samples_seen += batch_size

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc*100:.1f}%")
            
            if mode == "TRAIN":
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            functional.reset_net(model)
        
        avg_epoch_loss = epoch_loss / len(loader)
        avg_epoch_acc = epoch_corrects / samples_seen if samples_seen > 0 else 0
        if local_rank == 0:
            print(f"Epoch {epoch + 1}, {mode} Loss: {avg_epoch_loss:.4f}, Acc: {avg_epoch_acc*100:.2f}%")

    return avg_epoch_loss, avg_epoch_acc