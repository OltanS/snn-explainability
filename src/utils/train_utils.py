from tqdm import tqdm
import contextlib

import torch
import snntorch.functional as SF

def apply_criterion(criterion, output, labels):
    # output will by default be of shape (B, T, num_classes)
    # print(f"Output shape: {output.shape}")
    # print(f"labels shape: {labels.shape}")
    # print(f"Example output of first item of batch, last timestep: {output[0, -1, :]}")
    # print(f"Highest value across all timesteps: {output.max().item()}")
    if isinstance(criterion, torch.nn.CrossEntropyLoss):
        # For CrossEntropyLoss, output should be raw logits and labels should be class indices
        outputs = output.mean(dim=1)
        # print(f"Outputs are {outputs}")
        return criterion(outputs, labels)

    # Time first losses from here down
    outputs = output.permute(1, 0, 2)  # permute to (T, B, num_classes)

    if isinstance(criterion, SF.ce_max_membrane_loss):
        return criterion(outputs, labels)

    else:
        raise ValueError(f"Unsupported criterion type: {type(criterion)}.")

def run_epoch(model, loader, local_rank, epoch, optimizer, criterion, mode):
    model.train() if mode == "TRAIN" else model.eval()
    epoch_loss = 0
    context = torch.no_grad() if mode == "TEST" else contextlib.nullcontext()

    with context:
        pbar  = tqdm(loader, total=len(loader), desc=f"Epoch {epoch + 1}", unit="batch", disable=local_rank != 0)
        for event_batches, labels in pbar:
            event_batches = event_batches.to(local_rank) # B, T, C, H, W
            # print(f"Event batch shape: {event_batches.shape}")
            labels = labels.to(local_rank)
            optimizer.zero_grad()
            # for models that return e.g mem_history

            output, _ = model(event_batches)
            
            loss = apply_criterion(criterion, output, labels)
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.2f}")
            
            if mode == "TRAIN":
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        
        avg_epoch_loss = epoch_loss / len(loader)
        # avg_epoch_train_accuracy = epoch_train_accuracy / len(train_loader)
        if local_rank == 0:
            print(f"Epoch {epoch + 1}, {mode} Loss: {avg_epoch_loss:.2f}")

    return avg_epoch_loss