import torch
import glob
import os

def save_model(model, directory, optimizer, epoch, loss):
    '''
    Saves the model in the given location, DELETES OLD FILES!
    args:
        directory: str, the directory to save the model
        model: nn.Module, the model to save
        optimizer: torch.optim, the optimizer to save
        epoch: int, the epoch of the model
        loss: float, the loss of the model
        best_test_loss: float, the best test loss
        clean_old: bool, whether to clean the old models
    '''
    existing_files = glob.glob(f"{directory}/*")
    if existing_files:
        for file in existing_files:
            try:
                os.remove(file)
            except OSError as e:
                print(f"Error deleting file {file}: {e}")

    state = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': loss,
    }
    
    torch.save(state, f"{directory}/e{epoch + 1}_L{loss:.2f}.pth")


def load_model(model, directory, optimizer=None):
    '''
    Loads the model from the given location.
    args:
        directory: str, the directory to load the model from
        model: nn.Module, the model to load
        optimizer: torch.optim, the optimizer to load
    '''
    last_model_file = glob.glob(f"{directory}/*")[-1]
    checkpoint = torch.load(last_model_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    return epoch, loss