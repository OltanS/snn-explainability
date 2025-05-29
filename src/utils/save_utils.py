import torch
import glob
import os

from src.utils.model_utils import get_model_name

def save_model(model, directory, optimizer, epoch, loss, dataset_name):
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
    state = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': loss,
    }
    model_name = get_model_name(model)
    torch.save(state, f"{directory}/e{epoch + 1}_L{loss:.2f}_{dataset_name}_{model_name}.pth")

def load_model(model, directory, dataset_name, optimizer=None):
    '''
    Loads the model from the given location.
    args:
        directory: str, the directory to load the model from
        model: nn.Module, the model to load
        optimizer: torch.optim, the optimizer to load
    '''
    model_name = get_model_name(model)
    model_files = [f for f in glob.glob(f"{directory}/*") if dataset_name in f and model_name in f]
    if not model_files:
        raise FileNotFoundError(f"No model files found in {directory} for dataset '{dataset_name}' and model '{model_name}'.")
    last_model_file = max(model_files, key=os.path.getctime)
    print(f"Loading model from {last_model_file}")
    if not torch.cuda.is_available():
        checkpoint = torch.load(last_model_file, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(last_model_file)

    model_state_dict = checkpoint['model_state_dict']
    if any(key.startswith('module.') for key in model_state_dict.keys()):
        # For models saved with DataParallel or DistributedDataParallel
        model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}

    model.load_state_dict(model_state_dict)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    return epoch, loss