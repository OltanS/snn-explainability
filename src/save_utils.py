import torch
import glob

def save_model(directory, model, optimizer, epoch, loss):
    '''
    Saves the model in the given location.
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
    
    loss_steering_angle_rmse = 180 * (loss ** 0.5)
    torch.save(state, f"{directory}/e{epoch + 1}_L{loss_steering_angle_rmse:.2f}.pth")


def load_model(directory, model, optimizer):
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
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    return epoch, loss