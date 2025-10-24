import glob
import os

import torch

from IPython.display import display, clear_output
import ipywidgets as widgets

from src.utils.model_utils import get_model_name

def save_model(model, directory, optimizer, epoch, loss, dataset_name, norm_type):
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
    model = model.module if hasattr(model, 'module') else model  # Handle DataParallel or DistributedDataParallel
    model_name = get_model_name(model)
    torch.save(state, f"{directory}/e{epoch + 1}_L{loss:.2f}_{dataset_name}_{model_name}_norm_{norm_type}.pth")

def load_model(model, path_or_dir, dataset_name, optimizer=None, device=None):
    """
    Loads the model from either:
      • a single file (if path_or_dir is a file), or
      • the newest matching file in a directory.

    args:
      path_or_dir: str, path to a model‐file or a directory to scan
      model:       nn.Module to load weights into
      optimizer:   optional torch.optim to restore state
      device:      optional torch.device to map all tensors onto (e.g., cuda:local_rank)
    returns:
      (epoch, loss)
    """
    # determine whether we're given a file or a directory
    if os.path.isfile(path_or_dir):
        model_file = path_or_dir
    else:
        model_name = get_model_name(model)
        candidates = glob.glob(os.path.join(path_or_dir, "*"))
        model_files = [
            f for f in candidates
            if dataset_name in os.path.basename(f)
               and model_name in os.path.basename(f)
        ]
        if not model_files:
            raise FileNotFoundError(
                f"No model files found in {path_or_dir!r} for "
                f"dataset={dataset_name!r}, model={model_name!r}"
            )
        model_file = max(model_files, key=os.path.getctime)

    print(f"Loading model from {model_file}")

    # load checkpoint with proper device mapping
    if device is not None:
        ckpt = torch.load(model_file, map_location=device)
    elif not torch.cuda.is_available():
        ckpt = torch.load(model_file, map_location="cpu")
    else:
        ckpt = torch.load(model_file)

    # strip DataParallel prefix if needed
    sd = ckpt["model_state_dict"]
    if any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}

    model.load_state_dict(sd)

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        # move optimizer state tensors to the target device if provided
        if device is not None:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)

    return ckpt.get("epoch"), ckpt.get("loss")

def interactive_load_model(model, directory, dataset_name, optimizer=None):
    """
    Jupyter-friendly model loader that lets user select and load a checkpoint.
    Shows result in output widget.
    """
    print("Make sure that both ENABLE_BNTT matches the state the model was trained with.")
    model_name = get_model_name(model)
    files = [f for f in glob.glob(os.path.join(directory, '*'))
             if dataset_name in f and model_name in f]
    if not files:
        raise FileNotFoundError(f"No checkpoints in {directory!r} for {dataset_name!r}, {model_name!r}")

    files.sort(key=os.path.getmtime, reverse=True)  # Newest first
    titles = [os.path.basename(f) for f in files]

    dd = widgets.Dropdown(options=list(zip(titles, files)), description='Checkpoint:')
    btn = widgets.Button(description="Load", button_style='primary')
    out = widgets.Output()

    def on_load(b):
        btn.disabled = True
        with out:
            clear_output()
            try:
                path = dd.value
                print(f"Loading {os.path.basename(path)}...")
                epoch, loss = load_model(model, path, dataset_name, optimizer)
                print(f"✓ Loaded {os.path.basename(path)} → epoch={epoch}, loss={loss:.4f}")
            except Exception as e:
                print(f"✗ Error: {e}")

    btn._click_handlers.callbacks.clear()
    btn.on_click(on_load)
    display(widgets.VBox([dd, btn, out]))