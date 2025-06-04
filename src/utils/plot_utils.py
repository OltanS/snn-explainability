import matplotlib.pyplot as plt
import numpy as np

def plot_frames(frames):
    # Accept frames in (B, H, W, C) or (B, C, H, W)
    arr = frames
    if arr.ndim == 4:
        # if channels are in axis=1, move to last axis
        if arr.shape[-1] not in (1, 2, 3) and arr.shape[1] in (1, 2, 3):
            arr = np.transpose(arr, (0, 2, 3, 1))
    else:
        raise ValueError(f"Unsupported frames shape {arr.shape}, expected 4D array.")
    
    B, H, W, C = arr.shape
    n_frames = B
    num_channels = C

    if num_channels == 1:
        fig, axes = plt.subplots(1, n_frames, figsize=(n_frames * 3, 3), squeeze=False)
        for i in range(n_frames):
            frame = arr[i, :, :, 0]
            axes[i].imshow(frame, cmap='gray')
            axes[i].set_title(f"Frame {i}")
            axes[i].axis('off')

    elif num_channels == 2:
        fig, axes = plt.subplots(2, n_frames, figsize=(n_frames * 3, 6), squeeze=False)
        for i in range(n_frames):
            off_frame = arr[i, :, :, 0]
            on_frame  = arr[i, :, :, 1]
            axes[0, i].imshow(off_frame, cmap='gray')
            axes[0, i].set_title(f"Frame {i} - Polarity OFF")
            axes[1, i].imshow(on_frame,  cmap='gray')
            axes[1, i].set_title(f"Frame {i} - Polarity ON")

    elif num_channels == 3:
        fig, axes = plt.subplots(1, n_frames, figsize=(n_frames * 3, 3), squeeze=False)
        for i in range(n_frames):
            axes[i].imshow(arr[i])
            axes[i].set_title(f"Frame {i}")
            axes[i].axis('off')

    else:
        raise ValueError(f"Unsupported number of channels: {num_channels}. Expected 1, 2, or 3.")

    plt.tight_layout()
    plt.show()

def combine_polarities(frames):
    # frames are of shape time_frame, polarity, height, width
    # we can concat both polarities (there might be some overlap, but that's ok!)
    # negative polarities are turned into negative numbers
    polarity_zero = -frames[:, :, :, 0]
    polarity_one = frames[:, :, :, 1]
    # Take the value with the largest magnitude
    mask = np.abs(polarity_zero) > np.abs(polarity_one)
    combined = np.where(mask, polarity_zero, polarity_one)
    
    # print(combined.shape)
    return combined

def plot_combined_frames(combined_frames):
    n_frames = len(combined_frames)
    h, w = combined_frames[0].shape
    _, ax = plt.subplots(1, n_frames, figsize=(n_frames * 3, 3))
    for i, combined_frame in enumerate(combined_frames):
        ax[i].imshow(combined_frame, cmap='gray')
        ax[i].set_title(f"Combined Polarity {i}")
        ax[i].axis('off')
    plt.show()

def plot_cams(cams, class_idx=0, frames=None, timesteps=None,
              cmap='jet', alpha=0.5, vmin=0.0, vmax=1.0):
    
    # B, T, NUM_CLASSES, H, W
    # Squeeze away if it has a singleton batch dimension
    if cams.ndim == 5:
        cams = cams.squeeze(0)
    # convert to numpy
    if hasattr(cams, 'cpu'):
        cams = cams.cpu().detach().numpy()
    # select class if needed
    if cams.ndim == 4:
        # shape (T, C, H, W)
        cams = cams[:, class_idx, ...]
    T, H, W = cams.shape

    # choose timesteps
    if timesteps is None:
        indices = list(range(T))
    else:
        indices = [t for t in range(timesteps) if 0 <= t < T]

    n = len(indices)
    # decide grid: if overlay, two rows; else one
    fig, axes = plt.subplots(1, n, figsize=(n * 6, 5))
    axes = np.atleast_2d(axes)

    for col, t in enumerate(indices):
        if frames is not None:
            frame = frames[t]
            ax_frame = axes[-1, col]
            ax_frame.imshow(frame)
        heat = cams[t]
        ax_heat = axes[-1, col]
        im = ax_heat.imshow(heat, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha)
        ax_heat.set_title(f"t={t}")

    # colorbar on the last column of heatmaps
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.82, 0.15, 0.005, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Activation') 
    plt.show()

def plot_activations(activations, frames=None, timesteps=None,
              cmap='jet', alpha=0.5, vmin=0.0, vmax=1.0):
    '''
    activations are: b, t, num_channels, h, w
    plot_cams takes b, t, num_classes, h, w, with a target class index.

    If we're plotting one channel at a time, we can set the class index to 0, as a hack to reuse the function.
    '''

    plot_cams(activations, class_idx=0, frames=frames, timesteps=timesteps,
              cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)