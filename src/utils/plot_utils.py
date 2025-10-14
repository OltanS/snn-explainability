import torch
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
            axes[0, i].imshow(frame, cmap='gray')
            axes[0, i].set_title(f"Frame {i}")
            axes[0, i].axis('off')

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
        # denormalize assuming ImageNet stats
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        m = np.array(mean, dtype=arr.dtype).reshape(1, 1, 1, 3)
        s = np.array(std, dtype=arr.dtype).reshape(1, 1, 1, 3)
        arr = arr * s + m
        fig, axes = plt.subplots(1, n_frames, figsize=(n_frames * 3, 3), squeeze=False)
        for i in range(n_frames):
            axes[0, i].imshow(arr[i])
            axes[0, i].set_title(f"Frame {i}")
            axes[0, i].axis('off')

    else:
        raise ValueError(f"Unsupported number of channels: {num_channels}. Expected 1, 2, or 3.")

    plt.tight_layout()
    plt.show()

def combine_polarities(frames):
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

def plot_cam_time_series(
    frames,
    cams,
    aggregate="mean",
    frame_stride=1,
    max_frames=24,
    overlay_alpha=0.3,
    cmap="jet",
    figsize=(14, 6),
    per_frame_cam_norm=True,
    show=True,
    save_path=None,
):
    """
    Plot a time series of aggregated CAM weights with selected frame overlays.

    frames: torch.Tensor or np.ndarray
        Shape (T, C, H, W) or (1, T, C, H, W). C = 1 or 3.
    cams: torch.Tensor or np.ndarray
        Shape (T, H, W) (output of compute_cam).
    Other args: same as before.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    try:
        import torch
    except Exception:
        torch = None

    # Convert frames -> numpy (T,C,H,W)
    if torch is not None and isinstance(frames, torch.Tensor):
        if frames.dim() == 5:
            assert frames.size(0) == 1, "Batch dimension >1 not supported."
            frames = frames[0]
        assert frames.dim() == 4, "Expected frames shape (T, C, H, W)."
        frames_np = frames.detach().cpu().numpy().astype(np.float32)
    else:
        frames_np = np.asarray(frames, dtype=np.float32)
        if frames_np.ndim == 5:
            assert frames_np.shape[0] == 1, "Batch dimension >1 not supported."
            frames_np = frames_np[0]
        assert frames_np.ndim == 4, "Expected frames shape (T, C, H, W)."

    T, C, H, W = frames_np.shape

    # Convert cams -> numpy (T,H,W)
    if torch is not None and isinstance(cams, torch.Tensor):
        cams_np = cams.detach().cpu().numpy().astype(np.float32)
    else:
        cams_np = np.asarray(cams, dtype=np.float32)

    assert cams_np.shape[0] == T, "cams time dimension mismatch."

    # Normalize frames to [0,1] per channel for visualization
    flat = frames_np.reshape(T, C, -1)
    f_min = flat.min(axis=2).reshape(T, C, 1, 1)
    f_max = flat.max(axis=2).reshape(T, C, 1, 1)
    frames_vis = ((frames_np - f_min) / (f_max - f_min + 1e-8)).clip(0, 1)

    # Compute per-timestep CAM weights
    if callable(aggregate):
        cam_weights = aggregate(cams_np)  # expect (T,)
    else:
        flatc = cams_np.reshape(T, -1)
        if aggregate == "mean":
            cam_weights = flatc.mean(axis=1)
        elif aggregate == "max":
            cam_weights = flatc.max(axis=1)
        else:
            raise ValueError("Unknown aggregate.")
    cam_weights = np.asarray(cam_weights).astype(float)

    # Select frames for overlay
    indices = list(range(0, T, frame_stride))
    if len(indices) > max_frames:
        lin_idx = np.linspace(0, len(indices) - 1, max_frames).round().astype(int)
        indices = [indices[i] for i in lin_idx]

    # CAM normalization (numpy)
    if per_frame_cam_norm:
        cam_show = np.empty_like(cams_np, dtype=np.float32)
        for t in range(T):
            m = cams_np[t].min()
            M = cams_np[t].max()
            if (M - m) < 1e-8:
                cam_show[t] = np.zeros_like(cams_np[t], dtype=np.float32)
            else:
                cam_show[t] = (cams_np[t] - m) / (M - m)
    else:
        gmin = cams_np.min()
        gmax = cams_np.max()
        if (gmax - gmin) < 1e-8:
            cam_show = np.zeros_like(cams_np, dtype=np.float32)
        else:
            cam_show = (cams_np - gmin) / (gmax - gmin)

    # Figure layout:
    n_frames = len(indices)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, n_frames, height_ratios=[3, 1])

    for ci, t in enumerate(indices):
        ax = fig.add_subplot(gs[0, ci])
        frame = frames_vis[t]
        if C == 1:
            base_img = frame[0]
            ax.imshow(base_img, cmap="gray", interpolation="nearest")
        else:
            base_img = frame.transpose(1, 2, 0)
            ax.imshow(base_img, interpolation="nearest")

        heat = cam_show[t]
        ax.imshow(heat, cmap=cmap, alpha=overlay_alpha, interpolation="nearest")
        ax.set_title(f"t={t}")
        ax.axis("off")

    ax_ts = fig.add_subplot(gs[1, :])
    ax_ts.plot(range(T), cam_weights, color="tab:red" if matplotlib_is_available() else "tabred", lw=2)
    ax_ts.set_xlabel("Time step")
    ax_ts.set_ylabel("CAM weight")
    ax_ts.set_xlim(0, T - 1)

    ax_ts.scatter(indices, cam_weights[indices], color="black", s=16, zorder=3)
    ax_ts.grid(alpha=0.3)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    return fig

# helper used above to avoid failing if matplotlib not imported globally
def matplotlib_is_available():
    try:
        import matplotlib
        return True
    except Exception:
        return False