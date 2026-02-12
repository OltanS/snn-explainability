import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def plot_frames(frames):
    arr = frames
    if arr.ndim != 4:
        raise ValueError(f"Unsupported frames shape {arr.shape}, expected 4D array.")
    if arr.shape[-1] not in (1, 2, 3) and arr.shape[1] in (1, 2, 3):
        arr = np.transpose(arr, (0, 2, 3, 1))

    B, H, W, C = arr.shape
    if C == 1:
        fig, axes = plt.subplots(1, B, figsize=(B * 3, 3), squeeze=False)
        for i in range(B):
            axes[0, i].imshow(arr[i, :, :, 0], cmap='gray')
            axes[0, i].set_title(f"Frame {i}")
            axes[0, i].axis('off')
    elif C == 2:
        fig, axes = plt.subplots(2, B, figsize=(B * 3, 6), squeeze=False)
        for i in range(B):
            axes[0, i].imshow(arr[i, :, :, 0], cmap='gray')
            axes[0, i].set_title(f"{i} - OFF")
            axes[1, i].imshow(arr[i, :, :, 1], cmap='gray')
            axes[1, i].set_title(f"{i} - ON")
    elif C == 3:
        mean = np.array((0.485, 0.456, 0.406), dtype=arr.dtype).reshape(1, 1, 1, 3)
        std  = np.array((0.229, 0.224, 0.225), dtype=arr.dtype).reshape(1, 1, 1, 3)
        arr = arr * std + mean
        fig, axes = plt.subplots(1, B, figsize=(B * 3, 3), squeeze=False)
        for i in range(B):
            axes[0, i].imshow(arr[i])
            axes[0, i].set_title(f"Frame {i}")
            axes[0, i].axis('off')
    else:
        raise ValueError(f"Unsupported number of channels: {C}.")
    plt.tight_layout()
    plt.show()

def combine_polarities(frames):
    p0 = -frames[:, :, :, 0]
    p1 = frames[:, :, :, 1]
    mask = np.abs(p0) > np.abs(p1)
    return np.where(mask, p0, p1)

def plot_combined_frames(combined_frames):
    n = len(combined_frames)
    _, ax = plt.subplots(1, n, figsize=(n * 3, 3))
    for i, cf in enumerate(combined_frames):
        ax[i].imshow(cf, cmap='gray')
        ax[i].set_title(f"Combined {i}")
        ax[i].axis('off')
    plt.show()

def plot_aggregated_cam(frames, cams, **kwargs):
    if isinstance(frames, torch.Tensor):
        if frames.dim() == 5:
            frames = frames[0]
        first = frames[0:1]
    else:
        frames = np.asarray(frames)
        if frames.ndim == 5:
            frames = frames[0]
        first = frames[0:1]

    if isinstance(cams, torch.Tensor):
        agg = cams.mean(dim=0)
        mn, mx = agg.min(), agg.max()
        agg = (agg - mn) / (mx - mn + 1e-8) if mx > mn else torch.zeros_like(agg)
        cams_1t = agg.unsqueeze(0)
    else:
        cams = np.asarray(cams, dtype=np.float32)
        agg = cams.mean(axis=0)
        mn, mx = float(agg.min()), float(agg.max())
        agg = (agg - mn) / (mx - mn + 1e-8) if mx > mn else np.zeros_like(agg, dtype=np.float32)
        cams_1t = agg[None]

    return plot_cam_time_series(
        frames=first,
        cams=cams_1t,
        frame_stride=1,
        max_frames=1,
        show=kwargs.get("show", True),
        save_path=kwargs.get("save_path", None),
        overlay_alpha=kwargs.get("overlay_alpha", 0.4),
        cmap=kwargs.get("cmap", "jet"),
        figsize=kwargs.get("figsize", (5, 4)),
    )

def _to_numpy(frames):
    if isinstance(frames, torch.Tensor):
        f = frames.detach().cpu()
        if f.dim() == 5:
            assert f.size(0) == 1
            f = f[0]
        return f.numpy().astype(np.float32)
    f = np.asarray(frames, dtype=np.float32)
    if f.ndim == 5:
        assert f.shape[0] == 1
        f = f[0]
    return f


def denormalize(tensor):
    """Un-normalize ImageNet-like tensors/arrays for display.

    Works with torch.Tensor or numpy.ndarray. Assumes the last two dimensions
    are (H, W) and the channel dimension is at -3. Returns the result with the
    same type and shape as the input.
    """
    is_torch = isinstance(tensor, torch.Tensor)

    if is_torch:
        x = tensor.detach()
        # operate on CPU to simplify broadcasting / numpy-like behavior
        if x.device.type != "cpu":
            x = x.cpu()
        dtype = x.dtype
        C = x.size(-3)
        if C not in (1, 3):
            return x
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=dtype)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=dtype)
        if C == 1:
            mean = mean[[0]]
            std = std[[0]]
        # build broadcastable shape: [1,...,C,1,1] matching x.ndim
        shape = [1] * x.dim()
        shape[-3] = C
        mean = mean.view(*shape)
        std = std.view(*shape)
        return std * x + mean

    # numpy path
    arr = np.asarray(tensor)
    C = arr.shape[-3] if arr.ndim >= 3 else 1
    if C not in (1, 3):
        return arr
    dtype = arr.dtype
    mean = np.array([0.485, 0.456, 0.406], dtype=dtype)
    std = np.array([0.229, 0.224, 0.225], dtype=dtype)
    if C == 1:
        mean = mean[[0]]
        std = std[[0]]
    # reshape for broadcasting
    shape = [1] * arr.ndim
    shape[-3] = C
    mean = mean.reshape(shape)
    std = std.reshape(shape)
    return std * arr + mean

def plot_cam_time_series(
    frames,
    cams,
    frame_stride=1,
    max_frames=24,
    overlay_alpha=0.3,
    cmap="jet",
    figsize=(14, 6),
    show=True,
    save_path=None,
    image_rows=1,           # number of rows for the image grid
    images_per_row=None,    # optionally fix number of images per row
    show_colorbar=False,     # show colorbar legend for CAM values
    colorbar_fraction=0.04, # fraction width of colorbar
    colorbar_pad=0.02,      # pad between plot and colorbar
):
    # frames: (T,C,H,W) or (1,T,C,H,W); cams: (T,H,W)
    frames_np = _to_numpy(frames)
    assert frames_np.ndim == 4
    T, C, H, W = frames_np.shape

    denormalized = False
    if C == 3 and frames_np.min() < 0:
        frames_np = denormalize(torch.from_numpy(frames_np)).numpy()
        denormalized = True

    cams_np = (
        cams.detach().cpu().numpy().astype(np.float32)
        if isinstance(cams, torch.Tensor)
        else np.asarray(cams, dtype=np.float32)
    )
    assert cams_np.shape[0] == T

    frames_vis = frames_np

    # Select which timesteps to show
    indices = list(range(0, T, frame_stride))
    if len(indices) > max_frames:
        sel = np.linspace(0, len(indices) - 1, max_frames).round().astype(int)
        indices = [indices[i] for i in sel]

    cam_show = cams_np
    n_frames = len(indices)

    # Determine grid layout
    if images_per_row is not None and images_per_row > 0:
        cols = max(1, min(images_per_row, n_frames))
        rows = int(np.ceil(n_frames / cols))
    else:
        rows = max(1, image_rows)
        cols = int(np.ceil(n_frames / rows))

    # Auto figure size if user didn't explicitly tune it
    auto_figsize = (figsize is None) or (image_rows > 1 and figsize == (14, 6))
    if auto_figsize:
        per_col_w = 3.0   # width (inches) per image column
        per_row_h = 3.0   # height (inches) per image row
        fig_w = max(6.0, cols * per_col_w)
        fig_h = rows * per_row_h
        figsize = (fig_w, fig_h)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle("CAM overlaid on input frames" + (" (denormalized)" if denormalized else ""), y=0.98)
    axes = np.atleast_2d(axes)

    for ci, t in enumerate(indices):
        r, c = divmod(ci, cols)
        ax = axes[r, c]
        frame = frames_vis[t]
        if C == 1:
            ax.imshow(frame[0], cmap="gray", interpolation="nearest")
        else:
            ax.imshow(frame.transpose(1, 2, 0), interpolation="nearest")
        ax.imshow(cam_show[t], cmap=cmap, alpha=overlay_alpha, interpolation="nearest")
        ax.set_title(f"t={t}")
        ax.axis("off")

    # Hide any unused subplots
    total_slots = rows * cols
    for ci in range(n_frames, total_slots):
        r, c = divmod(ci, cols)
        axes[r, c].axis("off")

    # Add a single colorbar for the CAMs if requested
    if show_colorbar:
        vmin = float(cam_show.min())
        vmax = float(cam_show.max())
        mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
        mappable.set_array(cam_show)
        # Place colorbar to the right of all axes
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(axes[0, -1])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(mappable, cax=cax, orientation='vertical', label='CAM')

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    return fig
