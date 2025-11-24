import torch
import matplotlib.pyplot as plt
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
        per_frame_cam_norm=False,
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

def _norm_frames(frames_np):
    T, C, _, _ = frames_np.shape
    flat = frames_np.reshape(T, C, -1)
    fmin = flat.min(axis=2).reshape(T, C, 1, 1)
    fmax = flat.max(axis=2).reshape(T, C, 1, 1)
    return ((frames_np - fmin) / (fmax - fmin + 1e-8)).clip(0, 1)

def _norm_cams(cams_np, per_frame=True):
    if per_frame:
        out = np.empty_like(cams_np, dtype=np.float32)
        for t in range(cams_np.shape[0]):
            m, M = cams_np[t].min(), cams_np[t].max()
            out[t] = (cams_np[t] - m) / (M - m + 1e-8) if M > m else 0.0
        return out
    m, M = cams_np.min(), cams_np.max()
    return (cams_np - m) / (M - m + 1e-8) if M > m else np.zeros_like(cams_np, dtype=np.float32)

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
    image_rows=1,           # new: number of rows for the image grid
    images_per_row=None,    # new: optionally fix number of images per row
):
    # frames: (T,C,H,W) or (1,T,C,H,W); cams: (T,H,W)
    frames_np = _to_numpy(frames)
    assert frames_np.ndim == 4
    T, C, H, W = frames_np.shape

    cams_np = cams.detach().cpu().numpy().astype(np.float32) if isinstance(cams, torch.Tensor) else np.asarray(cams, dtype=np.float32)
    assert cams_np.shape[0] == T

    frames_vis = _norm_frames(frames_np)

    flatc = cams_np.reshape(T, -1)
    if callable(aggregate):
        cam_weights = np.asarray(aggregate(cams_np), dtype=float)
    else:
        if aggregate == "mean":
            cam_weights = flatc.mean(axis=1)
        elif aggregate == "max":
            cam_weights = flatc.max(axis=1)
        else:
            raise ValueError("Unknown aggregate.")
    indices = list(range(0, T, frame_stride))
    if len(indices) > max_frames:
        sel = np.linspace(0, len(indices) - 1, max_frames).round().astype(int)
        indices = [indices[i] for i in sel]

    cam_show = _norm_cams(cams_np, per_frame_cam_norm)

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
        ts_h = 2.0        # time-series height
        fig_w = max(6.0, cols * per_col_w)
        fig_h = rows * per_row_h + ts_h
        figsize = (fig_w, fig_h)

    fig = plt.figure(figsize=figsize)

    # Outer grid: top = images grid, bottom = time-series
    outer = fig.add_gridspec(2, 1, height_ratios=[max(1, rows) * 5, 1])
    grid = outer[0].subgridspec(rows, cols)

    for ci, t in enumerate(indices):
        r, c = divmod(ci, cols)
        ax = fig.add_subplot(grid[r, c])
        frame = frames_vis[t]
        if C == 1:
            ax.imshow(frame[0], cmap="gray", interpolation="nearest")
        else:
            ax.imshow(frame.transpose(1, 2, 0), interpolation="nearest")
        ax.imshow(cam_show[t], cmap=cmap, alpha=overlay_alpha, interpolation="nearest")
        ax.set_title(f"t={t}")
        ax.axis("off")

    ax_ts = fig.add_subplot(outer[1])
    ax_ts.plot(range(T), cam_weights, color="tab:red")
    ax_ts.scatter(indices, cam_weights[indices], color="black", s=14, zorder=3)
    ax_ts.set_xlim(0, T - 1)
    ax_ts.set_xlabel("Time")
    ax_ts.set_ylabel("CAM")
    ax_ts.grid(alpha=0.25)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    return fig