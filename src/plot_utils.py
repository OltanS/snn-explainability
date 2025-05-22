import matplotlib.pyplot as plt

def plot_frames(frames):
    n_frames = len(frames)
    fig, axes = plt.subplots(2, n_frames, figsize=(n_frames * 3, 6))
    
    for i, frame in enumerate(frames):
        # frame[0] is polarity 0 and frame[1] is polarity 1
        axes[0, i].imshow(frame[1], cmap='gray')
        axes[0, i].set_title("Polarity OFF (0)")
        axes[1, i].imshow(frame[0], cmap='gray')
        axes[1, i].set_title("Polarity ON (1)")
    plt.tight_layout()
    plt.show()

def plot_combined_frames(combined_frames):
    n_frames = len(combined_frames)
    h, w = combined_frames[0].shape
    _, ax = plt.subplots(1, n_frames, figsize=(n_frames * 3, 3))
    for i, combined_frame in enumerate(combined_frames):
        ax[i].imshow(combined_frame, cmap='gray')
        ax[i].set_title(f"Combined Polarity {i}")
        ax[i].axis('off')
    plt.show()