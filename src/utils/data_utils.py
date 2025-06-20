import os
import random

import tonic
import tonic.transforms as T
import snntorch.spikegen as spikegen

import torch
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.transforms import transforms as v1

from src.utils.config_utils import setup_config_dictionary

class RandomTemporalMask:
    '''
    Inspired by the augmentation shown in the supplemental material of "Deep Residual Learning in Spiking Neural Networks",
    this augmentation randomly 0's out some of the input frames.
    This option is chosen over one that deletes to keep the input dimensions consistent.
    '''
    def __init__(self, mask_ratio, temporal_dim=0, train=True):
        self.mask_ratio = mask_ratio
        self.temporal_dim = temporal_dim
        self.train = train
        self.mask_value=0.0

    def __call__(self, x):
        if not self.train:
            return x

        T = x.shape[self.temporal_dim]
        num_to_mask = int(T * self.mask_ratio)


        # Check if input is a PyTorch tensor or NumPy array
        if hasattr(x, 'clone'):  # PyTorch tensor
            masked_x = x.clone()
        else:  # NumPy array
            masked_x = x.copy()

        mask_indices = random.sample(range(T), num_to_mask)
        index_list = [slice(None)] * len(x.shape)
        index_list[self.temporal_dim] = mask_indices
        masked_x[tuple(index_list)] = self.mask_value
        
        return masked_x
    
def get_dataset_transforms(dataset_name, is_training):
    config = setup_config_dictionary()
    
    if dataset_name == "tiny_imagenet":
        return v2.Compose([
            v2.Resize((80, 80)),
            v2.ToImage(),
            v2.RandomCrop((64, 64)),
            v2.RandomHorizontalFlip(),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            v2.RandomRotation(degrees=10),
            v2.ToDtype(torch.float32, scale=True),
            lambda x : spikegen.rate(x, num_steps=config["N_TIME_BINS"]),
        ])
    
    elif dataset_name == "dvs_gesture":
        sensor_size = tonic.datasets.DVSGesture.sensor_size
        return v2.Compose([
            #T.Denoise(filter_time=10000),
            T.ToFrame(sensor_size=sensor_size, n_time_bins=config["N_TIME_BINS"]),
            v2.Lambda(lambda x: torch.from_numpy(x)),
            v1.Resize((160,160)),
            v1.RandomCrop((128, 128)),
            v1.RandomRotation(degrees=10),
            v2.ToDtype(torch.float32, scale=False), # scale doesn't seem to work well here, as the range isn't fixed
            RandomTemporalMask(mask_ratio=config["MASK_RATIO"], temporal_dim=0, train=is_training),
        ])
    
    elif dataset_name == "ucf101":
        return v2.Compose([
            v2.Resize((256, 256)),
            v2.ToImage(),
            v2.RandomCrop((224, 224)),
            v2.RandomHorizontalFlip(),
            v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            v2.RandomRotation(degrees=10),
            v2.ToDtype(torch.float32, scale=True),
            RandomTemporalMask(mask_ratio=config["MASK_RATIO"], temporal_dim=0, train=is_training),
            lambda x: spikegen.rate(x, time_var_input=True)
        ])
    
    else:
        raise ValueError(f"Dataset {dataset_name} is not implemented.")

def get_datasets(dataset_name):
    config = setup_config_dictionary()
    train_transforms = get_dataset_transforms(dataset_name, is_training=True)
    test_transforms = get_dataset_transforms(dataset_name, is_training=False)
    if dataset_name == "tiny_imagenet":
        DATA_DIR = config["DATA_SAVE_DIR"] + "/tiny-imagenet-200"
        TRAIN_DIR = os.path.join(DATA_DIR, "train")
        VAL_DIR = os.path.join(DATA_DIR, "val")
        VAL_IMG_DIR = os.path.join(VAL_DIR, "images")
        VAL_ANN = os.path.join(VAL_DIR, "val_annotations.txt")

        train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
        test_ds = datasets.ImageFolder(VAL_DIR, transform=test_transforms)

    elif dataset_name == "dvs_gesture":
        train_ds = tonic.datasets.DVSGesture(save_to=config["DATA_SAVE_DIR"], transform=train_transforms, train=True)
        test_ds = tonic.datasets.DVSGesture(save_to=config["DATA_SAVE_DIR"], transform=test_transforms, train=False)

    elif dataset_name == "ucf101":
        # torchvision spams us with deprecation warnings for UCF101, telling us to go do torchcodec...
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            from torchvision.datasets import UCF101
        
        class VideoOnlyUCF101(UCF101):
            """UCF101 dataset wrapper that only returns video and labels"""   
            def __getitem__(self, idx):
                video, audio, label = super().__getitem__(idx)
                return video, label  # Drop audio completely
            
        train_ds = VideoOnlyUCF101(
            root=f'{config["DATA_SAVE_DIR"]}/UCF-101',
            annotation_path=f'{config["DATA_SAVE_DIR"]}/ucfTrainTestlist',
            frames_per_clip=config["N_TIME_BINS"],
            step_between_clips=config["N_TIME_BINS"],
            frame_rate=None,
            train=True,
            transform=train_transforms,
            output_format='TCHW'  # TCHW: Time, Channels, Height, Width
        )

        test_ds = VideoOnlyUCF101(
            root=f'{config["DATA_SAVE_DIR"]}/UCF-101',
            annotation_path=f'{config["DATA_SAVE_DIR"]}/ucfTrainTestlist',
            frames_per_clip=config["N_TIME_BINS"],
            step_between_clips=config["N_TIME_BINS"],
            frame_rate=None,
            train=False,
            transform=test_transforms,
            output_format='TCHW'  # TCHW: Time, Channels, Height, Width
        )

    else:
        raise ValueError(f"Dataset {dataset_name} is not implemented.")
        
    return train_ds, test_ds

def get_dataloaders(dataset_name, batch_size, on_ddp=False):
    '''
    Dataloaders always load data of shape batch_size x time_bins x num_polarities x height x width
    '''
    train_ds, test_ds = get_datasets(dataset_name)
    
    if on_ddp:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_ds)
        test_sampler = DistributedSampler(test_ds)
    else:
        train_sampler = None
        test_sampler = None
    
    # PadTensors actually doesn't do anything since we use a fixed N_TIME_BINS in the transforms.
    train_loader = torch.utils.data.DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=(train_sampler is None), 
        collate_fn=tonic.collation.PadTensors(), 
        num_workers=2, 
        pin_memory=False, 
        sampler=train_sampler,
        drop_last=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_ds, 
        batch_size=batch_size, 
        shuffle=(test_sampler is None), 
        collate_fn=tonic.collation.PadTensors(), 
        num_workers=2, 
        pin_memory=False, 
        sampler=test_sampler,
        drop_last=True
    )
    
    return train_loader, test_loader 


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from PIL import Image
    import random
    config = setup_config_dictionary()
    # test temporal mask
    def test_temporal_mask():
        x = torch.randn(1, 10, 3, 4, 4)  # B, T, C, H, W
        mask = RandomTemporalMask(mask_ratio=0.5, temporal_dim=1, train=True)
        masked_x = mask(x)
        print(x[0])
        print(masked_x[0])
    
    def test_dvs_transforms():
        dataset_name = "dvs_gesture"
        out_dir = f"img/{dataset_name}"
        is_training = True
        transforms = get_dataset_transforms(dataset_name, is_training)
        dataset = tonic.datasets.DVSGesture(save_to=config["DATA_SAVE_DIR"], train=True)
        x, y = dataset[random.randint(0, len(dataset) - 1)]
        x_copy = x.copy()
        os.makedirs(out_dir, exist_ok=True)
        for i, transform in enumerate(transforms.transforms):
            print("Applying:", transform)
            x_copy = transform(x_copy)
            
            was_numpy = isinstance(x_copy, np.ndarray)
            try:
                if was_numpy:
                    x_copy = torch.from_numpy(x_copy)
                
            except Exception as e:
                print("Error converting to tensor:", e)
                continue
             
            if was_numpy:
                x_copy = x_copy.numpy()
            if isinstance(x_copy, np.ndarray):
                print("max:", x_copy[0, 0].max(), "min:", x_copy[0, 0].min(), "shape:", x_copy.shape)
                pic_version = x_copy[0, 0]
            else:  # tensor
                print("max:", x_copy[0, 0].max().item(), "min:", x_copy[0, 0].min().item(), "shape:", tuple(x_copy.shape))
                pic_version = x_copy[0, 0].numpy()

            fig = plt.figure(figsize=(10, 5))
            plt.imshow(pic_version, cmap='gray')
            fig.savefig(os.path.join(out_dir, f"{i}_{type(transform).__name__}.png"))

    def test_tiny_imagenet_transforms():
        dataset_name = "tiny_imagenet"
        out_dir = f"img/{dataset_name}"
        os.makedirs(out_dir, exist_ok=True)
        is_training = True
        transforms = get_dataset_transforms(dataset_name, is_training)
        DATA_DIR = config["DATA_SAVE_DIR"] + "/tiny-imagenet-200"
        TRAIN_DIR = os.path.join(DATA_DIR, "train")
        dataset = datasets.ImageFolder(TRAIN_DIR)
        x, y = dataset[random.randint(0, len(dataset) - 1)]  # PIL.Image initially
        for i, transform in enumerate(transforms.transforms):
            print("Applying:", transform)
            x = transform(x)
            try:
                x_copy = x.copy()
            except AttributeError:
                import copy
                x_copy = copy.copy(x)

            if isinstance(x, Image.Image):
                x = np.array(x)
            was_numpy = isinstance(x, np.ndarray)
            try:
                if was_numpy:
                    x = torch.from_numpy(x)
            except Exception as e:
                print("Error converting to tensor:", e)
                continue
            if was_numpy:
                x = x.numpy()
            # now x is either np.ndarray or torch.Tensor
            if isinstance(x, np.ndarray):
                print("max:", x.max(), "min:", x.min(), "shape:", x.shape)
                pic_version = x
            else:  # tensor
                print("max:", x.max().item(), "min:", x.min().item(), "shape:", tuple(x.shape))
                pic_version = x.numpy()

            if pic_version.ndim < 4:
                pic_version = np.expand_dims(pic_version, axis=0)

            n_imgs = pic_version.shape[0]
            fig = plt.figure(figsize=(5 * n_imgs, 5))
            for j in range(n_imgs):
                img = pic_version[j]
                # if channels‐first (C,H,W), move to HWC
                if img.ndim == 3 and img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                ax = fig.add_subplot(1, n_imgs, j + 1)
                ax.imshow(img)
                ax.axis("off")

            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"{i}_{type(transform).__name__}.png"))
            plt.close(fig)
            
            x = x_copy
    def test_ucf101_transforms():
        dataset_name = "ucf101"
        out_dir = f"img/{dataset_name}"
        os.makedirs(out_dir, exist_ok=True)
        is_training = True
        transforms = get_dataset_transforms(dataset_name, is_training)

        # torchvision spams us with deprecation warnings for UCF101, telling us to go do torchcodec...
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            from torchvision.datasets import UCF101

        class VideoOnlyUCF101(UCF101):
            """UCF101 dataset wrapper that only returns video and labels"""   
            def __getitem__(self, idx):
                video, audio, label = super().__getitem__(idx)
                return video, label  # Drop audio completely

        dataset = VideoOnlyUCF101(
            root=f'{config["DATA_SAVE_DIR"]}/UCF-101',
            annotation_path=f'{config["DATA_SAVE_DIR"]}/ucfTrainTestlist',
            frames_per_clip=config["N_TIME_BINS"],
            step_between_clips=config["N_TIME_BINS"] // 2,
            frame_rate=None,
            train=True,
            output_format='TCHW'  # TCHW: Time, Channels, Height, Width
        )
        x, y = dataset[random.randint(0, len(dataset) - 1)]  # x is a tensor with shape (T, C, H, W)
        
        print(f"Original video shape: {x.shape}")
        for i, transform in enumerate(transforms.transforms):
            print(f"Applying transform {i}: {transform}")
            x = transform(x)
            
            try:
                x_copy = x.clone() if hasattr(x, 'clone') else x.copy()
            except AttributeError:
                import copy
                x_copy = copy.copy(x)
            
            # Convert to numpy for visualization if it's a tensor
            if isinstance(x, torch.Tensor):
                print(f"max: {x.max().item():.4f}, min: {x.min().item():.4f}, shape: {tuple(x.shape)}")
                pic_version = x.numpy()
            else:
                print(f"max: {x.max():.4f}, min: {x.min():.4f}, shape: {x.shape}")
                pic_version = x
            
            # Handle different tensor shapes
            if pic_version.ndim == 4:  # (T, C, H, W)
                # Take first few frames for visualization
                n_frames = pic_version.shape[0]
                fig, axes = plt.subplots(1, n_frames, figsize=(5 * n_frames, 5))
                if n_frames == 1:
                    axes = [axes]
                
                for j in range(n_frames):
                    frame = pic_version[j]  # (C, H, W)
                    # Convert from CHW to HWC for matplotlib
                    frame = np.transpose(frame, (1, 2, 0))
                    
                    axes[j].imshow(frame)
                    axes[j].set_title(f'Frame {j}')
                    axes[j].axis('off')
            
            else:
                # Fallback for unexpected shapes
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                ax.imshow(pic_version.squeeze(), cmap='gray')
                ax.axis('off')
            
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, f"{i}_{type(transform).__name__}.png"))
            plt.close(fig)
            
            x = x_copy

    test_tiny_imagenet_transforms()
    test_dvs_transforms()
    test_ucf101_transforms()