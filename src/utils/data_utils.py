import os

from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets.n_mnist import NMNIST
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import v2

from src.utils.config_utils import setup_config_dictionary
    
def get_dataset_transforms(dataset_name, is_training):
    config = setup_config_dictionary()

    if dataset_name == "tiny_imagenet":
        return v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomResizedCrop((64, 64), scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0)) if is_training else v2.Resize((64, 64)),
            v2.RandomHorizontalFlip(),
            (v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1) if is_training else v2.Lambda(lambda x: x)),
            # we must scale the images to [0, 1] for encoding to work
            v2.Lambda(lambda x: x / 255.0),
        ])
    
    elif dataset_name == "nmnist":
        # this is 34 x 34.
        return v2.Compose([
            v2.Lambda(lambda x: torch.from_numpy(x)),
            v2.ToDtype(torch.float32, scale=False),
        ])

    elif dataset_name == "cifar10_dvs":
        return v2.Compose([
            v2.Lambda(lambda x: torch.from_numpy(x)),
            v2.ToDtype(torch.float32, scale=False),
            v2.RandomHorizontalFlip() if is_training else v2.Lambda(lambda x: x),
            v2.RandomVerticalFlip() if is_training else v2.Lambda(lambda x: x),
        ])
    
    elif dataset_name == "dvs_gesture":
        return v2.Compose([
            v2.Lambda(lambda x: torch.from_numpy(x)),
            v2.ToDtype(torch.float32, scale=False),
        ])
    
    elif dataset_name == "ucf101":
        return v2.Compose([
            v2.Resize((256, 256)),
            v2.ToImage(),
            v2.RandomCrop((224, 224)),
            v2.RandomHorizontalFlip(),
            (v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1) if is_training else v2.Lambda(lambda x: x)),
            v2.RandomRotation(degrees=10),
            v2.ToDtype(torch.float32, scale=True),
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
        
        class TinyImageNetVal(Dataset):
            def __init__(self, root, transform=None):
                self.root = root
                self.transform = transform
                images_dir = os.path.join(root, "images")
                ann_path = os.path.join(root, "val_annotations.txt")

                with open(ann_path, "r") as f:
                    lines = [l.strip().split("\t") for l in f if l.strip()]
                # filename -> wnid
                fname_to_cls = {fname: wnid for (fname, wnid, *_ ) in lines}

                classes = sorted(set(fname_to_cls.values()))
                self.class_to_idx = {c: i for i, c in enumerate(classes)}
                self.classes = classes

                self.samples = []
                for fname, wnid in fname_to_cls.items():
                    path = os.path.join(images_dir, fname)
                    if os.path.isfile(path):
                        self.samples.append((path, self.class_to_idx[wnid]))

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                path, target = self.samples[idx]
                img = Image.open(path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                return img, target

        test_ds = TinyImageNetVal(VAL_DIR, transform=test_transforms)

    elif dataset_name == "nmnist":
        train_ds = NMNIST(root=config["DATA_SAVE_DIR"] + "/NMNIST", data_type="frame", frames_number=config["N_TIME_BINS"], split_by='number', transform=train_transforms, train=True)
        test_ds = NMNIST(root=config["DATA_SAVE_DIR"] + "/NMNIST", data_type="frame", frames_number=config["N_TIME_BINS"], split_by='number', transform=test_transforms, train=False)

    elif dataset_name == "cifar10_dvs":
        # assumes same transforms for train and test, which currently is the case.
        full_ds = CIFAR10DVS(root=config["DATA_SAVE_DIR"] + "/CIFAR10DVS", data_type="frame", frames_number=config["N_TIME_BINS"], split_by='number', transform=train_transforms)
        n_train = int(0.8 * len(full_ds))
        n_val = len(full_ds) - n_train

        # deterministic split of the dataset
        generator = torch.Generator().manual_seed(42)
        train_ds, test_ds = torch.utils.data.random_split(full_ds, [n_train, n_val], generator=generator)

    elif dataset_name == "dvs_gesture":
        train_ds = DVS128Gesture(root=config["DATA_SAVE_DIR"] + "/DVS128Gesture", data_type="frame", frames_number=config["N_TIME_BINS"], split_by='number', transform=train_transforms, train=True)
        test_ds = DVS128Gesture(root=config["DATA_SAVE_DIR"] + "/DVS128Gesture", data_type="frame", frames_number=config["N_TIME_BINS"], split_by='number', transform=test_transforms, train=False)

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
        # collate_fn=tonic.collation.PadTensors(), 
        num_workers=4,
        prefetch_factor=5,
        persistent_workers=True,
        # pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_ds, 
        batch_size=batch_size, 
        shuffle=(test_sampler is None), 
        # collate_fn=tonic.collation.PadTensors(), 
        num_workers=4,
        prefetch_factor=5,
        persistent_workers=True,
        # pin_memory=True,
        sampler=test_sampler,
    )
    
    return train_loader, test_loader 
