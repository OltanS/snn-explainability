import os

from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets.n_mnist import NMNIST
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS

from PIL import Image
from pycocotools.coco import COCO

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
            v2.RandomResizedCrop((64, 64), scale=(0.25, 1.0), ratio=(3.0/4.0, 4.0/3.0)) if is_training else v2.Resize((64, 64)),
            v2.RandomHorizontalFlip() if is_training else v2.Lambda(lambda x: x),
            (v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1) if is_training else v2.Lambda(lambda x: x)),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
    
    # standard augmentations for image datasets
    elif "coco" in dataset_name or "voc" in dataset_name:
        return v2.Compose([
            v2.Resize((256, 256)),
            v2.ToImage(),
            v2.RandomResizedCrop((224, 224), scale=(0.5, 1)) if is_training else v2.Resize((224, 224)),
            v2.RandomHorizontalFlip() if is_training else v2.Lambda(lambda x: x),
            (v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1) if is_training else v2.Lambda(lambda x: x)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    elif "imagenet" in dataset_name:
        return v2.Compose([
            v2.Resize((256, 256)),
            v2.ToImage(),
            v2.RandomResizedCrop((224, 224)) if is_training else v2.Resize((224, 224)),
            v2.RandomHorizontalFlip() if is_training else v2.Lambda(lambda x: x),
            (v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1) if is_training else v2.Lambda(lambda x: x)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    else:
        raise ValueError(f"Dataset {dataset_name} is not implemented.")

def get_datasets(dataset_name, n_time_bins):
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
        train_ds = NMNIST(root=config["DATA_SAVE_DIR"] + "/NMNIST", data_type="frame", frames_number=n_time_bins, split_by='number', transform=train_transforms, train=True)
        test_ds = NMNIST(root=config["DATA_SAVE_DIR"] + "/NMNIST", data_type="frame", frames_number=n_time_bins, split_by='number', transform=test_transforms, train=False)

    elif dataset_name == "cifar10_dvs":
        # assumes same transforms for train and test, which currently is the case.
        full_ds = CIFAR10DVS(root=config["DATA_SAVE_DIR"] + "/CIFAR10DVS", data_type="frame", frames_number=n_time_bins, split_by='number', transform=train_transforms)
        n_train = int(0.8 * len(full_ds))
        n_val = len(full_ds) - n_train

        # deterministic split of the dataset
        generator = torch.Generator().manual_seed(42)
        train_ds, test_ds = torch.utils.data.random_split(full_ds, [n_train, n_val], generator=generator)

    elif dataset_name == "dvs_gesture":
        train_ds = DVS128Gesture(root=config["DATA_SAVE_DIR"] + "/DVS128Gesture", data_type="frame", frames_number=n_time_bins, split_by='number', transform=train_transforms, train=True)
        test_ds = DVS128Gesture(root=config["DATA_SAVE_DIR"] + "/DVS128Gesture", data_type="frame", frames_number=n_time_bins, split_by='number', transform=test_transforms, train=False)

    elif dataset_name == "coco":
        train_ds = datasets.CocoDetection(
            root=f'{config["DATA_SAVE_DIR"]}/CocoDetection/train2014',
            annFile=f'{config["DATA_SAVE_DIR"]}/CocoDetection/annotations/instances_train2014.json',
            transform=train_transforms,
        )
        
        test_ds = datasets.CocoDetection(
            root=f'{config["DATA_SAVE_DIR"]}/CocoDetection/val2014',
            annFile=f'{config["DATA_SAVE_DIR"]}/CocoDetection/annotations/instances_val2014.json',
            transform=test_transforms,
        )

    elif dataset_name == "coco_multilabel":
        class COCOMultiLabel(torch.utils.data.Dataset):
            def __init__(self, root_imgs, ann_file, transform=None):
                self.root_imgs = root_imgs
                self.transform = transform
                self.coco = COCO(ann_file)
                # COCO category ids are not contiguous; build mapping to [0..N-1]
                cats = self.coco.loadCats(self.coco.getCatIds())
                self.classes = [c['name'] for c in cats]
                self.cat_id_to_idx = {c['id']: i for i, c in enumerate(cats)}
                self.img_ids = self.coco.getImgIds()

            def __len__(self):
                return len(self.img_ids)

            def __getitem__(self, idx):
                img_id = self.img_ids[idx]
                info = self.coco.loadImgs([img_id])[0]
                path = os.path.join(self.root_imgs, info['file_name'])
                img = Image.open(path).convert("RGB")
                ann_ids = self.coco.getAnnIds(imgIds=[img_id])
                anns = self.coco.loadAnns(ann_ids)
                multi_hot = torch.zeros(len(self.classes), dtype=torch.float32)
                for a in anns:
                    cid = a['category_id']
                    if cid in self.cat_id_to_idx:
                        multi_hot[self.cat_id_to_idx[cid]] = 1.0
                if self.transform:
                    img = self.transform(img)
                return img, multi_hot

        train_ds = COCOMultiLabel(
            root_imgs=f'{config["DATA_SAVE_DIR"]}/CocoDetection/train2014',
            ann_file=f'{config["DATA_SAVE_DIR"]}/CocoDetection/annotations/instances_train2014.json',
            transform=train_transforms
        )
        test_ds = COCOMultiLabel(
            root_imgs=f'{config["DATA_SAVE_DIR"]}/CocoDetection/val2014',
            ann_file=f'{config["DATA_SAVE_DIR"]}/CocoDetection/annotations/instances_val2014.json',
            transform=test_transforms
        )

    elif dataset_name == "voc_detection":
        train_ds = datasets.VOCDetection(
            root=f'{config["DATA_SAVE_DIR"]}',
            year='2007',
            image_set='train',
            download=False,
            transform=train_transforms,
        )
        
        test_ds = datasets.VOCDetection(
            root=f'{config["DATA_SAVE_DIR"]}',
            year='2007',
            image_set='val',
            download=False,
            transform=test_transforms,
        )

    elif dataset_name == "voc_multilabel":
        class VOCMultiLabel(datasets.VOCDetection):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.classes = [
                    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
                    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
                    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
                ]
                self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

            def __getitem__(self, index):
                img, target = super().__getitem__(index)
                objects = target['annotation'].get('object', [])
                if not isinstance(objects, list):
                    objects = [objects]
                multi_hot = torch.zeros(len(self.classes), dtype=torch.float32)
                for obj in objects:
                    name = obj['name']
                    if name in self.class_to_idx:
                        multi_hot[self.class_to_idx[name]] = 1.0
                return img, multi_hot

        train_ds = VOCMultiLabel(
            root=f'{config["DATA_SAVE_DIR"]}',
            year='2007',
            image_set='train',
            download=False,
            transform=train_transforms,
        )
        
        test_ds = VOCMultiLabel(
            root=f'{config["DATA_SAVE_DIR"]}',
            year='2007',
            image_set='val',
            download=False,
            transform=test_transforms,
        )
    
    elif dataset_name == "imagenet":
        train_ds = datasets.ImageNet(
            root=f'{config["DATA_SAVE_DIR"]}/ImageNet',
            split='train',
            transform=train_transforms,
        )
        
        test_ds = datasets.ImageNet(
            root=f'{config["DATA_SAVE_DIR"]}/ImageNet',
            split='val',
            transform=test_transforms,
        )

    else:
        raise ValueError(f"Dataset {dataset_name} is not implemented.")
        
    return train_ds, test_ds

def get_dataloaders(dataset_name, batch_size, n_time_bins, on_ddp=False):
    '''
    Dataloaders always load data of shape batch_size x time_bins x num_polarities x height x width
    '''
    train_ds, test_ds = get_datasets(dataset_name, n_time_bins)

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
