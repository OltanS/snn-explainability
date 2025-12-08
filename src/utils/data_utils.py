import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import v2
from PIL import Image
from pycocotools.coco import COCO

from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets.n_mnist import NMNIST
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS

_BASE_DATASETS = [
    "tiny_imagenet", 
    "nmnist", 
    "cifar10_dvs", 
    "dvs_gesture", 
    "coco", 
    "coco_multilabel", 
    "voc_detection", 
    "voc_multilabel", 
    "imagenet"
]

AVAILABLE_DATASETS = _BASE_DATASETS + [f"{name}_composite" for name in _BASE_DATASETS]

class CompositeDataset(Dataset):
    def __init__(self, base_dataset, num_classes, double_width=True, force_different_class=True):
        self.base_dataset = base_dataset
        self.num_classes = num_classes
        self.double_width = double_width
        self.force_different_class = force_different_class
        
        if hasattr(base_dataset, 'targets'):
            self.targets = base_dataset.targets
        elif hasattr(base_dataset, 'dataset') and hasattr(base_dataset.dataset, 'targets'):
             self.targets = [base_dataset.dataset.targets[i] for i in base_dataset.indices]
        else:
            raise ValueError("Base dataset must have 'targets' attribute for efficient class checking.")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img_a, target_a = self.base_dataset[idx]
        label_a = int(target_a) if isinstance(target_a, (int, torch.Tensor)) else target_a

        while True:
            rand_idx = torch.randint(0, len(self.base_dataset), (1,)).item()
            if rand_idx == idx: continue
            
            if self.force_different_class:
                if self.targets is not None:
                    label_b = int(self.targets[rand_idx])
                    if label_a == label_b: continue
                else:
                    _, target_b = self.base_dataset[rand_idx]
                    label_b = int(target_b) if isinstance(target_b, (int, torch.Tensor)) else target_b
                    if label_a == label_b: continue
            break

        img_b, target_b = self.base_dataset[rand_idx]
        composite_img = torch.cat([img_a, img_b], dim=-1)

        label = torch.zeros(self.num_classes, dtype=torch.float32)
        label[label_a] = 1.0
        label[int(target_b)] = 1.0
            
        return composite_img, label

def get_dataset_transforms(dataset_name, is_training):
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

def get_datasets(dataset_name, data_dir, n_time_bins):    
    is_composite = False
    if dataset_name.endswith("_composite"):
        is_composite = True
        base_name = dataset_name.replace("_composite", "")
    else:
        base_name = dataset_name
    
    train_transforms = get_dataset_transforms(base_name, is_training=True)
    test_transforms = get_dataset_transforms(base_name, is_training=False)

    if base_name == "tiny_imagenet":
        DATA_DIR = os.path.join(data_dir, "tiny-imagenet-200")
        TRAIN_DIR = os.path.join(DATA_DIR, "train")
        VAL_DIR = os.path.join(DATA_DIR, "val")
        
        train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
        
        class TinyImageNetVal(Dataset):
            def __init__(self, root, transform=None):
                self.root = root
                self.transform = transform
                images_dir = os.path.join(root, "images")
                ann_path = os.path.join(root, "val_annotations.txt")

                with open(ann_path, "r") as f:
                    lines = [l.strip().split("\t") for l in f if l.strip()]
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

    elif base_name == "nmnist":
        train_ds = NMNIST(root=os.path.join(data_dir, "NMNIST"), data_type="frame", frames_number=n_time_bins, split_by='number', transform=train_transforms, train=True)
        test_ds = NMNIST(root=os.path.join(data_dir, "NMNIST"), data_type="frame", frames_number=n_time_bins, split_by='number', transform=test_transforms, train=False)

    elif base_name == "cifar10_dvs":
        full_ds = CIFAR10DVS(root=os.path.join(data_dir, "CIFAR10DVS"), data_type="frame", frames_number=n_time_bins, split_by='number', transform=train_transforms)
        n_train = int(0.8 * len(full_ds))
        n_val = len(full_ds) - n_train
        generator = torch.Generator().manual_seed(42)
        train_ds, test_ds = torch.utils.data.random_split(full_ds, [n_train, n_val], generator=generator)

    elif base_name == "dvs_gesture":
        train_ds = DVS128Gesture(root=os.path.join(data_dir, "DVS128Gesture"), data_type="frame", frames_number=n_time_bins, split_by='number', transform=train_transforms, train=True)
        test_ds = DVS128Gesture(root=os.path.join(data_dir, "DVS128Gesture"), data_type="frame", frames_number=n_time_bins, split_by='number', transform=test_transforms, train=False)

    elif base_name == "coco":
        train_ds = datasets.CocoDetection(
            root=os.path.join(data_dir, 'CocoDetection/train2014'),
            annFile=os.path.join(data_dir, 'CocoDetection/annotations/instances_train2014.json'),
            transform=train_transforms,
        )
        test_ds = datasets.CocoDetection(
            root=os.path.join(data_dir, 'CocoDetection/val2014'),
            annFile=os.path.join(data_dir, 'CocoDetection/annotations/instances_val2014.json'),
            transform=test_transforms,
        )
    elif base_name == "coco_multilabel":
        class COCOMultiLabel(torch.utils.data.Dataset):
            def __init__(self, root_imgs, ann_file, transform=None):
                self.root_imgs = root_imgs
                self.transform = transform
                self.coco = COCO(ann_file)
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
            root_imgs=os.path.join(data_dir, 'CocoDetection/train2014'),
            ann_file=os.path.join(data_dir, 'CocoDetection/annotations/instances_train2014.json'),
            transform=train_transforms
        )
        test_ds = COCOMultiLabel(
            root_imgs=os.path.join(data_dir, 'CocoDetection/val2014'),
            ann_file=os.path.join(data_dir, 'CocoDetection/annotations/instances_val2014.json'),
            transform=test_transforms
        )
    elif base_name == "voc_detection":
        train_ds = datasets.VOCDetection(
            root=data_dir,
            year='2007',
            image_set='train',
            download=False,
            transform=train_transforms,
        )
        test_ds = datasets.VOCDetection(
            root=data_dir,
            year='2007',
            image_set='val',
            download=False,
            transform=test_transforms,
        )
    elif base_name == "voc_multilabel":
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
            root=data_dir,
            year='2007',
            image_set='train',
            download=False,
            transform=train_transforms,
        )
        test_ds = VOCMultiLabel(
            root=data_dir,
            year='2007',
            image_set='val',
            download=False,
            transform=test_transforms,
        )
    
    elif base_name == "imagenet":
        train_ds = datasets.ImageNet(
            root=os.path.join(data_dir, 'ImageNet'),
            split='train',
            transform=train_transforms,
        )
        test_ds = datasets.ImageNet(
            root=os.path.join(data_dir, 'ImageNet'),
            split='val',
            transform=test_transforms,
        )

    else:
        raise ValueError(f"Dataset {dataset_name} is not implemented.")
        
    if is_composite:
        num_classes = len(train_ds.classes)
        train_ds = CompositeDataset(train_ds, num_classes=num_classes)
        test_ds = CompositeDataset(test_ds, num_classes=num_classes)

    return train_ds, test_ds

def get_dataloaders(dataset_name, data_dir, batch_size, n_time_bins, on_ddp=False, num_workers=4):
    # Added data_dir, num_workers arguments
    train_ds, test_ds = get_datasets(dataset_name, data_dir, n_time_bins)

    if on_ddp:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_ds)
        test_sampler = DistributedSampler(test_ds)
    else:
        train_sampler = None
        test_sampler = None
    
    train_loader = torch.utils.data.DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=(train_sampler is None), 
        num_workers=num_workers,
        prefetch_factor=5 if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
        sampler=train_sampler,
        drop_last=True,
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_ds, 
        batch_size=batch_size, 
        shuffle=(test_sampler is None), 
        num_workers=num_workers,
        prefetch_factor=5 if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
        sampler=test_sampler,
    )
    
    return train_loader, test_loader