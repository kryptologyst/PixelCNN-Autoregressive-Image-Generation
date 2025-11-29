"""
Data loading and preprocessing utilities for PixelCNN training.
"""

from typing import Tuple, Optional, Dict, Any, List
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import os


class CIFAR10Dataset(Dataset):
    """CIFAR-10 dataset with proper preprocessing for PixelCNN."""
    
    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        download: bool = True,
        transform: Optional[Any] = None
    ):
        """
        Initialize CIFAR-10 dataset.
        
        Args:
            root: Root directory for dataset
            train: Whether to use training set
            download: Whether to download dataset
            transform: Optional transform to apply
        """
        self.dataset = datasets.CIFAR10(
            root=root, train=train, download=download, transform=None
        )
        self.transform = transform
        
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get item from dataset."""
        image, label = self.dataset[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class CelebADataset(Dataset):
    """CelebA dataset for higher resolution training."""
    
    def __init__(
        self,
        root: str = "./data/celeba",
        split: str = "train",
        transform: Optional[Any] = None,
        max_samples: Optional[int] = None
    ):
        """
        Initialize CelebA dataset.
        
        Args:
            root: Root directory for dataset
            split: Dataset split ("train", "val", "test")
            transform: Optional transform to apply
            max_samples: Maximum number of samples to use
        """
        self.root = root
        self.split = split
        self.transform = transform
        
        # Load image paths
        split_file = os.path.join(root, f"list_eval_partition.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"CelebA split file not found at {split_file}")
        
        self.image_paths = []
        split_mapping = {"train": 0, "val": 1, "test": 2}
        target_split = split_mapping[split]
        
        with open(split_file, 'r') as f:
            for line in f:
                img_name, split_id = line.strip().split()
                if int(split_id) == target_split:
                    self.image_paths.append(os.path.join(root, "img_align_celeba", img_name))
        
        if max_samples:
            self.image_paths = self.image_paths[:max_samples]
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get item from dataset."""
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, 0  # CelebA doesn't have meaningful labels


def get_transforms(
    image_size: int = 32,
    is_training: bool = True,
    use_augmentation: bool = True
) -> transforms.Compose:
    """
    Get data transforms for training/validation.
    
    Args:
        image_size: Target image size
        is_training: Whether transforms are for training
        use_augmentation: Whether to use data augmentation
        
    Returns:
        Composed transforms
    """
    transform_list = []
    
    if is_training and use_augmentation:
        # Training transforms with augmentation
        transform_list.extend([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        ])
    else:
        # Validation transforms without augmentation
        transform_list.append(transforms.Resize((image_size, image_size)))
    
    # Common transforms
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    
    return transforms.Compose(transform_list)


def get_albumentations_transforms(
    image_size: int = 32,
    is_training: bool = True,
    use_augmentation: bool = True
) -> A.Compose:
    """
    Get Albumentations transforms for more advanced augmentation.
    
    Args:
        image_size: Target image size
        is_training: Whether transforms are for training
        use_augmentation: Whether to use data augmentation
        
    Returns:
        Albumentations compose transform
    """
    transform_list = []
    
    if is_training and use_augmentation:
        transform_list.extend([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
                A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=1.0),
            ], p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ], p=0.3),
        ])
    else:
        transform_list.append(A.Resize(image_size, image_size))
    
    transform_list.extend([
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ])
    
    return A.Compose(transform_list)


def create_dataloaders(
    dataset_name: str = "cifar10",
    batch_size: int = 64,
    num_workers: int = 4,
    image_size: int = 32,
    data_root: str = "./data",
    use_augmentation: bool = True,
    max_samples: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        dataset_name: Name of dataset ("cifar10", "celeba")
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        image_size: Target image size
        data_root: Root directory for data
        use_augmentation: Whether to use data augmentation
        max_samples: Maximum number of samples to use
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if dataset_name.lower() == "cifar10":
        # CIFAR-10 dataset
        train_transform = get_transforms(image_size, is_training=True, use_augmentation=use_augmentation)
        val_transform = get_transforms(image_size, is_training=False, use_augmentation=False)
        
        train_dataset = CIFAR10Dataset(root=data_root, train=True, transform=train_transform)
        val_dataset = CIFAR10Dataset(root=data_root, train=False, transform=val_transform)
        test_dataset = CIFAR10Dataset(root=data_root, train=False, transform=val_transform)
        
    elif dataset_name.lower() == "celeba":
        # CelebA dataset
        train_transform = get_transforms(image_size, is_training=True, use_augmentation=use_augmentation)
        val_transform = get_transforms(image_size, is_training=False, use_augmentation=False)
        
        train_dataset = CelebADataset(root=data_root, split="train", transform=train_transform, max_samples=max_samples)
        val_dataset = CelebADataset(root=data_root, split="val", transform=val_transform, max_samples=max_samples//10 if max_samples else None)
        test_dataset = CelebADataset(root=data_root, split="test", transform=val_transform, max_samples=max_samples//10 if max_samples else None)
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader


def denormalize_image(tensor: torch.Tensor, mean: List[float] = [0.5, 0.5, 0.5], std: List[float] = [0.5, 0.5, 0.5]) -> torch.Tensor:
    """
    Denormalize a tensor image.
    
    Args:
        tensor: Normalized tensor image
        mean: Mean values used for normalization
        std: Standard deviation values used for normalization
        
    Returns:
        Denormalized tensor image
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return tensor * std + mean
