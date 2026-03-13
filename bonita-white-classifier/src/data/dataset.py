"""
PyTorch Dataset for Bonita White Frame Classification

This module provides a custom Dataset class for loading and preprocessing
frames extracted from videos for the EfficientNet-B0 classifier.
"""

import os
import cv2
import torch
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms


logger = logging.getLogger(__name__)


class FrameDataset(Dataset):
    """
    Dataset for frame classification with support for train/val/test modes.

    Args:
        data_dir: Root directory containing class folders
        transform: Optional transform to be applied on images
        mode: 'train', 'val', or 'test'
        cache_images: Whether to cache images in memory
    """

    # ImageNet normalization for EfficientNet
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        mode: str = "train",
        cache_images: bool = False,
        img_size: int = 224,
    ):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.cache_images = cache_images
        self.img_size = img_size

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        # Get class names and create mappings
        self.classes = self._get_classes()
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        # Get all image paths and labels
        self.samples = self._load_samples()

        # Setup default transforms if none provided
        self.transform = transform or self._get_default_transforms()

        # Cache for images (optional)
        self._cache = {} if cache_images else None

        logger.info(
            f"Loaded {mode} dataset: {len(self.samples)} samples, {len(self.classes)} classes"
        )

    def _get_classes(self) -> List[str]:
        """Get sorted list of class names from directory structure."""
        classes = [d.name for d in self.data_dir.iterdir() if d.is_dir()]
        return sorted(classes)

    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load all image paths and their corresponding labels."""
        samples = []
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            class_idx = self.class_to_idx[class_name]

            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in valid_extensions:
                    samples.append((str(img_path), class_idx))

        return samples

    def _get_default_transforms(self) -> Callable:
        """Get default transforms based on mode."""
        if self.mode == "train":
            return transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((self.img_size, self.img_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=15),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                    ),
                    transforms.RandomAffine(
                        degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD
                    ),
                    transforms.RandomErasing(p=0.3),
                ]
            )
        else:  # val or test
            return transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((self.img_size, self.img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD
                    ),
                ]
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.

        Returns:
            Tuple of (image_tensor, label)
        """
        img_path, label = self.samples[idx]

        # Check cache
        if self._cache is not None and img_path in self._cache:
            image = self._cache[img_path]
        else:
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Failed to load image: {img_path}")

            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Cache if enabled
            if self._cache is not None:
                self._cache[img_path] = image

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for handling imbalanced datasets.

        Returns:
            Tensor of class weights
        """
        class_counts = [0] * len(self.classes)
        for _, label in self.samples:
            class_counts[label] += 1

        total_samples = sum(class_counts)
        weights = [
            total_samples / (len(self.classes) * count) for count in class_counts
        ]

        return torch.tensor(weights, dtype=torch.float32)

    def get_sampler(self) -> Optional[WeightedRandomSampler]:
        """Get weighted random sampler for handling class imbalance."""
        if self.mode != "train":
            return None

        weights = self.get_class_weights()
        sample_weights = [weights[label] for _, label in self.samples]

        return WeightedRandomSampler(
            weights=sample_weights, num_samples=len(self.samples), replacement=True
        )


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224,
    pin_memory: bool = True,
    use_weighted_sampler: bool = False,
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for train, val, and test sets.

    Args:
        data_dir: Root directory containing train/val/test folders
        batch_size: Batch size for all loaders
        num_workers: Number of worker processes
        img_size: Input image size
        pin_memory: Whether to pin memory for GPU training
        use_weighted_sampler: Whether to use weighted sampling for imbalance

    Returns:
        Dictionary with 'train', 'val', and 'test' DataLoaders
    """
    dataloaders = {}

    for split in ["train", "val", "test"]:
        split_dir = Path(data_dir) / split

        if not split_dir.exists():
            logger.warning(f"Directory not found: {split_dir}")
            continue

        # Create dataset
        dataset = FrameDataset(data_dir=str(split_dir), mode=split, img_size=img_size)

        # Create sampler if training and requested
        sampler = None
        if split == "train" and use_weighted_sampler:
            sampler = dataset.get_sampler()

        # Create dataloader
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train" and sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split == "train"),
        )

        logger.info(f"Created {split} dataloader: {len(dataset)} samples")

    return dataloaders


if __name__ == "__main__":
    # Test the dataset
    logging.basicConfig(level=logging.INFO)

    # Example usage
    data_dir = "data/splits/train"
    if os.path.exists(data_dir):
        dataset = FrameDataset(data_dir, mode="train")
        print(f"Classes: {dataset.classes}")
        print(f"Number of samples: {len(dataset)}")
        print(f"Class to index: {dataset.class_to_idx}")

        # Test loading a sample
        if len(dataset) > 0:
            img, label = dataset[0]
            print(f"Image shape: {img.shape}")
            print(f"Label: {label} ({dataset.idx_to_class[label]})")
    else:
        print(f"Directory not found: {data_dir}")
