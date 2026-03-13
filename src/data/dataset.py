"""
Dataset class for Bonita White Classifier.

This module provides a PyTorch Dataset class for loading and preprocessing
image frames for classification tasks. It supports training, validation, and
test modes with appropriate data augmentation and normalization.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# EfficientNet-B0 normalization values (ImageNet statistics)
EFFICIENTNET_MEAN = [0.485, 0.456, 0.406]
EFFICIENTNET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Get training transforms with data augmentation.

    Args:
        img_size: Target image size (default: 224 for EfficientNet-B0)

    Returns:
        Composed transform pipeline for training
    """
    return transforms.Compose(
        [
            transforms.Resize(
                (img_size, img_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=EFFICIENTNET_MEAN, std=EFFICIENTNET_STD),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
        ]
    )


def get_val_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Get validation/test transforms (no augmentation).

    Args:
        img_size: Target image size (default: 224 for EfficientNet-B0)

    Returns:
        Composed transform pipeline for validation/testing
    """
    return transforms.Compose(
        [
            transforms.Resize(
                (img_size, img_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=EFFICIENTNET_MEAN, std=EFFICIENTNET_STD),
        ]
    )


class FrameDataset(Dataset):
    """
    PyTorch Dataset for loading image frames.

    This dataset loads images from a directory structure where each subdirectory
    represents a class. It supports train, validation, and test modes with
    appropriate transforms for each mode.

    Args:
        data_dir: Root directory containing class folders
        mode: One of 'train', 'val', or 'test'
        img_size: Target image size for resizing
        transform: Optional custom transform (overrides mode-based transforms)
        cache_images: Whether to cache images in memory (use with caution for large datasets)

    Example:
        >>> dataset = FrameDataset('data/splits/train', mode='train')
        >>> image, label = dataset[0]
    """

    def __init__(
        self,
        data_dir: str,
        mode: str = "train",
        img_size: int = 224,
        transform: Optional[Callable] = None,
        cache_images: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.img_size = img_size
        self.cache_images = cache_images

        # Validate mode
        if mode not in ["train", "val", "test"]:
            raise ValueError(f"mode must be 'train', 'val', or 'test', got '{mode}'")

        # Check if data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        # Get transforms
        if transform is not None:
            self.transform = transform
        elif mode == "train":
            self.transform = get_train_transforms(img_size)
        else:  # val or test
            self.transform = get_val_transforms(img_size)

        # Load image paths and labels
        self.samples: List[Tuple[Path, int]] = []
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}
        self.classes: List[str] = []

        self._load_data()

        # Image cache (optional)
        self._cache: Dict[int, torch.Tensor] = {}

        logger.info(
            f"Initialized {mode} dataset with {len(self)} samples from {data_dir}"
        )

    def _load_data(self) -> None:
        """
        Load all image paths and create class mappings.

        Walks through the data directory, identifies class folders,
        and builds mappings between class names and indices.
        """
        # Get all class directories
        class_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])

        if not class_dirs:
            raise ValueError(f"No class directories found in {self.data_dir}")

        # Create class mappings
        self.classes = [d.name for d in class_dirs]
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.idx_to_class = {
            idx: cls_name for cls_name, idx in self.class_to_idx.items()
        }

        # Load image paths
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

        for class_dir in class_dirs:
            class_name = class_dir.name
            class_idx = self.class_to_idx[class_name]

            # Find all images in class directory
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in valid_extensions:
                    self.samples.append((img_path, class_idx))

        if not self.samples:
            raise ValueError(f"No valid images found in {self.data_dir}")

        logger.info(f"Found {len(self.classes)} classes: {', '.join(self.classes)}")
        logger.info(f"Class distribution: {self._get_class_distribution()}")

    def _get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of samples across classes."""
        distribution = {}
        for img_path, class_idx in self.samples:
            class_name = self.idx_to_class[class_idx]
            distribution[class_name] = distribution.get(class_name, 0) + 1
        return distribution

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (image_tensor, label)
        """
        # Check cache first
        if self.cache_images and idx in self._cache:
            return self._cache[idx], self.samples[idx][1]

        # Load image
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image {img_path}: {e}")
            # Return a blank image as fallback
            image = Image.new("RGB", (self.img_size, self.img_size), color="black")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Cache if enabled
        if self.cache_images:
            self._cache[idx] = image

        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for imbalanced datasets.

        Uses the inverse frequency method to compute weights that can be
        used in the loss function to handle class imbalance.

        Returns:
            Tensor of class weights with shape (num_classes,)
        """
        distribution = self._get_class_distribution()
        total_samples = len(self)

        weights = []
        for class_name in self.classes:
            class_count = distribution[class_name]
            weight = total_samples / (len(self.classes) * class_count)
            weights.append(weight)

        return torch.tensor(weights, dtype=torch.float)

    def get_sample_count_by_class(self) -> Dict[str, int]:
        """Get the number of samples per class."""
        return self._get_class_distribution()


def create_dataloaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224,
    cache_images: bool = False,
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        data_root: Root directory containing splits (train/val/test folders)
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        img_size: Target image size
        cache_images: Whether to cache images in memory

    Returns:
        Dictionary with 'train', 'val', and 'test' dataloaders

    Example:
        >>> dataloaders = create_dataloaders('data/splits')
        >>> train_loader = dataloaders['train']
        >>> images, labels = next(iter(train_loader))
    """
    data_root = Path(data_root)

    # Create datasets
    splits = {}
    for mode in ["train", "val", "test"]:
        split_dir = data_root / mode
        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}")
            continue

        try:
            dataset = FrameDataset(
                str(split_dir), mode=mode, img_size=img_size, cache_images=cache_images
            )
            splits[mode] = dataset
        except Exception as e:
            logger.error(f"Failed to create {mode} dataset: {e}")

    if not splits:
        raise ValueError("No valid splits found in data root")

    # Create dataloaders
    dataloaders = {}
    for mode, dataset in splits.items():
        shuffle = mode == "train"  # Only shuffle training data
        dataloaders[mode] = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(mode == "train"),  # Drop incomplete batches only for training
        )

        logger.info(
            f"Created {mode} dataloader: {len(dataloaders[mode])} batches, "
            f"{len(dataset)} samples"
        )

    return dataloaders


if __name__ == "__main__":
    # Test the dataset class
    import sys

    # Test with split data if available
    data_root = Path("data/splits")

    if data_root.exists():
        print("=" * 60)
        print("Testing FrameDataset with split data")
        print("=" * 60)

        try:
            # Create dataloaders
            dataloaders = create_dataloaders(
                str(data_root),
                batch_size=8,
                num_workers=0,
                img_size=224,
                cache_images=False,
            )

            # Test loading a batch
            for mode, loader in dataloaders.items():
                print(f"\n{mode.upper()} Set:")
                print(f"  Batches: {len(loader)}")
                print(f"  Samples: {len(loader.dataset)}")
                print(f"  Classes: {', '.join(loader.dataset.classes)}")
                print(
                    f"  Class distribution: {loader.dataset.get_sample_count_by_class()}"
                )

                # Load first batch
                images, labels = next(iter(loader))
                print(f"  Batch shape: {images.shape}")
                print(f"  Labels shape: {labels.shape}")
                print(f"  Value range: [{images.min():.3f}, {images.max():.3f}]")

            print("\n✅ All tests passed!")

        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            sys.exit(1)
    else:
        print(f"⚠️  Data directory not found: {data_root}")
        print("Please run split_dataset.py first to create the splits.")
