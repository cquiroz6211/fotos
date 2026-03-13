"""
Script to split the dataset into train, validation, and test sets.

This module provides functionality to divide a dataset of image frames
into train (70%), validation (15%), and test (15%) splits while maintaining
class distribution. It ensures reproducibility through random seeding and
creates organized directory structures with metadata CSV files.
"""

import os
import shutil
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import random

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DatasetSplitter:
    """
    Dataset splitter for creating train/val/test splits.

    This class handles the splitting of image datasets while preserving
    class distribution and creating organized output directories.

    Args:
        source_dir: Source directory containing class folders with images
        output_dir: Root output directory for splits
        train_ratio: Ratio for training set (default: 0.7)
        val_ratio: Ratio for validation set (default: 0.15)
        test_ratio: Ratio for test set (default: 0.15)
        random_seed: Random seed for reproducibility (default: 42)

    Example:
        >>> splitter = DatasetSplitter('data/processed/frames', 'data/splits')
        >>> splitter.split()
    """

    def __init__(
        self,
        source_dir: str,
        output_dir: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
    ):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed

        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if not abs(total_ratio - 1.0) < 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

        # Valid image extensions
        self.valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

        # Metadata storage
        self.metadata: List[Dict] = []
        self.class_stats: Dict[str, Dict[str, int]] = {}

    def _collect_images(self) -> Dict[str, List[Path]]:
        """
        Collect all image paths organized by class.

        Returns:
            Dictionary mapping class names to lists of image paths
        """
        class_images: Dict[str, List[Path]] = {}

        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {self.source_dir}")

        # Get class directories
        class_dirs = sorted([d for d in self.source_dir.iterdir() if d.is_dir()])

        if not class_dirs:
            raise ValueError(f"No class directories found in {self.source_dir}")

        logger.info(f"Found {len(class_dirs)} class directories")

        # Collect images for each class
        for class_dir in class_dirs:
            class_name = class_dir.name
            images = []

            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in self.valid_extensions:
                    images.append(img_path)

            if images:
                class_images[class_name] = sorted(images)
                logger.info(f"  {class_name}: {len(images)} images")
            else:
                logger.warning(f"  {class_name}: No valid images found")

        total_images = sum(len(imgs) for imgs in class_images.values())
        logger.info(f"Total images collected: {total_images}")

        return class_images

    def _split_class_images(
        self, images: List[Path]
    ) -> Tuple[List[Path], List[Path], List[Path]]:
        """
        Split a list of images into train, val, and test sets.

        Args:
            images: List of image paths for a single class

        Returns:
            Tuple of (train_images, val_images, test_images)
        """
        random.shuffle(images)

        train_count = int(len(images) * self.train_ratio)
        val_count = int(len(images) * self.val_ratio)
        test_count = len(images) - train_count - val_count

        train_images = images[:train_count]
        val_images = images[train_count : train_count + val_count]
        test_images = images[train_count + val_count :]

        return train_images, val_images, test_images

    def _create_output_directories(self, class_names: List[str]) -> None:
        """
        Create output directory structure for all splits.

        Args:
            class_names: List of class names to create subdirectories for
        """
        for split in ["train", "val", "test"]:
            split_dir = self.output_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)

            for class_name in class_names:
                class_dir = split_dir / class_name
                class_dir.mkdir(exist_ok=True)

        logger.info(f"Created output directory structure in {self.output_dir}")

    def _copy_images(self, images: List[Path], split: str, class_name: str) -> None:
        """
        Copy images to the appropriate split directory.

        Args:
            images: List of image paths to copy
            split: Split name ('train', 'val', or 'test')
            class_name: Class name for the images
        """
        dest_dir = self.output_dir / split / class_name

        for img_path in images:
            dest_path = dest_dir / img_path.name

            # Copy file
            shutil.copy2(img_path, dest_path)

            # Add to metadata
            self.metadata.append(
                {
                    "filename": img_path.name,
                    "original_path": str(img_path),
                    "class": class_name,
                    "split": split,
                    "file_size": img_path.stat().st_size,
                }
            )

    def _calculate_statistics(self) -> None:
        """
        Calculate and store class distribution statistics.
        """
        df = pd.DataFrame(self.metadata)

        for split in ["train", "val", "test"]:
            split_df = df[df["split"] == split]
            self.class_stats[split] = {}

            if len(split_df) > 0:
                class_counts = split_df["class"].value_counts().to_dict()
                self.class_stats[split] = class_counts

    def _print_statistics(self) -> None:
        """Print detailed statistics about the dataset split."""
        df = pd.DataFrame(self.metadata)

        print("\n" + "=" * 70)
        print("DATASET SPLIT STATISTICS")
        print("=" * 70)

        # Overall statistics
        print(f"\nTotal images: {len(df)}")
        print(f"Classes: {df['class'].nunique()}")
        print(f"Classes: {', '.join(sorted(df['class'].unique()))}")

        # Split distribution
        print("\n--- Split Distribution ---")
        split_counts = df["split"].value_counts()
        for split in ["train", "val", "test"]:
            count = split_counts.get(split, 0)
            ratio = count / len(df) * 100
            print(f"  {split:6s}: {count:5d} images ({ratio:5.1f}%)")

        # Class distribution per split
        print("\n--- Class Distribution per Split ---")
        df_stats = df.groupby(["split", "class"]).size().unstack(fill_value=0)

        print(df_stats.to_string())
        print()

        # Percentages per class
        print("--- Class Percentages (relative to total class size) ---")
        for class_name in sorted(df["class"].unique()):
            class_df = df[df["class"] == class_name]
            total = len(class_df)
            print(f"\n  {class_name}:")
            for split in ["train", "val", "test"]:
                count = len(class_df[class_df["split"] == split])
                ratio = count / total * 100 if total > 0 else 0
                print(f"    {split:6s}: {count:4d} images ({ratio:5.1f}%)")

        print("\n" + "=" * 70)

    def _save_metadata(self) -> None:
        """Save metadata to CSV files."""
        # Save complete metadata
        metadata_file = self.output_dir / "metadata.csv"
        df = pd.DataFrame(self.metadata)
        df.to_csv(metadata_file, index=False)
        logger.info(f"Saved metadata to {metadata_file}")

        # Save statistics
        stats_file = self.output_dir / "statistics.csv"
        df_stats = df.groupby(["split", "class"]).size().unstack(fill_value=0)
        df_stats.to_csv(stats_file)
        logger.info(f"Saved statistics to {stats_file}")

        # Save class distribution per split as JSON for easy loading
        import json

        dist_file = self.output_dir / "distribution.json"
        with open(dist_file, "w") as f:
            json.dump(self.class_stats, f, indent=2)
        logger.info(f"Saved distribution to {dist_file}")

    def split(self) -> Dict[str, Dict[str, int]]:
        """
        Execute the dataset splitting process.

        Returns:
            Dictionary with split statistics {split: {class: count}}
        """
        logger.info("=" * 70)
        logger.info("Starting dataset split")
        logger.info("=" * 70)
        logger.info(f"Source: {self.source_dir}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(
            f"Ratios - Train: {self.train_ratio:.2f}, Val: {self.val_ratio:.2f}, Test: {self.test_ratio:.2f}"
        )
        logger.info(f"Random seed: {self.random_seed}")

        # Set random seed for reproducibility
        random.seed(self.random_seed)

        # Collect all images
        class_images = self._collect_images()

        if not class_images:
            raise ValueError("No images found in source directory")

        # Create output directories
        class_names = list(class_images.keys())
        self._create_output_directories(class_names)

        # Split and copy images for each class
        for class_name, images in class_images.items():
            logger.info(f"\nProcessing class: {class_name} ({len(images)} images)")

            # Split images
            train_imgs, val_imgs, test_imgs = self._split_class_images(images)

            logger.info(
                f"  Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}"
            )

            # Copy images to output directories
            self._copy_images(train_imgs, "train", class_name)
            self._copy_images(val_imgs, "val", class_name)
            self._copy_images(test_imgs, "test", class_name)

        # Calculate and save statistics
        self._calculate_statistics()
        self._save_metadata()
        self._print_statistics()

        logger.info("Dataset split completed successfully!")

        return self.class_stats


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Split dataset into train, validation, and test sets"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/processed/frames",
        help="Source directory containing class folders with images",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/splits",
        help="Output directory for split datasets",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Ratio for training set (default: 0.7)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Ratio for validation set (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Ratio for test set (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory if it exists",
    )

    args = parser.parse_args()

    # Check if output directory exists
    output_dir = Path(args.output)
    if output_dir.exists():
        if args.overwrite:
            logger.warning(f"Output directory exists, will overwrite: {output_dir}")
            shutil.rmtree(output_dir)
        else:
            logger.error(f"Output directory already exists: {output_dir}")
            logger.error("Use --overwrite to overwrite existing data")
            return 1

    try:
        # Create splitter and execute
        splitter = DatasetSplitter(
            source_dir=args.source,
            output_dir=args.output,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_seed=args.seed,
        )

        splitter.split()

        return 0

    except Exception as e:
        logger.error(f"Error during dataset split: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
