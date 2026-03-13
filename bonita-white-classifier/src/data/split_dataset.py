"""
Dataset Splitting Script for Bonita White Classification

This script splits the extracted frames into train/val/test sets
with stratification to maintain class distribution.
"""

import os
import csv
import json
import shutil
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)


class DatasetSplitter:
    """
    Splits dataset into train/val/test sets with stratification.

    Args:
        source_dir: Directory containing class folders with images
        output_dir: Directory to save split datasets
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        source_dir: str,
        output_dir: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        # Validate ratios
        total = train_ratio + val_ratio + test_ratio
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"Ratios must sum to 1.0, got {total}")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Supported image extensions
        self.valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def _collect_samples(self) -> Dict[str, List[Path]]:
        """
        Collect all image samples organized by class.

        Returns:
            Dictionary mapping class names to lists of image paths
        """
        samples_by_class = defaultdict(list)

        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {self.source_dir}")

        for class_dir in self.source_dir.iterdir():
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            images = [
                f
                for f in class_dir.iterdir()
                if f.suffix.lower() in self.valid_extensions
            ]

            samples_by_class[class_name].extend(images)
            logger.info(f"Found {len(images)} images in class '{class_name}'")

        return dict(samples_by_class)

    def _split_class(
        self, images: List[Path], class_name: str
    ) -> Tuple[List[Path], List[Path], List[Path]]:
        """
        Split images of a single class into train/val/test.

        Args:
            images: List of image paths
            class_name: Name of the class

        Returns:
            Tuple of (train_images, val_images, test_images)
        """
        if len(images) == 0:
            return [], [], []

        # First split: separate test set
        train_val, test = train_test_split(
            images, test_size=self.test_ratio, random_state=self.seed, shuffle=True
        )

        # Second split: separate train and val
        # Adjust val ratio to account for already removed test set
        val_ratio_adjusted = self.val_ratio / (self.train_ratio + self.val_ratio)

        train, val = train_test_split(
            train_val,
            test_size=val_ratio_adjusted,
            random_state=self.seed,
            shuffle=True,
        )

        logger.info(
            f"Class '{class_name}': {len(train)} train, {len(val)} val, {len(test)} test"
        )

        return train, val, test

    def _copy_files(self, files: List[Path], split: str, class_name: str):
        """
        Copy files to the appropriate split directory.

        Args:
            files: List of files to copy
            split: 'train', 'val', or 'test'
            class_name: Class name for subdirectory
        """
        split_dir = self.output_dir / split / class_name
        split_dir.mkdir(parents=True, exist_ok=True)

        for file_path in files:
            dest_path = split_dir / file_path.name
            shutil.copy2(file_path, dest_path)

    def _save_metadata(self, all_splits: Dict[str, Dict[str, List[Path]]]):
        """
        Save metadata CSV and statistics JSON.

        Args:
            all_splits: Dictionary with split -> class -> files structure
        """
        # Save CSV metadata
        metadata_path = self.output_dir / "metadata.csv"
        with open(metadata_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "class", "split", "source_path"])

            for split, classes in all_splits.items():
                for class_name, files in classes.items():
                    for file_path in files:
                        writer.writerow(
                            [file_path.name, class_name, split, str(file_path)]
                        )

        logger.info(f"Saved metadata to {metadata_path}")

        # Calculate and save statistics
        stats = {}
        for split in ["train", "val", "test"]:
            if split in all_splits:
                split_data = all_splits[split]
                stats[split] = {
                    "total_samples": sum(len(files) for files in split_data.values()),
                    "classes": {
                        class_name: len(files)
                        for class_name, files in split_data.items()
                    },
                }

        stats_path = self.output_dir / "statistics.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Saved statistics to {stats_path}")

        # Print summary
        self._print_summary(stats)

    def _print_summary(self, stats: Dict):
        """Print summary statistics."""
        print("\n" + "=" * 60)
        print("Dataset Split Summary")
        print("=" * 60)

        total_samples = sum(s["total_samples"] for s in stats.values())
        print(f"Total samples: {total_samples}")
        print()

        for split, data in stats.items():
            print(f"{split.upper()}:")
            print(f"  Total: {data['total_samples']} samples")
            for class_name, count in sorted(data["classes"].items()):
                percentage = (count / data["total_samples"]) * 100
                print(f"  - {class_name}: {count} ({percentage:.1f}%)")
            print()

        print("=" * 60)

    def split(self, clear_existing: bool = False):
        """
        Execute the dataset splitting.

        Args:
            clear_existing: Whether to clear existing split directories
        """
        # Clear existing if requested
        if clear_existing and self.output_dir.exists():
            logger.info(f"Clearing existing directory: {self.output_dir}")
            shutil.rmtree(self.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Collect all samples
        logger.info(f"Collecting samples from {self.source_dir}")
        samples_by_class = self._collect_samples()

        if not samples_by_class:
            raise ValueError("No samples found in source directory")

        # Split each class
        all_splits = {
            "train": defaultdict(list),
            "val": defaultdict(list),
            "test": defaultdict(list),
        }

        for class_name, images in samples_by_class.items():
            train, val, test = self._split_class(images, class_name)

            all_splits["train"][class_name] = train
            all_splits["val"][class_name] = val
            all_splits["test"][class_name] = test

            # Copy files
            logger.info(f"Copying {class_name} files...")
            self._copy_files(train, "train", class_name)
            self._copy_files(val, "val", class_name)
            self._copy_files(test, "test", class_name)

        # Save metadata
        self._save_metadata(all_splits)

        logger.info("Dataset splitting completed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Split dataset into train/val/test sets"
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default="data/processed/frames",
        help="Directory containing class folders with images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/splits",
        help="Directory to save split datasets",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Proportion of data for training (default: 0.7)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Proportion of data for validation (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Proportion of data for testing (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing split directories before splitting",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create splitter and execute
    splitter = DatasetSplitter(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    splitter.split(clear_existing=args.clear)


if __name__ == "__main__":
    main()
