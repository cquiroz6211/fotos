"""
Group Split Dataset Script for Bonita White Classification

This script splits the extracted frames into train/val/test sets using
GROUP-BASED splitting to prevent data leakage.

WHY GROUP SPLIT?
- Videos have naming convention: C###_H.MP4 and C###_L.MP4
- C### = camera number (e.g., C193, C195, C197)
- _H = high altitude, _L = low altitude
- C193_H and C193_L are the SAME flower bed from different altitudes
- Random split would cause data leakage: C193_H in train, C193_L in test
- Group split ensures ALL frames from the same camera go to ONE split

The script:
1. Extracts camera number from frame filename
2. Groups frames by camera (all C193_* frames = group "C193")
3. Performs stratified group split (maintains class distribution per split)
4. Copies frames to new structure: data/splits/{train,val,test}/{class}/
"""

import os
import re
import csv
import json
import shutil
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


logger = logging.getLogger(__name__)


@dataclass
class FrameInfo:
    """Information about a single frame."""
    path: Path
    class_name: str
    camera_group: str
    original_video: str


@dataclass
class GroupInfo:
    """Information about a camera group."""
    group_id: str
    frames: List[FrameInfo] = field(default_factory=list)
    class_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def add_frame(self, frame: FrameInfo):
        self.frames.append(frame)
        self.class_counts[frame.class_name] += 1

    @property
    def primary_class(self) -> str:
        """Return the class with most frames (for stratification)."""
        return max(self.class_counts.items(), key=lambda x: x[1])[0]


class GroupDatasetSplitter:
    """
    Splits dataset into train/val/test sets using GROUP-BASED splitting.

    This prevents data leakage by ensuring all frames from the same camera
    go to the same split.

    Args:
        source_dir: Directory containing class folders with images
        output_dir: Directory to save split datasets
        train_ratio: Proportion of groups for training
        val_ratio: Proportion of groups for validation
        test_ratio: Proportion of groups for testing
        seed: Random seed for reproducibility
    """

    # Pattern to extract camera number from filename
    # Matches: C193_H (2)_frame_000000.jpg -> C193
    # Matches: C195_L_frame_000001.jpg -> C195
    CAMERA_PATTERN = re.compile(r'^C(\d+)_')

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

    def _extract_camera_group(self, filename: str) -> str:
        """
        Extract camera group from frame filename.

        Args:
            filename: Frame filename (e.g., "C193_H (2)_frame_000000.jpg")

        Returns:
            Camera group (e.g., "C193")
        """
        match = self.CAMERA_PATTERN.match(filename)
        if match:
            return f"C{match.group(1)}"
        else:
            logger.warning(f"Could not extract camera group from: {filename}")
            # Fallback: use entire prefix before first underscore
            return filename.split('_')[0] if '_' in filename else filename

    def _extract_original_video(self, filename: str) -> str:
        """
        Extract original video name from frame filename.

        Args:
            filename: Frame filename (e.g., "C193_H (2)_frame_000000.jpg")

        Returns:
            Original video name (e.g., "C193_H (2)")
        """
        # Remove "_frame_XXXXXX.jpg" suffix
        return re.sub(r'_frame_\d+\.jpg$', '', filename, flags=re.IGNORECASE)

    def _collect_frames(self) -> Tuple[List[FrameInfo], Dict[str, GroupInfo]]:
        """
        Collect all frames and organize by camera group.

        Returns:
            Tuple of (all_frames, groups_dict)
        """
        all_frames: List[FrameInfo] = []
        groups: Dict[str, GroupInfo] = {}

        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {self.source_dir}")

        for class_dir in sorted(self.source_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            logger.info(f"Processing class: {class_name}")

            for image_file in sorted(class_dir.iterdir()):
                if image_file.suffix.lower() not in self.valid_extensions:
                    continue

                filename = image_file.name
                camera_group = self._extract_camera_group(filename)
                original_video = self._extract_original_video(filename)

                frame_info = FrameInfo(
                    path=image_file,
                    class_name=class_name,
                    camera_group=camera_group,
                    original_video=original_video,
                )

                all_frames.append(frame_info)

                # Add to group
                if camera_group not in groups:
                    groups[camera_group] = GroupInfo(group_id=camera_group)
                groups[camera_group].add_frame(frame_info)

        logger.info(f"Collected {len(all_frames)} frames in {len(groups)} camera groups")
        return all_frames, groups

    def _stratified_group_split(
        self,
        groups: Dict[str, GroupInfo]
    ) -> Tuple[Set[str], Set[str], Set[str]]:
        """
        Perform stratified group split.

        Uses StratifiedGroupKFold to ensure:
        1. All frames from same camera go to same split
        2. Class distribution is maintained across splits

        For small numbers of groups, uses a simpler proportional split.

        Args:
            groups: Dictionary of camera_group -> GroupInfo

        Returns:
            Tuple of (train_groups, val_groups, test_groups)
        """
        group_ids = list(groups.keys())
        n_groups = len(group_ids)

        # Create labels for stratification (use primary class of each group)
        labels = [groups[gid].primary_class for gid in group_ids]

        # For small number of groups, use proportional split
        # StratifiedGroupKFold requires n_splits <= n_samples
        if n_groups < 5:
            logger.warning(
                f"Only {n_groups} groups found. Using proportional split instead of "
                "StratifiedGroupKFold. Consider collecting more camera data."
            )
            return self._proportional_group_split(groups)

        # Calculate target counts
        n_train = max(1, int(round(n_groups * self.train_ratio)))
        n_test = max(1, int(round(n_groups * self.test_ratio)))
        n_val = n_groups - n_train - n_test

        logger.info(f"Target split: {n_train} train, {n_val} val, {n_test} test groups")

        # Create group indices (each group is a single unit)
        X = np.zeros(len(group_ids))  # Dummy features
        y = np.array(labels)
        group_array = np.array(group_ids)

        # First split: separate test set using StratifiedGroupKFold
        sgkf = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=self.seed)

        # Get indices for train_val vs test
        splits = list(sgkf.split(X, y, group_array))
        train_val_idx, test_idx = splits[0]

        train_val_groups = set(group_ids[i] for i in train_val_idx)
        test_groups = set(group_ids[i] for i in test_idx)

        # Second split: separate train and val from train_val
        # Recalculate for the subset
        train_val_ids = [group_ids[i] for i in train_val_idx]
        train_val_labels = [labels[i] for i in train_val_idx]
        n_tv = len(train_val_ids)

        # Ensure we have enough groups for the second split
        if n_tv < 2:
            logger.warning("Not enough groups for second split. Assigning all to train.")
            return train_val_groups, set(), test_groups

        X_tv = np.zeros(n_tv)
        y_tv = np.array(train_val_labels)
        groups_tv = np.array(train_val_ids)

        # Use 2-fold split for train/val (gives ~50/50, adjust manually if needed)
        # For better control with small datasets, use proportional split
        n_val_target = max(1, int(round(n_tv * self.val_ratio / (self.train_ratio + self.val_ratio))))

        if n_val_target >= n_tv:
            n_val_target = max(1, n_tv // 3)  # Fallback: 1/3 for val

        # Use StratifiedGroupKFold only if we have enough samples
        n_splits_possible = min(3, n_tv)  # Max 3-fold for safety
        if n_tv >= n_splits_possible:
            sgkf_val = StratifiedGroupKFold(n_splits=n_splits_possible, shuffle=True, random_state=self.seed)
            try:
                val_splits = list(sgkf_val.split(X_tv, y_tv, groups_tv))
                # Take the first fold as validation
                train_idx, val_idx = val_splits[0]
            except ValueError as e:
                logger.warning(f"StratifiedGroupKFold failed: {e}. Using random split.")
                rng = np.random.RandomState(self.seed)
                indices = np.arange(n_tv)
                rng.shuffle(indices)
                val_idx = indices[:n_val_target]
                train_idx = indices[n_val_target:]
        else:
            # Very small dataset: random split
            rng = np.random.RandomState(self.seed)
            indices = np.arange(n_tv)
            rng.shuffle(indices)
            val_idx = indices[:n_val_target]
            train_idx = indices[n_val_target:]

        train_groups = set(train_val_ids[i] for i in train_idx)
        val_groups = set(train_val_ids[i] for i in val_idx)

        logger.info(f"Group split: {len(train_groups)} train, {len(val_groups)} val, {len(test_groups)} test groups")

        return train_groups, val_groups, test_groups

    def _proportional_group_split(
        self,
        groups: Dict[str, GroupInfo]
    ) -> Tuple[Set[str], Set[str], Set[str]]:
        """
        Split groups proportionally without stratification.

        Used when number of groups is too small for StratifiedGroupKFold.
        Attempts to balance class distribution as best as possible.

        Args:
            groups: Dictionary of camera_group -> GroupInfo

        Returns:
            Tuple of (train_groups, val_groups, test_groups)
        """
        group_ids = list(groups.keys())
        n_groups = len(group_ids)

        # Calculate target counts
        n_test = max(1, int(round(n_groups * self.test_ratio)))
        n_val = max(1, int(round(n_groups * self.val_ratio)))
        n_train = n_groups - n_test - n_val

        # Ensure at least 1 in train
        if n_train < 1:
            n_train = 1
            if n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1

        logger.info(f"Proportional split: {n_train} train, {n_val} val, {n_test} test groups")

        # Sort groups by frame count (largest first) for more balanced splits
        sorted_groups = sorted(
            group_ids,
            key=lambda gid: len(groups[gid].frames),
            reverse=True
        )

        # Assign groups to splits (round-robin for balance)
        train_groups = set()
        val_groups = set()
        test_groups = set()

        rng = np.random.RandomState(self.seed)
        rng.shuffle(sorted_groups)  # Shuffle for randomness

        for i, gid in enumerate(sorted_groups):
            if len(train_groups) < n_train:
                train_groups.add(gid)
            elif len(val_groups) < n_val:
                val_groups.add(gid)
            else:
                test_groups.add(gid)

        return train_groups, val_groups, test_groups

    def _copy_files(
        self,
        frames: List[FrameInfo],
        split: str,
        stats: Dict,
    ):
        """
        Copy frames to the appropriate split directory.

        Args:
            frames: List of frames to copy
            split: 'train', 'val', or 'test'
            stats: Statistics dict to update
        """
        for frame in frames:
            split_dir = self.output_dir / split / frame.class_name
            split_dir.mkdir(parents=True, exist_ok=True)

            dest_path = split_dir / frame.path.name
            shutil.copy2(frame.path, dest_path)

            # Update stats
            stats[split]["total"] += 1
            stats[split]["classes"][frame.class_name] += 1
            stats[split]["groups"].add(frame.camera_group)

    def _save_metadata(
        self,
        all_frames: List[FrameInfo],
        split_assignment: Dict[str, str],
        groups: Dict[str, GroupInfo]
    ):
        """
        Save detailed metadata about the split.

        Args:
            all_frames: All frame info
            split_assignment: Dict mapping frame path -> split name
            groups: Group information
        """
        # Save CSV metadata
        metadata_path = self.output_dir / "metadata.csv"
        with open(metadata_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "filename", "class", "split", "camera_group",
                "original_video", "source_path"
            ])

            for frame in all_frames:
                split = split_assignment.get(str(frame.path), "unknown")
                writer.writerow([
                    frame.path.name,
                    frame.class_name,
                    split,
                    frame.camera_group,
                    frame.original_video,
                    str(frame.path),
                ])

        logger.info(f"Saved metadata to {metadata_path}")

        # Save group information
        groups_path = self.output_dir / "groups_info.json"
        groups_data = {}
        for gid, ginfo in sorted(groups.items()):
            groups_data[gid] = {
                "frame_count": len(ginfo.frames),
                "class_counts": dict(ginfo.class_counts),
                "primary_class": ginfo.primary_class,
            }

        with open(groups_path, "w", encoding="utf-8") as f:
            json.dump(groups_data, f, indent=2)

        logger.info(f"Saved groups info to {groups_path}")

    def _print_summary(self, stats: Dict, groups: Dict[str, GroupInfo], group_counts: Dict[str, int] = None):
        """Print summary statistics.

        Args:
            stats: Statistics dictionary with split info
            groups: Original groups dictionary
            group_counts: Optional dict with group counts per split (if stats was serialized)
        """
        print("\n" + "=" * 70)
        print("GROUP-BASED DATASET SPLIT SUMMARY")
        print("=" * 70)

        total_frames = sum(stats[split]["total"] for split in ["train", "val", "test"])
        print(f"\nTotal frames: {total_frames}")
        print(f"Total camera groups: {len(groups)}")

        # Group distribution
        print("\n--- Group Distribution ---")
        for split in ["train", "val", "test"]:
            # Handle both cases: stats with set (before serialization) or int (after)
            n_groups = group_counts[split] if group_counts else len(stats[split]["groups"])
            n_frames = stats[split]["total"]
            pct = (n_frames / total_frames * 100) if total_frames > 0 else 0
            print(f"{split.upper():5s}: {n_groups:3d} groups, {n_frames:5d} frames ({pct:.1f}%)")

        # Class distribution per split
        print("\n--- Class Distribution ---")
        all_classes = set()
        for split in ["train", "val", "test"]:
            all_classes.update(stats[split]["classes"].keys())

        # Header
        header = f"{'Class':<30}"
        for split in ["train", "val", "test"]:
            header += f" {split.upper():>8s}"
        header += "   Total"
        print(header)
        print("-" * len(header))

        # Per class
        for class_name in sorted(all_classes):
            row = f"{class_name:<30}"
            class_total = 0
            for split in ["train", "val", "test"]:
                count = stats[split]["classes"].get(class_name, 0)
                class_total += count
                row += f" {count:8d}"
            row += f" {class_total:8d}"
            print(row)

        print("-" * len(header))

        # Totals
        row = f"{'TOTAL':<30}"
        for split in ["train", "val", "test"]:
            row += f" {stats[split]['total']:8d}"
        row += f" {total_frames:8d}"
        print(row)

        # Class percentages per split
        print("\n--- Class Percentages (verify stratification) ---")
        for split in ["train", "val", "test"]:
            split_total = stats[split]["total"]
            if split_total == 0:
                continue
            print(f"\n{split.upper()}:")
            for class_name in sorted(all_classes):
                count = stats[split]["classes"].get(class_name, 0)
                pct = (count / split_total * 100) if split_total > 0 else 0
                print(f"  {class_name}: {pct:.1f}%")

        print("\n" + "=" * 70)

    def split(self, clear_existing: bool = False):
        """
        Execute the group-based dataset splitting.

        Args:
            clear_existing: Whether to clear existing split directories
        """
        # Clear existing if requested
        if clear_existing and self.output_dir.exists():
            logger.info(f"Clearing existing directory: {self.output_dir}")
            shutil.rmtree(self.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Collect all frames and group info
        logger.info(f"Collecting frames from {self.source_dir}")
        all_frames, groups = self._collect_frames()

        if not all_frames:
            raise ValueError("No frames found in source directory")

        if len(groups) < 3:
            raise ValueError(
                f"Need at least 3 groups for train/val/test split, found {len(groups)}"
            )

        # Perform stratified group split
        logger.info("Performing stratified group split...")
        train_groups, val_groups, test_groups = self._stratified_group_split(groups)

        # Initialize stats
        stats = {
            "train": {"total": 0, "classes": defaultdict(int), "groups": set()},
            "val": {"total": 0, "classes": defaultdict(int), "groups": set()},
            "test": {"total": 0, "classes": defaultdict(int), "groups": set()},
        }

        # Assign frames to splits
        split_assignment: Dict[str, str] = {}
        train_frames: List[FrameInfo] = []
        val_frames: List[FrameInfo] = []
        test_frames: List[FrameInfo] = []

        for frame in all_frames:
            group = frame.camera_group
            if group in train_groups:
                split_assignment[str(frame.path)] = "train"
                train_frames.append(frame)
            elif group in val_groups:
                split_assignment[str(frame.path)] = "val"
                val_frames.append(frame)
            elif group in test_groups:
                split_assignment[str(frame.path)] = "test"
                test_frames.append(frame)
            else:
                logger.warning(f"Frame {frame.path} has unassigned group {group}")

        # Copy files
        logger.info(f"Copying {len(train_frames)} train frames...")
        self._copy_files(train_frames, "train", stats)

        logger.info(f"Copying {len(val_frames)} val frames...")
        self._copy_files(val_frames, "val", stats)

        logger.info(f"Copying {len(test_frames)} test frames...")
        self._copy_files(test_frames, "test", stats)

        # Store group counts before converting sets to ints
        group_counts = {
            split: len(stats[split]["groups"]) for split in ["train", "val", "test"]
        }

        # Convert defaultdicts to dicts for JSON serialization
        for split in stats:
            stats[split]["classes"] = dict(stats[split]["classes"])
            stats[split]["groups"] = len(stats[split]["groups"])

        # Save statistics
        stats_path = self.output_dir / "statistics.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved statistics to {stats_path}")

        # Save detailed metadata
        self._save_metadata(all_frames, split_assignment, groups)

        # Print summary
        self._print_summary(stats, groups, group_counts)

        # Save configuration
        config = {
            "source_dir": str(self.source_dir),
            "output_dir": str(self.output_dir),
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "seed": self.seed,
            "split_type": "stratified_group_split",
            "group_description": "Camera number extracted from filename (C### pattern)",
        }
        config_path = self.output_dir / "split_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        logger.info("Group-based dataset splitting completed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Split dataset into train/val/test using GROUP-BASED splitting "
                    "to prevent data leakage from same camera going to different splits."
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default="data/processed/frames",
        help="Directory containing class folders with images (default: data/processed/frames)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/splits",
        help="Directory to save split datasets (default: data/splits)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Proportion of GROUPS for training (default: 0.7)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Proportion of GROUPS for validation (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Proportion of GROUPS for testing (default: 0.15)",
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
    splitter = GroupDatasetSplitter(
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