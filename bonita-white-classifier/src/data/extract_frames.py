"""
Video Frame Extraction Script for Bonita White Classification

This script extracts frames from MP4 videos organized by day folders,
resizes them to 224x224 (EfficientNet-B0 input size), and organizes
them by class in a structured directory.

Author: Computer Vision Team
Date: 2025
"""

import os
import cv2
import yaml
import numpy as np
import logging
import argparse
import shutil
from pathlib import Path
from typing import Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime


# Configure logging
def setup_logging(log_dir: str, log_level: str = "INFO") -> logging.Logger:
    """
    Configure logging with file and console handlers.

    Args:
        log_dir: Directory to store log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured logger instance
    """
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"extract_frames_{timestamp}.log")

    logger = logging.getLogger("frame_extraction")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    logger.handlers = []

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_video_files(
    video_dir: str, video_extensions: Tuple[str, ...] = (".mp4", ".avi", ".mov")
) -> list:
    """
    Recursively find all video files in directory.

    Args:
        video_dir: Root directory containing video folders
        video_extensions: Tuple of valid video file extensions

    Returns:
        List of video file paths
    """
    video_files = []

    for ext in video_extensions:
        video_files.extend(Path(video_dir).rglob(f"*{ext}"))

    return sorted([str(f) for f in video_files])


def determine_class_from_path(video_path: str, class_mapping: dict) -> Optional[str]:
    """
    Determine the class label based on video path, filename, or parent folder.

    For Bonita White project, uses folder names like 'dia1-4', 'dia5-8', 'dia9-11'
    to determine the phenological state class.

    Args:
        video_path: Path to video file
        class_mapping: Dictionary mapping patterns to class names

    Returns:
        Class name or None if no match found
    """
    video_path_lower = video_path.lower()
    filename = os.path.basename(video_path_lower)
    parent_folder = os.path.basename(os.path.dirname(video_path_lower))

    # Folder-based mapping for Bonita White project
    folder_to_class = {
        "dia1-4": "Estado_0_Prefloracion",
        "dia5-8": "Estado_1_Floracion_Intermedia",
        "dia9-11": "Estado_2_Floracion_Maxima",
    }

    # Check parent folder first (highest priority)
    for folder_pattern, class_name in folder_to_class.items():
        if folder_pattern in parent_folder:
            return class_name

    # Fallback to filename matching
    for pattern, class_name in class_mapping.items():
        if pattern.lower() in filename:
            return class_name

    return None


def resize_frame(
    frame: cv2.typing.MatLike, target_size: Tuple[int, int] = (224, 224)
) -> cv2.typing.MatLike:
    """
    Resize frame to target size maintaining aspect ratio with padding.

    Args:
        frame: Input frame (numpy array)
        target_size: Target dimensions (width, height)

    Returns:
        Resized frame
    """
    height, width = frame.shape[:2]
    target_width, target_height = target_size

    # Calculate scaling factor to fit within target size while maintaining aspect ratio
    scale = min(target_width / width, target_height / height)

    # Resize frame
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_frame = cv2.resize(
        frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR
    )

    # Create canvas with target size and center the resized frame
    canvas = np.full((target_height, target_width, 3), 0, dtype=np.uint8)

    # Calculate padding to center the frame
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    canvas[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = (
        resized_frame
    )

    return canvas


def extract_frames_single_video(
    video_path: str,
    output_dir: str,
    class_name: str,
    config: dict,
    logger: logging.Logger,
) -> dict:
    """
    Extract frames from a single video file.

    Args:
        video_path: Path to video file
        output_dir: Directory to save extracted frames
        class_name: Class name for organizing output
        config: Configuration dictionary
        logger: Logger instance

    Returns:
        Dictionary with extraction statistics
    """
    stats = {
        "video_path": video_path,
        "success": False,
        "total_frames": 0,
        "extracted_frames": 0,
        "error": None,
    }

    # Extract configuration parameters
    resize_size = tuple(config.get("resize_size", [224, 224]))
    extract_mode = config.get("extract_mode", "seconds")  # 'seconds' or 'frames'
    interval = config.get("interval", 1)  # N seconds or N frames
    max_frames = config.get("max_frames", None)
    min_confidence = config.get("min_confidence", 0.5)

    # Create output directory for this class
    class_output_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_output_dir, exist_ok=True)

    # Generate output filename prefix
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    try:
        # Open video capture
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if video_fps <= 0:
            raise ValueError(f"Invalid FPS for video: {video_path}")

        stats["total_frames"] = total_frames

        # Calculate frame indices to extract
        if extract_mode == "seconds":
            frame_indices = [
                int(i * video_fps)
                for i in range(0, int(total_frames / video_fps), interval)
            ]
        else:  # 'frames' mode
            frame_indices = list(range(0, total_frames, interval))

        # Limit number of frames if specified
        if max_frames is not None and len(frame_indices) > max_frames:
            frame_indices = frame_indices[:max_frames]

        logger.debug(
            f"Processing {video_name}: {total_frames} frames, FPS={video_fps:.2f}"
        )
        logger.debug(
            f"Extracting {len(frame_indices)} frames (mode={extract_mode}, interval={interval})"
        )

        # Extract frames
        extracted_count = 0
        for frame_idx in frame_indices:
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            # Read frame
            ret, frame = cap.read()

            if not ret:
                logger.warning(f"Failed to read frame {frame_idx} from {video_name}")
                continue

            # Resize frame
            resized_frame = resize_frame(frame, resize_size)

            # Save frame
            output_filename = f"{video_name}_frame_{frame_idx:06d}.jpg"
            output_path = os.path.join(class_output_dir, output_filename)
            cv2.imwrite(output_path, resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

            extracted_count += 1

        stats["extracted_frames"] = extracted_count
        stats["success"] = True

        logger.info(f"Extracted {extracted_count} frames from {video_name}")

        # Release video capture
        cap.release()

    except Exception as e:
        stats["error"] = str(e)
        logger.error(f"Error processing {video_name}: {e}")
        return stats

    return stats


def process_videos_parallel(
    video_paths: list,
    output_dir: str,
    class_mapping: dict,
    config: dict,
    logger: logging.Logger,
    num_workers: int = 4,
) -> dict:
    """
    Process multiple videos in parallel.

    Args:
        video_paths: List of video file paths
        output_dir: Directory to save extracted frames
        class_mapping: Dictionary mapping patterns to class names
        config: Configuration dictionary
        logger: Logger instance
        num_workers: Number of parallel workers

    Returns:
        Dictionary with overall processing statistics
    """
    overall_stats = {
        "total_videos": len(video_paths),
        "successful": 0,
        "failed": 0,
        "total_frames_extracted": 0,
        "failed_videos": [],
    }

    logger.info(f"Processing {len(video_paths)} videos with {num_workers} workers")

    # Process videos in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {}
        for video_path in video_paths:
            class_name = determine_class_from_path(video_path, class_mapping)

            if class_name is None:
                logger.warning(f"Could not determine class for: {video_path}, skipping")
                overall_stats["failed"] += 1
                overall_stats["failed_videos"].append(video_path)
                continue

            future = executor.submit(
                extract_frames_single_video,
                video_path,
                output_dir,
                class_name,
                config,
                logger,
            )
            futures[future] = video_path

        # Process results with progress bar
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Extracting frames"
        ):
            video_path = futures[future]
            try:
                stats = future.result()

                if stats["success"]:
                    overall_stats["successful"] += 1
                    overall_stats["total_frames_extracted"] += stats["extracted_frames"]
                else:
                    overall_stats["failed"] += 1
                    overall_stats["failed_videos"].append(video_path)

            except Exception as e:
                overall_stats["failed"] += 1
                overall_stats["failed_videos"].append(video_path)
                logger.error(f"Unexpected error processing {video_path}: {e}")

    return overall_stats


def main():
    """
    Main function to orchestrate frame extraction process.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Extract frames from MP4 videos for Bonita White classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="bonita-white-classifier/config/extract_frames_config.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="bonita-white-classifier/data/raw/videos",
        help="Directory containing video subfolders",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="bonita-white-classifier/data/processed/frames",
        help="Directory to save extracted frames",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Clear output directory before extraction",
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Config file not found: {args.config}")
        print("Using default configuration...")
        config = {
            "resize_size": [224, 224],
            "extract_mode": "seconds",
            "interval": 1,
            "max_frames": None,
            "min_confidence": 0.5,
            "class_mapping": {
                "prefloracion": "Estado_0_Prefloracion",
                "floracion_intermedia": "Estado_1_Floracion_Intermedia",
                "floracion_maxima": "Estado_2_Floracion_Maxima",
            },
        }

    # Override config with CLI arguments
    if args.video_dir:
        video_dir = args.video_dir
    else:
        video_dir = config.get("video_dir", "bonita-white-classifier/data/raw/videos")

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = config.get(
            "output_dir", "bonita-white-classifier/data/processed/frames"
        )

    num_workers = args.workers if args.workers else config.get("workers", 4)
    log_dir = config.get("log_dir", "bonita-white-classifier/logs")

    # Setup logging
    logger = setup_logging(log_dir, args.log_level)

    # Clear output directory if requested
    if args.clear_output:
        if os.path.exists(output_dir):
            logger.info(f"Clearing output directory: {output_dir}")
            shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get list of video files
    logger.info(f"Searching for videos in: {video_dir}")
    video_paths = get_video_files(video_dir)

    if not video_paths:
        logger.error(f"No video files found in {video_dir}")
        return

    logger.info(f"Found {len(video_paths)} video files")

    # Get class mapping from config
    class_mapping = config.get(
        "class_mapping",
        {
            "prefloracion": "Estado_0_Prefloracion",
            "floracion_intermedia": "Estado_1_Floracion_Intermedia",
            "floracion_maxima": "Estado_2_Floracion_Maxima",
        },
    )

    # Create class directories
    for class_name in class_mapping.values():
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

    # Process videos
    start_time = datetime.now()
    logger.info(
        f"Starting frame extraction at {start_time.strftime('%Y-%m-%d %H:%M:%S')}"
    )

    stats = process_videos_parallel(
        video_paths, output_dir, class_mapping, config, logger, num_workers
    )

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Log summary
    logger.info("=" * 60)
    logger.info("Frame Extraction Summary")
    logger.info("=" * 60)
    logger.info(f"Total videos processed: {stats['total_videos']}")
    logger.info(f"Successful extractions: {stats['successful']}")
    logger.info(f"Failed extractions: {stats['failed']}")
    logger.info(f"Total frames extracted: {stats['total_frames_extracted']}")
    logger.info(f"Processing time: {duration:.2f} seconds")
    logger.info(
        f"Average time per video: {duration / stats['total_videos']:.2f} seconds"
    )

    if stats["failed_videos"]:
        logger.warning(f"Failed videos ({len(stats['failed_videos'])}):")
        for video_path in stats["failed_videos"][:10]:  # Show first 10
            logger.warning(f"  - {video_path}")
        if len(stats["failed_videos"]) > 10:
            logger.warning(f"  ... and {len(stats['failed_videos']) - 10} more")

    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Logs saved to: {log_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    import numpy as np  # Import numpy for resize_frame function

    main()
