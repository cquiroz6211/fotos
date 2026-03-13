"""
Device selection utilities for PyTorch.
"""

import logging
import torch

logger = logging.getLogger(__name__)


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Get the best available device for training/inference.

    Priority order:
    1. CUDA (NVIDIA GPU)
    2. MPS (Apple Silicon GPU)
    3. CPU

    Args:
        prefer_gpu: Whether to prefer GPU over CPU

    Returns:
        torch.device object
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif prefer_gpu and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")

    return device
