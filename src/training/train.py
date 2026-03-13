"""
Training script for EfficientNet-B0 model
Estado fenológico Bonita White - 3 Class Classification
"""

import os
import sys
import yaml
import logging
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.efficientnet_model import EfficientNetClassifier
from utils.metrics import MetricsTracker, compute_class_weights, plot_training_history
from data.dataset import BonitaWhiteDataset


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/training.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/efficientnet_b0.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


def setup_device(config: Dict) -> torch.device:
    """Setup computing device (CUDA/MPS/CPU)."""
    device_config = config.get("device", {})

    if device_config.get("auto_detect", True):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using MPS device (Apple Silicon)")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")
    else:
        preferred = device_config.get("preferred_device", "cpu")
        device = torch.device(preferred)
        logger.info(f"Using specified device: {device}")

    return device


def get_data_transforms(config: Dict) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get data augmentation transforms for training and validation.

    Args:
        config (Dict): Configuration dictionary

    Returns:
        Tuple[transforms.Compose, transforms.Compose]: Train and val transforms
    """
    aug_config = config["data"].get("augmentation", {})
    val_aug_config = config["data"].get("val_augmentation", {})

    # Training transforms
    train_transforms_list = []

    if aug_config.get("random_resized_crop", False):
        image_size = config["data"]["image_size"]
        train_transforms_list.append(
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0))
        )

    if aug_config.get("random_horizontal_flip", False):
        train_transforms_list.append(transforms.RandomHorizontalFlip())

    if aug_config.get("random_rotation", 0) > 0:
        train_transforms_list.append(
            transforms.RandomRotation(aug_config["random_rotation"])
        )

    if "random_affine" in aug_config:
        affine_config = aug_config["random_affine"]
        train_transforms_list.append(
            transforms.RandomAffine(
                degrees=affine_config.get("degrees", 0),
                translate=affine_config.get("translate", [0, 0]),
                scale=affine_config.get("scale", [1.0, 1.0]),
            )
        )

    if "color_jitter" in aug_config:
        jitter_config = aug_config["color_jitter"]
        train_transforms_list.append(
            transforms.ColorJitter(
                brightness=jitter_config.get("brightness", 0),
                contrast=jitter_config.get("contrast", 0),
                saturation=jitter_config.get("saturation", 0),
                hue=jitter_config.get("hue", 0),
            )
        )

    train_transforms_list.append(transforms.ToTensor())

    if "normalize" in aug_config:
        norm_config = aug_config["normalize"]
        train_transforms_list.append(
            transforms.Normalize(mean=norm_config["mean"], std=norm_config["std"])
        )

    # Validation transforms
    val_transforms_list = []
    image_size = config["data"]["image_size"]
    val_transforms_list.append(transforms.Resize((image_size, image_size)))
    val_transforms_list.append(transforms.ToTensor())

    if "normalize" in val_aug_config:
        norm_config = val_aug_config["normalize"]
        val_transforms_list.append(
            transforms.Normalize(mean=norm_config["mean"], std=norm_config["std"])
        )

    train_transforms = transforms.Compose(train_transforms_list)
    val_transforms = transforms.Compose(val_transforms_list)

    return train_transforms, val_transforms


def create_dataloaders(
    config: Dict,
    train_transform: transforms.Compose,
    val_transform: transforms.Compose,
    device: torch.device,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.

    Args:
        config (Dict): Configuration dictionary
        train_transform (transforms.Compose): Training transforms
        val_transform (transforms.Compose): Validation transforms
        device (torch.device): Computing device

    Returns:
        Tuple[DataLoader, DataLoader]: Training and validation dataloaders
    """
    data_config = config["data"]
    root_dir = data_config["root_dir"]

    # Create datasets
    train_dataset = BonitaWhiteDataset(
        root_dir=os.path.join(root_dir, data_config["train_split"]),
        transform=train_transform,
    )

    val_dataset = BonitaWhiteDataset(
        root_dir=os.path.join(root_dir, data_config["val_split"]),
        transform=val_transform,
    )

    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config["batch_size"],
        shuffle=True,
        num_workers=data_config.get("num_workers", 4),
        pin_memory=device.type == "cuda",
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config.get("val_batch_size", data_config["batch_size"]),
        shuffle=False,
        num_workers=data_config.get("num_workers", 4),
        pin_memory=device.type == "cuda",
    )

    return train_loader, val_loader


def setup_model(config: Dict, device: torch.device) -> EfficientNetClassifier:
    """
    Setup EfficientNet model.

    Args:
        config (Dict): Configuration dictionary
        device (torch.device): Computing device

    Returns:
        EfficientNetClassifier: Model instance
    """
    model_config = config["model"]

    model = EfficientNetClassifier(
        num_classes=model_config["num_classes"],
        pretrained=model_config.get("pretrained", True),
        dropout_rate=model_config.get("dropout_rate", 0.3),
        freeze_base=model_config.get("freeze_base", False),
    )

    model = model.to(device)
    model.print_model_info()

    return model


def setup_optimizer_and_scheduler(config: Dict, model: nn.Module, device: torch.device):
    """
    Setup optimizer and learning rate scheduler.

    Args:
        config (Dict): Configuration dictionary
        model (nn.Module): Model to optimize
        device (torch.device): Computing device

    Returns:
        Tuple: Optimizer and scheduler
    """
    train_config = config["training"]
    optimizer_config = train_config["optimizer"]
    scheduler_config = train_config["scheduler"]

    # Setup optimizer
    if optimizer_config["name"] == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_config["learning_rate"],
            weight_decay=optimizer_config.get("weight_decay", 0.01),
            betas=optimizer_config.get("betas", [0.9, 0.999]),
            eps=optimizer_config.get("eps", 1e-8),
        )
    elif optimizer_config["name"] == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_config["learning_rate"],
            weight_decay=optimizer_config.get("weight_decay", 0.0),
        )
    elif optimizer_config["name"] == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=train_config["learning_rate"],
            momentum=optimizer_config.get("momentum", 0.9),
            weight_decay=optimizer_config.get("weight_decay", 0.0),
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_config['name']}")

    logger.info(f"Optimizer: {optimizer_config['name']}")

    # Setup scheduler
    if scheduler_config["name"] == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config["T_max"],
            eta_min=scheduler_config.get("eta_min", 1e-6),
        )
    elif scheduler_config["name"] == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config["mode"],
            factor=scheduler_config.get("factor", 0.5),
            patience=scheduler_config.get("patience", 10),
            min_lr=scheduler_config.get("min_lr", 1e-6),
        )
    else:
        scheduler = None
        logger.warning("No scheduler configured")

    logger.info(f"Scheduler: {scheduler_config.get('name', 'None')}")

    return optimizer, scheduler


def setup_loss_function(
    config: Dict, train_loader: DataLoader, device: torch.device
) -> nn.Module:
    """
    Setup loss function.

    Args:
        config (Dict): Configuration dictionary
        train_loader (DataLoader): Training dataloader
        device (torch.device): Computing device

    Returns:
        nn.Module: Loss function
    """
    loss_config = config["training"]["loss"]

    # Compute class weights if needed
    if loss_config.get("use_class_weights", False):
        # Get all labels from training set
        all_labels = []
        for _, labels in train_loader:
            all_labels.extend(labels.numpy())
        all_labels = np.array(all_labels)

        class_weights = compute_class_weights(
            all_labels, config["model"]["num_classes"]
        )
        class_weights = class_weights.to(device)

        loss_fn = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=loss_config.get("label_smoothing", 0.0),
        )
        logger.info("Using class-weighted CrossEntropyLoss")
    else:
        loss_fn = nn.CrossEntropyLoss(
            label_smoothing=loss_config.get("label_smoothing", 0.0)
        )
        logger.info("Using standard CrossEntropyLoss")

    return loss_fn


class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        """
        Args:
            patience (int): Number of epochs to wait before stopping
            min_delta (float): Minimum change to qualify as an improvement
            mode (str): 'min' for metrics to minimize, 'max' for metrics to maximize
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if should stop training.

        Args:
            score (float): Current score

        Returns:
            bool: True if should stop training
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            improvement = self.best_score - score
        else:
            improvement = score - self.best_score

        if improvement > self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    config: Dict,
    writer: Optional[SummaryWriter] = None,
    debug: bool = False,
) -> Tuple[float, MetricsTracker]:
    """
    Train for one epoch.

    Args:
        model (nn.Module): Model to train
        dataloader (DataLoader): Training dataloader
        optimizer (torch.optim.Optimizer): Optimizer
        loss_fn (nn.Module): Loss function
        device (torch.device): Computing device
        epoch (int): Current epoch
        config (Dict): Configuration dictionary
        writer (Optional[SummaryWriter]): TensorBoard writer
        debug (bool): Debug mode flag

    Returns:
        Tuple[float, MetricsTracker]: Average loss and metrics tracker
    """
    model.train()
    metrics = MetricsTracker(
        num_classes=config["model"]["num_classes"],
        class_names=config.get("class_names"),
    )

    total_loss = 0.0
    num_batches = 0

    gradient_clipping = config["training"].get("gradient_clipping", {})
    clip_enabled = gradient_clipping.get("enabled", False)
    max_norm = gradient_clipping.get("max_norm", 1.0)

    max_batches = None
    if debug and config.get("debug", {}).get("max_batches_per_epoch"):
        max_batches = config["debug"]["max_batches_per_epoch"]

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")

    for batch_idx, (images, labels) in enumerate(pbar):
        if max_batches and batch_idx >= max_batches:
            break

        images, labels = images.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if clip_enabled:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()

        # Update metrics
        metrics.update(outputs, labels, loss.item())
        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix(
            {"loss": f"{loss.item():.4f}", "acc": f"{metrics.get_accuracy():.4f}"}
        )

        # Log to tensorboard
        if writer and (batch_idx % config["logging"]["log_frequency"] == 0):
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar("Train/BatchLoss", loss.item(), global_step)
            writer.add_scalar(
                "Train/BatchAccuracy", metrics.get_accuracy(), global_step
            )

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    return avg_loss, metrics


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    config: Dict,
    writer: Optional[SummaryWriter] = None,
) -> Tuple[float, MetricsTracker]:
    """
    Validate for one epoch.

    Args:
        model (nn.Module): Model to validate
        dataloader (DataLoader): Validation dataloader
        loss_fn (nn.Module): Loss function
        device (torch.device): Computing device
        epoch (int): Current epoch
        config (Dict): Configuration dictionary
        writer (Optional[SummaryWriter]): TensorBoard writer

    Returns:
        Tuple[float, MetricsTracker]: Average loss and metrics tracker
    """
    model.eval()
    metrics = MetricsTracker(
        num_classes=config["model"]["num_classes"],
        class_names=config.get("class_names"),
    )

    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")

    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            metrics.update(outputs, labels, loss.item())
            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{metrics.get_accuracy():.4f}"}
            )

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    return avg_loss, metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_path: str,
    config: Dict,
) -> None:
    """Save training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
        "config": config,
    }

    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")


def train(config: Dict, args: argparse.Namespace) -> None:
    """
    Main training loop.

    Args:
        config (Dict): Configuration dictionary
        args (argparse.Namespace): Command line arguments
    """
    # Setup
    device = setup_device(config)
    seed = args.seed or config.get("seed", 42)
    set_seed(seed)

    # Create directories
    checkpoint_dir = Path(config["checkpointing"]["save_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(config["logging"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    # Setup tensorboard
    writer = SummaryWriter(config["logging"]["tensorboard_dir"])

    # Get data transforms and dataloaders
    train_transform, val_transform = get_data_transforms(config)
    train_loader, val_loader = create_dataloaders(
        config, train_transform, val_transform, device
    )

    # Setup model
    model = setup_model(config, device)

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        logger.info(f"Resuming from epoch {start_epoch}")

    # Setup optimizer and scheduler
    optimizer, scheduler = setup_optimizer_and_scheduler(config, model, device)

    # Setup loss function
    loss_fn = setup_loss_function(config, train_loader, device)

    # Setup early stopping
    early_stopping_config = config.get("early_stopping", {})
    if early_stopping_config.get("enabled", False):
        early_stopping = EarlyStopping(
            patience=early_stopping_config.get("patience", 10),
            min_delta=early_stopping_config.get("min_delta", 0.001),
            mode=early_stopping_config.get("mode", "min"),
        )
        logger.info("Early stopping enabled")
    else:
        early_stopping = None

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "train_f1": [],
        "val_f1": [],
        "learning_rate": [],
    }

    # Training loop
    num_epochs = config["training"]["num_epochs"]
    best_metric = (
        -np.inf if early_stopping_config.get("mode", "min") == "max" else np.inf
    )

    logger.info(f"Starting training for {num_epochs} epochs")
    logger.info("=" * 80)

    for epoch in range(start_epoch, num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Training
        train_loss, train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            epoch,
            config,
            writer,
            args.debug,
        )

        # Validation
        val_loss, val_metrics = validate_epoch(
            model, val_loader, loss_fn, device, epoch, config, writer
        )

        # Log epoch metrics
        train_dict = train_metrics.get_metrics_dict()
        val_dict = val_metrics.get_metrics_dict()

        logger.info(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_dict['accuracy']:.4%} | Train F1: {train_dict['f1']:.4%}"
        )
        logger.info(
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_dict['accuracy']:.4%} | Val F1: {val_dict['f1']:.4%}"
        )

        # Log to tensorboard
        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Train/Accuracy", train_dict["accuracy"], epoch)
        writer.add_scalar("Train/F1", train_dict["f1"], epoch)
        writer.add_scalar("Val/Loss", val_loss, epoch)
        writer.add_scalar("Val/Accuracy", val_dict["accuracy"], epoch)
        writer.add_scalar("Val/F1", val_dict["f1"], epoch)
        writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)

        # Update history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_dict["accuracy"])
        history["val_accuracy"].append(val_dict["accuracy"])
        history["train_f1"].append(train_dict["f1"])
        history["val_f1"].append(val_dict["f1"])
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])

        # Update scheduler
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_dict["f1"])
            else:
                scheduler.step()

        # Save checkpoint
        checkpointing_config = config["checkpointing"]
        save_frequency = checkpointing_config.get("save_frequency", 1)

        if (epoch + 1) % save_frequency == 0:
            checkpoint_path = (
                checkpoint_dir
                / f"{checkpointing_config['checkpoint_name']}_epoch_{epoch + 1}.pth"
            )
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                {**train_dict, **{f"val_{k}": v for k, v in val_dict.items()}},
                str(checkpoint_path),
                config,
            )

        # Save best model
        if checkpointing_config.get("save_best_only", True):
            monitor_metric = checkpointing_config.get("monitor", "val_f1")
            metric_mode = checkpointing_config.get("mode", "max")

            current_metric = (
                val_dict[monitor_metric.replace("val_", "")]
                if monitor_metric.startswith("val_")
                else val_dict[monitor_metric]
            )

            is_better = (metric_mode == "max" and current_metric > best_metric) or (
                metric_mode == "min" and current_metric < best_metric
            )

            if is_better:
                best_metric = current_metric
                best_path = (
                    checkpoint_dir
                    / f"{checkpointing_config['checkpoint_name']}_best.pth"
                )
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    {**train_dict, **{f"val_{k}": v for k, v in val_dict.items()}},
                    str(best_path),
                    config,
                )
                logger.info(
                    f"Best model saved with {monitor_metric}={current_metric:.4f}"
                )

        # Save plots
        if config["logging"].get("save_plots", False) and (
            (epoch + 1) % config["logging"].get("plot_frequency", 5) == 0
        ):
            plot_path = log_dir / f"training_history_epoch_{epoch + 1}.png"
            plot_training_history(history, save_path=str(plot_path))
            plt.close("all")

        # Early stopping
        if early_stopping:
            monitor = early_stopping_config.get("monitor", "val_loss")
            monitor_metric = (
                val_dict[monitor.replace("val_", "")]
                if monitor.startswith("val_")
                else val_dict[monitor]
            )

            if early_stopping(monitor_metric):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

    # Final plot
    if config["logging"].get("save_plots", False):
        plot_path = log_dir / "training_history_final.png"
        plot_training_history(history, save_path=str(plot_path))
        logger.info(f"Final training history plot saved to {plot_path}")

    # Close tensorboard writer
    writer.close()

    logger.info("=" * 80)
    logger.info("Training completed!")
    logger.info(f"Best {checkpointing_config['monitor']}: {best_metric:.4f}")
    logger.info("=" * 80)


def main():
    """Main entry point."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override debug mode
    if args.debug:
        config["debug"]["enabled"] = True
        logger.setLevel(logging.DEBUG)

    # Override seed
    if args.seed:
        config["seed"] = args.seed

    # Start training
    try:
        train(config, args)
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
