"""
Training Script for Bonita White Classifier

Simplified training script that delegates to utility modules.
"""

import os
import sys
import time
import torch
import logging
import argparse
import numpy as np
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.factory import create_model, list_available_models
from data.dataset import create_dataloaders
from utils.metrics import MetricsTracker, plot_training_history
from utils import (
    setup_logging,
    load_config,
    get_device,
    EarlyStopping,
    compute_class_weights,
)


logger = logging.getLogger(__name__)


def train_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> dict:
    """Train for one epoch."""
    model.train()
    tracker = MetricsTracker(num_classes=3)

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # Track metrics
        tracker.update(outputs, labels, loss.item())

        if batch_idx % 10 == 0:
            logger.info(
                f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}"
            )

    return tracker.get_metrics_dict()


def validate_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> dict:
    """Validate for one epoch."""
    model.eval()
    tracker = MetricsTracker(num_classes=3)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            tracker.update(outputs, labels, loss.item())

    return tracker.get_metrics_dict()


def train(config: dict, model_name: str, resume_from: str = None, debug: bool = False):
    """
    Main training function.

    Args:
        config: Training configuration dictionary
        model_name: Name of the model architecture (from factory)
        resume_from: Path to checkpoint to resume from (optional)
        debug: Run in debug mode with limited batches
    """
    # Setup device
    device = get_device(config.get("training", {}).get("use_gpu", True))

    # Create dataloaders
    logger.info("Creating dataloaders...")
    data_config = config.get("data", {})
    dataloaders = create_dataloaders(
        data_dir=data_config.get("splits_dir", "data/splits"),
        batch_size=data_config.get("batch_size", 32),
        num_workers=data_config.get("num_workers", 4),
        img_size=data_config.get("image_size", 224),
        pin_memory=data_config.get("pin_memory", True),
    )

    train_loader = dataloaders.get("train")
    val_loader = dataloaders.get("val")

    if train_loader is None:
        raise ValueError("Training dataloader not found")

    # Guardar número de muestras antes de posible conversión a lista
    num_train_samples = len(train_loader.dataset)
    num_val_samples = len(val_loader.dataset) if val_loader else 0

    if debug:
        logger.info("Debug mode: limiting batches")
        train_loader = list(train_loader)[:5]
        if val_loader:
            val_loader = list(val_loader)[:2]

    # Create model
    logger.info(f"Creating model: {model_name}")
    model_config = config.get("model", {})
    model = create_model(
        model_name=model_name,
        num_classes=model_config.get("num_classes", 3),
        dropout_rate=model_config.get("dropout_rate", 0.3),
        pretrained=model_config.get("pretrained", True),
        freeze_base=model_config.get("freeze_base", False),
        device=str(device),
    )

    # Load checkpoint if resuming
    start_epoch = 0
    best_val_loss = float("inf")
    if resume_from:
        logger.info(f"Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    # Setup loss and optimizer
    class_weights = None
    if config.get("training", {}).get("use_class_weights", False):
        # Calculate class weights from training data
        all_labels = []
        for _, labels in train_loader:
            all_labels.extend(labels.numpy())
        from utils.metrics import compute_class_weights

        class_weights = compute_class_weights(all_labels, num_classes=3)
        class_weights = class_weights.to(device)
        logger.info(f"Using class weights: {class_weights}")

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    training_config = config.get("training", {})
    optimizer = AdamW(
        model.parameters(),
        lr=training_config.get("learning_rate", 0.001),
        weight_decay=training_config.get("weight_decay", 0.0001),
    )

    # Setup scheduler
    scheduler_type = training_config.get("scheduler", "cosine")
    if scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=training_config.get("num_epochs", 100),
            eta_min=training_config.get("min_lr", 1e-6),
        )
    elif scheduler_type == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=training_config.get("scheduler_patience", 5),
            factor=0.5,
        )
    else:
        scheduler = None

    # Setup early stopping
    early_stopping = EarlyStopping(
        patience=training_config.get("early_stopping_patience", 15), mode="min"
    )

    # Setup tensorboard
    log_dir = config.get("logging", {}).get("log_dir", "logs")
    writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))

    # Training history
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # Training loop
    num_epochs = training_config.get("num_epochs", 100)
    checkpoint_dir = config.get("checkpoint", {}).get("save_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger.info(f"Starting training for {num_epochs} epochs")
    logger.info(f"Training samples: {num_train_samples}")
    if val_loader:
        logger.info(f"Validation samples: {num_val_samples}")

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        logger.info(
            f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}"
        )

        # Validate
        val_metrics = None
        if val_loader:
            val_metrics = validate_epoch(model, val_loader, criterion, device)
            logger.info(
                f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}"
            )

        # Update scheduler
        if scheduler:
            if scheduler_type == "plateau" and val_metrics:
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()

        # Log to tensorboard
        writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
        writer.add_scalar("Accuracy/train", train_metrics["accuracy"], epoch)
        if val_metrics:
            writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
            writer.add_scalar("Accuracy/val", val_metrics["accuracy"], epoch)

        # Update history
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        if val_metrics:
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])

        # Save best model
        current_val_loss = val_metrics["loss"] if val_metrics else train_metrics["loss"]
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            model.save_checkpoint(
                os.path.join(checkpoint_dir, "best_model.pth"),
                epoch=epoch,
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict() if scheduler else None,
                metrics=val_metrics if val_metrics else train_metrics,
            )
            logger.info("Saved best model checkpoint")

        # Save periodic checkpoint
        if (epoch + 1) % config.get("checkpoint", {}).get("save_every", 10) == 0:
            model.save_checkpoint(
                os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth"),
                epoch=epoch,
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict() if scheduler else None,
            )

        # Early stopping
        if val_loader and early_stopping(current_val_loss):
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch time: {epoch_time:.2f}s")

    # Save final model
    model.save_checkpoint(
        os.path.join(checkpoint_dir, "final_model.pth"),
        epoch=num_epochs,
        optimizer_state=optimizer.state_dict(),
    )

    # Plot training history
    if history["train_loss"]:
        plot_path = os.path.join(log_dir, "training_history.png")
        plot_training_history(history, save_path=plot_path)
        logger.info(f"Saved training history plot to {plot_path}")

    writer.close()
    logger.info("Training completed!")


def main():
    parser = argparse.ArgumentParser(description="Train classifier")
    parser.add_argument(
        "--model",
        type=str,
        default="efficientnet_b0",
        choices=list_available_models(),
        help=f"Model architecture to use",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (optional, defaults to model's config)",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Run in debug mode with limited batches"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load config
    if args.config is None:
        # Use default config based on model
        config_path = f"configs/{args.model.replace('_', '/')}.yaml"
    else:
        config_path = args.config

    config = load_config(config_path)

    # Setup logging
    log_dir = config.get("logging", {}).get("log_dir", "logs")
    setup_logging(log_dir)

    logger.info(f"Using model: {args.model}")
    logger.info(f"Configuration loaded from {config_path}")
    logger.info(f"Random seed: {args.seed}")

    # Start training
    try:
        train(config, model_name=args.model, resume_from=args.resume, debug=args.debug)
    except Exception as e:
        logger.exception("Training failed")
        raise


if __name__ == "__main__":
    main()
