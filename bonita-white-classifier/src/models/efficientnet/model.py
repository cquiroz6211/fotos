"""
EfficientNet-B0 Model for Bonita White Classification

This module implements a fine-tuned EfficientNet-B0 model for
classifying the phenological state of Bonita White crops.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-B0 classifier for phenological state classification.

    Uses transfer learning with pretrained ImageNet weights.

    Args:
        num_classes: Number of output classes (default: 3)
        dropout_rate: Dropout probability (default: 0.3)
        pretrained: Whether to use pretrained weights (default: True)
    """

    def __init__(
        self, num_classes: int = 3, dropout_rate: float = 0.3, pretrained: bool = True
    ):
        super(EfficientNetClassifier, self).__init__()

        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Load pretrained EfficientNet-B0
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.efficientnet = models.efficientnet_b0(weights=weights)

        # Get the number of features from the original classifier
        in_features = self.efficientnet.classifier[1].in_features

        # Replace the classifier head
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate / 2, inplace=True),
            nn.Linear(512, num_classes),
        )

        logger.info(f"Initialized EfficientNet-B0 with {num_classes} classes")
        logger.info(f"Dropout rate: {dropout_rate}")
        logger.info(f"Pretrained: {pretrained}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, H, W)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        return self.efficientnet(x)

    def freeze_base(self) -> None:
        """Freeze all layers in the base model (feature extractor)."""
        for param in self.efficientnet.features.parameters():
            param.requires_grad = False
        logger.info("Base model (feature extractor) frozen")

    def unfreeze_base(self) -> None:
        """Unfreeze all layers in the base model."""
        for param in self.efficientnet.features.parameters():
            param.requires_grad = True
        logger.info("Base model (feature extractor) unfrozen")

    def freeze_classifier(self) -> None:
        """Freeze only the classifier head."""
        for param in self.efficientnet.classifier.parameters():
            param.requires_grad = False
        logger.info("Classifier head frozen")

    def unfreeze_classifier(self) -> None:
        """Unfreeze only the classifier head."""
        for param in self.efficientnet.classifier.parameters():
            param.requires_grad = True
        logger.info("Classifier head unfrozen")

    def get_model_info(self) -> Dict:
        """
        Get information about the model architecture.

        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        return {
            "model_name": "EfficientNet-B0",
            "num_classes": self.num_classes,
            "dropout_rate": self.dropout_rate,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": frozen_params,
            "input_size": (3, 224, 224),
        }

    def print_model_info(self) -> None:
        """Print model architecture information."""
        info = self.get_model_info()
        print("\n" + "=" * 60)
        print("Model Information")
        print("=" * 60)
        for key, value in info.items():
            if "parameters" in key:
                print(f"{key}: {value:,}")
            else:
                print(f"{key}: {value}")
        print("=" * 60 + "\n")

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        optimizer_state: Optional[dict] = None,
        scheduler_state: Optional[dict] = None,
        metrics: Optional[dict] = None,
    ) -> None:
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch number
            optimizer_state: Optional optimizer state dict
            scheduler_state: Optional scheduler state dict
            metrics: Optional metrics dictionary
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "model_config": {
                "num_classes": self.num_classes,
                "dropout_rate": self.dropout_rate,
            },
            "epoch": epoch,
        }

        if optimizer_state is not None:
            checkpoint["optimizer_state_dict"] = optimizer_state
        if scheduler_state is not None:
            checkpoint["scheduler_state_dict"] = scheduler_state
        if metrics is not None:
            checkpoint["metrics"] = metrics

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        device: str = "cpu",
        load_optimizer: bool = False,
        load_scheduler: bool = False,
    ) -> Tuple["EfficientNetClassifier", Optional[dict]]:
        """
        Load model from checkpoint.

        Args:
            path: Path to checkpoint file
            device: Device to load model on
            load_optimizer: Whether to load optimizer state
            load_scheduler: Whether to load scheduler state

        Returns:
            Tuple of (model, additional_info)
        """
        checkpoint = torch.load(path, map_location=device)

        # Create model with saved configuration
        config = checkpoint.get("model_config", {})
        model = cls(
            num_classes=config.get("num_classes", 3),
            dropout_rate=config.get("dropout_rate", 0.3),
            pretrained=False,  # Don't load pretrained weights
        )

        # Load model weights
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        logger.info(f"Checkpoint loaded from {path}")
        logger.info(f"Resuming from epoch {checkpoint.get('epoch', 0)}")

        # Prepare additional info
        info = {
            "epoch": checkpoint.get("epoch", 0),
            "metrics": checkpoint.get("metrics", {}),
        }

        if load_optimizer and "optimizer_state_dict" in checkpoint:
            info["optimizer_state_dict"] = checkpoint["optimizer_state_dict"]
        if load_scheduler and "scheduler_state_dict" in checkpoint:
            info["scheduler_state_dict"] = checkpoint["scheduler_state_dict"]

        return model, info


def create_model(
    num_classes: int = 3,
    dropout_rate: float = 0.3,
    pretrained: bool = True,
    freeze_base: bool = False,
    device: str = "cpu",
) -> EfficientNetClassifier:
    """
    Factory function to create and configure the model.

    Args:
        num_classes: Number of output classes
        dropout_rate: Dropout probability
        pretrained: Whether to use pretrained weights
        freeze_base: Whether to freeze the base model
        device: Device to move model to

    Returns:
        Configured EfficientNetClassifier model
    """
    model = EfficientNetClassifier(
        num_classes=num_classes, dropout_rate=dropout_rate, pretrained=pretrained
    )

    if freeze_base:
        model.freeze_base()

    model.to(device)

    return model


if __name__ == "__main__":
    # Test the model
    logging.basicConfig(level=logging.INFO)

    # Create model
    model = create_model(num_classes=3, pretrained=True)

    # Print model info
    model.print_model_info()

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    output = model(x)

    print(f"\nTest forward pass:")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output sample: {output[0]}")
