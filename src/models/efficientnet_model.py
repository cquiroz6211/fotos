"""
EfficientNet-B0 Model Wrapper for 3-Class Classification
Estado fenológico Bonita White classification task
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-B0 classifier for 3-class phenological state classification.

    Args:
        num_classes (int): Number of output classes. Default: 3
        pretrained (bool): Whether to use ImageNet pretrained weights. Default: True
        dropout_rate (float): Dropout probability in classifier head. Default: 0.3
        freeze_base (bool): Whether to freeze base model initially. Default: False
    """

    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        freeze_base: bool = False,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.pretrained = pretrained
        self.dropout_rate = dropout_rate

        try:
            # Load pretrained EfficientNet-B0
            weights = (
                models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            )
            self.base_model = models.efficientnet_b0(weights=weights)
            logger.info(f"EfficientNet-B0 loaded (pretrained={pretrained})")
        except Exception as e:
            logger.error(f"Failed to load EfficientNet-B0: {e}")
            raise

        # Get number of features from the base model
        num_features = self.base_model.classifier[1].in_features

        # Create custom classifier head
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes),
        )

        # Initialize classifier weights
        self._initialize_classifier()

        # Optionally freeze base model
        if freeze_base:
            self.freeze_base()

    def _initialize_classifier(self) -> None:
        """Initialize classifier layers with Xavier initialization."""
        for module in self.base_model.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        return self.base_model(x)

    def freeze_base(self) -> None:
        """Freeze all parameters in the base model (features)."""
        for param in self.base_model.features.parameters():
            param.requires_grad = False
        logger.info("Base model (features) frozen")

    def unfreeze_base(self) -> None:
        """Unfreeze all parameters in the base model (features)."""
        for param in self.base_model.features.parameters():
            param.requires_grad = True
        logger.info("Base model (features) unfrozen")

    def freeze_classifier(self) -> None:
        """Freeze classifier parameters only."""
        for param in self.base_model.classifier.parameters():
            param.requires_grad = False
        logger.info("Classifier frozen")

    def unfreeze_classifier(self) -> None:
        """Unfreeze classifier parameters."""
        for param in self.base_model.classifier.parameters():
            param.requires_grad = True
        logger.info("Classifier unfrozen")

    def get_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Save model checkpoint.

        Args:
            path (str): Path to save checkpoint
            epoch (int): Current epoch number
            optimizer (torch.optim.Optimizer, optional): Optimizer state
            scheduler: Scheduler state
            **kwargs: Additional metrics or info to save

        Returns:
            Dict[str, Any]: Checkpoint dictionary
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "num_classes": self.num_classes,
            "pretrained": self.pretrained,
            "dropout_rate": self.dropout_rate,
            **kwargs,
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        try:
            torch.save(checkpoint, path)
            logger.info(f"Checkpoint saved to {path}")
            return checkpoint
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise

    @classmethod
    def load_checkpoint(
        cls, path: str, device: torch.device
    ) -> "EfficientNetClassifier":
        """
        Load model from checkpoint.

        Args:
            path (str): Path to checkpoint file
            device (torch.device): Device to load model on

        Returns:
            EfficientNetClassifier: Loaded model
        """
        try:
            checkpoint = torch.load(path, map_location=device)
            logger.info(f"Loading checkpoint from {path}")

            # Create model with saved hyperparameters
            model = cls(
                num_classes=checkpoint.get("num_classes", 3),
                pretrained=checkpoint.get("pretrained", True),
                dropout_rate=checkpoint.get("dropout_rate", 0.3),
            )

            # Load state dict
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)

            logger.info(
                f"Model loaded successfully (epoch {checkpoint.get('epoch', 'unknown')})"
            )
            return model

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information summary."""
        total_params = self.get_total_params()
        trainable_params = self.get_trainable_params()

        return {
            "model_name": "EfficientNet-B0",
            "num_classes": self.num_classes,
            "pretrained": self.pretrained,
            "dropout_rate": self.dropout_rate,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": total_params - trainable_params,
            "input_size": (3, 224, 224),
        }

    def print_model_info(self) -> None:
        """Print model information summary."""
        info = self.get_model_info()
        print("\n" + "=" * 50)
        print("EfficientNet-B0 Model Information")
        print("=" * 50)
        for key, value in info.items():
            if isinstance(value, tuple):
                print(f"{key:20s}: {value}")
            else:
                print(
                    f"{key:20s}: {value:,}"
                    if isinstance(value, int)
                    else f"{key:20s}: {value}"
                )
        print("=" * 50 + "\n")
