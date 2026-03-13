"""
Model Factory - Centralized model creation for all architectures.

This module provides a unified interface to create any supported model
by name, eliminating the need to hardcode imports in training scripts.
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

AVAILABLE_MODELS = {
    "efficientnet_b0": {
        "class": "EfficientNetClassifier",
        "module": "src.models.efficientnet.model",
    },
    "mobilenet_v3_small": {
        "class": "MobileNetV3Classifier",
        "module": "src.models.mobilenet.model",
    },
    "mobilenet_v3_large": {
        "class": "MobileNetV3Classifier",
        "module": "src.models.mobilenet.model",
    },
}


def create_model(
    model_name: str,
    num_classes: int = 3,
    dropout_rate: float = 0.3,
    pretrained: bool = True,
    freeze_base: bool = False,
    device: str = "cpu",
    variant: Optional[str] = None,
):
    """
    Create a model by name.

    Args:
        model_name: Name of the model architecture (efficientnet_b0, mobilenet_v3_small, mobilenet_v3_large)
        num_classes: Number of output classes
        dropout_rate: Dropout probability
        pretrained: Whether to use pretrained weights
        freeze_base: Whether to freeze the base model
        device: Device to move model to
        variant: For mobilenet, specify "small" or "large"

    Returns:
        Initialized model

    Raises:
        ValueError: If model_name is not supported
    """
    model_name = model_name.lower()

    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}\n"
            f"Available models: {list(AVAILABLE_MODELS.keys())}"
        )

    model_info = AVAILABLE_MODELS[model_name]

    if model_name == "efficientnet_b0":
        from .efficientnet.model import create_model as create_efficientnet

        model = create_efficientnet(
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            pretrained=pretrained,
            freeze_base=freeze_base,
            device=device,
        )

    elif model_name in ("mobilenet_v3_small", "mobilenet_v3_large"):
        from .mobilenet.model import create_model as create_mobilenet

        if model_name == "mobilenet_v3_small":
            mobilenet_variant = "small"
        else:
            mobilenet_variant = "large"

        model = create_mobilenet(
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            pretrained=pretrained,
            freeze_base=freeze_base,
            device=device,
            variant=mobilenet_variant,
        )

    else:
        raise ValueError(f"Model {model_name} not implemented")

    logger.info(f"Created model: {model_name} with {num_classes} classes")

    return model


def list_available_models():
    """Return list of available model names."""
    return list(AVAILABLE_MODELS.keys())


def get_model_info(model_name: str) -> Dict:
    """
    Get basic information about a model without instantiating it.

    Args:
        model_name: Name of the model

    Returns:
        Dictionary with model metadata
    """
    model_name = model_name.lower()

    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    info = {
        "efficientnet_b0": {
            "parameters": "~5.3M",
            "macs": "390M",
            "input_size": (3, 224, 224),
            "description": "EfficientNet-B0 - Good balance of accuracy and efficiency",
        },
        "mobilenet_v3_small": {
            "parameters": "~2.5M",
            "macs": "56M",
            "input_size": (3, 224, 224),
            "description": "MobileNetV3-Small - Optimized for edge devices, fastest inference",
        },
        "mobilenet_v3_large": {
            "parameters": "~5.5M",
            "macs": "218M",
            "input_size": (3, 224, 224),
            "description": "MobileNetV3-Large - More accurate than Small, still efficient",
        },
    }

    return info.get(model_name, {})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Available models:")
    for name in list_available_models():
        info = get_model_info(name)
        print(f"  - {name}")
        print(f"    Params: {info.get('parameters')}, MACs: {info.get('macs')}")
        print(f"    {info.get('description')}")
        print()

    print("\nTesting model creation:")
    for name in list_available_models():
        model = create_model(name, num_classes=3, pretrained=False)
        print(f"  {name}: ✓")
