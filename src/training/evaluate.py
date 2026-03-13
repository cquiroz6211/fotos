"""
Evaluation script for EfficientNet-B0 model
Estado fenológico Bonita White - 3 Class Classification
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.efficientnet_model import EfficientNetClassifier
from utils.metrics import (
    MetricsTracker,
    calculate_metrics,
    plot_confusion_matrix,
    plot_class_wise_metrics,
    print_classification_report,
)
from data.dataset import BonitaWhiteDataset


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/evaluation.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate EfficientNet-B0 model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/efficientnet_b0.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Override batch size"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Force device to use",
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


def setup_device(preferred_device: Optional[str] = None) -> torch.device:
    """Setup computing device (CUDA/MPS/CPU)."""
    if preferred_device:
        device = torch.device(preferred_device)
        logger.info(f"Using specified device: {device}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")

    return device


def get_val_transforms(config: Dict) -> transforms.Compose:
    """
    Get validation/test transforms.

    Args:
        config (Dict): Configuration dictionary

    Returns:
        transforms.Compose: Transforms for validation/test
    """
    val_aug_config = config["data"].get("val_augmentation", {})
    image_size = config["data"]["image_size"]

    transform_list = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]

    if "normalize" in val_aug_config:
        norm_config = val_aug_config["normalize"]
        transform_list.append(
            transforms.Normalize(mean=norm_config["mean"], std=norm_config["std"])
        )

    return transforms.Compose(transform_list)


def create_dataloader(
    config: Dict,
    split: str,
    transform: transforms.Compose,
    batch_size: Optional[int] = None,
    device: torch.device = None,
) -> DataLoader:
    """
    Create dataloader for evaluation.

    Args:
        config (Dict): Configuration dictionary
        split (str): Dataset split ('train', 'val', or 'test')
        transform (transforms.Compose): Data transforms
        batch_size (Optional[int]): Override batch size
        device (torch.device): Computing device

    Returns:
        DataLoader: Dataloader for evaluation
    """
    data_config = config["data"]
    root_dir = data_config["root_dir"]

    # Create dataset
    dataset = BonitaWhiteDataset(
        root_dir=os.path.join(root_dir, split), transform=transform
    )

    logger.info(f"{split.capitalize()} samples: {len(dataset)}")

    # Create dataloader
    batch_size = batch_size or data_config.get(
        "val_batch_size", data_config["batch_size"]
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_config.get("num_workers", 4),
        pin_memory=device.type == "cuda" if device else False,
    )

    return dataloader


def load_model(
    checkpoint_path: str, device: torch.device
) -> Tuple[EfficientNetClassifier, Dict]:
    """
    Load model from checkpoint.

    Args:
        checkpoint_path (str): Path to checkpoint file
        device (torch.device): Computing device

    Returns:
        Tuple[EfficientNetClassifier, Dict]: Model and checkpoint info
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        logger.info(f"Checkpoint loaded from {checkpoint_path}")

        # Load config from checkpoint if available
        config = checkpoint.get("config", {})

        # Create model
        model = EfficientNetClassifier(
            num_classes=checkpoint.get("num_classes", 3),
            pretrained=False,  # Don't load pretrained weights
            dropout_rate=checkpoint.get("dropout_rate", 0.3),
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()

        logger.info(f"Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")

        if "metrics" in checkpoint:
            logger.info("Checkpoint training metrics:")
            for key, value in checkpoint["metrics"].items():
                logger.info(f"  {key}: {value:.4f}")

        return model, checkpoint

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    class_names: Optional[list] = None,
) -> Tuple[np.ndarray, np.ndarray, MetricsTracker]:
    """
    Evaluate model on dataset.

    Args:
        model (nn.Module): Model to evaluate
        dataloader (DataLoader): Dataloader for evaluation
        device (torch.device): Computing device
        num_classes (int): Number of classes
        class_names (Optional[list]): Class names

    Returns:
        Tuple[np.ndarray, np.ndarray, MetricsTracker]: Predictions, targets, and metrics tracker
    """
    model.eval()
    metrics = MetricsTracker(num_classes=num_classes, class_names=class_names)

    all_predictions = []
    all_targets = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = outputs.argmax(dim=1)

            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_targets.extend(labels.numpy().tolist())
            all_probabilities.extend(probabilities.cpu().numpy().tolist())

            metrics.update(outputs, labels)

    return (
        np.array(all_predictions),
        np.array(all_targets),
        np.array(all_probabilities),
        metrics,
    )


def save_evaluation_results(
    predictions: np.ndarray,
    targets: np.ndarray,
    probabilities: np.ndarray,
    metrics: MetricsTracker,
    checkpoint_path: str,
    split: str,
    output_dir: Path,
    class_names: list,
) -> Dict[str, any]:
    """
    Save comprehensive evaluation results.

    Args:
        predictions (np.ndarray): Model predictions
        targets (np.ndarray): Ground truth labels
        probabilities (np.ndarray): Prediction probabilities
        metrics (MetricsTracker): Metrics tracker
        checkpoint_path (str): Path to checkpoint
        split (str): Dataset split
        output_dir (Path): Output directory
        class_names (list): Class names

    Returns:
        Dict[str, any]: Evaluation results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate comprehensive metrics
    results = calculate_metrics(predictions, targets, len(class_names), class_names)

    # Print classification report
    print_classification_report(
        predictions,
        targets,
        class_names,
        f"Classification Report - {split.capitalize()}",
    )

    # Save confusion matrix
    cm_path = output_dir / f"confusion_matrix_{split}.png"
    plot_confusion_matrix(
        results["confusion_matrix"],
        class_names,
        title=f"Confusion Matrix - {split.capitalize()}",
        save_path=str(cm_path),
    )
    logger.info(f"Confusion matrix saved to {cm_path}")

    # Save class-wise metrics
    class_wise_metrics = metrics.get_class_wise_metrics()
    class_metrics_path = output_dir / f"class_wise_metrics_{split}.png"
    plot_class_wise_metrics(
        class_wise_metrics,
        title=f"Class-wise Metrics - {split.capitalize()}",
        save_path=str(class_metrics_path),
    )
    logger.info(f"Class-wise metrics plot saved to {class_metrics_path}")

    # Save detailed metrics to text file
    report_path = output_dir / f"classification_report_{split}.txt"
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"EVALUATION REPORT - {split.upper()}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Dataset split: {split}\n")
        f.write(f"Total samples: {len(targets)}\n\n")

        f.write("=" * 80 + "\n")
        f.write("OVERALL METRICS\n")
        f.write("=" * 80 + "\n")
        for key, value in results.items():
            if key not in ["confusion_matrix", "classification_report"]:
                f.write(
                    f"{key:20s}: {value:.4f}\n"
                    if isinstance(value, float)
                    else f"{key:20s}: {value}\n"
                )

        f.write("\n" + "=" * 80 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(results["classification_report"])

        f.write("\n" + "=" * 80 + "\n")
        f.write("CLASS-WISE METRICS\n")
        f.write("=" * 80 + "\n")
        for class_name, class_metrics in class_wise_metrics.items():
            f.write(f"\n{class_name}:\n")
            for metric_name, metric_value in class_metrics.items():
                if isinstance(metric_value, float):
                    f.write(f"  {metric_name:15s}: {metric_value:.4f}\n")
                else:
                    f.write(f"  {metric_name:15s}: {metric_value}\n")

    logger.info(f"Classification report saved to {report_path}")

    # Save predictions and probabilities as CSV
    results_path = output_dir / f"predictions_{split}.csv"
    with open(results_path, "w") as f:
        f.write("target,prediction")
        for i, class_name in enumerate(class_names):
            f.write(f",prob_{class_name}")
        f.write("\n")

        for target, pred, probs in zip(targets, predictions, probabilities):
            f.write(f"{target},{pred}")
            for prob in probs:
                f.write(f",{prob:.6f}")
            f.write("\n")

    logger.info(f"Predictions saved to {results_path}")

    results.update(
        {
            "class_wise_metrics": class_wise_metrics,
            "split": split,
            "checkpoint_path": checkpoint_path,
            "num_samples": len(targets),
        }
    )

    return results


def main():
    """Main entry point."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup device
    device = setup_device(args.device)

    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model, checkpoint = load_model(args.checkpoint, device)

    # Get transforms
    transform = get_val_transforms(config)

    # Create dataloader
    dataloader = create_dataloader(
        config, args.split, transform, args.batch_size, device
    )

    # Evaluate model
    logger.info(f"Evaluating on {args.split} split...")
    predictions, targets, probabilities, metrics = evaluate_model(
        model,
        dataloader,
        device,
        config["model"]["num_classes"],
        config.get(
            "class_names", [f"Class_{i}" for i in range(config["model"]["num_classes"])]
        ),
    )

    # Save results
    output_dir = Path(args.output_dir)
    results = save_evaluation_results(
        predictions,
        targets,
        probabilities,
        metrics,
        args.checkpoint,
        args.split,
        output_dir,
        config.get(
            "class_names", [f"Class_{i}" for i in range(config["model"]["num_classes"])]
        ),
    )

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Split: {args.split}")
    logger.info(f"Samples: {results['num_samples']}")
    logger.info(f"Accuracy: {results['accuracy']:.4%}")
    logger.info(f"Precision (weighted): {results['precision_weighted']:.4%}")
    logger.info(f"Recall (weighted): {results['recall_weighted']:.4%}")
    logger.info(f"F1 Score (weighted): {results['f1_weighted']:.4%}")
    logger.info(f"Precision (macro): {results['precision_macro']:.4%}")
    logger.info(f"Recall (macro): {results['recall_macro']:.4%}")
    logger.info(f"F1 Score (macro): {results['f1_macro']:.4%}")
    logger.info("=" * 80)

    logger.info(f"\nAll results saved to {output_dir}")
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nEvaluation interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)
