"""
Evaluation Script for EfficientNet-B0 Classifier

This script evaluates a trained model on test/validation data and
generates detailed metrics, confusion matrices, and reports.
"""

import os
import sys
import csv
import torch
import argparse
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.factory import create_model, list_available_models
from data.dataset import create_dataloaders
from utils.metrics import (
    MetricsTracker,
    calculate_metrics,
    plot_confusion_matrix,
    plot_class_wise_metrics,
)


logger = logging.getLogger(__name__)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: List[str],
) -> Dict:
    """
    Evaluate model on a dataset.

    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        class_names: List of class names

    Returns:
        Dictionary with evaluation results
    """
    model.eval()
    tracker = MetricsTracker(num_classes=len(class_names))

    all_probs = []
    all_preds = []
    all_labels = []

    logger.info(f"Evaluating on {len(dataloader)} batches...")

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            tracker.update(outputs, labels, 0)  # Loss not needed for eval

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                logger.info(f"  Processed {batch_idx + 1}/{len(dataloader)} batches")

    # Get metrics
    metrics = tracker.get_metrics_dict()

    # Get confusion matrix
    cm = tracker.get_confusion_matrix()

    # Get classification report
    report = tracker.get_classification_report(class_names)

    return {
        "metrics": metrics,
        "confusion_matrix": cm,
        "classification_report": report,
        "predictions": np.array(all_preds),
        "labels": np.array(all_labels),
        "probabilities": np.array(all_probs),
    }


def save_results(results: Dict, output_dir: str, split_name: str):
    """
    Save evaluation results to files.

    Args:
        results: Evaluation results dictionary
        output_dir: Directory to save results
        split_name: Name of the split (train/val/test)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save metrics to JSON
    import json

    metrics_path = os.path.join(output_dir, f"{split_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results["metrics"], f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")

    # Save classification report
    report_path = os.path.join(output_dir, f"{split_name}_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Classification Report - {split_name.upper()}\n")
        f.write("=" * 60 + "\n\n")
        f.write(results["classification_report"])
        f.write("\n\n")
        f.write("Metrics Summary:\n")
        for key, value in results["metrics"].items():
            f.write(f"  {key}: {value:.4f}\n")
    logger.info(f"Saved report to {report_path}")

    # Save predictions to CSV
    csv_path = os.path.join(output_dir, f"{split_name}_predictions.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "true_label", "predicted_label", "confidence"])
        for i in range(len(results["labels"])):
            confidence = np.max(results["probabilities"][i])
            writer.writerow(
                [
                    i,
                    results["labels"][i],
                    results["predictions"][i],
                    f"{confidence:.4f}",
                ]
            )
    logger.info(f"Saved predictions to {csv_path}")

    # Save confusion matrix plot
    cm_path = os.path.join(output_dir, f"{split_name}_confusion_matrix.png")
    plot_confusion_matrix(
        results["confusion_matrix"],
        class_names=["Estado_0", "Estado_1", "Estado_2"],
        save_path=cm_path,
        normalize=True,
        title=f"Confusion Matrix - {split_name.upper()}",
    )
    logger.info(f"Saved confusion matrix plot to {cm_path}")


def print_summary(results: Dict, split_name: str):
    """Print evaluation summary."""
    print("\n" + "=" * 60)
    print(f"Evaluation Results - {split_name.upper()}")
    print("=" * 60)
    print("\nMetrics:")
    for key, value in results["metrics"].items():
        print(f"  {key:25s}: {value:.4f}")
    print("\nClassification Report:")
    print(results["classification_report"])
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate EfficientNet-B0 model")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/splits",
        help="Path to data splits directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for evaluation",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Setup device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")

    # Load model
    logger.info(f"Loading model from {args.checkpoint}")

    # Detect model type from checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_config = checkpoint.get("model_config", {})

    # Determine model type from checkpoint or config
    if "variant" in model_config:
        # It's a MobileNet model
        if model_config.get("variant") == "small":
            model_name = "mobilenet_v3_small"
        else:
            model_name = "mobilenet_v3_large"
    else:
        # Default to EfficientNet
        model_name = "efficientnet_b0"

    logger.info(f"Detected model type: {model_name}")

    # Create model with factory
    model = create_model(
        model_name=model_name,
        num_classes=model_config.get("num_classes", 3),
        pretrained=False,  # Don't load pretrained, we'll load from checkpoint
        device=str(device),
    )

    # Load weights from checkpoint
    model.load_state_dict(checkpoint["model_state_dict"])

    # Prepare info
    info = {
        "epoch": checkpoint.get("epoch", 0),
        "metrics": checkpoint.get("metrics", {}),
    }

    logger.info(f"Model loaded. Epoch: {info.get('epoch', 'unknown')}")

    # Create dataloader
    logger.info(f"Loading {args.split} dataset...")
    dataloaders = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False,
    )

    dataloader = dataloaders.get(args.split)
    if dataloader is None:
        raise ValueError(f"Split '{args.split}' not found in {args.data_dir}")

    logger.info(f"Dataset loaded: {len(dataloader.dataset)} samples")

    # Get class names from dataset
    class_names = dataloader.dataset.classes
    logger.info(f"Classes: {class_names}")

    # Evaluate
    logger.info("Starting evaluation...")
    results = evaluate_model(model, dataloader, device, class_names)

    # Print and save results
    print_summary(results, args.split)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"eval_{args.split}_{timestamp}")
    save_results(results, output_dir, args.split)

    logger.info(f"Evaluation complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
