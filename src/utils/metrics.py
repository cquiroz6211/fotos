"""
Metrics utilities for model training and evaluation
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import logging

logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    Track and compute metrics during training and validation.

    Args:
        num_classes (int): Number of classes in the dataset
        class_names (Optional[List[str]]): List of class names for display
    """

    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class {i}" for i in range(num_classes)]
        self.reset()

    def reset(self) -> None:
        """Reset all tracked metrics."""
        self.predictions: List[int] = []
        self.targets: List[int] = []
        self.losses: List[float] = []

    def update(
        self, preds: torch.Tensor, targets: torch.Tensor, loss: Optional[float] = None
    ) -> None:
        """
        Update tracked metrics with new batch.

        Args:
            preds (torch.Tensor): Model predictions (logits)
            targets (torch.Tensor): Ground truth labels
            loss (Optional[float]): Loss value for this batch
        """
        # Get predicted classes
        pred_classes = preds.argmax(dim=1).cpu().numpy()
        target_classes = targets.cpu().numpy()

        self.predictions.extend(pred_classes.tolist())
        self.targets.extend(target_classes.tolist())

        if loss is not None:
            self.losses.append(loss)

    def get_accuracy(self) -> float:
        """Calculate accuracy."""
        if len(self.predictions) == 0:
            return 0.0
        return accuracy_score(self.targets, self.predictions)

    def get_precision(self, average: str = "weighted") -> float:
        """
        Calculate precision.

        Args:
            average (str): Averaging method ('micro', 'macro', 'weighted', None)

        Returns:
            float: Precision score
        """
        if len(self.predictions) == 0:
            return 0.0
        return precision_score(
            self.targets, self.predictions, average=average, zero_division=0
        )

    def get_recall(self, average: str = "weighted") -> float:
        """
        Calculate recall.

        Args:
            average (str): Averaging method ('micro', 'macro', 'weighted', None)

        Returns:
            float: Recall score
        """
        if len(self.predictions) == 0:
            return 0.0
        return recall_score(
            self.targets, self.predictions, average=average, zero_division=0
        )

    def get_f1(self, average: str = "weighted") -> float:
        """
        Calculate F1 score.

        Args:
            average (str): Averaging method ('micro', 'macro', 'weighted', None)

        Returns:
            float: F1 score
        """
        if len(self.predictions) == 0:
            return 0.0
        return f1_score(
            self.targets, self.predictions, average=average, zero_division=0
        )

    def get_average_loss(self) -> float:
        """Calculate average loss."""
        if len(self.losses) == 0:
            return 0.0
        return sum(self.losses) / len(self.losses)

    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        if len(self.predictions) == 0:
            return np.zeros((self.num_classes, self.num_classes), dtype=int)
        return confusion_matrix(self.targets, self.predictions)

    def get_classification_report(self) -> str:
        """Get classification report."""
        if len(self.predictions) == 0:
            return "No predictions made yet."
        return classification_report(
            self.targets,
            self.predictions,
            target_names=self.class_names,
            zero_division=0,
        )

    def get_metrics_dict(self) -> Dict[str, float]:
        """Get all metrics as a dictionary."""
        return {
            "accuracy": self.get_accuracy(),
            "precision": self.get_precision(average="weighted"),
            "recall": self.get_recall(average="weighted"),
            "f1": self.get_f1(average="weighted"),
            "loss": self.get_average_loss(),
        }

    def get_class_wise_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Get per-class metrics.

        Returns:
            Dict[str, Dict[str, float]]: Dictionary with class names as keys and
                                          metrics (precision, recall, f1) as values
        """
        if len(self.predictions) == 0:
            return {}

        class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            class_metrics[class_name] = {
                "precision": precision_score(
                    self.targets,
                    self.predictions,
                    labels=[i],
                    average="micro",
                    zero_division=0,
                ),
                "recall": recall_score(
                    self.targets,
                    self.predictions,
                    labels=[i],
                    average="micro",
                    zero_division=0,
                ),
                "f1": f1_score(
                    self.targets,
                    self.predictions,
                    labels=[i],
                    average="micro",
                    zero_division=0,
                ),
                "support": self.targets.count(i),
            }

        return class_metrics

    def get_summary(self) -> str:
        """Get formatted summary of all metrics."""
        metrics = self.get_metrics_dict()
        summary = []
        summary.append("=" * 60)
        summary.append("METRICS SUMMARY")
        summary.append("=" * 60)
        for metric, value in metrics.items():
            if metric == "loss":
                summary.append(f"{metric:20s}: {value:.4f}")
            else:
                summary.append(f"{metric:20s}: {value:.4%}")
        summary.append("=" * 60)
        return "\n".join(summary)


def calculate_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
    class_names: Optional[List[str]] = None,
) -> Dict[str, any]:
    """
    Calculate comprehensive metrics for model predictions.

    Args:
        predictions (np.ndarray): Model predictions
        targets (np.ndarray): Ground truth labels
        num_classes (int): Number of classes
        class_names (Optional[List[str]]): Class names for display

    Returns:
        Dict[str, any]: Dictionary containing all metrics
    """
    class_names = class_names or [f"Class_{i}" for i in range(num_classes)]

    metrics = {
        "accuracy": accuracy_score(targets, predictions),
        "precision_macro": precision_score(
            targets, predictions, average="macro", zero_division=0
        ),
        "precision_weighted": precision_score(
            targets, predictions, average="weighted", zero_division=0
        ),
        "recall_macro": recall_score(
            targets, predictions, average="macro", zero_division=0
        ),
        "recall_weighted": recall_score(
            targets, predictions, average="weighted", zero_division=0
        ),
        "f1_macro": f1_score(targets, predictions, average="macro", zero_division=0),
        "f1_weighted": f1_score(
            targets, predictions, average="weighted", zero_division=0
        ),
        "confusion_matrix": confusion_matrix(targets, predictions),
        "classification_report": classification_report(
            targets, predictions, target_names=class_names, zero_division=0
        ),
    }

    return metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot confusion matrix using seaborn.

    Args:
        cm (np.ndarray): Confusion matrix
        class_names (List[str]): List of class names
        title (str): Plot title
        cmap (str): Colormap for heatmap
        figsize (Tuple[int, int]): Figure size
        save_path (Optional[str]): Path to save the plot

    Returns:
        plt.Figure: The matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Normalize confusion matrix
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Create heatmap
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2%",
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Normalized Count"},
        ax=ax,
    )

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Rotate x-axis labels if needed
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()

    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Confusion matrix saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save confusion matrix: {e}")

    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    metrics: List[str] = ["loss", "accuracy"],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training and validation history.

    Args:
        history (Dict[str, List[float]]): Dictionary with training history
        metrics (List[str]): List of metrics to plot
        save_path (Optional[str]): Path to save the plot

    Returns:
        plt.Figure: The matplotlib figure object
    """
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 5))

    if num_metrics == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        if metric == "loss":
            ax.plot(history.get("train_loss", []), label="Train Loss", marker="o")
            ax.plot(history.get("val_loss", []), label="Validation Loss", marker="s")
            ax.set_ylabel("Loss")
            ax.set_title("Training and Validation Loss")
        else:
            ax.plot(
                history.get(f"train_{metric}", []),
                label=f"Train {metric.capitalize()}",
                marker="o",
            )
            ax.plot(
                history.get(f"val_{metric}", []),
                label=f"Validation {metric.capitalize()}",
                marker="s",
            )
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f"Training and Validation {metric.capitalize()}")

        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Training history plot saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save training history plot: {e}")

    return fig


def plot_class_wise_metrics(
    class_metrics: Dict[str, Dict[str, float]],
    title: str = "Class-wise Metrics",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot class-wise precision, recall, and F1 scores.

    Args:
        class_metrics (Dict[str, Dict[str, float]]): Class-wise metrics
        title (str): Plot title
        figsize (Tuple[int, int]): Figure size
        save_path (Optional[str]): Path to save the plot

    Returns:
        plt.Figure: The matplotlib figure object
    """
    class_names = list(class_metrics.keys())
    metrics_names = ["precision", "recall", "f1"]

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(class_names))
    width = 0.25

    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    for idx, metric in enumerate(metrics_names):
        values = [class_metrics[name][metric] for name in class_names]
        bars = ax.bar(
            x + idx * width, values, width, label=metric.capitalize(), color=colors[idx]
        )

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel("Classes", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Class-wise metrics plot saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save class-wise metrics plot: {e}")

    return fig


def print_classification_report(
    predictions: np.ndarray,
    targets: np.ndarray,
    class_names: List[str],
    title: str = "Classification Report",
) -> None:
    """
    Print a formatted classification report.

    Args:
        predictions (np.ndarray): Model predictions
        targets (np.ndarray): Ground truth labels
        class_names (List[str]): List of class names
        title (str): Report title
    """
    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80)

    report = classification_report(
        targets, predictions, target_names=class_names, zero_division=0
    )
    print(report)

    cm = confusion_matrix(targets, predictions)
    print("\nConfusion Matrix:")
    print(cm)

    print("=" * 80 + "\n")


def compute_class_weights(targets: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets.

    Args:
        targets (np.ndarray): Ground truth labels
        num_classes (int): Number of classes

    Returns:
        torch.Tensor: Class weights tensor
    """
    class_counts = np.bincount(targets, minlength=num_classes)
    total_samples = len(targets)

    # Compute inverse frequency weights
    class_weights = total_samples / (num_classes * class_counts)

    # Convert to tensor
    weights = torch.FloatTensor(class_weights)

    logger.info(f"Class weights: {dict(zip(range(num_classes), weights.tolist()))}")
    return weights
