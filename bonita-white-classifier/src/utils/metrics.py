"""
Metrics utilities for training and evaluation.

Provides functions for calculating classification metrics and
visualizing results.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns


class MetricsTracker:
    """
    Tracks metrics during training and validation.

    Args:
        num_classes: Number of classes in the classification task
    """

    def __init__(self, num_classes: int = 3):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """Reset all tracked metrics."""
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
        self.running_loss = 0.0
        self.num_batches = 0

    def update(self, preds: torch.Tensor, labels: torch.Tensor, loss: float):
        """
        Update metrics with a new batch.

        Args:
            preds: Model predictions (logits or probabilities)
            labels: Ground truth labels
            loss: Batch loss value
        """
        # Convert predictions to class indices
        if preds.dim() > 1 and preds.shape[1] > 1:
            pred_classes = torch.argmax(preds, dim=1)
        else:
            pred_classes = preds

        self.all_preds.extend(pred_classes.cpu().numpy())
        self.all_labels.extend(labels.cpu().numpy())

        # Store probabilities if available
        if preds.dim() > 1 and preds.shape[1] > 1:
            probs = torch.softmax(preds, dim=1)
            self.all_probs.extend(probs.detach().cpu().numpy())

        self.running_loss += loss
        self.num_batches += 1

    def get_loss(self) -> float:
        """Get average loss."""
        if self.num_batches == 0:
            return 0.0
        return self.running_loss / self.num_batches

    def get_accuracy(self) -> float:
        """Calculate accuracy."""
        if len(self.all_labels) == 0:
            return 0.0
        return accuracy_score(self.all_labels, self.all_preds)

    def get_precision(self, average: str = "macro") -> float:
        """Calculate precision."""
        if len(self.all_labels) == 0:
            return 0.0
        return precision_score(
            self.all_labels, self.all_preds, average=average, zero_division=0
        )

    def get_recall(self, average: str = "macro") -> float:
        """Calculate recall."""
        if len(self.all_labels) == 0:
            return 0.0
        return recall_score(
            self.all_labels, self.all_preds, average=average, zero_division=0
        )

    def get_f1(self, average: str = "macro") -> float:
        """Calculate F1 score."""
        if len(self.all_labels) == 0:
            return 0.0
        return f1_score(
            self.all_labels, self.all_preds, average=average, zero_division=0
        )

    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        if len(self.all_labels) == 0:
            return np.zeros((self.num_classes, self.num_classes))
        return confusion_matrix(
            self.all_labels, self.all_preds, labels=list(range(self.num_classes))
        )

    def get_classification_report(self, class_names: Optional[List[str]] = None) -> str:
        """Get detailed classification report."""
        if len(self.all_labels) == 0:
            return "No data available"

        target_names = class_names or [f"Class_{i}" for i in range(self.num_classes)]
        return classification_report(
            self.all_labels, self.all_preds, target_names=target_names, zero_division=0
        )

    def get_metrics_dict(self) -> Dict[str, float]:
        """Get all metrics as a dictionary."""
        return {
            "loss": self.get_loss(),
            "accuracy": self.get_accuracy(),
            "precision_macro": self.get_precision("macro"),
            "recall_macro": self.get_recall("macro"),
            "f1_macro": self.get_f1("macro"),
            "precision_weighted": self.get_precision("weighted"),
            "recall_weighted": self.get_recall("weighted"),
            "f1_weighted": self.get_f1("weighted"),
        }


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3
) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        num_classes: Number of classes

    Returns:
        Dictionary with all metrics
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def compute_class_weights(
    labels: List[int], num_classes: int = 3, method: str = "balanced"
) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets.

    Args:
        labels: List of class labels
        num_classes: Number of classes
        method: 'balanced' or 'inverse'

    Returns:
        Tensor of class weights
    """
    counts = np.bincount(labels, minlength=num_classes)

    if method == "balanced":
        # sklearn's balanced method: n_samples / (n_classes * count)
        total = sum(counts)
        weights = [
            total / (num_classes * count) if count > 0 else 0.0 for count in counts
        ]
    elif method == "inverse":
        # Simple inverse frequency
        max_count = max(counts)
        weights = [max_count / count if count > 0 else 0.0 for count in counts]
    else:
        weights = [1.0] * num_classes

    return torch.tensor(weights, dtype=torch.float32)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """
    Plot confusion matrix.

    Args:
        cm: Confusion matrix array
        class_names: List of class names
        save_path: Path to save figure (optional)
        normalize: Whether to normalize the matrix
        title: Plot title

    Returns:
        Matplotlib figure
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)  # Handle division by zero
        fmt = ".2%"
    else:
        fmt = "d"

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_training_history(
    history: Dict[str, List[float]], save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training history (loss and metrics).

    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc', etc.
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    if "train_loss" in history:
        axes[0].plot(history["train_loss"], label="Train Loss", marker="o")
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="Val Loss", marker="o")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    if "train_acc" in history:
        axes[1].plot(history["train_acc"], label="Train Acc", marker="o")
    if "val_acc" in history:
        axes[1].plot(history["val_acc"], label="Val Acc", marker="o")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_class_wise_metrics(
    metrics: Dict[str, Dict[str, float]],
    class_names: List[str],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot class-wise metrics comparison.

    Args:
        metrics: Dictionary with metric names as keys and class values
        class_names: List of class names
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure
    """
    metric_names = list(metrics.keys())
    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, metric_name in enumerate(metric_names):
        values = [metrics[metric_name][cls] for cls in class_names]
        offset = width * (i - len(metric_names) / 2)
        ax.bar(x + offset, values, width, label=metric_name.capitalize())

    ax.set_xlabel("Classes")
    ax.set_ylabel("Score")
    ax.set_title("Class-wise Metrics Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
