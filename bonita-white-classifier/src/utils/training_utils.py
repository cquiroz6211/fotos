"""
Training utilities and helpers.
"""

import numpy as np
import torch


class EarlyStopping:
    """
    Early stopping handler to prevent overfitting.

    Monitors a metric and stops training if it doesn't improve
    for a specified number of epochs (patience).

    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        mode: 'min' to minimize metric (e.g., loss), 'max' to maximize (e.g., accuracy)
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if should stop early.

        Args:
            score: Current metric value (e.g., validation loss)

        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def compute_class_weights(labels: list, num_classes: int) -> torch.Tensor:
    """
    Compute class weights to handle class imbalance.

    Formula: weight[class] = total_samples / (num_classes * count[class])

    Args:
        labels: List of all labels in training set
        num_classes: Number of classes

    Returns:
        Tensor with class weights
    """
    class_counts = np.bincount(labels, minlength=num_classes)
    total_samples = len(labels)

    weights = [
        total_samples / (num_classes * count) if count > 0 else 1.0
        for count in class_counts
    ]

    return torch.tensor(weights, dtype=torch.float32)
