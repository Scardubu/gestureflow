"""Training metrics for model evaluation."""

import numpy as np
from typing import List


def calculate_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calculate prediction accuracy.
    
    Args:
        predictions: Model predictions
        targets: Ground truth labels
        
    Returns:
        Accuracy score
    """
    correct = np.sum(predictions.argmax(axis=1) == targets.argmax(axis=1))
    return correct / len(targets)


def top_k_accuracy(predictions: np.ndarray, targets: np.ndarray, k: int = 5) -> float:
    """Calculate top-k accuracy.
    
    Args:
        predictions: Model predictions
        targets: Ground truth labels
        k: Number of top predictions to consider
        
    Returns:
        Top-k accuracy score
    """
    target_indices = targets.argmax(axis=1)
    top_k_preds = predictions.argsort(axis=1)[:, -k:]
    
    correct = 0
    for i, target_idx in enumerate(target_indices):
        if target_idx in top_k_preds[i]:
            correct += 1
            
    return correct / len(targets)


def calculate_perplexity(loss: float) -> float:
    """Calculate perplexity from cross-entropy loss.
    
    Args:
        loss: Cross-entropy loss value
        
    Returns:
        Perplexity score
    """
    return np.exp(loss)


class MetricsTracker:
    """Track training metrics over epochs."""
    
    def __init__(self):
        self.history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'top_k_accuracy': []
        }
    
    def update(self, metrics: dict):
        """Update metrics history.
        
        Args:
            metrics: Dictionary of metric values
        """
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
    
    def get_best_epoch(self, metric: str = 'val_accuracy') -> int:
        """Get epoch with best metric value.
        
        Args:
            metric: Metric name to evaluate
            
        Returns:
            Epoch index with best metric
        """
        if metric not in self.history or not self.history[metric]:
            return -1
        
        if 'loss' in metric:
            return int(np.argmin(self.history[metric]))
        else:
            return int(np.argmax(self.history[metric]))
