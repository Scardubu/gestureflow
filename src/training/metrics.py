"""
Evaluation metrics for GestureFlow.
"""
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def calculate_top_k_accuracy(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    k: int = 5
) -> float:
    """
    Calculate top-k accuracy.
    
    Args:
        y_true: True labels
        y_pred_probs: Predicted probabilities
        k: Number of top predictions to consider
        
    Returns:
        Top-k accuracy
    """
    top_k_preds = np.argsort(y_pred_probs, axis=1)[:, -k:]
    
    correct = 0
    for i, true_label in enumerate(y_true):
        if true_label in top_k_preds[i]:
            correct += 1
    
    return correct / len(y_true)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_probs: np.ndarray = None
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_probs: Optional predicted probabilities
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1_score'] = f1
    
    # Top-k accuracy if probabilities provided
    if y_pred_probs is not None:
        for k in [3, 5, 10]:
            metrics[f'top_{k}_accuracy'] = calculate_top_k_accuracy(
                y_true, y_pred_probs, k=k
            )
    
    return metrics


def print_metrics(metrics: Dict[str, float]):
    """
    Pretty print metrics.
    
    Args:
        metrics: Dictionary of metrics
    """
    print("\n" + "=" * 50)
    print("EVALUATION METRICS")
    print("=" * 50)
    
    for name, value in metrics.items():
        print(f"{name:20s}: {value:.4f}")
    
    print("=" * 50 + "\n")


def calculate_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str] = None
) -> np.ndarray:
    """
    Calculate confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Optional label names
        
    Returns:
        Confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    return cm


def calculate_per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None
) -> Dict[str, float]:
    """
    Calculate per-class accuracy.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional class names
        
    Returns:
        Dictionary mapping class to accuracy
    """
    unique_classes = np.unique(y_true)
    
    per_class_acc = {}
    for cls in unique_classes:
        mask = y_true == cls
        class_acc = accuracy_score(y_true[mask], y_pred[mask])
        
        class_name = class_names[cls] if class_names else str(cls)
        per_class_acc[class_name] = class_acc
    
    return per_class_acc


class MetricsTracker:
    """Track metrics during training."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
    
    def update(
        self,
        train_loss: float,
        train_accuracy: float,
        val_loss: float,
        val_accuracy: float
    ):
        """
        Update metrics.
        
        Args:
            train_loss: Training loss
            train_accuracy: Training accuracy
            val_loss: Validation loss
            val_accuracy: Validation accuracy
        """
        self.history['train_loss'].append(train_loss)
        self.history['train_accuracy'].append(train_accuracy)
        self.history['val_loss'].append(val_loss)
        self.history['val_accuracy'].append(val_accuracy)
    
    def get_best_epoch(self, metric: str = 'val_accuracy') -> int:
        """
        Get the best epoch based on a metric.
        
        Args:
            metric: Metric name
            
        Returns:
            Best epoch index
        """
        if metric.endswith('loss'):
            return int(np.argmin(self.history[metric]))
        else:
            return int(np.argmax(self.history[metric]))
    
    def plot_history(self):
        """Plot training history."""
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Loss
            ax1.plot(self.history['train_loss'], label='Train Loss')
            ax1.plot(self.history['val_loss'], label='Val Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Accuracy
            ax2.plot(self.history['train_accuracy'], label='Train Accuracy')
            ax2.plot(self.history['val_accuracy'], label='Val Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Training and Validation Accuracy')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for plotting")


def main():
    """Test metrics."""
    # Generate sample data
    np.random.seed(42)
    
    y_true = np.random.randint(0, 10, 100)
    y_pred = y_true.copy()
    
    # Add some errors
    error_indices = np.random.choice(100, 20, replace=False)
    y_pred[error_indices] = np.random.randint(0, 10, 20)
    
    y_pred_probs = np.random.rand(100, 10)
    y_pred_probs = y_pred_probs / y_pred_probs.sum(axis=1, keepdims=True)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_pred_probs)
    print_metrics(metrics)


if __name__ == "__main__":
    main()
