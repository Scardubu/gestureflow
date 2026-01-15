"""Metrics for GestureFlow training and evaluation."""

import torch
import numpy as np

def calculate_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """Calculate top-1 accuracy."""
    pred_classes = torch.argmax(preds, dim=1)
    return (pred_classes == labels).float().mean().item() * 100

def calculate_top_k_accuracy(
    preds: torch.Tensor, 
    labels: torch.Tensor, 
    k: int = 5
) -> float:
    """Calculate top-k accuracy."""
    top_k_preds = torch.topk(preds, k, dim=1)[1]
    labels_expanded = labels.unsqueeze(1).expand(-1, k)
    correct = (top_k_preds == labels_expanded).any(dim=1).float()
    return correct.mean().item() * 100