"""Predictor class for gesture-based text prediction."""

import numpy as np
import tensorflow as tf
from typing import List, Tuple


class Predictor:
    """Handles model inference for gesture predictions."""
    
    def __init__(self, model_path: str):
        """Initialize predictor with trained model.
        
        Args:
            model_path: Path to the trained model file
        """
        self.model = tf.keras.models.load_model(model_path)
        
    def predict(self, gesture_sequence: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Predict words from gesture sequence.
        
        Args:
            gesture_sequence: Input gesture coordinates
            top_k: Number of top predictions to return
            
        Returns:
            List of (word, confidence) tuples
        """
        # Normalize input
        normalized = self._normalize_gesture(gesture_sequence)
        
        # Get model predictions
        predictions = self.model.predict(np.expand_dims(normalized, axis=0))
        
        # Get top k predictions
        top_indices = np.argsort(predictions[0])[-top_k:][::-1]
        results = [(self._decode_prediction(idx), predictions[0][idx]) 
                   for idx in top_indices]
        
        return results
    
    def _normalize_gesture(self, gesture: np.ndarray) -> np.ndarray:
        """Normalize gesture coordinates."""
        # Normalize to [0, 1] range
        min_vals = gesture.min(axis=0)
        max_vals = gesture.max(axis=0)
        
        # Check for zero range (all points identical on an axis)
        range_vals = max_vals - min_vals
        epsilon = 1e-6
        
        # For axes with zero range, keep original values (they're already the same)
        # For axes with range, normalize
        normalized = np.where(
            range_vals < epsilon,
            0.5,  # Center value for constant dimensions
            (gesture - min_vals) / (range_vals + epsilon)
        )
        
        return normalized
    
    def _decode_prediction(self, index: int) -> str:
        """Decode prediction index to word."""
        # This would use the vocabulary mapping
        return f"word_{index}"
