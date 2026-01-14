"""
Gesture prediction for inference.
"""
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import List, Dict, Tuple
import json

from ..data.processor import GestureProcessor


class GesturePredictor:
    """Predict words from gesture trajectories."""
    
    def __init__(
        self,
        model_path: Path,
        vocab_path: Path = None
    ):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model
            vocab_path: Path to vocabulary JSON
        """
        self.model = tf.keras.models.load_model(model_path)
        self.processor = GestureProcessor()
        
        # Load vocabulary
        if vocab_path and vocab_path.exists():
            with open(vocab_path, 'r') as f:
                vocab_data = json.load(f)
            self.idx_to_word = {int(k): v for k, v in vocab_data['idx_to_word'].items()}
            self.word_to_idx = vocab_data['word_to_idx']
        else:
            self.idx_to_word = {}
            self.word_to_idx = {}
    
    def predict(
        self,
        trajectory: np.ndarray,
        timestamps: np.ndarray,
        top_k: int = 5
    ) -> List[Dict[str, any]]:
        """
        Predict word from gesture trajectory.
        
        Args:
            trajectory: Trajectory array (N, 2)
            timestamps: Timestamps array (N,)
            top_k: Number of top predictions to return
            
        Returns:
            List of predictions with confidence scores
        """
        # Process trajectory
        features = self.processor.extract_features(trajectory, timestamps)
        padded_features, _ = self.processor.pad_sequence(features)
        
        # Add batch dimension
        input_data = np.expand_dims(padded_features, axis=0)
        
        # Predict
        predictions = self.model.predict(input_data, verbose=0)[0]
        
        # Get top-k predictions
        top_k_indices = np.argsort(predictions)[-top_k:][::-1]
        
        results = []
        for idx in top_k_indices:
            word = self.idx_to_word.get(idx, f"unknown_{idx}")
            confidence = float(predictions[idx])
            
            results.append({
                'word': word,
                'confidence': confidence,
                'index': int(idx)
            })
        
        return results
    
    def predict_batch(
        self,
        trajectories: np.ndarray,
        top_k: int = 5
    ) -> List[List[Dict[str, any]]]:
        """
        Predict words from batch of trajectories.
        
        Args:
            trajectories: Batch of trajectories (batch_size, sequence_length, features)
            top_k: Number of top predictions per sample
            
        Returns:
            List of prediction lists
        """
        # Predict
        predictions = self.model.predict(trajectories, verbose=0)
        
        results = []
        for pred in predictions:
            top_k_indices = np.argsort(pred)[-top_k:][::-1]
            
            sample_results = []
            for idx in top_k_indices:
                word = self.idx_to_word.get(idx, f"unknown_{idx}")
                confidence = float(pred[idx])
                
                sample_results.append({
                    'word': word,
                    'confidence': confidence,
                    'index': int(idx)
                })
            
            results.append(sample_results)
        
        return results


def main():
    """Test predictor."""
    import argparse
    from ..config import CHECKPOINT_DIR, PROCESSED_DATA_DIR
    
    parser = argparse.ArgumentParser(description="Test predictor")
    parser.add_argument(
        "--model",
        type=str,
        default="best_model.h5",
        help="Model file name"
    )
    parser.add_argument(
        "--vocab",
        type=str,
        default="vocab.json",
        help="Vocabulary file name"
    )
    
    args = parser.parse_args()
    
    model_path = CHECKPOINT_DIR / args.model
    vocab_path = PROCESSED_DATA_DIR / args.vocab
    
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return
    
    if not vocab_path.exists():
        print(f"Vocabulary not found: {vocab_path}")
        return
    
    # Create predictor
    predictor = GesturePredictor(
        model_path=model_path,
        vocab_path=vocab_path
    )
    
    print(f"Predictor loaded with vocabulary size: {len(predictor.idx_to_word)}")
    
    # Test with random trajectory
    trajectory = np.random.rand(30, 2)
    timestamps = np.linspace(0, 1, 30)
    
    predictions = predictor.predict(trajectory, timestamps, top_k=5)
    
    print("\nTest predictions:")
    for i, pred in enumerate(predictions, 1):
        print(f"{i}. {pred['word']:15s} (confidence: {pred['confidence']:.4f})")


if __name__ == "__main__":
    main()
