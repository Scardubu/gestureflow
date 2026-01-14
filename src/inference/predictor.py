# ================================

# src/inference/predictor.py

# ================================

“”“Production inference class for GestureFlow.”””
import torch
from typing import List, Dict
from pathlib import Path
import json
import numpy as np

from ..models.lstm_model import create_model
from ..config import CHECKPOINTS_DIR, MODEL_CONFIG

class SwipePredictor:
“”“Production-ready predictor for swipe gestures.”””

```
def __init__(self, language: str = "en", device: str = "cpu"):
    """
    Initialize predictor.
    
    Args:
        language: Language code
        device: Device to run inference on
    """
    self.language = language
    self.device = device
    self.model = None
    self.vocab = None
    self._load_model()

def _load_model(self):
    """Load trained model and vocabulary."""
    model_dir = CHECKPOINTS_DIR / self.language
    model_path = model_dir / "best_model.pt"
    vocab_path = model_dir / "vocabulary.json"
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found for language '{self.language}'. "
            f"Train it first: python scripts/train_model.py --language {self.language}"
        )
    
    # Load vocabulary
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
    
    self.vocab = {
        'word_to_idx': vocab_data['word_to_idx'],
        'idx_to_word': {int(k): v for k, v in vocab_data['idx_to_word'].items()},
        'vocab_size': vocab_data['vocab_size']
    }
    
    # Load model
    self.model = create_model(vocab_size=self.vocab['vocab_size'])
    checkpoint = torch.load(model_path, map_location=self.device)
    self.model.load_state_dict(checkpoint['model_state_dict'])
    self.model.eval()
    self.model.to(self.device)

def predict(self, trajectory: List[Dict], top_k: int = 5) -> List[Dict]:
    """
    Predict words from gesture trajectory.
    
    Args:
        trajectory: List of {x, y, timestamp} dicts
        top_k: Number of predictions to return
        
    Returns:
        List of prediction dictionaries
    """
    # Preprocess trajectory
    input_tensor = self._preprocess(trajectory)
    
    # Predict
    with torch.no_grad():
        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k=top_k)
    
    # Format results
    predictions = []
    for idx, prob in zip(top_indices[0].tolist(), top_probs[0].tolist()):
        word = self.vocab['idx_to_word'].get(idx, "<UNK>")
        if word != "<UNK>":
            predictions.append({
                'word': word,
                'confidence': float(prob)
            })
    
    return predictions

def _preprocess(self, trajectory: List[Dict]) -> torch.Tensor:
    """Preprocess trajectory for model input."""
    # Extract features
    features = []
    for point in trajectory:
        features.append([point['x'], point['y'], point['timestamp']])
    
    features = np.array(features)
    
    # Normalize timestamps
    if len(features) > 1:
        ts = features[:, 2]
        features[:, 2] = (ts - ts.min()) / (ts.max() - ts.min() + 1e-8)
    
    # Pad to max length
    max_len = MODEL_CONFIG["sequence_length"]
    if len(features) < max_len:
        padding = np.zeros((max_len - len(features), 3))
        features = np.vstack([features, padding])
    else:
        features = features[:max_len]
    
    return torch.FloatTensor(features).unsqueeze(0).to(self.device)
```
