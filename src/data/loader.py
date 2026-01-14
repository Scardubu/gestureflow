"""
Data loader for training and inference.
"""
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import tensorflow as tf

from ..config import PROCESSED_DATA_DIR, TRAINING_CONFIG


class GestureDataLoader:
    """Load and prepare gesture data for training."""
    
    def __init__(
        self,
        data_path: Path,
        batch_size: int = 32,
        validation_split: float = 0.2,
        shuffle: bool = True
    ):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to processed data JSON
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
            shuffle: Whether to shuffle data
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.shuffle = shuffle
        
        # Load data
        self.data = self._load_data()
        
        # Build vocabulary
        self.word_to_idx, self.idx_to_word = self._build_vocabulary()
        self.vocab_size = len(self.word_to_idx)
    
    def _load_data(self) -> List[Dict]:
        """Load data from JSON file."""
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        return data
    
    def _build_vocabulary(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Build vocabulary from data."""
        words = sorted(set(sample['word'] for sample in self.data))
        word_to_idx = {word: idx for idx, word in enumerate(words)}
        idx_to_word = {idx: word for word, idx in word_to_idx.items()}
        return word_to_idx, idx_to_word
    
    def _prepare_sample(self, sample: Dict) -> Tuple[np.ndarray, int]:
        """
        Prepare a single sample.
        
        Args:
            sample: Data sample
            
        Returns:
            (features, label) tuple
        """
        features = np.array(sample['features'], dtype=np.float32)
        label = self.word_to_idx[sample['word']]
        return features, label
    
    def get_datasets(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Get training and validation datasets.
        
        Returns:
            (train_dataset, val_dataset) tuple
        """
        # Prepare all samples
        X = []
        y = []
        
        for sample in self.data:
            features, label = self._prepare_sample(sample)
            X.append(features)
            y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        
        # Split into train and validation
        if self.shuffle:
            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]
        
        split_idx = int(len(X) * (1 - self.validation_split))
        
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Create TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        return train_dataset, val_dataset
    
    def save_vocabulary(self, output_path: Path):
        """
        Save vocabulary to JSON file.
        
        Args:
            output_path: Output file path
        """
        vocab_data = {
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word
        }
        
        with open(output_path, 'w') as f:
            json.dump(vocab_data, f, indent=2)
        
        print(f"Vocabulary saved to {output_path}")


def main():
    """Test data loader."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test data loader")
    parser.add_argument(
        "--data-file",
        type=str,
        default="processed_en.json",
        help="Data file name"
    )
    
    args = parser.parse_args()
    
    data_path = PROCESSED_DATA_DIR / args.data_file
    
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return
    
    loader = GestureDataLoader(
        data_path=data_path,
        batch_size=TRAINING_CONFIG["batch_size"],
        validation_split=TRAINING_CONFIG["validation_split"]
    )
    
    print(f"Vocabulary size: {loader.vocab_size}")
    print(f"Total samples: {len(loader.data)}")
    
    train_ds, val_ds = loader.get_datasets()
    
    print(f"\nDataset shapes:")
    for x, y in train_ds.take(1):
        print(f"  Features: {x.shape}")
        print(f"  Labels: {y.shape}")
    
    # Save vocabulary
    vocab_path = PROCESSED_DATA_DIR / "vocab.json"
    loader.save_vocabulary(vocab_path)


if __name__ == "__main__":
    main()
