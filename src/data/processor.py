"""
Data preprocessing and augmentation for GestureFlow.
"""
import numpy as np
from typing import List, Dict, Tuple
import json
from pathlib import Path
from tqdm import tqdm

from ..config import PROCESSED_DATA_DIR, MODEL_CONFIG


class GestureProcessor:
    """Process and augment gesture data for training."""

    def __init__(self, sequence_length: int = 50):
        """
        Initialize processor.
        
        Args:
            sequence_length: Maximum sequence length for padding
        """
        self.sequence_length = sequence_length

    def normalize_trajectory(
        self, 
        trajectory: np.ndarray
    ) -> np.ndarray:
        """
        Normalize trajectory coordinates to [0, 1] range.
        
        Args:
            trajectory: Array of shape (N, 2) with x, y coordinates
            
        Returns:
            Normalized trajectory
        """
        # Already normalized in generator, but ensure range
        return np.clip(trajectory, 0, 1)

    def normalize_timestamps(
        self,
        timestamps: np.ndarray
    ) -> np.ndarray:
        """
        Normalize timestamps to [0, 1] range.
        
        Args:
            timestamps: Array of timestamps
            
        Returns:
            Normalized timestamps
        """
        if len(timestamps) <= 1:
            return np.zeros_like(timestamps)
        
        min_t = timestamps.min()
        max_t = timestamps.max()
        
        if max_t - min_t < 1e-8:
            return np.zeros_like(timestamps)
        
        return (timestamps - min_t) / (max_t - min_t)

    def augment_trajectory(
        self,
        trajectory: np.ndarray,
        noise_std: float = 0.02,
        time_warp: bool = True
    ) -> np.ndarray:
        """
        Augment trajectory with noise and time warping.
        
        Args:
            trajectory: Input trajectory (N, 3)
            noise_std: Standard deviation for Gaussian noise
            time_warp: Whether to apply time warping
            
        Returns:
            Augmented trajectory
        """
        augmented = trajectory.copy()
        
        # Add spatial noise to x, y
        spatial_noise = np.random.normal(0, noise_std, (len(trajectory), 2))
        augmented[:, :2] += spatial_noise
        augmented[:, :2] = np.clip(augmented[:, :2], 0, 1)
        
        # Time warping (stretch or compress)
        if time_warp:
            warp_factor = np.random.uniform(0.8, 1.2)
            augmented[:, 2] *= warp_factor
            augmented[:, 2] = self.normalize_timestamps(augmented[:, 2])
        
        return augmented

    def pad_sequence(
        self,
        sequence: np.ndarray,
        max_length: int = None
    ) -> Tuple[np.ndarray, int]:
        """
        Pad sequence to maximum length.
        
        Args:
            sequence: Input sequence (N, 3)
            max_length: Maximum length (uses self.sequence_length if None)
            
        Returns:
            (padded_sequence, actual_length)
        """
        if max_length is None:
            max_length = self.sequence_length
        
        actual_length = len(sequence)
        
        if len(sequence) < max_length:
            padding = np.zeros((max_length - len(sequence), sequence.shape[1]))
            padded = np.vstack([sequence, padding])
        else:
            padded = sequence[:max_length]
            actual_length = max_length
        
        return padded, actual_length

    def extract_features(
        self,
        trajectory: np.ndarray,
        timestamps: np.ndarray
    ) -> np.ndarray:
        """
        Extract additional features from gesture.
        
        Args:
            trajectory: Array of (x, y) coordinates
            timestamps: Array of timestamps
            
        Returns:
            Feature array (N, 3+) with x, y, t, and optional features
        """
        # Basic features: x, y, normalized_t
        normalized_t = self.normalize_timestamps(timestamps).reshape(-1, 1)
        basic_features = np.concatenate([trajectory, normalized_t], axis=1)
        
        # Optional: Add velocity features
        # velocity = np.diff(trajectory, axis=0, prepend=trajectory[0:1])
        # speed = np.linalg.norm(velocity, axis=1, keepdims=True)
        # features = np.concatenate([basic_features, speed], axis=1)
        
        return basic_features

    def process_dataset(
        self,
        input_file: Path,
        output_file: Path = None,
        augmentation: bool = True,
        augmentation_factor: int = 3
    ) -> List[Dict]:
        """
        Process entire dataset with optional augmentation.
        
        Args:
            input_file: Path to raw JSON dataset
            output_file: Optional path to save processed dataset
            augmentation: Whether to augment data
            augmentation_factor: Number of augmented versions per sample
            
        Returns:
            Processed dataset
        """
        print(f"Loading dataset from {input_file}...")
        with open(input_file, 'r') as f:
            raw_data = json.load(f)
        
        processed_data = []
        
        desc = "Processing with augmentation" if augmentation else "Processing"
        for sample in tqdm(raw_data, desc=desc):
            # Original sample
            trajectory = np.array(sample['trajectory'])
            timestamps = np.array(sample['timestamps'])
            
            # Extract features
            features = self.extract_features(trajectory, timestamps)
            
            # Pad sequence
            padded_features, actual_length = self.pad_sequence(features)
            
            processed_sample = {
                'word': sample['word'],
                'features': padded_features.tolist(),
                'actual_length': actual_length,
                'layout': sample.get('layout', 'qwerty')
            }
            
            processed_data.append(processed_sample)
            
            # Augmentation
            if augmentation:
                for _ in range(augmentation_factor - 1):
                    aug_features = self.augment_trajectory(features)
                    padded_aug, aug_length = self.pad_sequence(aug_features)
                    
                    aug_sample = {
                        'word': sample['word'],
                        'features': padded_aug.tolist(),
                        'actual_length': aug_length,
                        'layout': sample.get('layout', 'qwerty'),
                        'augmented': True
                    }
                    processed_data.append(aug_sample)
        
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(processed_data, f)
            print(f"Processed dataset saved to {output_file}")
            print(f"Original samples: {len(raw_data)}")
            print(f"Total samples (with augmentation): {len(processed_data)}")
        
        return processed_data


def main():
    """Process datasets."""
    import argparse

    parser = argparse.ArgumentParser(description="Process GestureFlow datasets")
    parser.add_argument(
        "--language", "-l",
        type=str,
        default="en",
        help="Language code"
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply data augmentation"
    )
    parser.add_argument(
        "--aug-factor",
        type=int,
        default=3,
        help="Augmentation factor"
    )

    args = parser.parse_args()

    processor = GestureProcessor(
        sequence_length=MODEL_CONFIG["sequence_length"]
    )

    input_file = PROCESSED_DATA_DIR / f"gestures_{args.language}.json"
    output_file = PROCESSED_DATA_DIR / f"processed_{args.language}.json"

    processor.process_dataset(
        input_file=input_file,
        output_file=output_file,
        augmentation=args.augment,
        augmentation_factor=args.aug_factor
    )


if __name__ == "__main__":
    main()
