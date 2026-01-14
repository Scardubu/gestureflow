"""
Generate synthetic gesture data for training.
"""
import numpy as np
from typing import List, Dict, Tuple
import json
from pathlib import Path
from tqdm import tqdm

from ..config import QWERTY_LAYOUT, DICTIONARY_DIR, PROCESSED_DATA_DIR, GENERATION_CONFIG


class GestureGenerator:
    """Generate synthetic gesture trajectories."""
    
    def __init__(
        self,
        layout: Dict[str, Tuple[float, float]] = None,
        noise_std: float = 0.02,
        min_points: int = 10,
        max_points: int = 50
    ):
        """
        Initialize generator.
        
        Args:
            layout: Keyboard layout dictionary
            noise_std: Standard deviation for noise
            min_points: Minimum points in trajectory
            max_points: Maximum points in trajectory
        """
        self.layout = layout or QWERTY_LAYOUT
        self.noise_std = noise_std
        self.min_points = min_points
        self.max_points = max_points
    
    def generate_trajectory(
        self,
        word: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a gesture trajectory for a word.
        
        Args:
            word: Target word
            
        Returns:
            (trajectory, timestamps) as numpy arrays
        """
        word = word.lower()
        
        # Get key positions
        key_positions = []
        for char in word:
            if char in self.layout:
                key_positions.append(self.layout[char])
        
        if len(key_positions) < 2:
            raise ValueError(f"Word '{word}' has insufficient keys in layout")
        
        # Generate interpolated trajectory
        num_points = np.random.randint(self.min_points, self.max_points + 1)
        trajectory = []
        
        # Interpolate between key positions
        for i in range(len(key_positions) - 1):
            start = np.array(key_positions[i])
            end = np.array(key_positions[i + 1])
            
            # Number of points for this segment
            segment_points = max(2, num_points // len(key_positions))
            
            # Linear interpolation with bezier-like curve
            t = np.linspace(0, 1, segment_points)
            
            # Add some curvature
            control_point = (start + end) / 2 + np.random.normal(0, 0.05, 2)
            
            for ti in t[:-1] if i < len(key_positions) - 2 else t:
                # Quadratic bezier curve
                p = (1 - ti)**2 * start + 2 * (1 - ti) * ti * control_point + ti**2 * end
                
                # Add noise
                noise = np.random.normal(0, self.noise_std, 2)
                p = p + noise
                
                # Clip to valid range
                p = np.clip(p, 0, 1)
                trajectory.append(p)
        
        trajectory = np.array(trajectory)
        
        # Generate timestamps (normalized)
        timestamps = np.linspace(0, 1, len(trajectory))
        
        # Add some temporal variation
        timestamps += np.random.normal(0, 0.01, len(timestamps))
        timestamps = np.clip(timestamps, 0, 1)
        timestamps = np.sort(timestamps)
        
        return trajectory, timestamps
    
    def generate_dataset(
        self,
        words: List[str],
        samples_per_word: int = 100,
        output_file: Path = None
    ) -> List[Dict]:
        """
        Generate a dataset of gesture trajectories.
        
        Args:
            words: List of words to generate
            samples_per_word: Number of samples per word
            output_file: Optional output file path
            
        Returns:
            List of gesture samples
        """
        dataset = []
        
        print(f"Generating dataset for {len(words)} words...")
        for word in tqdm(words):
            for _ in range(samples_per_word):
                try:
                    trajectory, timestamps = self.generate_trajectory(word)
                    
                    sample = {
                        'word': word,
                        'trajectory': trajectory.tolist(),
                        'timestamps': timestamps.tolist(),
                        'layout': 'qwerty'
                    }
                    
                    dataset.append(sample)
                except ValueError as e:
                    print(f"Skipping word '{word}': {e}")
                    break
        
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(dataset, f, indent=2)
            print(f"Dataset saved to {output_file}")
            print(f"Total samples: {len(dataset)}")
        
        return dataset


def load_dictionary(language: str = "en") -> List[str]:
    """
    Load dictionary for a language.
    
    Args:
        language: Language code (en, es, fr)
        
    Returns:
        List of words
    """
    lang_map = {
        "en": "en_US.txt",
        "es": "es_ES.txt",
        "fr": "fr_FR.txt"
    }
    
    dict_file = DICTIONARY_DIR / lang_map.get(language, "en_US.txt")
    
    if not dict_file.exists():
        raise FileNotFoundError(f"Dictionary file not found: {dict_file}")
    
    with open(dict_file, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f if line.strip()]
    
    return words


def main():
    """Generate gesture dataset."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate gesture dataset")
    parser.add_argument(
        "--language", "-l",
        type=str,
        default="en",
        help="Language code (en, es, fr)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Samples per word"
    )
    
    args = parser.parse_args()
    
    # Load dictionary
    words = load_dictionary(args.language)
    print(f"Loaded {len(words)} words for language '{args.language}'")
    
    # Generate dataset
    generator = GestureGenerator(
        noise_std=GENERATION_CONFIG["noise_std"],
        min_points=GENERATION_CONFIG["min_points"],
        max_points=GENERATION_CONFIG["max_points"]
    )
    
    output_file = PROCESSED_DATA_DIR / f"gestures_{args.language}.json"
    generator.generate_dataset(
        words=words,
        samples_per_word=args.samples,
        output_file=output_file
    )


if __name__ == "__main__":
    main()
