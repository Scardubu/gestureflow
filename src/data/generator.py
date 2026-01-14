“””
Synthetic swipe gesture data generator.
Generates realistic swipe trajectories for training the LSTM model.
“””
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path
import json
from tqdm import tqdm
import argparse

from ..config import (
KEYBOARD_LAYOUTS, DATA_CONFIG, DICTIONARIES_DIR,
PROCESSED_DATA_DIR, SUPPORTED_LANGUAGES
)

class SwipeGestureGenerator:
“”“Generate synthetic swipe gesture data from text.”””

```
def __init__(self, layout: str = "qwerty", noise_std: float = 5.0):
    """
    Initialize gesture generator.
    
    Args:
        layout: Keyboard layout name
        noise_std: Standard deviation of Gaussian noise (in pixels)
    """
    self.layout = KEYBOARD_LAYOUTS[layout]
    self.noise_std = noise_std / 100.0  # Normalize to 0-1 space
    
def get_char_position(self, char: str) -> Tuple[float, float]:
    """Get normalized keyboard position for a character."""
    char = char.lower()
    if char not in self.layout:
        # Return center position for unknown chars
        return (0.5, 0.5)
    return self.layout[char]

def generate_bezier_curve(
    self, 
    start: Tuple[float, float], 
    end: Tuple[float, float],
    num_points: int = 10
) -> np.ndarray:
    """
    Generate smooth Bezier curve between two points.
    
    Args:
        start: Starting (x, y) coordinates
        end: Ending (x, y) coordinates
        num_points: Number of interpolation points
        
    Returns:
        Array of shape (num_points, 2) with interpolated coordinates
    """
    # Create control point slightly off the direct line
    control_x = (start[0] + end[0]) / 2 + np.random.normal(0, 0.05)
    control_y = (start[1] + end[1]) / 2 + np.random.normal(0, 0.05)
    control = (control_x, control_y)
    
    # Quadratic Bezier curve: B(t) = (1-t)²P0 + 2(1-t)tP1 + t²P2
    t = np.linspace(0, 1, num_points)
    
    # Vectorized Bezier calculation
    x = (1 - t)**2 * start[0] + 2 * (1 - t) * t * control[0] + t**2 * end[0]
    y = (1 - t)**2 * start[1] + 2 * (1 - t) * t * control[1] + t**2 * end[1]
    
    return np.column_stack([x, y])

def add_human_noise(self, trajectory: np.ndarray) -> np.ndarray:
    """
    Add human-like imprecision to trajectory.
    
    Args:
        trajectory: Array of (x, y) coordinates
        
    Returns:
        Noisy trajectory
    """
    noise = np.random.normal(0, self.noise_std, trajectory.shape)
    noisy_trajectory = trajectory + noise
    
    # Clip to valid range [0, 1]
    return np.clip(noisy_trajectory, 0, 1)

def generate_timestamps(
    self, 
    num_points: int,
    speed_range: Tuple[int, int] = (100, 300)
) -> np.ndarray:
    """
    Generate realistic timestamps for gesture points.
    
    Args:
        num_points: Number of points in trajectory
        speed_range: (min_ms, max_ms) per character
        
    Returns:
        Array of timestamps in milliseconds
    """
    # Variable speed throughout gesture
    speed = np.random.uniform(speed_range[0], speed_range[1])
    total_time = speed * num_points
    
    # Non-uniform time intervals (faster in middle, slower at start/end)
    t = np.linspace(0, 1, num_points)
    ease_curve = t * (2 - t)  # Ease-out curve
    timestamps = ease_curve * total_time
    
    return timestamps

def word_to_gesture(
    self, 
    word: str,
    points_per_char: int = 8
) -> Dict:
    """
    Convert a word into a synthetic swipe gesture.
    
    Args:
        word: Input word
        points_per_char: Number of interpolation points per character
        
    Returns:
        Dictionary with gesture data
    """
    word = word.lower()
    
    # Get character positions
    positions = [self.get_char_position(char) for char in word]
    
    # Generate smooth trajectory through all characters
    trajectory_segments = []
    
    for i in range(len(positions) - 1):
        segment = self.generate_bezier_curve(
            positions[i], 
            positions[i + 1],
            num_points=points_per_char
        )
        trajectory_segments.append(segment)
    
    # Combine all segments
    if trajectory_segments:
        trajectory = np.vstack(trajectory_segments)
    else:
        # Single character word
        pos = positions[0]
        trajectory = np.array([pos] * points_per_char)
    
    # Add human-like noise
    trajectory = self.add_human_noise(trajectory)
    
    # Generate timestamps
    timestamps = self.generate_timestamps(
        len(trajectory),
        speed_range=DATA_CONFIG["swipe_speed_range"]
    )
    
    return {
        "word": word,
        "trajectory": trajectory.tolist(),
        "timestamps": timestamps.tolist(),
        "num_points": len(trajectory),
        "layout": "qwerty"
    }

def generate_dataset(
    self,
    words: List[str],
    samples_per_word: int = 10,
    output_file: Path = None
) -> List[Dict]:
    """
    Generate complete dataset from word list.
    
    Args:
        words: List of words to generate gestures for
        samples_per_word: Number of synthetic samples per word
        output_file: Optional path to save dataset
        
    Returns:
        List of gesture samples
    """
    dataset = []
    
    print(f"Generating {len(words) * samples_per_word} gesture samples...")
    
    for word in tqdm(words):
        for _ in range(samples_per_word):
            gesture = self.word_to_gesture(word)
            dataset.append(gesture)
    
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"Dataset saved to {output_file}")
    
    return dataset
```

def load_dictionary(language: str = “en”) -> List[str]:
“””
Load dictionary file for a language.

```
Args:
    language: Language code (en, es, fr)
    
Returns:
    List of words
"""
lang_config = SUPPORTED_LANGUAGES[language]
dict_file = DICTIONARIES_DIR / lang_config["dictionary_file"]

if not dict_file.exists():
    raise FileNotFoundError(
        f"Dictionary not found: {dict_file}\n"
        f"Run: python scripts/download_dictionaries.py"
    )

with open(dict_file, 'r', encoding='utf-8') as f:
    words = [line.strip().lower() for line in f if line.strip()]

# Filter by length
words = [
    w for w in words 
    if DATA_CONFIG["min_word_length"] <= len(w) <= DATA_CONFIG["max_word_length"]
]

# Use only alphabetic words
words = [w for w in words if w.isalpha()]

return words
```

def main():
“”“Generate synthetic dataset for training.”””
parser = argparse.ArgumentParser(
description=“Generate synthetic swipe gesture data”
)
parser.add_argument(
“–language”, “-l”,
type=str,
default=“en”,
choices=list(SUPPORTED_LANGUAGES.keys()),
help=“Language code”
)
parser.add_argument(
“–samples”, “-s”,
type=int,
default=DATA_CONFIG[“samples_per_word”],
help=“Samples per word”
)
parser.add_argument(
“–max-words”, “-m”,
type=int,
default=5000,
help=“Maximum number of words to use”
)

```
args = parser.parse_args()

# Load dictionary
print(f"Loading {SUPPORTED_LANGUAGES[args.language]['name']} dictionary...")
words = load_dictionary(args.language)

# Limit dataset size for faster training
if len(words) > args.max_words:
    # Randomly sample for diversity
    np.random.seed(42)
    words = np.random.choice(words, args.max_words, replace=False).tolist()

print(f"Using {len(words)} words")

# Generate dataset
generator = SwipeGestureGenerator(
    layout="qwerty",
    noise_std=DATA_CONFIG["noise_std"]
)

output_file = PROCESSED_DATA_DIR / f"gestures_{args.language}.json"
dataset = generator.generate_dataset(
    words=words,
    samples_per_word=args.samples,
    output_file=output_file
)

print(f"\nDataset Statistics:")
print(f"  Total samples: {len(dataset)}")
print(f"  Unique words: {len(set(d['word'] for d in dataset))}")
print(f"  Avg points per gesture: {np.mean([d['num_points'] for d in dataset]):.1f}")
```

if **name** == “**main**”:
main()
