“””
GestureFlow Test Suite
“””

# tests/test_generator.py

import pytest
import numpy as np
from src.data.generator import SwipeGestureGenerator

def test_gesture_generator_init():
“”“Test generator initialization.”””
generator = SwipeGestureGenerator(layout=“qwerty”)
assert generator.layout is not None
assert generator.noise_std > 0

def test_get_char_position():
“”“Test character position retrieval.”””
generator = SwipeGestureGenerator()
pos = generator.get_char_position(‘a’)
assert isinstance(pos, tuple)
assert len(pos) == 2
assert 0 <= pos[0] <= 1
assert 0 <= pos[1] <= 1

def test_generate_bezier_curve():
“”“Test Bezier curve generation.”””
generator = SwipeGestureGenerator()
start = (0.0, 0.0)
end = (1.0, 1.0)
curve = generator.generate_bezier_curve(start, end, num_points=10)

```
assert curve.shape == (10, 2)
assert np.allclose(curve[0], start, atol=0.1)
assert np.allclose(curve[-1], end, atol=0.1)
```

def test_add_human_noise():
“”“Test noise addition.”””
generator = SwipeGestureGenerator()
trajectory = np.array([[0.5, 0.5], [0.6, 0.6]])
noisy = generator.add_human_noise(trajectory)

```
assert noisy.shape == trajectory.shape
assert not np.array_equal(noisy, trajectory)
assert np.all(noisy >= 0) and np.all(noisy <= 1)
```

def test_generate_timestamps():
“”“Test timestamp generation.”””
generator = SwipeGestureGenerator()
timestamps = generator.generate_timestamps(10)

```
assert len(timestamps) == 10
assert timestamps[0] == 0 or timestamps[0] < timestamps[-1]
assert all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))
```

def test_word_to_gesture():
“”“Test complete word to gesture conversion.”””
generator = SwipeGestureGenerator()
gesture = generator.word_to_gesture(“hello”)

```
assert gesture['word'] == "hello"
assert len(gesture['trajectory']) > 0
assert len(gesture['timestamps']) == len(gesture['trajectory'])
assert gesture['num_points'] == len(gesture['trajectory'])
assert gesture['layout'] == "qwerty"
```

def test_word_to_gesture_single_char():
“”“Test single character word.”””
generator = SwipeGestureGenerator()
gesture = generator.word_to_gesture(“a”)

```
assert gesture['word'] == "a"
assert len(gesture['trajectory']) > 0
```

def test_generate_dataset():
“”“Test dataset generation.”””
generator = SwipeGestureGenerator()
words = [“hello”, “world”, “test”]
dataset = generator.generate_dataset(words, samples_per_word=2, output_file=None)

```
assert len(dataset) == 6  # 3 words * 2 samples
assert all('word' in sample for sample in dataset)
assert all('trajectory' in sample for sample in dataset)
```



