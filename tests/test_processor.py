import pytest
import numpy as np
from src.data.processor import GestureProcessor

def test_processor_init():
“”“Test processor initialization.”””
processor = GestureProcessor(sequence_length=50)
assert processor.sequence_length == 50

def test_normalize_trajectory():
“”“Test trajectory normalization.”””
processor = GestureProcessor()
trajectory = np.array([[0.5, 0.5], [1.5, -0.5]])
normalized = processor.normalize_trajectory(trajectory)

```
assert np.all(normalized >= 0)
assert np.all(normalized <= 1)
```

def test_normalize_timestamps():
“”“Test timestamp normalization.”””
processor = GestureProcessor()
timestamps = np.array([0, 100, 200, 300])
normalized = processor.normalize_timestamps(timestamps)

```
assert normalized[0] == 0
assert normalized[-1] == 1
```

def test_augment_trajectory():
“”“Test trajectory augmentation.”””
processor = GestureProcessor()
trajectory = np.random.rand(10, 3)
augmented = processor.augment_trajectory(trajectory)

```
assert augmented.shape == trajectory.shape
assert not np.array_equal(augmented, trajectory)
```

def test_pad_sequence():
“”“Test sequence padding.”””
processor = GestureProcessor(sequence_length=50)
sequence = np.random.rand(30, 3)

```
padded, actual_length = processor.pad_sequence(sequence)
assert padded.shape == (50, 3)
assert actual_length == 30
```

def test_pad_sequence_truncate():
“”“Test sequence truncation.”””
processor = GestureProcessor(sequence_length=50)
sequence = np.random.rand(60, 3)

```
padded, actual_length = processor.pad_sequence(sequence)
assert padded.shape == (50, 3)
assert actual_length == 50
```
