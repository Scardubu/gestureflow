“””
Data loaders and dataset utilities for GestureFlow.
“””
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple
import json
import numpy as np
from pathlib import Path

from ..config import PROCESSED_DATA_DIR, MODEL_CONFIG, TRAINING_CONFIG

class GestureDataset(Dataset):
“”“PyTorch Dataset for gesture sequences.”””

```
def __init__(
    self,
    data: List[Dict],
    word_to_idx: Dict[str, int],
    max_length: int = 50,
    return_metadata: bool = False
):
    """
    Initialize dataset.
    
    Args:
        data: List of gesture dictionaries
        word_to_idx: Word to index mapping
        max_length: Maximum sequence length
        return_metadata: Whether to return additional metadata
    """
    self.data = data
    self.word_to_idx = word_to_idx
    self.max_length = max_length
    self.return_metadata = return_metadata

def __len__(self) -> int:
    return len(self.data)

def __getitem__(self, idx: int) -> Tuple:
    """
    Get a single sample.
    
    Returns:
        (features, label, actual_length) or with metadata
    """
    sample = self.data[idx]
    
    # Get features
    if 'features' in sample:
        # Already processed
        features = np.array(sample['features'])
    else:
        # Raw format - process on the fly
        trajectory = np.array(sample['trajectory'])
        timestamps = np.array(sample['timestamps']).reshape(-1, 1)
        
        # Normalize timestamps
        if len(timestamps) > 1:
            timestamps = (timestamps - timestamps.min()) / \
                        (timestamps.max() - timestamps.min() + 1e-8)
        
        features = np.concatenate([trajectory, timestamps], axis=1)
        
        # Pad to max_length
        if len(features) < self.max_length:
            padding = np.zeros((self.max_length - len(features), 3))
            features = np.vstack([features, padding])
        else:
            features = features[:self.max_length]
    
    # Get label
    word = sample['word']
    label = self.word_to_idx.get(word, 0)  # 0 is <UNK>
    
    # Get actual length
    actual_length = sample.get('actual_length', len(features))
    
    # Convert to tensors
    features_tensor = torch.FloatTensor(features)
    label_tensor = torch.LongTensor([label])[0]
    length_tensor = torch.LongTensor([actual_length])[0]
    
    if self.return_metadata:
        metadata = {
            'word': word,
            'layout': sample.get('layout', 'qwerty'),
            'augmented': sample.get('augmented', False)
        }
        return features_tensor, label_tensor, length_tensor, metadata
    
    return features_tensor, label_tensor, length_tensor

@classmethod
def from_file(
    cls,
    file_path: Path,
    word_to_idx: Dict[str, int],
    **kwargs
):
    """
    Load dataset from file.
    
    Args:
        file_path: Path to JSON dataset file
        word_to_idx: Word to index mapping
        **kwargs: Additional arguments for __init__
        
    Returns:
        GestureDataset instance
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    return cls(data=data, word_to_idx=word_to_idx, **kwargs)
```

def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, …]:
“””
Custom collate function for DataLoader.
Handles variable-length sequences.

```
Args:
    batch: List of (features, label, length) tuples
    
Returns:
    (batched_features, batched_labels, batched_lengths)
"""
features, labels, lengths = zip(*batch)

# Stack into batches
features_batch = torch.stack(features)
labels_batch = torch.stack(labels)
lengths_batch = torch.stack(lengths)

# Sort by length (descending) for pack_padded_sequence
sorted_lengths, sorted_indices = torch.sort(lengths_batch, descending=True)
sorted_features = features_batch[sorted_indices]
sorted_labels = labels_batch[sorted_indices]

return sorted_features, sorted_labels, sorted_lengths
```

def create_dataloaders(
train_data: List[Dict],
val_data: List[Dict],
test_data: List[Dict],
word_to_idx: Dict[str, int],
batch_size: int = 32,
num_workers: int = 2,
pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
“””
Create train, validation, and test dataloaders.

```
Args:
    train_data: Training samples
    val_data: Validation samples
    test_data: Test samples
    word_to_idx: Word to index mapping
    batch_size: Batch size
    num_workers: Number of data loading workers
    pin_memory: Whether to pin memory for GPU transfer
    
Returns:
    (train_loader, val_loader, test_loader)
"""
# Create datasets
train_dataset = GestureDataset(
    data=train_data,
    word_to_idx=word_to_idx,
    max_length=MODEL_CONFIG["sequence_length"]
)

val_dataset = GestureDataset(
    data=val_data,
    word_to_idx=word_to_idx,
    max_length=MODEL_CONFIG["sequence_length"]
)

test_dataset = GestureDataset(
    data=test_data,
    word_to_idx=word_to_idx,
    max_length=MODEL_CONFIG["sequence_length"]
)

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
    collate_fn=collate_fn,
    drop_last=True  # Drop incomplete batches for training
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    collate_fn=collate_fn
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    collate_fn=collate_fn
)

return train_loader, val_loader, test_loader
```

def build_vocabulary(data: List[Dict]) -> Tuple[Dict, Dict]:
“””
Build vocabulary from dataset.

```
Args:
    data: List of gesture samples
    
Returns:
    (word_to_idx, idx_to_word) dictionaries
"""
# Get unique words
unique_words = sorted(set(sample['word'] for sample in data))

# Create mappings (0 is reserved for <UNK>)
word_to_idx = {'<UNK>': 0}
word_to_idx.update({word: idx + 1 for idx, word in enumerate(unique_words)})

idx_to_word = {idx: word for word, idx in word_to_idx.items()}

return word_to_idx, idx_to_word
```

def split_dataset(
data: List[Dict],
val_split: float = 0.2,
test_split: float = 0.1,
seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
“””
Split dataset into train, validation, and test sets.

```
Args:
    data: Complete dataset
    val_split: Validation set proportion
    test_split: Test set proportion
    seed: Random seed
    
Returns:
    (train_data, val_data, test_data)
"""
np.random.seed(seed)

# Shuffle data
indices = np.random.permutation(len(data))

# Calculate split sizes
test_size = int(len(data) * test_split)
val_size = int(len(data) * val_split)
train_size = len(data) - val_size - test_size

# Split indices
train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

# Create splits
train_data = [data[i] for i in train_indices]
val_data = [data[i] for i in val_indices]
test_data = [data[i] for i in test_indices]

return train_data, val_data, test_data
```

if **name** == “**main**”:
# Test data loading
data_file = PROCESSED_DATA_DIR / “gestures_en.json”

```
if data_file.exists():
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples")
    
    # Build vocabulary
    word_to_idx, idx_to_word = build_vocabulary(data)
    print(f"Vocabulary size: {len(word_to_idx)}")
    
    # Split dataset
    train_data, val_data, test_data = split_dataset(data)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data, word_to_idx
    )
    
    # Test batch
    batch = next(iter(train_loader))
    print(f"Batch shapes: {[x.shape for x in batch]}")
else:
    print(f"Dataset not found: {data_file}")
    print("Run: python src/data/generator.py first")
```
