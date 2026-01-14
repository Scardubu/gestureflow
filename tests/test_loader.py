import pytest
import torch
from src.data.loader import GestureDataset, build_vocabulary

def test_build_vocabulary():
    """Test vocabulary building."""
    data = [
        {'word': 'hello'},
        {'word': 'world'},
        {'word': 'hello'},
    ]
    word_to_idx, idx_to_word = build_vocabulary(data)
    
    assert len(word_to_idx) == 3  # <UNK>, hello, world
    assert '<UNK>' in word_to_idx
    assert 'hello' in word_to_idx
    assert 'world' in word_to_idx

def test_dataset_creation():
    """Test dataset creation."""
    data = [
        {
            'word': 'test',
            'trajectory': [[0.5, 0.5], [0.6, 0.6]],
            'timestamps': [0, 100]
        }
    ]
    word_to_idx = {'<UNK>': 0, 'test': 1}
    
    dataset = GestureDataset(data, word_to_idx, max_length=50)
    assert len(dataset) == 1

def test_dataset_getitem():
    """Test dataset item retrieval."""
    data = [
        {
            'word': 'test',
            'trajectory': [[0.5, 0.5], [0.6, 0.6]],
            'timestamps': [0, 100]
        }
    ]
    word_to_idx = {'<UNK>': 0, 'test': 1}
    
    dataset = GestureDataset(data, word_to_idx, max_length=50)
    features, label, length = dataset[0]
    
    assert features.shape == (50, 3)
    assert isinstance(label, torch.Tensor)
    assert label.item() == 1

