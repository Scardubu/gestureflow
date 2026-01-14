# tests/test_model.py

import pytest
import torch
from src.models.lstm_model import SwipeLSTM, SwipeLSTMWithAttention, create_model

def test_model_creation():
“”“Test basic model creation.”””
model = create_model(vocab_size=1000, model_type=“lstm”)
assert model is not None
assert isinstance(model, SwipeLSTM)
assert model.vocab_size == 1000

def test_model_parameters():
“”“Test model parameter counting.”””
model = create_model(vocab_size=1000)
param_count = model.count_parameters()
assert param_count > 0
assert isinstance(param_count, int)

def test_model_size():
“”“Test model size calculation.”””
model = create_model(vocab_size=1000)
size_mb = model.get_model_size()
assert size_mb > 0
assert isinstance(size_mb, float)

def test_model_forward():
“”“Test forward pass.”””
model = create_model(vocab_size=1000)
batch_size, seq_length = 4, 50
x = torch.randn(batch_size, seq_length, 3)

```
output = model(x)
assert output.shape == (batch_size, 1000)
```

def test_model_forward_with_lengths():
“”“Test forward pass with sequence lengths.”””
model = create_model(vocab_size=1000)
batch_size, seq_length = 4, 50
x = torch.randn(batch_size, seq_length, 3)
lengths = torch.tensor([50, 40, 30, 20])

```
output = model(x, lengths)
assert output.shape == (batch_size, 1000)
```

def test_model_predict():
“”“Test prediction method.”””
model = create_model(vocab_size=1000)
x = torch.randn(1, 50, 3)

```
indices, probs = model.predict(x, top_k=5)
assert indices.shape == (1, 5)
assert probs.shape == (1, 5)
assert torch.all(probs >= 0) and torch.all(probs <= 1)
assert torch.allclose(probs.sum(dim=1), torch.ones(1), atol=1e-5) is False  # Top-k, not full distribution
```

def test_attention_model_creation():
“”“Test attention model creation.”””
model = create_model(vocab_size=1000, model_type=“lstm_attention”)
assert model is not None
assert isinstance(model, SwipeLSTMWithAttention)

def test_attention_model_forward():
“”“Test attention model forward pass.”””
model = create_model(vocab_size=1000, model_type=“lstm_attention”)
x = torch.randn(4, 50, 3)

```
output = model(x)
assert output.shape == (4, 1000)
```

assert features.shape == (50, 3)
assert isinstance(label, torch.Tensor)
assert label.item() == 1
```
