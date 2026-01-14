“””
LSTM-based sequence model for swipe gesture prediction.
“””
import torch
import torch.nn as nn
from typing import Tuple, List
import numpy as np

from ..config import MODEL_CONFIG

class SwipeLSTM(nn.Module):
“””
Bidirectional LSTM model for gesture-to-word prediction.

```
Architecture:
    Input: (batch, sequence_length, 3)  # x, y, timestamp
    -> Embedding
    -> Bi-LSTM Layer 1
    -> Dropout
    -> LSTM Layer 2  
    -> Dropout
    -> Dense
    -> Output: (batch, vocab_size)
"""

def __init__(
    self,
    vocab_size: int,
    embedding_dim: int = 32,
    lstm_units: List[int] = [128, 64],
    dropout_rate: float = 0.3,
    bidirectional: bool = True
):
    """
    Initialize LSTM model.
    
    Args:
        vocab_size: Size of vocabulary (output dimension)
        embedding_dim: Dimension of input embedding
        lstm_units: List of LSTM hidden units for each layer
        dropout_rate: Dropout probability
        bidirectional: Use bidirectional LSTM for first layer
    """
    super(SwipeLSTM, self).__init__()
    
    self.vocab_size = vocab_size
    self.embedding_dim = embedding_dim
    self.lstm_units = lstm_units
    self.dropout_rate = dropout_rate
    self.bidirectional = bidirectional
    
    # Input embedding: project (x, y, t) to higher dimension
    self.embedding = nn.Linear(3, embedding_dim)
    self.embedding_activation = nn.ReLU()
    
    # First LSTM layer (bidirectional)
    self.lstm1 = nn.LSTM(
        input_size=embedding_dim,
        hidden_size=lstm_units[0],
        num_layers=1,
        batch_first=True,
        bidirectional=bidirectional,
        dropout=0.0  # No dropout in single-layer LSTM
    )
    
    # Dropout after first LSTM
    self.dropout1 = nn.Dropout(dropout_rate)
    
    # Calculate input size for second LSTM
    lstm2_input = lstm_units[0] * 2 if bidirectional else lstm_units[0]
    
    # Second LSTM layer (unidirectional for efficiency)
    self.lstm2 = nn.LSTM(
        input_size=lstm2_input,
        hidden_size=lstm_units[1],
        num_layers=1,
        batch_first=True,
        bidirectional=False
    )
    
    # Dropout after second LSTM
    self.dropout2 = nn.Dropout(dropout_rate)
    
    # Output layer
    self.fc = nn.Linear(lstm_units[1], vocab_size)
    
    # Initialize weights
    self._init_weights()

def _init_weights(self):
    """Initialize model weights using Xavier initialization."""
    for name, param in self.named_parameters():
        if 'weight' in name:
            if 'lstm' in name:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.xavier_normal_(param)
        elif 'bias' in name:
            nn.init.constant_(param, 0.0)

def forward(
    self, 
    x: torch.Tensor,
    lengths: torch.Tensor = None
) -> torch.Tensor:
    """
    Forward pass.
    
    Args:
        x: Input tensor of shape (batch, sequence_length, 3)
        lengths: Actual lengths of sequences (for packing)
        
    Returns:
        Output logits of shape (batch, vocab_size)
    """
    batch_size = x.size(0)
    
    # Embedding layer
    x = self.embedding(x)  # (batch, seq_len, embedding_dim)
    x = self.embedding_activation(x)
    
    # Pack padded sequences if lengths provided
    if lengths is not None:
        x = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
    
    # First LSTM layer
    x, (h1, c1) = self.lstm1(x)
    
    # Unpack if we packed
    if lengths is not None:
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
    
    x = self.dropout1(x)
    
    # Second LSTM layer
    x, (h2, c2) = self.lstm2(x)
    
    # Use final hidden state
    x = h2.squeeze(0)  # (batch, lstm_units[1])
    x = self.dropout2(x)
    
    # Output layer
    logits = self.fc(x)  # (batch, vocab_size)
    
    return logits

def predict(
    self, 
    x: torch.Tensor, 
    top_k: int = 5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Make predictions with confidence scores.
    
    Args:
        x: Input tensor
        top_k: Number of top predictions to return
        
    Returns:
        (indices, probabilities) for top-k predictions
    """
    self.eval()
    with torch.no_grad():
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)
    
    return top_indices, top_probs

def count_parameters(self) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in self.parameters() if p.requires_grad)

def get_model_size(self) -> float:
    """
    Calculate model size in MB.
    
    Returns:
        Model size in megabytes
    """
    param_size = sum(p.nelement() * p.element_size() 
                    for p in self.parameters())
    buffer_size = sum(b.nelement() * b.element_size() 
                     for b in self.buffers())
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb
```

class SwipeLSTMWithAttention(SwipeLSTM):
“””
Enhanced LSTM model with attention mechanism.
Better for longer sequences and multi-language support.
“””

```
def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    
    # Attention mechanism
    lstm2_hidden = self.lstm_units[1]
    self.attention = nn.MultiheadAttention(
        embed_dim=lstm2_hidden,
        num_heads=4,
        dropout=self.dropout_rate,
        batch_first=True
    )
    
    # Update output layer to use attention output
    self.fc = nn.Linear(lstm2_hidden, self.vocab_size)

def forward(
    self, 
    x: torch.Tensor,
    lengths: torch.Tensor = None
) -> torch.Tensor:
    """Forward pass with attention."""
    # Embedding
    x = self.embedding(x)
    x = self.embedding_activation(x)
    
    # First LSTM
    x, _ = self.lstm1(x)
    x = self.dropout1(x)
    
    # Second LSTM  
    x, _ = self.lstm2(x)
    
    # Self-attention over sequence
    attn_output, _ = self.attention(x, x, x)
    
    # Use mean of attended sequence
    x = torch.mean(attn_output, dim=1)
    x = self.dropout2(x)
    
    # Output
    logits = self.fc(x)
    
    return logits
```

def create_model(
vocab_size: int,
model_type: str = “lstm”,
**kwargs
) -> nn.Module:
“””
Factory function to create model.

```
Args:
    vocab_size: Vocabulary size
    model_type: "lstm" or "lstm_attention"
    **kwargs: Additional model parameters
    
Returns:
    Initialized model
"""
# Default config
config = {
    "embedding_dim": MODEL_CONFIG["embedding_dim"],
    "lstm_units": MODEL_CONFIG["lstm_units"],
    "dropout_rate": MODEL_CONFIG["dropout_rate"],
    "bidirectional": MODEL_CONFIG["bidirectional"],
}
config.update(kwargs)

if model_type == "lstm":
    model = SwipeLSTM(vocab_size=vocab_size, **config)
elif model_type == "lstm_attention":
    model = SwipeLSTMWithAttention(vocab_size=vocab_size, **config)
else:
    raise ValueError(f"Unknown model type: {model_type}")

print(f"\nModel Architecture:")
print(f"  Type: {model_type}")
print(f"  Vocabulary size: {vocab_size}")
print(f"  Parameters: {model.count_parameters():,}")
print(f"  Size: {model.get_model_size():.2f} MB")

return model
```

if **name** == “**main**”:
# Test model creation
model = create_model(vocab_size=10000, model_type=“lstm”)

```
# Test forward pass
batch_size = 16
seq_length = 50
x = torch.randn(batch_size, seq_length, 3)

output = model(x)
print(f"\nTest forward pass:")
print(f"  Input shape: {x.shape}")
print(f"  Output shape: {output.shape}")

# Test prediction
indices, probs = model.predict(x[:1], top_k=5)
print(f"\nTest prediction:")
print(f"  Top-5 indices: {indices[0]}")
print(f"  Top-5 probabilities: {probs[0]}")
```
