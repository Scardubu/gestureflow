"""
LSTM model for gesture recognition.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Optional

from ..config import MODEL_CONFIG


class GestureLSTM(keras.Model):
    """LSTM model for gesture-to-text prediction."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        lstm_units: int = 256,
        dropout_rate: float = 0.3,
        **kwargs
    ):
        """
        Initialize LSTM model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embedding layer
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate
        """
        super().__init__(**kwargs)
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        
        # Layers
        self.masking = layers.Masking(mask_value=0.0)
        
        self.lstm1 = layers.LSTM(
            lstm_units,
            return_sequences=True,
            dropout=dropout_rate
        )
        
        self.lstm2 = layers.LSTM(
            lstm_units // 2,
            return_sequences=False,
            dropout=dropout_rate
        )
        
        self.dense1 = layers.Dense(256, activation='relu')
        self.dropout = layers.Dropout(dropout_rate)
        self.dense2 = layers.Dense(128, activation='relu')
        self.output_layer = layers.Dense(vocab_size, activation='softmax')
    
    def call(self, inputs, training=False):
        """
        Forward pass.
        
        Args:
            inputs: Input tensor (batch, sequence_length, features)
            training: Whether in training mode
            
        Returns:
            Output predictions (batch, vocab_size)
        """
        x = self.masking(inputs)
        x = self.lstm1(x, training=training)
        x = self.lstm2(x, training=training)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        output = self.output_layer(x)
        
        return output
    
    def get_config(self):
        """Get model configuration."""
        return {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate
        }


def create_model(vocab_size: int, config: dict = None) -> GestureLSTM:
    """
    Create and compile a GestureLSTM model.
    
    Args:
        vocab_size: Size of vocabulary
        config: Optional configuration dict
        
    Returns:
        Compiled model
    """
    if config is None:
        config = MODEL_CONFIG
    
    model = GestureLSTM(
        vocab_size=vocab_size,
        embedding_dim=config.get('embedding_dim', 128),
        lstm_units=config.get('lstm_units', 256),
        dropout_rate=config.get('dropout_rate', 0.3)
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=config.get('learning_rate', 0.001)
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def main():
    """Test model creation."""
    print("Creating test model...")
    
    model = create_model(vocab_size=1000)
    
    # Build model with sample input
    sample_input = tf.random.normal((32, 50, 3))
    output = model(sample_input)
    
    print(f"Model created successfully!")
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    
    model.summary()


if __name__ == "__main__":
    main()
