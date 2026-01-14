"""
Tests for LSTM model.
"""
import pytest
import numpy as np
import tensorflow as tf

from src.models.lstm_model import GestureLSTM, create_model
from src.config import MODEL_CONFIG


class TestGestureLSTM:
    """Test GestureLSTM model."""
    
    def test_model_creation(self):
        """Test model creation."""
        vocab_size = 100
        model = GestureLSTM(
            vocab_size=vocab_size,
            embedding_dim=128,
            lstm_units=256,
            dropout_rate=0.3
        )
        
        assert model.vocab_size == vocab_size
        assert model.embedding_dim == 128
        assert model.lstm_units == 256
        assert model.dropout_rate == 0.3
    
    def test_model_forward_pass(self):
        """Test model forward pass."""
        vocab_size = 100
        batch_size = 8
        sequence_length = 50
        features = 3
        
        model = GestureLSTM(vocab_size=vocab_size)
        
        # Create sample input
        sample_input = tf.random.normal((batch_size, sequence_length, features))
        
        # Forward pass
        output = model(sample_input, training=False)
        
        # Check output shape
        assert output.shape == (batch_size, vocab_size)
        
        # Check output is probability distribution
        assert np.allclose(tf.reduce_sum(output, axis=1).numpy(), 1.0, atol=1e-5)
    
    def test_model_training_mode(self):
        """Test model in training mode."""
        vocab_size = 100
        batch_size = 8
        sequence_length = 50
        features = 3
        
        model = GestureLSTM(vocab_size=vocab_size)
        
        sample_input = tf.random.normal((batch_size, sequence_length, features))
        
        # Forward pass in training mode
        output_train = model(sample_input, training=True)
        
        # Check output shape
        assert output_train.shape == (batch_size, vocab_size)
    
    def test_create_model(self):
        """Test create_model helper function."""
        vocab_size = 100
        
        model = create_model(vocab_size=vocab_size)
        
        # Check model is compiled
        assert model.optimizer is not None
        assert model.loss is not None
        assert len(model.metrics) > 0
    
    def test_model_with_custom_config(self):
        """Test model creation with custom config."""
        vocab_size = 50
        config = {
            'embedding_dim': 64,
            'lstm_units': 128,
            'dropout_rate': 0.2,
            'learning_rate': 0.0005
        }
        
        model = create_model(vocab_size=vocab_size, config=config)
        
        assert model.embedding_dim == 64
        assert model.lstm_units == 128
        assert model.dropout_rate == 0.2
    
    def test_model_masking(self):
        """Test that model handles padded sequences with masking."""
        vocab_size = 100
        batch_size = 4
        sequence_length = 50
        features = 3
        
        model = GestureLSTM(vocab_size=vocab_size)
        
        # Create input with padding (zeros)
        sample_input = tf.random.normal((batch_size, sequence_length, features))
        # Set last 10 timesteps to zero (padding)
        sample_input = tf.concat([
            sample_input[:, :40, :],
            tf.zeros((batch_size, 10, features))
        ], axis=1)
        
        # Forward pass should handle padding
        output = model(sample_input, training=False)
        
        assert output.shape == (batch_size, vocab_size)


class TestModelArchitecture:
    """Test model architecture details."""
    
    def test_model_layers(self):
        """Test model has expected layers."""
        vocab_size = 100
        model = GestureLSTM(vocab_size=vocab_size)
        
        # Build model
        model.build(input_shape=(None, 50, 3))
        
        # Check layers exist
        assert hasattr(model, 'masking')
        assert hasattr(model, 'lstm1')
        assert hasattr(model, 'lstm2')
        assert hasattr(model, 'dense1')
        assert hasattr(model, 'dropout')
        assert hasattr(model, 'dense2')
        assert hasattr(model, 'output_layer')
    
    def test_model_trainable_parameters(self):
        """Test model has trainable parameters."""
        vocab_size = 100
        model = GestureLSTM(vocab_size=vocab_size)
        
        model.build(input_shape=(None, 50, 3))
        
        # Check model has trainable parameters
        assert len(model.trainable_variables) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
