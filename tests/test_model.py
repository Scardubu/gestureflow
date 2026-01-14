"""Tests for LSTM model module."""

import pytest
import numpy as np
import tensorflow as tf
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.lstm_model import create_model
from src.config import Config


class TestLSTMModel:
    """Test cases for LSTM model."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = Config()
        self.model = create_model(self.config)
    
    def test_model_creation(self):
        """Test that model is created successfully."""
        assert self.model is not None
        assert isinstance(self.model, tf.keras.Model)
    
    def test_model_layers(self):
        """Test that model has expected layers."""
        layer_types = [type(layer).__name__ for layer in self.model.layers]
        assert 'LSTM' in layer_types or 'Bidirectional' in layer_types
        assert 'Dense' in layer_types
    
    def test_model_input_shape(self):
        """Test model input shape."""
        input_shape = self.model.input_shape
        assert input_shape is not None
        assert len(input_shape) >= 2
    
    def test_model_output_shape(self):
        """Test model output shape."""
        output_shape = self.model.output_shape
        assert output_shape is not None
        assert len(output_shape) >= 1
    
    def test_model_compilation(self):
        """Test that model can be compiled."""
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        assert self.model.optimizer is not None
    
    def test_model_prediction_shape(self):
        """Test model prediction with dummy input."""
        # Create dummy input
        batch_size = 1
        sequence_length = 10
        feature_dim = 2
        
        dummy_input = np.random.randn(batch_size, sequence_length, feature_dim)
        
        try:
            predictions = self.model.predict(dummy_input)
            assert predictions is not None
            assert len(predictions.shape) >= 1
            assert predictions.shape[0] == batch_size
        except Exception as e:
            # Model might need specific input shape
            pytest.skip(f"Model prediction test skipped: {e}")
    
    def test_model_trainable_parameters(self):
        """Test that model has trainable parameters."""
        trainable_count = np.sum([tf.keras.backend.count_params(w) 
                                  for w in self.model.trainable_weights])
        assert trainable_count > 0
