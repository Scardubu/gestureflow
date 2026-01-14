"""Tests for data generator module."""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.generator import DataGenerator
from src.config import Config


class TestDataGenerator:
    """Test cases for DataGenerator class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = Config()
        self.generator = DataGenerator(self.config)
    
    def test_generator_initialization(self):
        """Test that generator initializes correctly."""
        assert self.generator is not None
        assert self.generator.config == self.config
    
    def test_generate_single_gesture(self):
        """Test generating a single gesture sequence."""
        word = "hello"
        gesture = self.generator.generate_gesture(word)
        
        assert gesture is not None
        assert len(gesture) > 0
        assert all(isinstance(point, tuple) for point in gesture)
    
    def test_gesture_coordinates_valid(self):
        """Test that generated coordinates are valid."""
        word = "test"
        gesture = self.generator.generate_gesture(word)
        
        for point in gesture:
            x, y = point
            assert isinstance(x, (int, float))
            assert isinstance(y, (int, float))
            assert x >= 0
            assert y >= 0
    
    def test_generate_dataset_shape(self):
        """Test that generated dataset has correct shape."""
        # This is a placeholder test
        # Actual implementation would depend on DataGenerator API
        pass
    
    def test_empty_word_handling(self):
        """Test handling of empty word input."""
        with pytest.raises(ValueError):
            self.generator.generate_gesture("")
    
    def test_invalid_characters(self):
        """Test handling of invalid characters."""
        # Generator should handle or filter invalid characters
        word_with_invalid = "test123!@#"
        gesture = self.generator.generate_gesture(word_with_invalid)
        assert gesture is not None
