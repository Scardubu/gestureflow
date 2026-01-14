"""
Tests for gesture generator.
"""
import pytest
import numpy as np
from pathlib import Path
import tempfile
import json

from src.data.generator import GestureGenerator, load_dictionary
from src.config import QWERTY_LAYOUT


class TestGestureGenerator:
    """Test GestureGenerator class."""
    
    def test_init(self):
        """Test generator initialization."""
        generator = GestureGenerator()
        assert generator.layout == QWERTY_LAYOUT
        assert generator.noise_std == 0.02
        assert generator.min_points == 10
        assert generator.max_points == 50
    
    def test_generate_trajectory(self):
        """Test trajectory generation."""
        generator = GestureGenerator()
        
        word = "hello"
        trajectory, timestamps = generator.generate_trajectory(word)
        
        # Check shapes
        assert trajectory.shape[1] == 2  # x, y coordinates
        assert len(timestamps) == len(trajectory)
        
        # Check value ranges
        assert np.all(trajectory >= 0) and np.all(trajectory <= 1)
        assert np.all(timestamps >= 0) and np.all(timestamps <= 1)
        
        # Check minimum length
        assert len(trajectory) >= generator.min_points
        assert len(trajectory) <= generator.max_points
    
    def test_generate_trajectory_invalid_word(self):
        """Test trajectory generation with invalid word."""
        generator = GestureGenerator()
        
        with pytest.raises(ValueError):
            generator.generate_trajectory("123")  # No letters in layout
    
    def test_generate_dataset(self):
        """Test dataset generation."""
        generator = GestureGenerator()
        
        words = ["the", "and", "for"]
        samples_per_word = 5
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_dataset.json"
            
            dataset = generator.generate_dataset(
                words=words,
                samples_per_word=samples_per_word,
                output_file=output_file
            )
            
            # Check dataset size
            assert len(dataset) == len(words) * samples_per_word
            
            # Check file was created
            assert output_file.exists()
            
            # Check file contents
            with open(output_file, 'r') as f:
                loaded_data = json.load(f)
            
            assert len(loaded_data) == len(dataset)
            
            # Check sample structure
            sample = dataset[0]
            assert 'word' in sample
            assert 'trajectory' in sample
            assert 'timestamps' in sample
            assert 'layout' in sample
    
    def test_load_dictionary(self):
        """Test dictionary loading."""
        # This test assumes dictionary files exist
        try:
            words = load_dictionary("en")
            assert isinstance(words, list)
            assert len(words) > 0
            assert all(isinstance(word, str) for word in words)
        except FileNotFoundError:
            pytest.skip("Dictionary file not found")


class TestDataGeneration:
    """Test data generation utilities."""
    
    def test_trajectory_normalization(self):
        """Test that generated trajectories are normalized."""
        generator = GestureGenerator()
        
        trajectory, _ = generator.generate_trajectory("test")
        
        assert np.all(trajectory >= 0)
        assert np.all(trajectory <= 1)
    
    def test_timestamp_ordering(self):
        """Test that timestamps are ordered."""
        generator = GestureGenerator()
        
        _, timestamps = generator.generate_trajectory("test")
        
        assert np.all(np.diff(timestamps) >= 0)  # Non-decreasing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
