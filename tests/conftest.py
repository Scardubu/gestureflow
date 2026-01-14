import pytest
import numpy as np

@pytest.fixture
def sample_trajectory():
    """Fixture for sample trajectory."""
    return np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

@pytest.fixture
def sample_timestamps():
    """Fixture for sample timestamps."""
    return np.array([0, 100, 200])

@pytest.fixture
def sample_gesture():
    """Fixture for sample gesture."""
    return {
        'word': 'test',
        'trajectory': [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        'timestamps': [0, 100, 200],
        'num_points': 3,
        'layout': 'qwerty'
    }