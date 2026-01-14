"""
Configuration settings for GestureFlow.
"""
from pathlib import Path

# Project directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DICTIONARY_DIR = DATA_DIR / "dictionaries"
MODEL_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"

# Ensure directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                  DICTIONARY_DIR, MODEL_DIR, CHECKPOINT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    "sequence_length": 50,
    "embedding_dim": 128,
    "lstm_units": 256,
    "dropout_rate": 0.3,
    "learning_rate": 0.001,
}

# Training configuration
TRAINING_CONFIG = {
    "batch_size": 32,
    "epochs": 50,
    "validation_split": 0.2,
    "early_stopping_patience": 5,
}

# Data generation configuration
GENERATION_CONFIG = {
    "samples_per_word": 100,
    "noise_std": 0.02,
    "min_points": 10,
    "max_points": 50,
}

# Supported languages
SUPPORTED_LANGUAGES = {
    "en": "en_US",
    "es": "es_ES",
    "fr": "fr_FR",
}

# Keyboard layout
QWERTY_LAYOUT = {
    'q': (0.0, 0.0), 'w': (0.1, 0.0), 'e': (0.2, 0.0), 'r': (0.3, 0.0),
    't': (0.4, 0.0), 'y': (0.5, 0.0), 'u': (0.6, 0.0), 'i': (0.7, 0.0),
    'o': (0.8, 0.0), 'p': (0.9, 0.0),
    'a': (0.05, 0.33), 's': (0.15, 0.33), 'd': (0.25, 0.33), 'f': (0.35, 0.33),
    'g': (0.45, 0.33), 'h': (0.55, 0.33), 'j': (0.65, 0.33), 'k': (0.75, 0.33),
    'l': (0.85, 0.33),
    'z': (0.15, 0.67), 'x': (0.25, 0.67), 'c': (0.35, 0.67), 'v': (0.45, 0.67),
    'b': (0.55, 0.67), 'n': (0.65, 0.67), 'm': (0.75, 0.67),
}
