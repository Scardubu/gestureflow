"""
Configuration settings for GestureFlow.
"""
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_DIR = DATA_DIR / "raw"

# Model configuration
MODEL_CONFIG = {
    "sequence_length": 50,
    "embedding_dim": 128,
    "hidden_dim": 256,
    "num_layers": 2,
    "dropout": 0.2,
    "learning_rate": 0.001,
    "batch_size": 32,
    "num_epochs": 100
}

# Training configuration
TRAIN_CONFIG = {
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
    "random_seed": 42
}

# Data augmentation configuration
AUGMENTATION_CONFIG = {
    "noise_std": 0.02,
    "time_warp": True,
    "augmentation_factor": 3
}
