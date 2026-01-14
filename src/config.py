“””
Configuration file for SwipePredict project.
Contains all hyperparameters, paths, and system settings.
“””
import os
from pathlib import Path

# Project paths

PROJECT_ROOT = Path(**file**).parent.parent
DATA_DIR = PROJECT_ROOT / “data”
DICTIONARIES_DIR = DATA_DIR / “dictionaries”
RAW_DATA_DIR = DATA_DIR / “raw”
PROCESSED_DATA_DIR = DATA_DIR / “processed”
MODELS_DIR = PROJECT_ROOT / “models”
CHECKPOINTS_DIR = MODELS_DIR / “checkpoints”
LOGS_DIR = PROJECT_ROOT / “logs”

# Create directories if they don’t exist

for dir_path in [DATA_DIR, DICTIONARIES_DIR, RAW_DATA_DIR,
PROCESSED_DATA_DIR, MODELS_DIR, CHECKPOINTS_DIR, LOGS_DIR]:
dir_path.mkdir(parents=True, exist_ok=True)

# Model Architecture Configuration

MODEL_CONFIG = {
“lstm_units”: [128, 64],  # Two-layer LSTM
“dropout_rate”: 0.3,
“bidirectional”: True,
“sequence_length”: 50,  # Max points in gesture
“embedding_dim”: 32,
“batch_size”: 32,
“learning_rate”: 0.001,
“weight_decay”: 1e-5,
}

# Training Configuration

TRAINING_CONFIG = {
“epochs”: 50,
“early_stopping_patience”: 10,
“reduce_lr_patience”: 5,
“reduce_lr_factor”: 0.5,
“min_lr”: 1e-6,
“validation_split”: 0.2,
“test_split”: 0.1,
“shuffle”: True,
“seed”: 42,
}

# Data Generation Configuration

DATA_CONFIG = {
“samples_per_word”: 10,  # Synthetic samples per dictionary word
“min_word_length”: 3,
“max_word_length”: 15,
“noise_std”: 5.0,  # Pixels of noise to add
“swipe_speed_range”: (100, 300),  # ms per character
“num_workers”: 4,  # Parallel processing workers
}

# QWERTY Keyboard Layout (normalized coordinates 0-1)

KEYBOARD_LAYOUTS = {
“qwerty”: {
“q”: (0.05, 0.0), “w”: (0.15, 0.0), “e”: (0.25, 0.0),
“r”: (0.35, 0.0), “t”: (0.45, 0.0), “y”: (0.55, 0.0),
“u”: (0.65, 0.0), “i”: (0.75, 0.0), “o”: (0.85, 0.0),
“p”: (0.95, 0.0),

```
    "a": (0.08, 0.33), "s": (0.18, 0.33), "d": (0.28, 0.33),
    "f": (0.38, 0.33), "g": (0.48, 0.33), "h": (0.58, 0.33),
    "j": (0.68, 0.33), "k": (0.78, 0.33), "l": (0.88, 0.33),
    
    "z": (0.15, 0.66), "x": (0.25, 0.66), "c": (0.35, 0.66),
    "v": (0.45, 0.66), "b": (0.55, 0.66), "n": (0.65, 0.66),
    "m": (0.75, 0.66),
}
```

}

# Optimization Configuration

OPTIMIZATION_CONFIG = {
“quantization”: “int8”,  # int8, float16, or None
“pruning_threshold”: 0.01,  # Remove weights below this value
“target_latency_ms”: 50,
“target_model_size_mb”: 5,
}

# API Configuration

API_CONFIG = {
“host”: “0.0.0.0”,
“port”: 8000,
“reload”: True,
“workers”: 4,
“log_level”: “info”,
“max_request_size”: 1024 * 1024,  # 1MB
“timeout”: 30,
}

# Inference Configuration

INFERENCE_CONFIG = {
“top_k”: 5,  # Return top-5 predictions
“min_confidence”: 0.1,  # Minimum confidence threshold
“batch_inference”: False,
“use_gpu”: False,  # For edge deployment, typically CPU
}

# Language Support

SUPPORTED_LANGUAGES = {
“en”: {
“name”: “English”,
“dictionary_url”: “https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt”,
“dictionary_file”: “en_US.txt”,
“min_frequency”: 100,  # Minimum word frequency for training
},
“es”: {
“name”: “Spanish”,
“dictionary_url”: “https://raw.githubusercontent.com/lorenbrichter/Words/master/Spanish.txt”,
“dictionary_file”: “es_ES.txt”,
“min_frequency”: 50,
},
“fr”: {
“name”: “French”,
“dictionary_url”: “https://raw.githubusercontent.com/lorenbrichter/Words/master/French.txt”,
“dictionary_file”: “fr_FR.txt”,
“min_frequency”: 50,
}
}

# Logging Configuration

LOGGING_CONFIG = {
“version”: 1,
“disable_existing_loggers”: False,
“formatters”: {
“standard”: {
“format”: “%(asctime)s [%(levelname)s] %(name)s: %(message)s”
},
},
“handlers”: {
“console”: {
“class”: “logging.StreamHandler”,
“level”: “INFO”,
“formatter”: “standard”,
“stream”: “ext://sys.stdout”,
},
“file”: {
“class”: “logging.FileHandler”,
“level”: “DEBUG”,
“formatter”: “standard”,
“filename”: str(LOGS_DIR / “swipepredict.log”),
“mode”: “a”,
},
},
“loggers”: {
“”: {
“handlers”: [“console”, “file”],
“level”: “INFO”,
“propagate”: False,
}
},
}

# Environment-specific overrides

ENV = os.getenv(“SWIPEPREDICT_ENV”, “development”)

if ENV == “production”:
API_CONFIG[“reload”] = False
API_CONFIG[“workers”] = 8
LOGGING_CONFIG[“handlers”][“console”][“level”] = “WARNING”
elif ENV == “testing”:
TRAINING_CONFIG[“epochs”] = 2
DATA_CONFIG[“samples_per_word”] = 2
