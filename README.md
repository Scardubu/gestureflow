# GestureFlow

A gesture-based text input system that processes and analyzes gesture data for machine learning applications.

## Project Structure

```
gestureflow/
├── src/
│   ├── __init__.py         # Package initialization
│   ├── config.py           # Configuration settings
│   └── data/
│       ├── __init__.py     # Data module initialization
│       └── processor.py    # Data preprocessing and augmentation
├── requirements.txt        # Python dependencies
└── .gitignore             # Git ignore patterns
```

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Data Processing

The `GestureProcessor` class provides functionality for processing and augmenting gesture data:

```python
from src.data.processor import GestureProcessor
from src.config import MODEL_CONFIG

# Initialize processor
processor = GestureProcessor(
    sequence_length=MODEL_CONFIG["sequence_length"]
)

# Process dataset with augmentation
processor.process_dataset(
    input_file="path/to/input.json",
    output_file="path/to/output.json",
    augmentation=True,
    augmentation_factor=3
)
```

### Command Line Interface

Process datasets from the command line:

```bash
python -m src.data.processor --language en --augment --aug-factor 3
```

## Features

- **Data Normalization**: Normalize trajectory coordinates and timestamps
- **Data Augmentation**: Apply noise and time warping for data augmentation
- **Sequence Padding**: Pad sequences to fixed length for batch processing
- **Feature Extraction**: Extract features from gesture trajectories

## Configuration

Edit `src/config.py` to modify:
- Model parameters (sequence length, embedding dimensions, etc.)
- Training configuration (split ratios, random seed)
- Data augmentation settings

## Dependencies

- numpy >= 1.21.0
- tqdm >= 4.62.0
