# GestureFlow

A machine learning system for gesture-based text input using swipe/gesture typing.

## Features

- Gesture trajectory generation and processing
- LSTM-based prediction model
- Multi-language support (English, Spanish, French)
- REST API for predictions
- Web-based demo interface

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python scripts/train_model.py --language en
```

### API Server

```bash
uvicorn api.main:app --reload
```

### Web Interface

```bash
cd web
npm install
npm run dev
```

## Project Structure

- `data/` - Training data and dictionaries
- `src/` - Core library code
- `api/` - REST API server
- `web/` - Web interface
- `scripts/` - Utility scripts
- `notebooks/` - Jupyter notebooks for exploration
- `tests/` - Unit tests
- `models/` - Trained model checkpoints

## License

MIT
