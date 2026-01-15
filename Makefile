# Makefile for GestureFlow

# Convenient commands for development and deployment

.PHONY: help install clean test lint format data train api web deploy-api deploy-web docker

# Default target

help:
@echo “GestureFlow - Available Commands”
@echo “=================================”
@echo “Setup & Installation:”
@echo “  make install        - Install all dependencies”
@echo “  make install-dev    - Install with dev dependencies”
@echo “”
@echo “Development:”
@echo “  make data           - Download dictionaries and generate data”
@echo “  make train          - Train LSTM model”
@echo “  make api            - Start API server”
@echo “  make web            - Start frontend dev server”
@echo “  make test           - Run test suite”
@echo “  make lint           - Run linters”
@echo “  make format         - Format code with black”
@echo “”
@echo “Deployment:”
@echo “  make docker         - Build Docker containers”
@echo “  make deploy-api     - Deploy API to Railway”
@echo “  make deploy-web     - Deploy frontend to Vercel”
@echo “”
@echo “Utilities:”
@echo “  make clean          - Clean generated files”
@echo “  make benchmark      - Run performance benchmarks”

# Setup & Installation

install:
python -m pip install –upgrade pip
pip install -r requirements.txt

install-dev:
python -m pip install –upgrade pip
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy

# Data preparation

data:
@echo “Downloading dictionaries…”
python scripts/download_dictionaries.py
@echo “Generating training data…”
python src/data/generator.py –language en –max-words 5000 –samples 10

data-quick:
@echo “Quick data generation (3000 words)…”
python src/data/generator.py –language en –max-words 3000 –samples 5

# Training

train:
@echo “Training LSTM model…”
python scripts/train_model.py –language en –epochs 50 –batch-size 32

train-quick:
@echo “Quick training (20 epochs)…”
python scripts/train_model.py –language en –epochs 20 –batch-size 64

train-gpu:
@echo “Training with GPU…”
python scripts/train_model.py –language en –epochs 50 –device cuda

# Development servers

api:
@echo “Starting API server…”
cd api && uvicorn main:app –reload –host 0.0.0.0 –port 8000

web:
@echo “Starting frontend dev server…”
cd web && npm run dev

# Testing

test:
@echo “Running test suite…”
pytest tests/ -v

test-cov:
@echo “Running tests with coverage…”
pytest tests/ -v –cov=src –cov-report=html –cov-report=term

test-watch:
@echo “Running tests in watch mode…”
pytest-watch tests/ -v

# Code quality

lint:
@echo “Running flake8…”
flake8 src/ api/ scripts/ –max-line-length=100
@echo “Running mypy…”
mypy src/ –ignore-missing-imports

format:
@echo “Formatting code with black…”
black src/ api/ scripts/ tests/

format-check:
@echo “Checking code format…”
black –check src/ api/ scripts/ tests/

# Utilities

benchmark:
@echo “Running performance benchmarks…”
python scripts/benchmark.py –language en –device cpu

benchmark-gpu:
@echo “Running GPU benchmarks…”
python scripts/benchmark.py –language en –device cuda

quantize:
@echo “Quantizing model…”
python scripts/quantize_model.py –language en

clean:
@echo “Cleaning generated files…”
find . -type d -name **pycache** -exec rm -rf {} + 2>/dev/null || true
find . -type f -name “*.pyc” -delete
find . -type f -name “*.pyo” -delete
find . -type f -name “*.log” -delete
rm -rf .pytest_cache
rm -rf .coverage
rm -rf htmlcov
rm -rf dist
rm -rf build
rm -rf *.egg-info

clean-data:
@echo “Cleaning data files (WARNING: destructive)…”
@read -p “Are you sure? [y/N] “ -n 1 -r;   
echo;   
if [[ $$REPLY =~ ^[Yy]$$ ]]; then   
rm -rf data/processed/*;   
rm -rf data/raw/*;   
echo “Data cleaned.”;   
fi

clean-models:
@echo “Cleaning model checkpoints (WARNING: destructive)…”
@read -p “Are you sure? [y/N] “ -n 1 -r;   
echo;   
if [[ $$REPLY =~ ^[Yy]$$ ]]; then   
rm -rf models/checkpoints/*;   
echo “Models cleaned.”;   
fi

# Docker

docker:
@echo “Building Docker containers…”
docker-compose build

docker-up:
@echo “Starting Docker containers…”
docker-compose up -d

docker-down:
@echo “Stopping Docker containers…”
docker-compose down

docker-logs:
@echo “Viewing Docker logs…”
docker-compose logs -f

# Deployment

deploy-api:
@echo “Deploying API to Railway…”
railway up

deploy-web:
@echo “Deploying frontend to Vercel…”
cd web && vercel –prod

deploy-all: deploy-api deploy-web
@echo “Full deployment complete!”

# Development workflow shortcuts

dev-setup: install data
@echo “Development environment ready!”

quick-start: install data-quick train-quick
@echo “Quick start complete! Run ‘make api’ and ‘make web’ to start servers.”

full-setup: install data train
@echo “Full setup complete! Run ‘make api’ and ‘make web’ to start servers.”

# Continuous Integration

ci: lint test
@echo “CI checks passed!”

# Pre-commit checks

pre-commit: format lint test
@echo “Pre-commit checks passed!”