“””
Main training script for GestureFlow model.
“””
import argparse
import json
import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import sys

# Add src to path

sys.path.append(str(Path(**file**).parent.parent))

from src.config import (
PROCESSED_DATA_DIR, TRAINING_CONFIG, MODEL_CONFIG,
CHECKPOINTS_DIR, SUPPORTED_LANGUAGES
)
from src.models.lstm_model import create_model
from src.training.trainer import Trainer, GestureDataset

def load_data(language: str = “en”) -> tuple:
“””
Load and prepare dataset.

```
Returns:
    (dataset, word_to_idx, idx_to_word)
"""
data_file = PROCESSED_DATA_DIR / f"gestures_{language}.json"

if not data_file.exists():
    raise FileNotFoundError(
        f"Dataset not found: {data_file}\n"
        f"Run: python src/data/generator.py --language {language}"
    )

print(f"Loading dataset from {data_file}...")
with open(data_file, 'r') as f:
    data = json.load(f)

print(f"Loaded {len(data)} gesture samples")

# Build vocabulary
unique_words = sorted(set(sample['word'] for sample in data))
word_to_idx = {word: idx + 1 for idx, word in enumerate(unique_words)}
word_to_idx['<UNK>'] = 0  # Unknown token
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

print(f"Vocabulary size: {len(word_to_idx)}")

return data, word_to_idx, idx_to_word
```

def prepare_dataloaders(
data: list,
word_to_idx: dict,
batch_size: int = 32,
val_split: float = 0.2,
test_split: float = 0.1
) -> tuple:
“””
Create train/val/test dataloaders.

```
Returns:
    (train_loader, val_loader, test_loader)
"""
# Create dataset
dataset = GestureDataset(
    data=data,
    word_to_idx=word_to_idx,
    max_length=MODEL_CONFIG["sequence_length"]
)

# Calculate splits
total_size = len(dataset)
test_size = int(total_size * test_split)
val_size = int(total_size * val_split)
train_size = total_size - val_size - test_size

# Split dataset
train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(TRAINING_CONFIG["seed"])
)

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print(f"Dataset splits:")
print(f"  Train: {train_size} samples")
print(f"  Val: {val_size} samples")
print(f"  Test: {test_size} samples")

return train_loader, val_loader, test_loader
```

def main():
“”“Main training function.”””
parser = argparse.ArgumentParser(description=“Train SwipePredict model”)
parser.add_argument(
“–language”, “-l”,
type=str,
default=“en”,
choices=list(SUPPORTED_LANGUAGES.keys()),
help=“Language to train on”
)
parser.add_argument(
“–epochs”, “-e”,
type=int,
default=TRAINING_CONFIG[“epochs”],
help=“Number of training epochs”
)
parser.add_argument(
“–batch-size”, “-b”,
type=int,
default=MODEL_CONFIG[“batch_size”],
help=“Batch size”
)
parser.add_argument(
“–learning-rate”, “-lr”,
type=float,
default=MODEL_CONFIG[“learning_rate”],
help=“Learning rate”
)
parser.add_argument(
“–model-type”,
type=str,
default=“lstm”,
choices=[“lstm”, “lstm_attention”],
help=“Model architecture”
)
parser.add_argument(
“–device”,
type=str,
default=“cuda” if torch.cuda.is_available() else “cpu”,
help=“Device to train on”
)
parser.add_argument(
“–resume”,
type=str,
default=None,
help=“Path to checkpoint to resume from”
)

```
args = parser.parse_args()

print("=" * 80)
print("SwipePredict Model Training")
print("=" * 80)
print(f"\nConfiguration:")
print(f"  Language: {SUPPORTED_LANGUAGES[args.language]['name']}")
print(f"  Model: {args.model_type}")
print(f"  Epochs: {args.epochs}")
print(f"  Batch size: {args.batch_size}")
print(f"  Learning rate: {args.learning_rate}")
print(f"  Device: {args.device}")
print()

# Load data
data, word_to_idx, idx_to_word = load_data(args.language)

# Prepare dataloaders
train_loader, val_loader, test_loader = prepare_dataloaders(
    data=data,
    word_to_idx=word_to_idx,
    batch_size=args.batch_size,
    val_split=TRAINING_CONFIG["validation_split"],
    test_split=TRAINING_CONFIG["test_split"]
)

# Create model
model = create_model(
    vocab_size=len(word_to_idx),
    model_type=args.model_type
)

# Create trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=args.device,
    learning_rate=args.learning_rate,
    checkpoint_dir=CHECKPOINTS_DIR / args.language
)

# Resume from checkpoint if specified
if args.resume:
    trainer.load_checkpoint(Path(args.resume))

# Train model
history = trainer.train(
    num_epochs=args.epochs,
    early_stopping_patience=TRAINING_CONFIG["early_stopping_patience"]
)

# Save vocabulary and config
output_dir = CHECKPOINTS_DIR / args.language
output_dir.mkdir(parents=True, exist_ok=True)

# Save vocabulary
vocab_file = output_dir / "vocabulary.json"
with open(vocab_file, 'w') as f:
    json.dump({
        'word_to_idx': word_to_idx,
        'idx_to_word': idx_to_word,
        'vocab_size': len(word_to_idx)
    }, f, indent=2)
print(f"\nVocabulary saved to {vocab_file}")

# Save training history
history_file = output_dir / "training_history.json"
with open(history_file, 'w') as f:
    json.dump(history, f, indent=2)
print(f"Training history saved to {history_file}")

# Test final model
print("\n" + "=" * 80)
print("Evaluating on test set...")
print("=" * 80)

test_loss, test_top1, test_top5 = trainer.validate()

print(f"\nFinal Test Results:")
print(f"  Loss: {test_loss:.4f}")
print(f"  Top-1 Accuracy: {test_top1:.2f}%")
print(f"  Top-5 Accuracy: {test_top5:.2f}%")

# Save test results
results_file = output_dir / "test_results.json"
with open(results_file, 'w') as f:
    json.dump({
        'test_loss': test_loss,
        'test_top1_accuracy': test_top1,
        'test_top5_accuracy': test_top5,
        'model_type': args.model_type,
        'language': args.language
    }, f, indent=2)

print(f"\nTraining complete! Model saved to {output_dir}")
```

if **name** == “**main**”:
main()
