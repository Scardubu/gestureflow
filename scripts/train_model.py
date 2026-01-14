#!/usr/bin/env python3
"""
Train GestureFlow model.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from src.data.generator import load_dictionary, GestureGenerator
from src.data.processor import GestureProcessor
from src.data.loader import GestureDataLoader
from src.models.lstm_model import create_model
from src.training.trainer import ModelTrainer
from src.config import (
    PROCESSED_DATA_DIR,
    GENERATION_CONFIG,
    MODEL_CONFIG,
    TRAINING_CONFIG
)


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train GestureFlow model")
    
    # Data arguments
    parser.add_argument(
        '--language', '-l',
        type=str,
        default='en',
        help='Language code (en, es, fr)'
    )
    parser.add_argument(
        '--samples-per-word',
        type=int,
        default=100,
        help='Number of samples to generate per word'
    )
    parser.add_argument(
        '--skip-generation',
        action='store_true',
        help='Skip data generation step'
    )
    parser.add_argument(
        '--skip-processing',
        action='store_true',
        help='Skip data processing step'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=TRAINING_CONFIG['epochs'],
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=TRAINING_CONFIG['batch_size'],
        help='Batch size'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=MODEL_CONFIG['learning_rate'],
        help='Learning rate'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("GestureFlow Model Training")
    print("=" * 70)
    print(f"Language: {args.language}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 70)
    print()
    
    # Step 1: Generate gesture data
    if not args.skip_generation:
        print("Step 1: Generating gesture data...")
        print("-" * 70)
        
        words = load_dictionary(args.language)
        print(f"Loaded {len(words)} words")
        
        generator = GestureGenerator(
            noise_std=GENERATION_CONFIG['noise_std'],
            min_points=GENERATION_CONFIG['min_points'],
            max_points=GENERATION_CONFIG['max_points']
        )
        
        raw_data_file = PROCESSED_DATA_DIR / f"gestures_{args.language}.json"
        generator.generate_dataset(
            words=words,
            samples_per_word=args.samples_per_word,
            output_file=raw_data_file
        )
        print()
    
    # Step 2: Process data
    if not args.skip_processing:
        print("Step 2: Processing and augmenting data...")
        print("-" * 70)
        
        processor = GestureProcessor(
            sequence_length=MODEL_CONFIG['sequence_length']
        )
        
        raw_data_file = PROCESSED_DATA_DIR / f"gestures_{args.language}.json"
        processed_data_file = PROCESSED_DATA_DIR / f"processed_{args.language}.json"
        
        processor.process_dataset(
            input_file=raw_data_file,
            output_file=processed_data_file,
            augmentation=True,
            augmentation_factor=3
        )
        print()
    
    # Step 3: Load data
    print("Step 3: Loading processed data...")
    print("-" * 70)
    
    processed_data_file = PROCESSED_DATA_DIR / f"processed_{args.language}.json"
    
    loader = GestureDataLoader(
        data_path=processed_data_file,
        batch_size=args.batch_size,
        validation_split=TRAINING_CONFIG['validation_split']
    )
    
    print(f"Vocabulary size: {loader.vocab_size}")
    print(f"Total samples: {len(loader.data)}")
    
    # Save vocabulary
    vocab_file = PROCESSED_DATA_DIR / "vocab.json"
    loader.save_vocabulary(vocab_file)
    
    train_ds, val_ds = loader.get_datasets()
    print()
    
    # Step 4: Create model
    print("Step 4: Creating model...")
    print("-" * 70)
    
    model = create_model(
        vocab_size=loader.vocab_size,
        config=MODEL_CONFIG
    )
    
    model.summary()
    print()
    
    # Step 5: Train model
    print("Step 5: Training model...")
    print("-" * 70)
    
    trainer = ModelTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds
    )
    
    history = trainer.train(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        early_stopping_patience=TRAINING_CONFIG['early_stopping_patience']
    )
    print()
    
    # Step 6: Evaluate
    print("Step 6: Final evaluation...")
    print("-" * 70)
    
    metrics = trainer.evaluate()
    print()
    
    print("=" * 70)
    print("Training completed successfully!")
    print("=" * 70)
    print(f"Best model saved to: models/checkpoints/best_model.h5")
    print(f"Vocabulary saved to: {vocab_file}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
