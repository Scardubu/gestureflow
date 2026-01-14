#!/usr/bin/env python3
"""
Benchmark model performance.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import time
import numpy as np
from src.inference.predictor import GesturePredictor
from src.data.loader import GestureDataLoader
from src.training.metrics import calculate_metrics, print_metrics
from src.config import CHECKPOINT_DIR, PROCESSED_DATA_DIR


def main():
    """Benchmark model."""
    parser = argparse.ArgumentParser(description="Benchmark GestureFlow model")
    
    parser.add_argument(
        '--model',
        type=str,
        default='best_model.h5',
        help='Model file name'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='processed_en.json',
        help='Test data file name'
    )
    parser.add_argument(
        '--vocab',
        type=str,
        default='vocab.json',
        help='Vocabulary file name'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=1000,
        help='Number of samples to benchmark'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("GestureFlow Model Benchmark")
    print("=" * 70)
    print()
    
    model_path = CHECKPOINT_DIR / args.model
    vocab_path = PROCESSED_DATA_DIR / args.vocab
    data_path = PROCESSED_DATA_DIR / args.data
    
    # Check files exist
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return 1
    
    if not vocab_path.exists():
        print(f"Error: Vocabulary not found at {vocab_path}")
        return 1
    
    if not data_path.exists():
        print(f"Error: Data not found at {data_path}")
        return 1
    
    # Load predictor
    print("Loading predictor...")
    predictor = GesturePredictor(
        model_path=model_path,
        vocab_path=vocab_path
    )
    print(f"✓ Predictor loaded with vocabulary size: {len(predictor.idx_to_word)}")
    print()
    
    # Load test data
    print("Loading test data...")
    loader = GestureDataLoader(
        data_path=data_path,
        batch_size=32,
        validation_split=0.2,
        shuffle=False
    )
    _, val_ds = loader.get_datasets()
    print(f"✓ Test data loaded")
    print()
    
    # Benchmark
    print("Running benchmark...")
    print("-" * 70)
    
    y_true = []
    y_pred = []
    y_pred_probs = []
    inference_times = []
    
    num_samples = 0
    for X_batch, y_batch in val_ds:
        if num_samples >= args.num_samples:
            break
        
        # Time inference
        start_time = time.time()
        predictions = predictor.model.predict(X_batch, verbose=0)
        inference_time = time.time() - start_time
        
        # Collect results
        y_true.extend(y_batch.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))
        y_pred_probs.extend(predictions)
        inference_times.append(inference_time)
        
        num_samples += len(y_batch)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_probs = np.array(y_pred_probs)
    
    # Calculate metrics
    print()
    print("Performance Metrics:")
    print("-" * 70)
    
    metrics = calculate_metrics(y_true, y_pred, y_pred_probs)
    print_metrics(metrics)
    
    # Timing statistics
    print("\nTiming Statistics:")
    print("-" * 70)
    total_time = sum(inference_times)
    avg_time_per_batch = np.mean(inference_times)
    avg_time_per_sample = total_time / num_samples
    
    print(f"Total samples: {num_samples}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per batch: {avg_time_per_batch*1000:.2f} ms")
    print(f"Average time per sample: {avg_time_per_sample*1000:.2f} ms")
    print(f"Throughput: {num_samples/total_time:.1f} samples/second")
    
    print()
    print("=" * 70)
    print("Benchmark completed!")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
