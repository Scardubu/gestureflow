#!/usr/bin/env python3
"""
Quantize trained model for deployment.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from src.models.optimizer import ModelOptimizer
from src.config import CHECKPOINT_DIR, MODEL_DIR


def main():
    """Quantize model."""
    parser = argparse.ArgumentParser(description="Quantize GestureFlow model")
    
    parser.add_argument(
        '--model',
        type=str,
        default='best_model.h5',
        help='Model file name in checkpoints directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file name (default: quantized_<type>.tflite)'
    )
    parser.add_argument(
        '--type',
        type=str,
        choices=['float16', 'int8'],
        default='float16',
        help='Quantization type'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("GestureFlow Model Quantization")
    print("=" * 70)
    print(f"Input model: {args.model}")
    print(f"Quantization type: {args.type}")
    print("=" * 70)
    print()
    
    model_path = CHECKPOINT_DIR / args.model
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return 1
    
    if args.output:
        output_path = MODEL_DIR / args.output
    else:
        output_path = MODEL_DIR / f"quantized_{args.type}.tflite"
    
    optimizer = ModelOptimizer()
    
    try:
        optimizer.quantize_model(
            model_path=model_path,
            output_path=output_path,
            quantization_type=args.type
        )
        
        print()
        print("=" * 70)
        print("Quantization completed successfully!")
        print(f"Quantized model saved to: {output_path}")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print(f"Error during quantization: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
