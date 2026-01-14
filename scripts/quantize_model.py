"""Script to quantize trained models for optimized inference."""

import argparse
import os
import sys
import tensorflow as tf
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def quantize_model(model_path: str, output_path: str, quantization_type: str = 'dynamic'):
    """Quantize a trained TensorFlow model.
    
    Args:
        model_path: Path to the trained model
        output_path: Path to save quantized model
        quantization_type: Type of quantization ('dynamic', 'int8', 'float16')
    """
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantization_type == 'dynamic':
        print("Applying dynamic range quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    elif quantization_type == 'int8':
        print("Applying full integer quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    
    elif quantization_type == 'float16':
        print("Applying float16 quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    
    else:
        print(f"Unknown quantization type: {quantization_type}")
        print("Using dynamic range quantization as fallback...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Convert model
    print("Converting model...")
    tflite_model = converter.convert()
    
    # Save quantized model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    # Print size comparison
    original_size = os.path.getsize(model_path) / (1024 * 1024)
    quantized_size = os.path.getsize(output_path) / (1024 * 1024)
    compression_ratio = original_size / quantized_size
    
    print(f"\nQuantization complete!")
    print(f"Original model size: {original_size:.2f} MB")
    print(f"Quantized model size: {quantized_size:.2f} MB")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Quantized model saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Quantize trained models for optimized inference'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to the trained model (.h5 file)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save quantized model (.tflite file)'
    )
    parser.add_argument(
        '--type',
        type=str,
        default='dynamic',
        choices=['dynamic', 'int8', 'float16'],
        help='Type of quantization to apply'
    )
    
    args = parser.parse_args()
    
    # Generate output path if not provided
    if args.output is None:
        model_dir = os.path.dirname(args.model)
        model_name = os.path.splitext(os.path.basename(args.model))[0]
        args.output = os.path.join(model_dir, f"{model_name}_quantized.tflite")
    
    # Validate input path
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    # Perform quantization
    try:
        quantize_model(args.model, args.output, args.type)
    except Exception as e:
        print(f"Error during quantization: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
