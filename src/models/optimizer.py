"""
Model optimization utilities.
"""
import tensorflow as tf
from pathlib import Path
from typing import Optional


class ModelOptimizer:
    """Optimize and quantize models."""
    
    @staticmethod
    def quantize_model(
        model_path: Path,
        output_path: Path,
        quantization_type: str = "float16"
    ):
        """
        Quantize a TensorFlow model.
        
        Args:
            model_path: Path to input model
            output_path: Path to save quantized model
            quantization_type: Type of quantization (float16, int8)
        """
        # Load model
        model = tf.keras.models.load_model(model_path)
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        if quantization_type == "float16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        elif quantization_type == "int8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            # For full integer quantization, you would need representative dataset
        
        tflite_model = converter.convert()
        
        # Save quantized model
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"Quantized model saved to {output_path}")
        
        # Print size comparison
        import os
        original_size = os.path.getsize(model_path)
        quantized_size = os.path.getsize(output_path)
        
        print(f"Original size: {original_size / 1024 / 1024:.2f} MB")
        print(f"Quantized size: {quantized_size / 1024 / 1024:.2f} MB")
        print(f"Reduction: {(1 - quantized_size / original_size) * 100:.1f}%")
    
    @staticmethod
    def prune_model(
        model: tf.keras.Model,
        target_sparsity: float = 0.5
    ) -> tf.keras.Model:
        """
        Apply pruning to a model.
        
        Args:
            model: Input model
            target_sparsity: Target sparsity level (0.0 to 1.0)
            
        Returns:
            Pruned model
        """
        try:
            import tensorflow_model_optimization as tfmot
            
            prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
            
            pruning_params = {
                'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                    initial_sparsity=0.0,
                    final_sparsity=target_sparsity,
                    begin_step=0,
                    end_step=1000
                )
            }
            
            model_for_pruning = prune_low_magnitude(model, **pruning_params)
            
            print(f"Model pruned with target sparsity: {target_sparsity}")
            
            return model_for_pruning
            
        except ImportError:
            print("Warning: tensorflow_model_optimization not installed. Skipping pruning.")
            return model


def main():
    """Test optimizer."""
    import argparse
    from ..config import MODEL_DIR, CHECKPOINT_DIR
    
    parser = argparse.ArgumentParser(description="Optimize model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="best_model.h5",
        help="Model file name"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="float16",
        choices=["float16", "int8"],
        help="Quantization type"
    )
    
    args = parser.parse_args()
    
    model_path = CHECKPOINT_DIR / args.model_path
    output_path = MODEL_DIR / f"quantized_{args.quantization}.tflite"
    
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return
    
    optimizer = ModelOptimizer()
    optimizer.quantize_model(
        model_path=model_path,
        output_path=output_path,
        quantization_type=args.quantization
    )


if __name__ == "__main__":
    main()
