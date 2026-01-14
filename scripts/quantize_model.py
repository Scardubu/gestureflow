"""Quantize trained model for edge deployment."""
import torch
from pathlib import Path
import sys
import argparse

sys.path.append(str(Path(__file__).parent.parent))

from src.models.lstm_model import create_model
from src.config import CHECKPOINTS_DIR
import json


def quantize_model(language: str = "en", output_name: str = "quantized_model.pt"):
    """
    Quantize model to INT8.
    
    Args:
        language: Language code
        output_name: Output filename
    """
    model_dir = CHECKPOINTS_DIR / language
    model_path = model_dir / "best_model.pt"
    vocab_path = model_dir / "vocabulary.json"
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print(f"Train a model first: python scripts/train_model.py --language {language}")
        return
    
    # Load vocabulary to get size
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
    vocab_size = vocab_data['vocab_size']
    
    print(f"Loading model from {model_path}...")
    
    # Load model
    model = create_model(vocab_size=vocab_size)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get original size
    original_size = model_path.stat().st_size / (1024 ** 2)
    
    print("\nQuantizing model to INT8...")
    
    # Apply dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear, torch.nn.LSTM}, 
        dtype=torch.qint8
    )
    
    # Save quantized model
    output_path = model_dir / output_name
    torch.save({
        'model_state_dict': quantized_model.state_dict(),
        'vocab_size': vocab_size,
        'quantized': True,
        'original_model': str(model_path)
    }, output_path)
    
    # Get quantized size
    quantized_size = output_path.stat().st_size / (1024 ** 2)
    reduction = 100 * (1 - quantized_size / original_size)
    
    print(f"\n{'='*60}")
    print("Quantization Results")
    print(f"{'='*60}")
    print(f"Original size:    {original_size:.2f} MB")
    print(f"Quantized size:   {quantized_size:.2f} MB")
    print(f"Size reduction:   {reduction:.1f}%")
    print(f"\nQuantized model saved to: {output_path}")
    print(f"{'='*60}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Quantize GestureFlow model")
    parser.add_argument(
        "--language", "-l",
        type=str,
        default="en",
        help="Language code"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="quantized_model.pt",
        help="Output filename"
    )
    
    args = parser.parse_args()
    quantize_model(args.language, args.output)


if __name__ == "__main__":
    main()