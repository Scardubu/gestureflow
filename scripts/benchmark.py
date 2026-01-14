“””
Benchmark inference performance for GestureFlow models.
“””
import torch
import time
import json
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
import argparse

sys.path.append(str(Path(**file**).parent.parent))

from src.models.lstm_model import create_model
from src.config import CHECKPOINTS_DIR, MODEL_CONFIG

class PerformanceBenchmark:
“”“Benchmark model inference performance.”””

```
def __init__(self, model_path: Path, vocab_size: int, device: str = "cpu"):
    """
    Initialize benchmark.
    
    Args:
        model_path: Path to model checkpoint
        vocab_size: Size of vocabulary
        device: Device to run on
    """
    self.device = device
    
    # Load model
    self.model = create_model(vocab_size=vocab_size, model_type="lstm")
    checkpoint = torch.load(model_path, map_location=device)
    self.model.load_state_dict(checkpoint['model_state_dict'])
    self.model.eval()
    self.model.to(device)
    
    print(f"Model loaded on {device}")
    print(f"Parameters: {self.model.count_parameters():,}")
    print(f"Size: {self.model.get_model_size():.2f} MB")

def generate_random_batch(
    self,
    batch_size: int = 1,
    sequence_length: int = 50
) -> torch.Tensor:
    """Generate random input batch for testing."""
    return torch.randn(batch_size, sequence_length, 3).to(self.device)

def benchmark_latency(
    self,
    num_runs: int = 1000,
    batch_size: int = 1,
    warmup: int = 100
) -> dict:
    """
    Benchmark inference latency.
    
    Args:
        num_runs: Number of inference runs
        batch_size: Batch size for inference
        warmup: Number of warmup runs
        
    Returns:
        Dictionary with latency statistics
    """
    print(f"\nBenchmarking latency (batch_size={batch_size}, runs={num_runs})...")
    
    # Warmup
    for _ in range(warmup):
        input_batch = self.generate_random_batch(batch_size)
        with torch.no_grad():
            _ = self.model(input_batch)
    
    # Benchmark
    latencies = []
    
    for _ in tqdm(range(num_runs), desc="Running inference"):
        input_batch = self.generate_random_batch(batch_size)
        
        start_time = time.perf_counter()
        with torch.no_grad():
            _ = self.model(input_batch)
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
    
    # Calculate statistics
    stats = {
        'mean_ms': np.mean(latencies),
        'median_ms': np.median(latencies),
        'std_ms': np.std(latencies),
        'min_ms': np.min(latencies),
        'max_ms': np.max(latencies),
        'p50_ms': np.percentile(latencies, 50),
        'p90_ms': np.percentile(latencies, 90),
        'p95_ms': np.percentile(latencies, 95),
        'p99_ms': np.percentile(latencies, 99),
    }
    
    return stats

def benchmark_throughput(
    self,
    duration_seconds: int = 10,
    batch_size: int = 32
) -> dict:
    """
    Benchmark inference throughput.
    
    Args:
        duration_seconds: How long to run benchmark
        batch_size: Batch size for inference
        
    Returns:
        Dictionary with throughput statistics
    """
    print(f"\nBenchmarking throughput (batch_size={batch_size}, duration={duration_seconds}s)...")
    
    start_time = time.time()
    total_samples = 0
    
    with tqdm(desc="Processing samples") as pbar:
        while time.time() - start_time < duration_seconds:
            input_batch = self.generate_random_batch(batch_size)
            
            with torch.no_grad():
                _ = self.model(input_batch)
            
            total_samples += batch_size
            pbar.update(batch_size)
    
    elapsed_time = time.time() - start_time
    
    stats = {
        'total_samples': total_samples,
        'elapsed_seconds': elapsed_time,
        'samples_per_second': total_samples / elapsed_time,
        'batch_size': batch_size
    }
    
    return stats

def benchmark_memory(self) -> dict:
    """
    Benchmark memory usage.
    
    Returns:
        Dictionary with memory statistics
    """
    print("\nBenchmarking memory usage...")
    
    if self.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        input_batch = self.generate_random_batch(batch_size=1)
        
        with torch.no_grad():
            _ = self.model(input_batch)
        
        stats = {
            'allocated_mb': torch.cuda.memory_allocated() / 1024**2,
            'reserved_mb': torch.cuda.memory_reserved() / 1024**2,
            'peak_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2,
        }
    else:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024**2
        
        input_batch = self.generate_random_batch(batch_size=1)
        
        with torch.no_grad():
            _ = self.model(input_batch)
        
        mem_after = process.memory_info().rss / 1024**2
        
        stats = {
            'memory_before_mb': mem_before,
            'memory_after_mb': mem_after,
            'memory_used_mb': mem_after - mem_before,
        }
    
    return stats

def run_full_benchmark(
    self,
    output_file: Path = None
) -> dict:
    """
    Run complete benchmark suite.
    
    Args:
        output_file: Optional path to save results
        
    Returns:
        Complete benchmark results
    """
    results = {
        'model_info': {
            'parameters': self.model.count_parameters(),
            'size_mb': self.model.get_model_size(),
            'device': self.device
        },
        'latency': {},
        'throughput': {},
        'memory': {}
    }
    
    # Latency benchmarks
    for batch_size in [1, 8, 16, 32]:
        print(f"\n{'='*60}")
        stats = self.benchmark_latency(
            num_runs=500,
            batch_size=batch_size,
            warmup=50
        )
        results['latency'][f'batch_{batch_size}'] = stats
        
        print(f"\nLatency Results (batch_size={batch_size}):")
        print(f"  Mean: {stats['mean_ms']:.2f} ms")
        print(f"  Median: {stats['median_ms']:.2f} ms")
        print(f"  P95: {stats['p95_ms']:.2f} ms")
        print(f"  P99: {stats['p99_ms']:.2f} ms")
    
    # Throughput benchmark
    print(f"\n{'='*60}")
    throughput = self.benchmark_throughput(duration_seconds=10, batch_size=32)
    results['throughput'] = throughput
    
    print(f"\nThroughput Results:")
    print(f"  Samples/sec: {throughput['samples_per_second']:.2f}")
    print(f"  Total samples: {throughput['total_samples']}")
    
    # Memory benchmark
    print(f"\n{'='*60}")
    memory = self.benchmark_memory()
    results['memory'] = memory
    
    print(f"\nMemory Results:")
    for key, value in memory.items():
        print(f"  {key}: {value:.2f}")
    
    # Save results
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n{'='*60}")
        print(f"Results saved to {output_file}")
    
    return results
```

def main():
“”“Run benchmark.”””
parser = argparse.ArgumentParser(description=“Benchmark GestureFlow model”)
parser.add_argument(
“–language”, “-l”,
type=str,
default=“en”,
help=“Language model to benchmark”
)
parser.add_argument(
“–device”,
type=str,
default=“cpu”,
choices=[“cpu”, “cuda”],
help=“Device to run benchmark on”
)
parser.add_argument(
“–output”, “-o”,
type=str,
default=None,
help=“Output file for results”
)

```
args = parser.parse_args()

# Load model
model_dir = CHECKPOINTS_DIR / args.language
model_path = model_dir / "best_model.pt"
vocab_path = model_dir / "vocabulary.json"

if not model_path.exists():
    print(f"Error: Model not found at {model_path}")
    print(f"Train a model first: python scripts/train_model.py --language {args.language}")
    return

# Load vocabulary to get size
with open(vocab_path, 'r') as f:
    vocab_data = json.load(f)
vocab_size = vocab_data['vocab_size']

# Create benchmark
benchmark = PerformanceBenchmark(
    model_path=model_path,
    vocab_size=vocab_size,
    device=args.device
)

# Set output file
if args.output:
    output_file = Path(args.output)
else:
    output_file = model_dir / f"benchmark_results_{args.device}.json"

# Run benchmark
print("="*60)
print("GestureFlow Performance Benchmark")
print("="*60)

results = benchmark.run_full_benchmark(output_file=output_file)

# Print summary
print("\n" + "="*60)
print("BENCHMARK SUMMARY")
print("="*60)
print(f"\nModel: {args.language}")
print(f"Device: {args.device}")
print(f"Parameters: {results['model_info']['parameters']:,}")
print(f"Size: {results['model_info']['size_mb']:.2f} MB")
print(f"\nSingle Sample Latency: {results['latency']['batch_1']['mean_ms']:.2f} ms")
print(f"Throughput (batch=32): {results['throughput']['samples_per_second']:.2f} samples/sec")
print("="*60)
```

if **name** == “**main**”:
main()
