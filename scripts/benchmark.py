import subprocess
import argparse
import os
import time
import shutil

def run_benchmark(vector_count: int, query_count: int, use_gpu: bool) -> None:
    """Run a benchmark with specified parameters."""
    mode = "GPU" if use_gpu else "CPU"
    path = f"LESCdb_benchmark_{mode.lower()}"
    
    # Clean up previous benchmark data if it exists
    for file_pattern in [f"{path}.lmdb", f"{path}_staging.lmdb", f"{path}_index.faiss", f"{path}_evaluations.msgpack"]:
        if os.path.exists(file_pattern):
            if os.path.isdir(file_pattern):
                shutil.rmtree(file_pattern)
            else:
                os.remove(file_pattern)
    
    # Run the evaluation script
    cmd = [
        "python", "scripts/evaluate.py",
        "--vector-count", str(vector_count),
        "--query-count", str(query_count),
        "--path", path
    ]
    
    if use_gpu:
        cmd.append("--use-gpu")
    
    print(f"\nRunning {mode} benchmark with {vector_count} vectors and {query_count} queries...")
    start_time = time.time()
    subprocess.run(cmd)
    total_time = time.time() - start_time
    print(f"Total {mode} benchmark time: {total_time:.4f} seconds")

def main():
    parser = argparse.ArgumentParser(description="Compare CPU vs GPU performance for LESCdb")
    parser.add_argument("--vector-count", type=int, default=10000, help="Number of vectors to insert")
    parser.add_argument("--query-count", type=int, default=100, help="Number of search queries to perform")
    parser.add_argument("--skip-cpu", action="store_true", help="Skip CPU benchmark")
    parser.add_argument("--skip-gpu", action="store_true", help="Skip GPU benchmark")
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"LESCdb Performance Benchmark")
    print(f"Vectors: {args.vector_count}, Queries: {args.query_count}")
    print("=" * 70)
    
    # Run CPU benchmark
    if not args.skip_cpu:
        run_benchmark(args.vector_count, args.query_count, use_gpu=False)
    
    # Run GPU benchmark
    if not args.skip_gpu:
        run_benchmark(args.vector_count, args.query_count, use_gpu=True)
    
    print("\nBenchmark complete!")

if __name__ == "__main__":
    main() 
