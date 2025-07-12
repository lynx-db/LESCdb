#!/usr/bin/env python3
import argparse
import time
import numpy as np
from main import Database

def generate_random_vectors(count: int, dim: int = 128) -> list[list[float]]:
    """Generate random vectors for testing."""
    return np.random.rand(count, dim).astype(np.float32).tolist() # pyright: ignore[reportAny]

def build_database(use_gpu: bool, vector_count: int, path: str) -> tuple[Database, float]:
    """Build a database with random vectors and measure the time taken."""
    # Clean up any existing database files
    import os
    import shutil
    for suffix in ["_main.db", "_staging.db", "_evals.db", "_index.faiss"]:
        file_path = f"{path}{suffix}"
        if os.path.exists(file_path):
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)
    
    db = Database(path=path, use_gpu=use_gpu)
    
    # Generate random test data
    vectors = generate_random_vectors(vector_count)
    
    # Measure build time
    start_time = time.time()
    
    # Insert vectors into staging
    for i, vec in enumerate(vectors):
        # Use a unique key for each vector to avoid collisions
        key_vec = vec.copy()
        key_vec[0] = i  # Modify first element to make key unique
        db.insert_staging(key_vec, vec)
    
    # Build the database
    db.build()
    
    elapsed_time = time.time() - start_time
    return db, elapsed_time

def test_search(db: Database, query_count: int) -> tuple[float, float]:
    """Test search performance and measure time."""
    # Generate random queries
    queries = generate_random_vectors(query_count)
    
    # Measure search time
    start_time = time.time()
    results = []
    
    for query in queries:
        result = db.search(query, limit=10)
        results.append(result)
    
    search_time = time.time() - start_time
    avg_time = search_time / query_count
    
    return search_time, avg_time

def main():
    parser = argparse.ArgumentParser(description="Evaluate LESCdb performance")
    _ = parser.add_argument("--use-gpu", action="store_true", help="Use GPU acceleration")
    _ = parser.add_argument("--vector-count", type=int, default=10000, help="Number of vectors to insert")
    _ = parser.add_argument("--query-count", type=int, default=100, help="Number of search queries to perform")
    _ = parser.add_argument("--path", type=str, default="LESCdb_test", help="Database path")
    args = parser.parse_args()
    use_gpu: bool = args.use_gpu
    vector_count: int = args.vector_count
    query_count: int = args.query_count
    path: str = args.path
    
    print(f"Evaluating LESCdb with {'GPU' if use_gpu else 'CPU'} mode")
    print(f"Building database with {vector_count} vectors...")
    
    # Build database and measure time
    db, build_time = build_database(use_gpu, vector_count, path)
    print(f"Database built in {build_time:.4f} seconds")
    
    # Test search performance
    print(f"Testing search with {query_count} queries...")
    total_search_time, avg_search_time = test_search(db, query_count)
    print(f"Total search time: {total_search_time:.4f} seconds")
    print(f"Average search time: {avg_search_time:.4f} seconds per query")
    
    # Save and load test
    print("Testing save and load functionality...")
    save_start = time.time()
    db.save()
    save_time = time.time() - save_start
    print(f"Save completed in {save_time:.4f} seconds")
    
    # Load the database again
    load_start = time.time()
    new_db = Database(path=path, use_gpu=use_gpu)
    load_time = time.time() - load_start
    print(f"Load completed in {load_time:.4f} seconds")
    
    # Test search on loaded database
    print("Testing search on loaded database...")
    total_search_time_loaded, avg_search_time_loaded = test_search(new_db, query_count)
    print(f"Total search time (loaded): {total_search_time_loaded:.4f} seconds")
    print(f"Average search time (loaded): {avg_search_time_loaded:.4f} seconds per query")
    
    # Print summary
    print("\nPerformance Summary:")
    print(f"{'=' * 50}")
    print(f"Mode: {'GPU' if use_gpu else 'CPU'}")
    print(f"Vector count: {vector_count}")
    print(f"Query count: {query_count}")
    print(f"Build time: {build_time:.4f} seconds")
    print(f"Save time: {save_time:.4f} seconds")
    print(f"Load time: {load_time:.4f} seconds")
    print(f"Average search time (original): {avg_search_time:.4f} seconds/query")
    print(f"Average search time (loaded): {avg_search_time_loaded:.4f} seconds/query")
    print(f"{'=' * 50}")

if __name__ == "__main__":
    main() 