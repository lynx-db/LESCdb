#!/usr/bin/env python3
from lescdb.database import Database
import numpy as np
import os
import shutil

def cleanup_files(path):
    """Remove test database files."""
    for suffix in ["_main.db", "_staging.db", "_evals.db", "_index.faiss"]:
        file_path = f"{path}{suffix}"
        if os.path.exists(file_path):
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)

def main():
    # Test parameters
    path = "test_db"
    use_gpu = False
    
    # Clean up any existing files
    cleanup_files(path)
    
    print("Creating database...")
    db = Database(path=path, use_gpu=use_gpu)
    
    print("Generating test vectors...")
    # Generate a few test vectors
    vectors = np.random.rand(10, 128).astype(np.float32).tolist()
    
    print("Inserting vectors into staging...")
    # Insert vectors into staging
    for i, vec in enumerate(vectors):
        key_vec = vec.copy()
        key_vec[0] = i  # Make key unique
        db.insert_staging(key_vec, vec)
    
    print("Building database...")
    # Build the database
    db.build()
    
    print("Testing search...")
    # Test search
    results = db.search(vectors[0], limit=3)
    print(f"Search results: {len(results)} items found")
    for i, (distance, value) in enumerate(results):
        print(f"  Result {i}: distance={distance:.4f}")
    
    print("Saving database...")
    # Save the database
    db.save()
    
    print("Loading database...")
    # Load the database
    new_db = Database(path=path, use_gpu=use_gpu)
    
    print("Testing search on loaded database...")
    # Test search on loaded database
    results = new_db.search(vectors[0], limit=3)
    print(f"Search results: {len(results)} items found")
    for i, (distance, value) in enumerate(results):
        print(f"  Result {i}: distance={distance:.4f}")
    
    print("Test completed successfully!")

if __name__ == "__main__":
    main() 
