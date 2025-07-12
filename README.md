# LESCdb

LESCdb is a lightweight embedded similarity-based content database with GPU acceleration support. It uses FAISS for efficient similarity search and LMDB for persistent storage.

## Features

- Fast similarity search using FAISS
- GPU acceleration support
- Persistent storage with LMDB
- Serialization with MessagePack
- Save and load functionality

## Installation

This project uses [Pixi](https://pixi.sh/) for dependency management.

```bash
# Clone the repository
git clone https://github.com/yourusername/lescdb.git
cd lescdb

# Install dependencies with Pixi
pixi install
```

## Usage

### FAISS-based Similarity Database

```python
from main import Database

# Create a database instance
db = Database(path="my_database", use_gpu=True)  # Set use_gpu=False to use CPU only

# Insert data into staging
key = [0.1, 0.2, 0.3, ...]  # 128-dimensional vector
value = [0.4, 0.5, 0.6, ...]  # 128-dimensional vector
db.insert_staging(key, value)

# Build the database (processes staging data)
db.build()

# Search for similar items
results = db.search([0.1, 0.2, 0.3, ...], limit=10)
for distance, item in results:
    print(f"Distance: {distance}, Item: {item}")

# Save the database (automatically called after build)
db.save()
```

### B-tree Database

```python
from btree_db import BTreeDatabase

# Create a database instance
db = BTreeDatabase[str, dict](path="my_btree_db", order=5)

# Insert data
db.insert("user1", {"name": "Alice", "age": 30})
db.insert("user2", {"name": "Bob", "age": 25})

# Search for data
result = db.search("user1")
if result:
    print(f"Found: {result}")

# Delete data
db.delete("user2")

# Save the database
db.save()

# Load the database in a new instance
new_db = BTreeDatabase[str, dict](path="my_btree_db")
```

### Evaluation and Benchmarking

The project includes scripts for evaluation and benchmarking:

#### FAISS Database Evaluation

```bash
# Run evaluation with default settings
python evaluate.py

# Run with GPU acceleration
python evaluate.py --use-gpu

# Customize parameters
python evaluate.py --use-gpu --vector-count 50000 --query-count 200 --path custom_db
```

#### FAISS Database Benchmark (CPU vs GPU)

```bash
# Run benchmark with default settings
python benchmark.py

# Skip CPU or GPU tests
python benchmark.py --skip-cpu
python benchmark.py --skip-gpu

# Customize parameters
python benchmark.py --vector-count 50000 --query-count 200
```

#### B-tree Database Evaluation

```bash
# Run evaluation with default settings
python btree_evaluate.py

# Customize parameters
python btree_evaluate.py --order 20 --count 50000 --search-iterations 2000 --delete-percentage 0.5
```

## Performance Considerations

### FAISS Database
- GPU acceleration provides significant speedup for large datasets
- The database automatically falls back to CPU if GPU is not available
- Save/load operations always use the CPU index for compatibility

### B-tree Database
- B-tree order affects performance and memory usage
- Higher order values reduce tree height but increase node size
- Optimal order depends on your specific use case and data

## License

See the [LICENSE](LICENSE) file for details.
