from typing import final, Dict, Tuple, Optional, Any

from numpy import float32
from torch import Tensor
from faiss import IndexFlatL2, write_index, read_index, GpuIndexFlatL2, StandardGpuResources, index_cpu_to_gpu, index_gpu_to_cpu # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
import os
from btree_db import BTreeDatabase

@final
class Database:
  def __init__(self, path: str = "LESCdb", use_gpu: bool = True):
    self.path = path
    # Replace LMDB with BTreeDatabase
    self.database: BTreeDatabase[str, list[float]] = BTreeDatabase[str, list[float]](path=f"{path}_main")
    self.staging: BTreeDatabase[str, list[float]] = BTreeDatabase[str, list[float]](path=f"{path}_staging")
    
    # GPU setup
    self.use_gpu = use_gpu
    self.gpu_resources = None
    if self.use_gpu:
      try:
        self.gpu_resources = StandardGpuResources()
      except Exception:
        self.use_gpu = False
        print("GPU not available, falling back to CPU")
    
    # Create index
    self.index = IndexFlatL2(128)
    self.gpu_index: GpuIndexFlatL2 | None = None
    if self.use_gpu and self.gpu_resources is not None:
      self.gpu_index = index_cpu_to_gpu(self.gpu_resources, 0, self.index) # pyright: ignore[reportUnknownMemberType]
    
    self.evaluations: list[list[float]] = []
    self._load()

  def _insert_db(self, db: BTreeDatabase[str, list[float]], key: list[float], value: list[float]):
    # Convert list to string for B-tree key
    key_str = str(key)
    db.insert(key_str, value)

  def _select_db(self, db: BTreeDatabase[str, list[float]], key: list[float]) -> list[float]:
    # Convert list to string for B-tree key
    key_str = str(key)
    result = db.search(key_str)
    if result is None:
      raise KeyError(f"Key not found: {key}")
    return result

  def insert_staging(self, key: list[float], value: list[float]):
    self._insert_db(self.staging, key, value)

  def _train(self, original: Tensor, data: Tensor) -> Tensor:
    return data

  def _evaluate(self, data: Tensor) -> Tensor:
    return data

  def build(self):
    # Gather all data from staging as the actual data to build the index
    all_keys = []
    all_values = []
    
    # Get all keys from staging using a custom approach
    staging_keys = self._get_staging_keys()
    
    for key_str in staging_keys:
      value = self.staging.search(key_str)
      if value is not None:
        all_keys.append(eval(key_str))  # Convert string back to list[float]
        all_values.append(value)
    
    if not all_values:
      print("Warning: No data to build the database")
      return
    
    # Convert lists to tensors
    import numpy as np
    data = Tensor(np.array(all_values, dtype=np.float32))
    original = Tensor(np.array(all_keys, dtype=np.float32))
    
    _ = self._train(original, data)
    eval_result = self._evaluate(data)
    
    # Clear existing evaluations and reset index
    self.evaluations = []
    self.index = IndexFlatL2(128)
    if self.use_gpu and self.gpu_resources is not None:
      self.gpu_index = index_cpu_to_gpu(self.gpu_resources, 0, self.index) # pyright: ignore[reportUnknownMemberType]
    
    le = len(eval_result)
    if le == 0:
      print("Warning: No data to build the database")
      return
      
    for i in range(le):
      current_data: list[float] = data[i].tolist() # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
      current_eval: list[float] = eval_result[i].tolist() # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
      self._insert_db(self.database, current_eval, current_data)
      
      # Add to the appropriate index - ensure proper shape for FAISS
      eval_numpy = eval_result[i].numpy().reshape(1, -1)  # Reshape to 2D array (1, dim)
      
      if self.use_gpu and self.gpu_index is not None:
        self.gpu_index.add(eval_numpy) # pyright: ignore[reportCallIssue, reportUnknownMemberType]
      else:
        self.index.add(eval_numpy) # pyright: ignore[reportCallIssue, reportUnknownMemberType]
      
      self.evaluations.append(current_eval)
    
    self.save()

  def search(self, value: list[float], limit: int = 10) -> list[tuple[float, list[float]]]:
    results: list[tuple[float, list[float]]] = []
    evaluation = self._evaluate(Tensor([value]))
    
    # Ensure proper shape for FAISS search
    eval_numpy = evaluation.numpy()
    
    # Use the appropriate index for search
    if self.use_gpu and self.gpu_index is not None:
      distances, indices = self.gpu_index.search(eval_numpy, limit) # pyright: ignore[reportCallIssue, reportUnknownMemberType, reportUnknownVariableType]
    else:
      distances, indices = self.index.search(eval_numpy, limit) # pyright: ignore[reportCallIssue, reportUnknownMemberType, reportUnknownVariableType]
    
    # Handle empty results
    if len(distances) == 0 or len(indices) == 0:
      return results
    
    for distance, indice in zip(distances[0], indices[0]): # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
      distance: float32
      indice: int
      # Skip invalid indices (that might occur if the index is not fully built)
      if indice < 0 or indice >= len(self.evaluations):
        continue
      try:
        result_value = self._select_db(self.database, self.evaluations[indice]) # pyright: ignore[reportUnknownArgumentType]
        results.append((float(distance), result_value))
      except KeyError:
        # Skip entries that don't exist in the database
        continue
    return results
  
  def save(self):
    """Save the index and evaluations to disk."""
    # If using GPU, copy the index back to CPU first
    if self.use_gpu and self.gpu_index is not None:
      self.index = index_gpu_to_cpu(self.gpu_index)
    
    # Save the index
    index_path = f"{self.path}_index.faiss"
    write_index(self.index, index_path)
    
    # Save evaluations to a separate B-tree database
    evals_db: BTreeDatabase[str, list[float]] = BTreeDatabase[str, list[float]](path=f"{self.path}_evals")
    for i, eval_data in enumerate(self.evaluations):
      evals_db.insert(str(i), eval_data)
    evals_db.save()
    
    # Save the main and staging databases
    self.database.save()
    self.staging.save()
  
  def _load(self):
    """Load the index and evaluations from disk if they exist."""
    index_path = f"{self.path}_index.faiss"
    evals_path = f"{self.path}_evals.db"
    
    # Load index if it exists
    if os.path.exists(index_path):
      self.index: IndexFlatL2 = read_index(index_path)
      
      # Move to GPU if needed
      if self.use_gpu and self.gpu_resources is not None:
        self.gpu_index = index_cpu_to_gpu(self.gpu_resources, 0, self.index) # pyright: ignore[reportUnknownMemberType]
    
    # Load evaluations if they exist
    if os.path.exists(evals_path):
      evals_db: BTreeDatabase[str, list[float]] = BTreeDatabase[str, list[float]](path=f"{self.path}_evals")
      self.evaluations = []
      
      # Find the maximum index to determine the number of evaluations
      max_index = -1
      for i in range(10000):  # Reasonable upper bound
        if evals_db.search(str(i)) is None:
          max_index = i - 1
          break
      
      # Load all evaluations in order
      for i in range(max_index + 1):
        eval_data = evals_db.search(str(i))
        if eval_data is not None:
          self.evaluations.append(eval_data)

  def _get_staging_keys(self) -> list[str]:
    """Get all keys from the staging database."""
    # Create a temporary file to store the keys
    import tempfile
    import pickle
    
    # Get the underlying BTree from the staging database
    btree = self.staging.btree
    
    # Collect all keys by traversing the BTree
    keys = []
    self._collect_keys_from_node(btree.root, keys)
    return keys
  
  def _collect_keys_from_node(self, node, keys: list[str]) -> None:
    """Recursively collect all keys from a node and its children."""
    if not node:
      return
    
    # For each key in this node
    for key in node.keys:
      keys.append(key)
    
    # If not a leaf, traverse all children
    if not node.leaf:
      for child in node.children:
        self._collect_keys_from_node(child, keys)
