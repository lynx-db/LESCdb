from typing import final, Dict, Tuple, Optional, Any, Union

from numpy import float32
from torch import Tensor
from faiss import IndexFlatL2, write_index, read_index, GpuIndexFlatL2, StandardGpuResources, index_cpu_to_gpu, index_gpu_to_cpu # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
import os
import torch
import torch.nn as nn
from distance_preservation_encoder.model import DPEncoder
from distance_preservation_encoder.loss import DPLoss
from btree_db import BTreeDatabase

# Define a type for our database values which can be either a list of floats or a dict with embedding and text
DBValue = Union[list[float], Dict[str, Any]]

@final
class Database:
  def __init__(self, 
               path: str = "LESCdb", 
               use_gpu: bool = True, 
               input_dim: Optional[int] = None, 
               latent_dim: Optional[int] = None,
               hidden_dims: list[int] = [768]):
    self.path = path
    # Replace LMDB with BTreeDatabase
    self.database: BTreeDatabase[str, DBValue] = BTreeDatabase[str, DBValue](path=f"{path}_main")
    self.staging: BTreeDatabase[str, DBValue] = BTreeDatabase[str, DBValue](path=f"{path}_staging")
    
    # GPU setup
    self.use_gpu = use_gpu
    self.gpu_resources = None
    if self.use_gpu:
      try:
        self.gpu_resources = StandardGpuResources()
      except Exception:
        self.use_gpu = False
        print("GPU not available, falling back to CPU")
    
    # Initialize dimensions (configurable or will be inferred from data)
    self.input_dim = input_dim  # Can be set or inferred from first data
    self.hidden_dims = hidden_dims
    self.latent_dim = latent_dim  # Can be set or inferred from first data
    
    # Default dimension for initial index creation
    self.default_dim = self.latent_dim if self.latent_dim is not None else 128
    
    # Create a default index (will be recreated with proper dimensions when data is available)
    self.index = IndexFlatL2(self.default_dim)
    self.gpu_index: GpuIndexFlatL2 | None = None
    if self.use_gpu and self.gpu_resources is not None:
      self.gpu_index = index_cpu_to_gpu(self.gpu_resources, 0, self.index) # pyright: ignore[reportUnknownMemberType]
    
    self.evaluations: list[list[float]] = []
    
    # Initialize DPEncoder model
    self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    self.model = None
    
    self._load()

  def _insert_db(self, db: BTreeDatabase[str, DBValue], key: list[float], value: DBValue):
    # Convert list to string for B-tree key
    key_str = str(key)
    db.insert(key_str, value)

  def _select_db(self, db: BTreeDatabase[str, DBValue], key: list[float]) -> DBValue:
    # Convert list to string for B-tree key
    key_str = str(key)
    result = db.search(key_str)
    if result is None:
      raise KeyError(f"Key not found: {key}")
    return result

  def insert_staging(self, key: list[float], value: DBValue):
    self._insert_db(self.staging, key, value)

  def _train(self, original: Tensor, data: Tensor) -> Tensor:
    """
    Train the distance preservation encoder using the provided data.
    
    Args:
        original: Original input data (not used in training)
        data: Input data to train the model on
        
    Returns:
        The encoded data after training
    """
    # Infer dimensions from data if not already set
    if self.input_dim is None:
      self.input_dim = data.shape[1]
      print(f"Inferred input dimension from data: {self.input_dim}")
    elif data.shape[1] != self.input_dim:
      print(f"Warning: Input data dimension ({data.shape[1]}) doesn't match configured dimension ({self.input_dim})")
      self.input_dim = data.shape[1]  # Adjust to actual data dimension
    
    # Set latent dimension if not already set
    if self.latent_dim is None:
      # Set latent dimension based on input dimension
      # Use a reasonable ratio or fixed size based on your needs
      self.latent_dim = min(512, max(128, self.input_dim // 3))
      print(f"Inferred latent dimension: {self.latent_dim}")
    
    print(f"Using dimensions - Input: {self.input_dim}, Hidden: {self.hidden_dims}, Latent: {self.latent_dim}")
    
    # Initialize model
    self.model = DPEncoder(self.input_dim, self.hidden_dims, self.latent_dim).to(self.device)
    
    # Training parameters
    batch_size = min(64, len(data))
    epochs = 100
    k = 5
    lambda_rank = 1.0
    lambda_pairdist = 0.3
    lr = 1e-3
    
    # Initialize loss and optimizer
    criterion = DPLoss(k=k, lambda_rank=lambda_rank, lambda_pairdist=lambda_pairdist).to(self.device)
    optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
    
    # Move data to device
    data_device = data.to(self.device)
    
    # Training loop
    for epoch in range(epochs):
      self.model.train()
      
      # Process in batches if data is large
      total_loss = 0.0
      num_batches = (len(data) + batch_size - 1) // batch_size
      
      for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(data))
        batch_data = data_device[start_idx:end_idx]
        
        # Forward pass
        encoded = self.model(batch_data)
        loss, rank_loss, pairdist_loss = criterion(batch_data, encoded)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(batch_data)
      
      # Print progress every 10 epochs
      if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(data):.4f}")
    
    # Save the trained model
    model_path = "model.pth"
    torch.save(self.model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Return encoded data
    self.model.eval()
    with torch.no_grad():
      encoded_data = self.model(data_device)
    
    return encoded_data.cpu()

  def _evaluate(self, data: Tensor) -> Tensor:
    """
    Encode the input data using the trained model.
    
    Args:
        data: Input data to encode
        
    Returns:
        The encoded data
    """
    # Check if model exists
    if self.model is None:
      # Try to load the model
      model_path = "model.pth"
      if os.path.exists(model_path):
        # Handle input dimension
        if self.input_dim is None:
          self.input_dim = data.shape[1]
          print(f"Inferred input dimension from data: {self.input_dim}")
        elif data.shape[1] != self.input_dim:
          print(f"Warning: Input data dimension ({data.shape[1]}) doesn't match configured dimension ({self.input_dim})")
          self.input_dim = data.shape[1]  # Adjust to actual data dimension
        
        # Handle latent dimension
        if self.latent_dim is None:
          try:
            # Try to infer from saved model
            state_dict = torch.load(model_path, map_location=self.device)
            # Extract dimensions from the model
            if isinstance(state_dict, dict) and "encoder.0.weight" in state_dict:
              # Get input dimension from the first layer weight
              model_input_dim = state_dict["encoder.0.weight"].shape[1]
              # Find the last layer's weight
              last_layer_keys = [k for k in state_dict.keys() if "weight" in k and "encoder" in k]
              last_layer_keys.sort(key=lambda x: int(x.split('.')[1]))
              last_layer_key = last_layer_keys[-1]
              model_latent_dim = state_dict[last_layer_key].shape[0]
              
              # Check if model dimensions match our data
              if self.input_dim != model_input_dim:
                print(f"Warning: Model input dimension ({model_input_dim}) doesn't match data dimension ({self.input_dim})")
                print("Using data dimensions instead of loading the model.")
                self.latent_dim = self.input_dim  # Use same dimension for output
                return data  # Return data as is without processing through model
              else:
                self.latent_dim = model_latent_dim
                print(f"Inferred latent dimension from model: {self.latent_dim}")
            else:
              # Default to a reasonable size
              self.latent_dim = min(512, max(128, self.input_dim // 3))
              print(f"Using default latent dimension: {self.latent_dim}")
          except Exception as e:
            # If we can't infer, use a reasonable default
            self.latent_dim = min(512, max(128, self.input_dim // 3))
            print(f"Error loading model for dimension inference: {e}")
            print(f"Using default latent dimension: {self.latent_dim}")
        
        # Initialize and load model
        print(f"Using dimensions - Input: {self.input_dim}, Hidden: {self.hidden_dims}, Latent: {self.latent_dim}")
        self.model = DPEncoder(self.input_dim, self.hidden_dims, self.latent_dim).to(self.device)
        
        try:
          self.model.load_state_dict(torch.load(model_path, map_location=self.device))
          print(f"Model loaded from {model_path}")
        except RuntimeError as e:
          print(f"Error loading model: {e}")
          print("Using data without model processing.")
          self.model = None
          return data  # Return data as is without processing through model
      else:
        # No model available, ensure dimensions are set and return the data as is
        if self.input_dim is None:
          self.input_dim = data.shape[1]
          print(f"Inferred input dimension from data: {self.input_dim}")
        
        if self.latent_dim is None:
          self.latent_dim = min(512, max(128, self.input_dim // 3))
          print(f"Using default latent dimension: {self.latent_dim}")
        
        print(f"Warning: No model available for encoding. Using raw data with dimensions - Input: {self.input_dim}, Latent: {self.latent_dim}")
        return data
    
    # Set model to evaluation mode
    self.model.eval()
    
    # Move data to device
    data_device = data.to(self.device)
    
    # Encode data
    with torch.no_grad():
      encoded_data = self.model(data_device)
    
    return encoded_data.cpu()

  def build(self, train_model: bool = True):
    """
    Build the database from staging data.
    
    Args:
        train_model: Whether to train the model or use the existing one.
                    If False and no model exists, will use raw data.
    """
    # Gather all data from staging as the actual data to build the index
    all_keys = []
    all_values = []
    all_original_data = []
    
    # Get all keys from staging using a custom approach
    staging_keys = self._get_staging_keys()
    
    for key_str in staging_keys:
      value = self.staging.search(key_str)
      if value is not None:
        all_keys.append(eval(key_str))  # Convert string back to list[float]
        
        # Handle both old format (list[float]) and new format (dict with embedding and text)
        if isinstance(value, dict) and "embedding" in value:
          all_values.append(value["embedding"])
          all_original_data.append(value)  # Store the complete dict
        else:
          all_values.append(value)  # Old format - value is the embedding
          all_original_data.append(value)  # Store the raw value for backward compatibility
    
    if not all_values:
      print("Warning: No data to build the database")
      return
    
    # Convert lists to tensors
    import numpy as np
    data = Tensor(np.array(all_values, dtype=np.float32))
    original = Tensor(np.array(all_keys, dtype=np.float32))
    
    # Train the model if requested, otherwise use existing model or raw data
    if train_model:
      _ = self._train(original, data)
    
    eval_result = self._evaluate(data)
    
    # Clear existing evaluations and reset index
    self.evaluations = []
    
    # Ensure latent_dim is set (should be set during training, but just in case)
    if self.latent_dim is None:
      # Default to a reasonable size if not set yet
      self.latent_dim = min(512, max(128, self.input_dim // 3 if self.input_dim else 128))
      print(f"Using default latent dimension for index: {self.latent_dim}")
    
    # Create index with proper dimension
    self.index = IndexFlatL2(self.latent_dim)
    if self.use_gpu and self.gpu_resources is not None:
      self.gpu_index = index_cpu_to_gpu(self.gpu_resources, 0, self.index) # pyright: ignore[reportUnknownMemberType]
    
    le = len(eval_result)
    if le == 0:
      print("Warning: No data to build the database")
      return
      
    for i in range(le):
      current_data = all_original_data[i]  # Use the original data (with text if available)
      current_eval: list[float] = eval_result[i].tolist() # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
      self._insert_db(self.database, current_eval, current_data)
      
      # Add to the appropriate index - ensure proper shape for FAISS
      eval_numpy = eval_result[i].numpy().reshape(1, -1)  # Reshape to 2D array (1, dim)
      
      if self.use_gpu and self.gpu_index is not None:
        self.gpu_index.add(eval_numpy) # pyright: ignore[reportCallIssue, reportUnknownMemberType]
      else:
        self.index.add(eval_numpy) # pyright: ignore[reportCallIssue, reportUnknownMemberType]
      
      self.evaluations.append(current_eval)
    
    self.staging.clear()
    self.save()

  def search(self, value: list[float], limit: int = 10) -> list[tuple[float, DBValue]]:
    results: list[tuple[float, DBValue]] = []
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
      result_value = self._select_db(self.database, self.evaluations[indice]) # pyright: ignore[reportUnknownArgumentType]
      results.append((float(distance), result_value))
    print(results)
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
    model_path = "model.pth"
    
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
      for i in range(100000000):  # Reasonable upper bound
        if evals_db.search(str(i)) is None:
          max_index = i - 1
          break
      
      # Load all evaluations in order
      for i in range(max_index + 1):
        eval_data = evals_db.search(str(i))
        if eval_data is not None:
          self.evaluations.append(eval_data)
    
    # Try to load the model if it exists
    model_path = "model.pth"
    if os.path.exists(model_path) and self.model is None:
      # We'll initialize the model later when we know the input dimension
      pass

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
