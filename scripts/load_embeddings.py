#!/usr/bin/env python3
import os

from datasets import Dataset, load_dataset # pyright: ignore[reportUnknownVariableType, reportMissingTypeStubs]
import torch
from tqdm import tqdm
from typer import run

from lescdb.database import Database

def is_nvidia_gpu_available():
  """Check if an NVIDIA GPU is available and compatible with CUDA."""
  return torch.cuda.is_available()

def get_model_dimensions(model_path: str):
  """Extract input and output dimensions from the pre-trained model."""
  if not os.path.exists(model_path):
    print(f"Warning: Model file {model_path} not found.")
    return None, None
  
  try:
    # Load the model state dict
    state_dict = torch.load(model_path, map_location=torch.device('cpu')) # pyright: ignore[reportAny]
    
    # Extract dimensions from the first and last layers
    if "encoder.0.weight" in state_dict:
      input_dim = state_dict["encoder.0.weight"].shape[1] # pyright: ignore[reportAny]
      # Find the last layer's weight
      last_layer_keys = [k for k in state_dict.keys() if "weight" in k and "encoder" in k] # pyright: ignore[reportAny]
      last_layer_keys.sort(key=lambda x: int(x.split('.')[1])) # pyright: ignore[reportAny]
      last_layer_key = last_layer_keys[-1] # pyright: ignore[reportAny]
      latent_dim = state_dict[last_layer_key].shape[0] # pyright: ignore[reportAny]
      
      print(f"Detected model dimensions - Input: {input_dim}, Latent: {latent_dim}")
      return input_dim, latent_dim
  except Exception as e:
    print(f"Error extracting model dimensions: {e}")

  return None, None

def load_embeddings(dataset_name: str, save_path: str = "LESCdb", train_model: bool = False):
  # Get dimensions from the pre-trained model
  input_dim, latent_dim = get_model_dimensions("model.pth")
  
  # Automatically detect if we should use GPU
  use_gpu = is_nvidia_gpu_available()
  
  # Initialize the database with the auto-detected GPU setting and dimensions
  db = Database(path=save_path, use_gpu=use_gpu, input_dim=input_dim, latent_dim=latent_dim)
  
  print(f"Loading embeddings from Hugging Face dataset: {dataset_name}")
  print(f"GPU acceleration: {is_nvidia_gpu_available()}")
  
  dataset: Dataset = load_dataset(dataset_name, split="train") # pyright: ignore[reportAssignmentType]
  print(f"Successfully loaded dataset: {dataset_name}")

  # Process the dataset
  batch_count = 0
  total_count = 0
  batch_embeddings: list[list[float]] = []
  batch_texts: list[str] = []
  
  # Determine embedding dimension from first item
  embedding_dim = None
  
  # For non-streaming datasets, we can access the first item directly
  first_item: dict[str, list[float]] = dataset[0] # pyright: ignore[reportUnknownVariableType]
  embedding: list[float] = first_item["embedding"]
  embedding_dim = len(embedding)
  print(f"Detected embedding dimension: {embedding_dim}")
  
  # If the model input dimension doesn't match the embedding dimension, adapt
  if input_dim is not None and input_dim != embedding_dim:
    print(f"Warning: Model input dimension ({input_dim}) doesn't match embedding dimension ({embedding_dim}).")
    print("Will use the embeddings directly without the pre-trained model.")
    db.input_dim = embedding_dim
    db.latent_dim = embedding_dim
  
  # Process items
  for item in tqdm(dataset, desc="Processing embeddings"): # pyright: ignore[reportUnknownVariableType]
    embedding = item["embedding"] # pyright: ignore[reportUnknownVariableType, reportArgumentType, reportCallIssue]
    text: str = item["text"] # pyright: ignore[reportUnknownVariableType, reportArgumentType, reportCallIssue]

    # Add to batch
    batch_embeddings.append(embedding) # pyright: ignore[reportUnknownArgumentType]
    batch_texts.append(text) # pyright: ignore[reportUnknownArgumentType]
    
    # Process batch when it reaches the specified size
    if len(batch_embeddings) >= 1000:
      process_batch(db, batch_embeddings, batch_texts)
      batch_count += 1
      total_count += len(batch_embeddings)
      batch_embeddings = []
      batch_texts = []
  
  # Process any remaining items
  if batch_embeddings:
    process_batch(db, batch_embeddings, batch_texts)
    total_count += len(batch_embeddings)
    print(f"Processed final batch, total items: {total_count}")
  
  # Build the database index
  print("Building database index...")
  db.build(train_model=train_model)
  
  # Save the database
  print(f"Saving database to {save_path}...")
  db.save()
  
  print(f"Successfully loaded {total_count} embeddings from Hugging Face dataset")

def process_batch(db: Database, embeddings: list[list[float]], texts: list[str]):
  for embedding, text in zip(embeddings, texts):
    # Store the embedding as key and a dictionary with embedding and text as value
    value_dict = {
      "embedding": embedding,
      "text": text
    }
    db.insert_staging(embedding, value_dict)

if __name__ == "__main__":
  run(load_embeddings)
