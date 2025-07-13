#!/usr/bin/env python3
import json
import torch
import argparse
import os
import numpy as np
from tqdm import tqdm
from main import Database
from distance_preservation_encoder.model import DPEncoder

def is_nvidia_gpu_available():
    """Check if an NVIDIA GPU is available and compatible with CUDA."""
    if not torch.cuda.is_available():
        return False
    
    try:
        # Try to get the device name to confirm it's an NVIDIA GPU
        device_name = torch.cuda.get_device_name(0)
        # Check if it's not an AMD GPU (which might report as available but fail)
        return "AMD" not in device_name.upper()
    except Exception:
        return False

def get_model_dimensions(model_path):
    """Extract input and output dimensions from the pre-trained model."""
    if not os.path.exists(model_path):
        print(f"Warning: Model file {model_path} not found.")
        return None, None
    
    try:
        # Load the model state dict
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Extract dimensions from the first and last layers
        if "encoder.0.weight" in state_dict:
            input_dim = state_dict["encoder.0.weight"].shape[1]
            # Find the last layer's weight
            last_layer_keys = [k for k in state_dict.keys() if "weight" in k and "encoder" in k]
            last_layer_keys.sort(key=lambda x: int(x.split('.')[1]))
            last_layer_key = last_layer_keys[-1]
            latent_dim = state_dict[last_layer_key].shape[0]
            
            print(f"Detected model dimensions - Input: {input_dim}, Latent: {latent_dim}")
            return input_dim, latent_dim
    except Exception as e:
        print(f"Error extracting model dimensions: {e}")
    
    return None, None

def load_embeddings_from_jsonl(file_path, batch_size=1000, model_path="model.pth"):
    """
    Load embeddings from a JSONL file and insert them into the database.
    Each line in the JSONL file should contain an 'embedding' key and a 'text' key.
    
    Args:
        file_path: Path to the JSONL file
        batch_size: Number of embeddings to process at once
        model_path: Path to the pre-trained model
    """
    # Get dimensions from the pre-trained model
    input_dim, latent_dim = get_model_dimensions(model_path)
    
    # Automatically detect if we should use GPU
    use_gpu = is_nvidia_gpu_available()
    
    # Initialize the database with the auto-detected GPU setting and dimensions
    db = Database(use_gpu=use_gpu, input_dim=input_dim, latent_dim=latent_dim)
    
    print(f"Loading embeddings from {file_path}...")
    print(f"GPU acceleration: {'Enabled' if use_gpu else 'Disabled'}")
    
    # Process the file in batches to avoid loading everything into memory
    batch_count = 0
    total_count = 0
    
    # First, check the dimension of embeddings in the file
    embedding_dim = None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if 'embedding' in data:
                    embedding_dim = len(data['embedding'])
                    print(f"Detected embedding dimension in file: {embedding_dim}")
                    break
    except Exception as e:
        print(f"Error detecting embedding dimension: {e}")
    
    # If the model input dimension doesn't match the embedding dimension, we need to adapt
    if input_dim is not None and embedding_dim is not None and input_dim != embedding_dim:
        print(f"Warning: Model input dimension ({input_dim}) doesn't match embedding dimension ({embedding_dim}).")
        print("Will use the embeddings directly without the pre-trained model.")
        # Override the database's dimensions to match the embeddings
        db.input_dim = embedding_dim
        db.latent_dim = embedding_dim  # Use same dimension for output
    
    with open(file_path, 'r', encoding='utf-8') as f:
        batch_embeddings = []
        batch_texts = []
        
        for line in tqdm(f):
            try:
                # Parse the JSON line
                data = json.loads(line)
                
                # Extract the embedding and text
                if 'embedding' not in data:
                    print(f"Warning: 'embedding' key not found in line, skipping")
                    continue
                
                if 'text' not in data:
                    print(f"Warning: 'text' key not found in line, using empty string")
                    text = ""
                else:
                    text = data['text']
                
                embedding = data['embedding']
                
                # Add to batch
                batch_embeddings.append(embedding)
                batch_texts.append(text)
                
                # Process batch when it reaches the specified size
                if len(batch_embeddings) >= batch_size:
                    process_batch(db, batch_embeddings, batch_texts)
                    batch_count += 1
                    total_count += len(batch_embeddings)
                    print(f"Processed batch {batch_count}, total items: {total_count}")
                    batch_embeddings = []
                    batch_texts = []
            
            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON line, skipping")
                continue
            except Exception as e:
                print(f"Error processing line: {e}")
                continue
        
        # Process any remaining items
        if batch_embeddings:
            process_batch(db, batch_embeddings, batch_texts)
            total_count += len(batch_embeddings)
            print(f"Processed final batch, total items: {total_count}")
    
    # Build the database index using the pre-trained model
    print("Building database index with pre-trained model...")
    db.build(train_model=False)
    
    print(f"Successfully loaded {total_count} embeddings into the database")

def process_batch(db, embeddings, texts):
    """
    Process a batch of embeddings and insert them into the database staging area.
    
    Args:
        db: Database instance
        embeddings: List of embeddings to insert
        texts: List of original texts corresponding to the embeddings
    """
    for embedding, text in zip(embeddings, texts):
        # Store the embedding as key and a dictionary with embedding and text as value
        value_dict = {
            "embedding": embedding,
            "text": text
        }
        db.insert_staging(embedding, value_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load embeddings from JSONL file into the database")
    parser.add_argument("--file", default="wikipedia-korean-20221001-embeddings-1k.jsonl", 
                        help="Path to the JSONL file containing embeddings")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="Number of embeddings to process at once")
    parser.add_argument("--model-path", default="model.pth",
                        help="Path to the pre-trained model")
    
    args = parser.parse_args()
    
    load_embeddings_from_jsonl(args.file, args.batch_size, args.model_path) 