#!/usr/bin/env python3
import json
import torch
import argparse
import os
import numpy as np
from tqdm import tqdm
from main import Database
from distance_preservation_encoder.model import DPEncoder

# Add Hugging Face datasets import
try:
    from datasets import load_dataset
    hf_datasets_available = True
except ImportError:
    load_dataset = None
    hf_datasets_available = False
    print("Warning: datasets library not installed. Install with: pip install datasets")

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

def load_embeddings_from_huggingface(dataset_name, batch_size=1000, model_path="model.pth", 
                                    embedding_column="embedding", text_column="text",
                                    config_name=None, split="train", streaming=False,
                                    save_path="database.db"):
    """
    Load embeddings from a Hugging Face dataset and insert them into the database.
    
    Args:
        dataset_name: Name of the Hugging Face dataset (e.g., "sentence-transformers/all-MiniLM-L6-v2")
        batch_size: Number of embeddings to process at once
        model_path: Path to the pre-trained model
        embedding_column: Column name containing embeddings
        text_column: Column name containing text
        config_name: Dataset configuration name (if applicable)
        split: Dataset split to load ("train", "test", etc.)
        streaming: Whether to stream the dataset (for large datasets)
        save_path: Path where the database will be saved
    """
    if not hf_datasets_available:
        raise ImportError("datasets library is required. Install with: pip install datasets")
    
    # Get dimensions from the pre-trained model
    input_dim, latent_dim = get_model_dimensions(model_path)
    
    # Automatically detect if we should use GPU
    use_gpu = is_nvidia_gpu_available()
    
    # Initialize the database with the auto-detected GPU setting and dimensions
    db = Database(path=save_path, use_gpu=use_gpu, input_dim=input_dim, latent_dim=latent_dim)
    
    print(f"Loading embeddings from Hugging Face dataset: {dataset_name}")
    print(f"Config: {config_name}, Split: {split}")
    print(f"GPU acceleration: {'Enabled' if use_gpu else 'Disabled'}")
    print(f"Streaming: {'Enabled' if streaming else 'Disabled'}")
    
    try:
        # Load the dataset
        if load_dataset is None:
            raise ImportError("datasets library is required. Install with: pip install datasets")
        
        dataset = load_dataset(
            dataset_name, 
            name=config_name, 
            split=split, 
            streaming=streaming
        )
        
        print(f"Successfully loaded dataset: {dataset_name}")
        
        # Check if the required columns exist
        if hasattr(dataset, 'features') and dataset.features is not None:
            features = dataset.features
            if hasattr(features, 'keys') and embedding_column not in features:
                raise ValueError(f"Embedding column '{embedding_column}' not found in dataset. Available columns: {list(features.keys())}")
            if text_column and hasattr(features, 'keys') and text_column not in features:
                print(f"Warning: Text column '{text_column}' not found. Available columns: {list(features.keys())}")
                text_column = None
        
        # Process the dataset
        batch_count = 0
        total_count = 0
        batch_embeddings = []
        batch_texts = []
        
        # Determine embedding dimension from first item
        embedding_dim = None
        first_item = None
        
        if streaming:
            # For streaming datasets, we need to peek at the first item
            dataset_iter = iter(dataset)
            first_item = next(dataset_iter)
            if embedding_column in first_item:
                embedding_dim = len(first_item[embedding_column])
                print(f"Detected embedding dimension: {embedding_dim}")
        else:
            # For non-streaming datasets, we can access the first item directly
            if len(dataset) > 0:
                first_item = dataset[0]
                if embedding_column in first_item:
                    embedding_dim = len(first_item[embedding_column])
                    print(f"Detected embedding dimension: {embedding_dim}")
        
        # If the model input dimension doesn't match the embedding dimension, adapt
        if input_dim is not None and embedding_dim is not None and input_dim != embedding_dim:
            print(f"Warning: Model input dimension ({input_dim}) doesn't match embedding dimension ({embedding_dim}).")
            print("Will use the embeddings directly without the pre-trained model.")
            db.input_dim = embedding_dim
            db.latent_dim = embedding_dim
        
        # Create iterator based on streaming or not
        if streaming:
            # For streaming, we need to create a new iterator and include the first item
            import itertools
            dataset_iter = itertools.chain([first_item], dataset)
        else:
            dataset_iter = dataset
        
        # Process items
        for item in tqdm(dataset_iter, desc="Processing embeddings"):
            try:
                # Extract the embedding
                if embedding_column not in item:
                    print(f"Warning: '{embedding_column}' not found in item, skipping")
                    continue
                
                embedding = item[embedding_column]
                
                # Extract the text (if available)
                text = ""
                if text_column and text_column in item:
                    text = item[text_column]
                elif text_column:
                    print(f"Warning: '{text_column}' not found in item, using empty string")
                
                # Add to batch
                batch_embeddings.append(embedding)
                batch_texts.append(text)
                
                # Process batch when it reaches the specified size
                if len(batch_embeddings) >= batch_size:
                    process_batch(db, batch_embeddings, batch_texts)
                    batch_count += 1
                    total_count += len(batch_embeddings)
                    batch_embeddings = []
                    batch_texts = []
                    
            except Exception as e:
                print(f"Error processing item: {e}")
                continue
        
        # Process any remaining items
        if batch_embeddings:
            process_batch(db, batch_embeddings, batch_texts)
            total_count += len(batch_embeddings)
            print(f"Processed final batch, total items: {total_count}")
        
        # Build the database index
        print("Building database index...")
        db.build(train_model=False)
        
        # Save the database
        print(f"Saving database to {save_path}...")
        db.save()
        
        print(f"Successfully loaded {total_count} embeddings from Hugging Face dataset")
        
    except Exception as e:
        print(f"Error loading dataset from Hugging Face: {e}")
        raise

def load_embeddings_from_jsonl(file_path, batch_size=1000, model_path="model.pth", save_path="database.db"):
    """
    Load embeddings from a JSONL file and insert them into the database.
    Each line in the JSONL file should contain an 'embedding' key and a 'text' key.
    
    Args:
        file_path: Path to the JSONL file
        batch_size: Number of embeddings to process at once
        model_path: Path to the pre-trained model
        save_path: Path where the database will be saved
    """
    # Get dimensions from the pre-trained model
    input_dim, latent_dim = get_model_dimensions(model_path)
    
    # Automatically detect if we should use GPU
    use_gpu = is_nvidia_gpu_available()
    
    # Initialize the database with the auto-detected GPU setting and dimensions
    db = Database(path=save_path, use_gpu=use_gpu, input_dim=input_dim, latent_dim=latent_dim)
    
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
    
    # Save the database
    print(f"Saving database to {save_path}...")
    db.save()
    
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
    parser = argparse.ArgumentParser(description="Load embeddings from JSONL file or Hugging Face dataset into the database")
    parser.add_argument("--source", choices=["jsonl", "huggingface"], default="jsonl",
                        help="Source of embeddings: 'jsonl' for JSONL file, 'huggingface' for HF dataset")
    parser.add_argument("--file", default="wikipedia-korean-20221001-embeddings-1k.jsonl", 
                        help="Path to the JSONL file containing embeddings (for jsonl source)")
    parser.add_argument("--dataset", 
                        help="Name of the Hugging Face dataset (for huggingface source)")
    parser.add_argument("--config", 
                        help="Dataset configuration name (optional)")
    parser.add_argument("--split", default="train",
                        help="Dataset split to load (default: train)")
    parser.add_argument("--embedding-column", default="embedding",
                        help="Column name containing embeddings (default: embedding)")
    parser.add_argument("--text-column", default="text",
                        help="Column name containing text (default: text)")
    parser.add_argument("--streaming", action="store_true",
                        help="Use streaming for large datasets")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="Number of embeddings to process at once")
    parser.add_argument("--model-path", default="model.pth",
                        help="Path to the pre-trained model")
    parser.add_argument("--save-path", default="database.db",
                        help="Path where the database will be saved")
    
    args = parser.parse_args()
    
    if args.source == "jsonl":
        load_embeddings_from_jsonl(args.file, args.batch_size, args.model_path, args.save_path)
    elif args.source == "huggingface":
        if not args.dataset:
            parser.error("--dataset is required when using huggingface source")
        load_embeddings_from_huggingface(
            args.dataset, 
            args.batch_size, 
            args.model_path,
            args.embedding_column,
            args.text_column,
            args.config,
            args.split,
            args.streaming,
            args.save_path
        ) 
