#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import torch
import numpy as np
import cohere
from main import Database
from distance_preservation_encoder.model import DPEncoder
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import os
import json
from tqdm import tqdm
import faiss
import time

# Add Hugging Face datasets import
try:
    from datasets import load_dataset
    hf_datasets_available = True
except ImportError:
    load_dataset = None
    hf_datasets_available = False

def ensure_embedding_is_float_list(embedding):
    """
    Ensure the embedding is a list of floats.
    
    Args:
        embedding: The embedding, which could be a string, list, or other format
        
    Returns:
        A list of floats
    """
    if isinstance(embedding, str):
        # Try to parse the string as a JSON array
        try:
            embedding = json.loads(embedding)
        except json.JSONDecodeError:
            # If it's not valid JSON, try to parse it as a space-separated string
            try:
                embedding = [float(x) for x in embedding.strip('[]').split()]
            except ValueError:
                # If that fails, try to parse it as a comma-separated string
                try:
                    embedding = [float(x) for x in embedding.strip('[]').split(',')]
                except ValueError:
                    print(f"Warning: Could not parse embedding string: {embedding[:50]}...")
                    return None
    
    # Convert numpy arrays to lists
    if isinstance(embedding, np.ndarray):
        embedding = embedding.tolist()
    
    # Ensure all elements are floats
    if isinstance(embedding, list):
        try:
            embedding = [float(x) for x in embedding]
        except (ValueError, TypeError):
            print(f"Warning: Could not convert all embedding elements to float")
            return None
    else:
        print(f"Warning: Embedding is not a list or string: {type(embedding)}")
        return None
        
    return embedding

def parse_args():
    parser = argparse.ArgumentParser(description='Search procedure using Cohere multilingual-22-12')
    parser.add_argument('--query', type=str, required=True, help='Natural language query text')
    parser.add_argument('--api_key', type=str, help='Cohere API key (or set COHERE_API_KEY env var)')
    parser.add_argument('--db_path', type=str, default='LESCdb', help='Path to database')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--search_type', type=str, choices=['cosine', 'l2'], default='l2', 
                        help='Search method: cosine similarity or L2 distance')
    parser.add_argument('--limit', type=int, default=1, help='Number of results to return')
    parser.add_argument('--compare', action='store_true', help='Compare DB search with direct embedding search')
    
    # Comparison data source options
    parser.add_argument('--compare_source', type=str, choices=['jsonl', 'huggingface'], default='jsonl',
                        help='Source for comparison embeddings: jsonl file or huggingface dataset')
    parser.add_argument('--jsonl_file', type=str, help='JSONL file with original embeddings for comparison')
    
    # Hugging Face dataset options
    parser.add_argument('--dataset', type=str, help='Hugging Face dataset name for comparison')
    parser.add_argument('--config', type=str, help='Dataset configuration name')
    parser.add_argument('--split', type=str, default='train', help='Dataset split to load')
    parser.add_argument('--embedding_column', type=str, default='embedding', help='Column name containing embeddings')
    parser.add_argument('--text_column', type=str, default='text', help='Column name containing text')
    parser.add_argument('--streaming', action='store_true', help='Use streaming for large datasets')
    parser.add_argument('--max_samples', type=int, help='Maximum number of samples to load for comparison')
    
    return parser.parse_args()

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

def get_embedding(text, api_key=None):
    """Get embedding from Cohere's multilingual-22-12 model"""
    client = cohere.Client(api_key=api_key)
    
    try:
        response = client.embed(
            texts=[text],
            model="multilingual-22-12",
            input_type="search_query"
        )
        # Fix linter error: access embeddings properly
        return list(response.embeddings)[0]
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

def get_model_dimensions(model_path):
    """Extract input and output dimensions from the pre-trained model."""
    if not os.path.exists(model_path):
        print(f"Warning: Model file {model_path} not found.")
        return 768, 512  # Default dimensions for Cohere multilingual-22-12
    
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
    
    return 768, 512  # Default dimensions for Cohere multilingual-22-12

def extract_text_from_result(result):
    """Extract text from result data which could be a dict or just an embedding"""
    if isinstance(result, dict) and "text" in result:
        return result["text"]
    elif isinstance(result, dict) and "embedding" in result:
        # If there's only embedding but no text, show a truncated version of the embedding
        return f"[Embedding only: {str(result['embedding'][:5])}...]"
    else:
        # For backwards compatibility with old data format
        return f"[Raw data: {str(result[:5])}...]"

def get_embedding_from_result(result):
    """Extract embedding from result data which could be a dict or just an embedding"""
    if isinstance(result, dict) and "embedding" in result:
        # Convert embedding to float list if it's a string
        embedding = ensure_embedding_is_float_list(result["embedding"])
        return embedding if embedding is not None else result["embedding"]
    else:
        # For backwards compatibility with old data format
        return result

def load_jsonl_embeddings(file_path):
    """Load embeddings from a JSONL file for direct comparison"""
    if not file_path or not os.path.exists(file_path):
        print(f"JSONL file not found: {file_path}")
        return [], []
    
    embeddings = []
    texts = []
    
    print(f"Loading embeddings from {file_path} for comparison...")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            try:
                data = json.loads(line)
                if 'embedding' in data:
                    # Convert embedding to float list if it's a string
                    embedding = ensure_embedding_is_float_list(data['embedding'])
                    if embedding is None:
                        print(f"Warning: Could not process embedding, skipping")
                        continue
                    
                    embeddings.append(embedding)
                    
                    # Also store text if available
                    if 'text' in data:
                        texts.append(data['text'])
                    else:
                        texts.append("")
            except Exception as e:
                print(f"Error loading line: {e}")
                continue
    
    print(f"Loaded {len(embeddings)} embeddings for comparison")
    return embeddings, texts

def load_huggingface_embeddings(dataset_name, config_name=None, split="train", 
                               embedding_column="embedding", text_column="text", 
                               streaming=False, max_samples=None):
    """Load embeddings from a Hugging Face dataset for direct comparison"""
    if not hf_datasets_available:
        print("Warning: datasets library not available. Install with: pip install datasets")
        return [], []
    
    if load_dataset is None:
        print("Warning: datasets library not properly imported")
        return [], []
    
    try:
        print(f"Loading embeddings from Hugging Face dataset: {dataset_name}")
        if config_name:
            print(f"Config: {config_name}")
        print(f"Split: {split}, Streaming: {streaming}")
        
        # Load the dataset
        dataset = load_dataset(
            dataset_name,
            name=config_name,
            split=split,
            streaming=streaming
        )
        
        embeddings = []
        texts = []
        count = 0
        
        # Process items
        for item in tqdm(dataset, desc="Loading embeddings for comparison"):
            try:
                # Extract the embedding
                if embedding_column not in item:
                    continue
                
                # Convert embedding to float list if it's a string
                embedding = ensure_embedding_is_float_list(item[embedding_column])
                if embedding is None:
                    print(f"Warning: Could not process embedding, skipping")
                    continue
                
                # Extract the text (if available)
                text = ""
                if text_column and text_column in item:
                    text = item[text_column]
                
                embeddings.append(embedding)
                texts.append(text)
                count += 1
                
                # Limit samples if specified
                if max_samples and count >= max_samples:
                    break
                    
            except Exception as e:
                print(f"Error processing item: {e}")
                continue
        
        print(f"Loaded {len(embeddings)} embeddings from Hugging Face dataset")
        return embeddings, texts
        
    except Exception as e:
        print(f"Error loading Hugging Face dataset: {e}")
        return [], []

def direct_search_faiss(query_embedding, embeddings, texts, search_type='l2', limit=1, use_gpu=True):
    """
    Perform direct search on the original embeddings using FAISS
    
    Args:
        query_embedding: The query embedding
        embeddings: List of embeddings to search
        texts: List of texts corresponding to embeddings
        search_type: 'l2' or 'cosine'
        limit: Number of results to return
        use_gpu: Whether to use GPU if available
    
    Returns:
        List of (metric, embedding, text) tuples and the metric name
    """
    if not embeddings:
        return [], "none"
    
    # Convert embeddings to numpy array
    embeddings_array = np.array(embeddings, dtype=np.float32)
    query_vector = np.array([query_embedding], dtype=np.float32)
    
    # Get dimension from the first embedding
    d = embeddings_array.shape[1]
    
    # Create appropriate index based on search type
    if search_type == 'cosine':
        # For cosine similarity, we need to normalize the vectors
        faiss.normalize_L2(embeddings_array)
        faiss.normalize_L2(query_vector)
        index = faiss.IndexFlatIP(d)  # Inner product for cosine similarity with normalized vectors
        metric_name = "similarity"
    else:  # l2 distance
        index = faiss.IndexFlatL2(d)  # L2 distance
        metric_name = "distance"
    
    # Use GPU if available and requested
    gpu_resources = None
    gpu_index = None
    
    if use_gpu and is_nvidia_gpu_available():
        try:
            print("Using GPU for FAISS search")
            gpu_resources = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(gpu_resources, 0, index)
            gpu_index.add(embeddings_array)
            
            # Measure search time
            start_time = time.time()
            distances, indices = gpu_index.search(query_vector, min(limit, len(embeddings)))
            search_time = time.time() - start_time
            print(f"GPU search completed in {search_time:.4f} seconds")
            
            # Process results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < 0 or idx >= len(embeddings):
                    continue
                results.append((float(distance), embeddings[idx], texts[idx]))
            
            return results, metric_name
            
        except Exception as e:
            print(f"Error using GPU for FAISS: {e}")
            print("Falling back to CPU")
            # Fall back to CPU if GPU fails
            gpu_resources = None
            gpu_index = None
    
    # CPU search
    print("Using CPU for FAISS search")
    index.add(embeddings_array)
    
    # Measure search time
    start_time = time.time()
    distances, indices = index.search(query_vector, min(limit, len(embeddings)))
    search_time = time.time() - start_time
    print(f"CPU search completed in {search_time:.4f} seconds")
    
    # Process results
    results = []
    for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
        if idx < 0 or idx >= len(embeddings):
            continue
        results.append((float(distance), embeddings[idx], texts[idx]))
    
    return results, metric_name

def calculate_overlap(db_results, direct_results):
    """Calculate overlap between database results and direct search results"""
    db_texts = [extract_text_from_result(r[1]) for r in db_results]
    direct_texts = [r[2] for r in direct_results]
    
    # Count matches
    matches = 0
    for text in db_texts:
        if text in direct_texts:
            matches += 1
    
    overlap_percentage = (matches / len(db_texts)) * 100 if db_texts else 0
    return overlap_percentage, matches, len(db_texts)

def main():
    args = parse_args()
    
    # Auto-detect GPU availability
    use_gpu = args.use_gpu or is_nvidia_gpu_available()
    if use_gpu:
        print("GPU acceleration enabled")
    else:
        print("Using CPU only")
    
    # Get API key from args or environment
    api_key = args.api_key or os.environ.get("COHERE_API_KEY")
    if not api_key:
        print("Error: Cohere API key is required. Provide it with --api_key or set COHERE_API_KEY environment variable.")
        return
    
    # Step 1: Get natural language input
    query_text = args.query
    print(f"Query: {query_text}")
    
    # Step 2: Generate embeddings using Cohere multilingual-22-12
    print("Generating embedding with Cohere multilingual-22-12...")
    embedding = get_embedding(query_text, api_key)
    if embedding is None:
        print("Failed to generate embedding. Exiting.")
        return
    
    # Ensure embedding is a list of floats
    embedding = ensure_embedding_is_float_list(embedding)
    if embedding is None:
        print("Failed to process embedding. Exiting.")
        return
    
    # Step 3: Search using Database (let the database handle processing)
    print(f"Searching with {args.search_type} similarity...")
    
    # Get model dimensions for database initialization
    model_path = "model.pth"
    input_dim, latent_dim = get_model_dimensions(model_path)
    db = Database(path=args.db_path, use_gpu=use_gpu, input_dim=input_dim, latent_dim=latent_dim)
    print(f"Staging entries: {len(db.staging.keys())}")
    print(f"Database entries: {len(db.database.keys())}")
    print(f"Index size: {db.index.ntotal}")
    
    # Variables to store results and metrics
    results = []
    similarities = []
    
    # Measure database search time
    db_start_time = time.time()
    
    if args.search_type == 'cosine':
        # For cosine similarity, we need to implement it manually since the Database uses L2
        # First get all data points
        # This is a simplified approach - in a real system, you'd want to optimize this
        # by implementing cosine similarity directly in the search method
        # Ensure embedding is a list of floats
        embedding_floats = embedding  # Already converted to list of floats above
        results = db.search(embedding_floats, limit=args.limit)
        
        # Convert results to vectors for cosine similarity
        query_vector = np.array(embedding).reshape(1, -1)
        result_vectors = np.array([get_embedding_from_result(r[1]) for r in results])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_vector, result_vectors)[0]
        
        # Sort by similarity (highest first)
        sorted_indices = np.argsort(-similarities)
        
        # Calculate database search time
        db_search_time = time.time() - db_start_time
        
        # Print results
        print(f"\nDatabase Search Results (Cosine Similarity) - completed in {db_search_time:.4f} seconds:")
        for i, idx in enumerate(sorted_indices):
            text = extract_text_from_result(results[idx][1])
            print(f"{i+1}. Similarity: {similarities[idx]:.4f}")
            print(f"   Text: {text[:200]}{'...' if len(text) > 200 else ''}")
            print()
    else:
        # L2 distance search (directly using the Database implementation)
        # Ensure embedding is a list of floats
        embedding_list = embedding  # Already converted to list of floats above
        results = db.search(embedding_list, limit=args.limit)
        
        # Calculate database search time
        db_search_time = time.time() - db_start_time
        
        # Print results
        print(f"\nDatabase Search Results (L2 Distance) - completed in {db_search_time:.4f} seconds:")
        for i, (distance, data) in enumerate(results):
            text = extract_text_from_result(data)
            print(f"{i+1}. Distance: {distance:.4f}")
            print(f"   Text: {text[:200]}{'...' if len(text) > 200 else ''}")
            print()
    
    # If comparison is requested
    if args.compare:
        original_embeddings = []
        original_texts = []
        
        # Load comparison data based on source type
        if args.compare_source == 'jsonl' and args.jsonl_file:
            original_embeddings, original_texts = load_jsonl_embeddings(args.jsonl_file)
        elif args.compare_source == 'huggingface' and args.dataset:
            original_embeddings, original_texts = load_huggingface_embeddings(
                args.dataset,
                args.config,
                args.split,
                args.embedding_column,
                args.text_column,
                args.streaming,
                args.max_samples
            )
        else:
            print("Warning: Comparison requested but no valid source specified.")
            print("For JSONL: use --compare_source jsonl --jsonl_file <file>")
            print("For HuggingFace: use --compare_source huggingface --dataset <dataset>")
        
        if original_embeddings:
            # Perform direct search on original embeddings using FAISS
            source_name = f"original {args.compare_source}"
            print(f"\nPerforming direct search on {source_name} embeddings using FAISS...")
            
            # Measure direct search time
            direct_start_time = time.time()
            direct_results, metric_name = direct_search_faiss(
                embedding, 
                original_embeddings, 
                original_texts, 
                args.search_type, 
                args.limit,
                use_gpu
            )
            direct_search_time = time.time() - direct_start_time
            
            # Print direct search results
            print(f"\nDirect Search Results ({args.search_type}) - completed in {direct_search_time:.4f} seconds:")
            for i, (metric, emb, text) in enumerate(direct_results):
                print(f"{i+1}. {metric_name.capitalize()}: {metric:.4f}")
                print(f"   Text: {text[:200]}{'...' if len(text) > 200 else ''}")
                print()
            
            # Calculate and print overlap
            overlap_percentage, matches, total = calculate_overlap(results, direct_results)
            print(f"\nResults Comparison:")
            print(f"Overlap between database and direct search: {matches}/{total} ({overlap_percentage:.1f}%)")
            
            # Print time comparison
            time_diff = direct_search_time - db_search_time
            speedup = (direct_search_time / db_search_time) if db_search_time > 0 else float('inf')
            print(f"\nTime Comparison:")
            print(f"Database search time: {db_search_time:.4f} seconds")
            print(f"Direct search time: {direct_search_time:.4f} seconds")
            print(f"Time difference: {time_diff:.4f} seconds ({speedup:.2f}x {'faster' if speedup > 1 else 'slower'} with compressed DB)")
            
            # Analyze differences if any
            if overlap_percentage < 100:
                print("\nAnalysis of differences:")
                print("The differences could be due to:")
                print("1. Dimension reduction in the database encoding")
                print("2. Approximation errors in the search algorithm")
                print("3. Different normalization of vectors")
                print("4. Different embedding models (Cohere vs. original)")
                
                # Calculate average metric difference
                db_metrics = [r[0] for r in results]
                direct_metrics = [r[0] for r in direct_results]
                
                if args.search_type == 'cosine':
                    # For cosine, higher is better
                    db_avg = np.mean(similarities) if similarities else np.mean(db_metrics)
                    direct_avg = np.mean([r[0] for r in direct_results])
                    print(f"Average similarity - Database: {db_avg:.4f}, Direct: {direct_avg:.4f}")
                else:
                    # For L2, lower is better
                    db_avg = np.mean(db_metrics)
                    direct_avg = np.mean(direct_metrics)
                    print(f"Average distance - Database: {db_avg:.4f}, Direct: {direct_avg:.4f}")

if __name__ == "__main__":
    main() 