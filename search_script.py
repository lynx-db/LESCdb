#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import torch
import numpy as np
from openai import OpenAI
from main import Database
from distance_preservation_encoder.model import DPEncoder
from sklearn.metrics.pairwise import cosine_similarity
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Search procedure using text-embedding-3-small and DPAE')
    parser.add_argument('--query', type=str, required=True, help='Natural language query text')
    parser.add_argument('--api_key', type=str, help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--model_path', type=str, default='model.pth', help='Path to DPAE model')
    parser.add_argument('--db_path', type=str, default='LESCdb', help='Path to database')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--search_type', type=str, choices=['cosine', 'l2'], default='l2', 
                        help='Search method: cosine similarity or L2 distance')
    parser.add_argument('--limit', type=int, default=10, help='Number of results to return')
    return parser.parse_args()

def get_embedding(text, api_key=None):
    """Get embedding from OpenAI's text-embedding-3-small model"""
    client = OpenAI(api_key=api_key)
    
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

def load_dpae_model(model_path, input_dim=1536, hidden_dims=[768], latent_dim=512):
    """Load the DPAE model"""
    if not os.path.exists(model_path):
        print(f"Warning: Model file {model_path} not found. Embeddings will not be processed through DPAE.")
        return None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DPEncoder(input_dim, hidden_dims, latent_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def process_through_dpae(embedding, model):
    """Process embedding through DPAE model"""
    if model is None:
        return embedding
    
    device = next(model.parameters()).device
    tensor = torch.tensor([embedding], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        output = model(tensor)
    
    return output[0].cpu().numpy().tolist()

def main():
    args = parse_args()
    
    # Get API key from args or environment
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key is required. Provide it with --api_key or set OPENAI_API_KEY environment variable.")
        return
    
    # Step 1: Get natural language input
    query_text = args.query
    print(f"Query: {query_text}")
    
    # Step 2: Generate embeddings using text-embedding-3-small
    print("Generating embedding with text-embedding-3-small...")
    embedding = get_embedding(query_text, api_key)
    if embedding is None:
        print("Failed to generate embedding. Exiting.")
        return
    
    # Step 3: Process through DPAE
    print("Processing through DPAE...")
    dpae_model = load_dpae_model(args.model_path)
    processed_embedding = process_through_dpae(embedding, dpae_model)
    
    # Step 4: Search using Database
    print(f"Searching with {args.search_type} similarity...")
    db = Database(path=args.db_path, use_gpu=args.use_gpu)
    
    if args.search_type == 'cosine':
        # For cosine similarity, we need to implement it manually since the Database uses L2
        # First get all data points
        # This is a simplified approach - in a real system, you'd want to optimize this
        # by implementing cosine similarity directly in the search method
        results = db.search(processed_embedding, limit=args.limit)
        
        # Convert results to vectors for cosine similarity
        query_vector = np.array(processed_embedding).reshape(1, -1)
        result_vectors = np.array([r[1] for r in results])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_vector, result_vectors)[0]
        
        # Sort by similarity (highest first)
        sorted_indices = np.argsort(-similarities)
        
        # Print results
        print("\nSearch Results (Cosine Similarity):")
        for i, idx in enumerate(sorted_indices):
            print(f"{i+1}. Similarity: {similarities[idx]:.4f}, Data: {results[idx][1][:10]}...")
    else:
        # L2 distance search (directly using the Database implementation)
        results = db.search(processed_embedding, limit=args.limit)
        
        # Print results
        print("\nSearch Results (L2 Distance):")
        for i, (distance, data) in enumerate(results):
            print(f"{i+1}. Distance: {distance:.4f}, Data: {data[:10]}...")

if __name__ == "__main__":
    main() 