#!/usr/bin/env python3
"""
Test dynamic weight calculation to debug why weights are all 0.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from search_client import VideoSearchClient

def main():
    print("=" * 80)
    print("Dynamic Weight Calculation Test")
    print("=" * 80)
    print()

    # Initialize client
    mongodb_uri = os.environ.get("MONGODB_URI")
    if not mongodb_uri:
        print("❌ MONGODB_URI not set")
        return

    client = VideoSearchClient(mongodb_uri=mongodb_uri)

    # Initialize anchors
    print("Initializing anchor embeddings...")
    anchors = client.initialize_anchors()
    print(f"Anchor embeddings initialized:")
    for modality, embedding in anchors.items():
        print(f"  {modality}: {len(embedding)} dimensions")
    print()

    # Test query
    query = "Ross says I take thee Rachel at a wedding"
    print(f"Query: {query}")
    print()

    # Get query embedding
    result = client.bedrock.get_text_query_embedding(query)
    query_embedding = result["embedding"]
    print(f"Query embedding: {len(query_embedding)} dimensions")
    print()

    # Compute dynamic weights
    dynamic_result = client.compute_dynamic_weights(query_embedding, temperature=10.0)
    weights = dynamic_result["weights"]
    similarities = dynamic_result["similarities"]

    print("Similarities:")
    for modality, sim in similarities.items():
        print(f"  {modality}: {sim:.4f}")
    print()

    print("Weights:")
    for modality, weight in weights.items():
        print(f"  {modality}: {weight:.4f}")
    print()

    print("=" * 80)

    if all(w == 0 for w in weights.values()):
        print("❌ PROBLEM: All weights are 0!")
    elif not weights:
        print("❌ PROBLEM: Weights dict is empty!")
    else:
        print("✅ Weights computed successfully")

if __name__ == "__main__":
    main()
