#!/usr/bin/env python3
"""
Debug score differences between MongoDB and S3 Vectors.

Compares raw scores from both backends for the same query to understand
why MongoDB returns higher scores than S3 Vectors.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from mongodb_client import MongoDBEmbeddingClient
from s3_vectors_client import S3VectorsClient
from bedrock_client import BedrockMarengoClient

def main():
    print("=" * 80)
    print("Score Comparison: MongoDB vs S3 Vectors")
    print("=" * 80)
    print()

    # Initialize clients
    mongodb_uri = os.environ.get("MONGODB_URI")
    if not mongodb_uri:
        print("❌ MONGODB_URI not set")
        return

    mongo_client = MongoDBEmbeddingClient(connection_string=mongodb_uri)
    s3v_client = S3VectorsClient(
        bucket_name="brice-video-search-multimodal",
        region="us-east-1"
    )
    bedrock_client = BedrockMarengoClient(region="us-east-1")

    # Test query
    query = "Ross says I take thee Rachel at a wedding"
    print(f"Query: {query}")
    print()

    # Get query embedding
    result = bedrock_client.get_text_query_embedding(query)
    query_embedding = result["embedding"]
    print(f"Query embedding: {len(query_embedding)} dimensions")
    print()

    # MongoDB scores (single-index mode with modality filter)
    print("-" * 80)
    print("MongoDB Single-Index (unified-embeddings with filters)")
    print("-" * 80)

    mongo_results = {}
    for modality in ["visual", "audio", "transcription"]:
        results = mongo_client.vector_search(
            query_embedding=query_embedding,
            limit=5,
            modality_filter=modality
        )
        mongo_results[modality] = results

        if results:
            print(f"\n{modality.upper()}:")
            for i, doc in enumerate(results[:3], 1):
                print(f"  Result {i}: score={doc['score']:.4f}, video={doc['video_id']}, segment={doc['segment_id']}")
        else:
            print(f"\n{modality.upper()}: No results")

    # S3 Vectors scores (multi-index mode)
    print()
    print("-" * 80)
    print("S3 Vectors Multi-Index (separate indexes per modality)")
    print("-" * 80)

    s3v_results = {}
    for modality in ["visual", "audio", "transcription"]:
        results = s3v_client.vector_search(
            query_embedding=query_embedding,
            modality=modality,
            limit=5
        )
        s3v_results[modality] = results

        if results:
            print(f"\n{modality.upper()}:")
            for i, doc in enumerate(results[:3], 1):
                print(f"  Result {i}: score={doc['score']:.4f}, video={doc['video_id']}, segment={doc['segment_id']}")
                # Also show the raw distance for debugging
                # Note: score = 1 - distance
        else:
            print(f"\n{modality.upper()}: No results")

    # Compare top result scores
    print()
    print("-" * 80)
    print("Score Comparison (Top Result Per Modality)")
    print("-" * 80)
    print()

    for modality in ["visual", "audio", "transcription"]:
        mongo_score = mongo_results[modality][0]["score"] if mongo_results[modality] else 0
        s3v_score = s3v_results[modality][0]["score"] if s3v_results[modality] else 0

        diff = mongo_score - s3v_score
        pct_diff = (diff / mongo_score * 100) if mongo_score > 0 else 0

        print(f"{modality.upper()}:")
        print(f"  MongoDB:    {mongo_score:.4f}")
        print(f"  S3 Vectors: {s3v_score:.4f}")
        print(f"  Difference: {diff:+.4f} ({pct_diff:+.1f}%)")
        print()

    # Analysis
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print()
    print("Potential causes for score differences:")
    print("1. Different distance metrics:")
    print("   - MongoDB Atlas: Uses cosine similarity (native vectorSearchScore)")
    print("   - S3 Vectors: May use different distance metric (check API docs)")
    print()
    print("2. Score normalization:")
    print("   - MongoDB: Returns vectorSearchScore directly (0-1 range)")
    print("   - S3 Vectors: Returns distance, we convert with `1 - distance`")
    print("   - Conversion formula may be incorrect for non-cosine metrics")
    print()
    print("3. Data consistency:")
    print("   - Are embeddings identical in both backends?")
    print("   - Were they imported from the same source?")
    print("   - Check embedding values match between backends")
    print()
    print("4. Index configuration:")
    print("   - MongoDB: unified-embeddings collection with filters")
    print("   - S3 Vectors: Separate indexes (visual, audio, transcription)")
    print("   - Different index configurations may affect scores")

if __name__ == "__main__":
    main()
