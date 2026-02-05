#!/usr/bin/env python3
"""
Test S3 Vectors unified-embeddings search with actual query embedding.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from s3_vectors_client import S3VectorsClient
from bedrock_client import BedrockMarengoClient

def main():
    s3v_client = S3VectorsClient(
        bucket_name="brice-video-search-multimodal",
        region="us-east-1"
    )

    bedrock_client = BedrockMarengoClient(region="us-east-1")

    print("=" * 80)
    print("S3 Vectors Unified Index Search Test")
    print("=" * 80)
    print()

    # Get embedding for test query
    query = "Ross says I take thee Rachel at a wedding"
    print(f"Query: {query}")
    print()

    result = bedrock_client.get_text_query_embedding(query)
    query_embedding = result["embedding"]

    print(f"Got query embedding: {len(query_embedding)} dimensions")
    print()

    # Search unified index
    try:
        response = s3v_client.client.query_vectors(
            vectorBucketName=s3v_client.bucket_name,
            indexName=s3v_client.UNIFIED_INDEX_NAME,
            queryVector={"float32": query_embedding},
            topK=100,
            returnMetadata=True,
            returnDistance=True
        )

        vectors = response.get("vectors", [])
        print(f"Retrieved {len(vectors)} vectors from unified-embeddings index")
        print()

        # Count by modality
        modality_counts = {"visual": 0, "audio": 0, "transcription": 0, "unknown": 0}

        for vector in vectors:
            metadata = vector.get("metadata", {})
            modality = metadata.get("modality_type", "unknown")

            if modality in modality_counts:
                modality_counts[modality] += 1
            else:
                modality_counts["unknown"] += 1

        print(f"Modality distribution in search results:")
        print(f"  Visual:        {modality_counts['visual']} results")
        print(f"  Audio:         {modality_counts['audio']} results")
        print(f"  Transcription: {modality_counts['transcription']} results")
        print(f"  Unknown:       {modality_counts['unknown']} results")
        print(f"  Total:         {len(vectors)} results")
        print()

        # Show top 10 results
        print("Top 10 results:")
        for i, vector in enumerate(vectors[:10]):
            key = vector.get("key", "")
            distance = vector.get("distance", 1.0)
            score = 1 - distance
            metadata = vector.get("metadata", {})
            modality = metadata.get("modality_type", "MISSING")

            print(f"\n  Result {i+1}:")
            print(f"    key: {key}")
            print(f"    score: {score:.4f}")
            print(f"    modality_type: {modality}")
            print(f"    video_id: {metadata.get('video_id', 'MISSING')}")
            print(f"    segment_id: {metadata.get('segment_id', 'MISSING')}")

        print()
        print("=" * 80)

        if modality_counts['visual'] == 0 or modality_counts['audio'] == 0:
            print("❌ PROBLEM: Search returns no visual or audio results!")
            if modality_counts['unknown'] > 0:
                print("   Metadata is MISSING from vectors - need to re-import with metadata!")
            else:
                print("   Only transcription vectors are semantically close to this query (expected for unified index)")
        else:
            print("✅ All modalities present in search results")

    except Exception as e:
        print(f"❌ Error searching S3 Vectors: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
