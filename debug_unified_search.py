#!/usr/bin/env python3
"""
Debug unified index search to see what's being returned.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from s3_vectors_client import S3VectorsClient
from bedrock_client import BedrockMarengoClient


def debug_unified_search():
    client = S3VectorsClient(bucket_name="brice-video-search-multimodal", region="us-east-1")
    bedrock = BedrockMarengoClient(region="us-east-1", output_bucket="tl-brice-media")

    query = "Ross says I take thee Rachel at a wedding"
    print(f"Query: '{query}'")
    print("=" * 80)

    query_result = bedrock.get_text_query_embedding(query)
    query_embedding = query_result["embedding"]

    # Manually query unified index
    print("\nQuerying unified-embeddings index...")
    topK = 100  # S3 Vectors API maximum

    response = client.client.query_vectors(
        vectorBucketName=client.bucket_name,
        indexName="unified-embeddings",
        queryVector={"float32": query_embedding},
        topK=topK,
        returnMetadata=True,
        returnDistance=True
    )

    vectors = response.get("vectors", [])
    print(f"Total vectors returned: {len(vectors)}")

    # Count by modality
    modality_counts = {}
    modality_positions = {}

    for i, vec in enumerate(vectors):
        metadata = vec.get("metadata", {})
        modality = metadata.get("modality_type", "MISSING")

        if modality not in modality_counts:
            modality_counts[modality] = 0
            modality_positions[modality] = []

        modality_counts[modality] += 1
        modality_positions[modality].append(i + 1)  # 1-indexed position

    print("\nModality Distribution:")
    print("-" * 80)
    for modality in ["visual", "audio", "transcription", "MISSING"]:
        count = modality_counts.get(modality, 0)
        positions = modality_positions.get(modality, [])
        print(f"{modality:15s}: {count:4d} vectors", end="")
        if positions:
            print(f" (positions: {positions[:5]}{'...' if len(positions) > 5 else ''})")
        else:
            print()

    # Show first 20 results in detail
    print("\nFirst 20 Results Detail:")
    print("-" * 80)
    for i, vec in enumerate(vectors[:20], 1):
        metadata = vec.get("metadata", {})
        distance = vec.get("distance", 1.0)
        score = 1 - distance
        modality = metadata.get("modality_type", "MISSING")
        video_id = metadata.get("video_id", "")[:8]

        print(f"{i:3d}. {modality:15s} score={score:.4f} video={video_id}")

    print("\n" + "=" * 80)
    print("DIAGNOSIS:")
    print("=" * 80)

    if modality_counts.get("visual", 0) == 0 and modality_counts.get("audio", 0) == 0:
        print("""
❌ Visual and audio have ZERO vectors in unified index!
   This means the Lambda function is NOT writing visual/audio to unified index.

   Solution: Reprocess videos with current Lambda code (which has dual_write=True)
""")
    elif modality_counts.get("visual", 0) > 0 or modality_counts.get("audio", 0) > 0:
        visual_pos = modality_positions.get("visual", [])
        audio_pos = modality_positions.get("audio", [])

        print(f"""
✅ Visual and audio ARE in unified index!
   Visual appears at positions: {visual_pos[:10]}
   Audio appears at positions: {audio_pos[:10]}

   But unified_multi_modality_search is returning zero results because:
   - The topK ({topK}) wasn't retrieving deep enough, OR
   - The filtering logic has a bug

   Check if any visual/audio vectors appear in first {topK} results.
""")
    else:
        print("⚠️ Unclear what the issue is from this data")


if __name__ == "__main__":
    debug_unified_search()
