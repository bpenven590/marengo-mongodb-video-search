#!/usr/bin/env python3
"""
Check what data is actually in S3 Vectors indexes.
"""

import sys
import os
import boto3

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from s3_vectors_client import S3VectorsClient


def check_index_data(client, index_name, description):
    """Check how many vectors are in an index."""
    print(f"\n{description}: {index_name}")
    print("-" * 60)

    try:
        # Query with a dummy vector to see what's there
        dummy_vector = [0.1] * 512

        response = client.client.query_vectors(
            vectorBucketName=client.bucket_name,
            indexName=index_name,
            queryVector={"float32": dummy_vector},
            topK=10,
            returnMetadata=True
        )

        vectors = response.get("vectors", [])
        print(f"  Sample results: {len(vectors)} vectors returned")

        if vectors:
            # Show first few vectors' metadata
            for i, vec in enumerate(vectors[:3], 1):
                metadata = vec.get("metadata", {})
                print(f"  Vector {i}:")
                print(f"    video_id: {metadata.get('video_id', 'N/A')}")
                print(f"    segment_id: {metadata.get('segment_id', 'N/A')}")
                print(f"    modality_type: {metadata.get('modality_type', 'N/A')}")

            # Count unique videos
            video_ids = set(v.get("metadata", {}).get("video_id") for v in vectors)
            print(f"  Unique videos in sample: {len(video_ids)}")

        else:
            print(f"  ❌ No vectors found in index")

    except Exception as e:
        print(f"  ❌ Error querying index: {e}")


def main():
    print("=" * 80)
    print("S3 Vectors Index Data Check")
    print("=" * 80)

    client = S3VectorsClient(bucket_name="brice-video-search-multimodal", region="us-east-1")

    # Check unified index
    check_index_data(client, "unified-embeddings", "Unified Index (Single-Index Mode)")

    # Check modality-specific indexes
    check_index_data(client, "visual-embeddings", "Visual Index (Multi-Index Mode)")
    check_index_data(client, "audio-embeddings", "Audio Index (Multi-Index Mode)")
    check_index_data(client, "transcription-embeddings", "Transcription Index (Multi-Index Mode)")

    print("\n" + "=" * 80)
    print("Diagnosis:")
    print("=" * 80)
    print("""
If unified-embeddings only has transcription vectors, this means:
1. Lambda function is NOT writing visual/audio to unified index, OR
2. Videos were processed before dual-write was implemented, OR
3. Lambda is calling store_all_segments with dual_write=False

Solution:
- Check lambda_function.py line 184 to ensure dual_write=True (default)
- Reprocess at least one video to populate all modality types in unified index
- Verify lambda logs show "unified" writes for visual and audio
""")


if __name__ == "__main__":
    main()
