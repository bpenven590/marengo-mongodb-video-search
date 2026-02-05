#!/usr/bin/env python3
"""
Check S3 Vectors unified-embeddings index to see what modalities are stored.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from s3_vectors_client import S3VectorsClient

def main():
    client = S3VectorsClient(
        bucket_name="brice-video-search-multimodal",
        region="us-east-1"
    )

    print("=" * 80)
    print("S3 Vectors unified-embeddings Index Check")
    print("=" * 80)
    print()

    # List vectors to see what's in the index
    try:
        response = client.client.list_vectors(
            vectorBucketName=client.bucket_name,
            indexName=client.UNIFIED_INDEX_NAME,
            maxResults=100
        )

        vectors = response.get("vectors", [])
        print(f"Retrieved {len(vectors)} vectors from unified-embeddings index")
        print()

        # Count by modality
        modality_counts = {"visual": 0, "audio": 0, "transcription": 0}
        video_ids = set()

        for vector in vectors:
            metadata = vector.get("metadata", {})
            modality = metadata.get("modality_type", "unknown")
            video_id = metadata.get("video_id", "unknown")

            if modality in modality_counts:
                modality_counts[modality] += 1

            video_ids.add(video_id)

        print(f"Modality distribution in unified-embeddings index:")
        print(f"  Visual:        {modality_counts['visual']} vectors")
        print(f"  Audio:         {modality_counts['audio']} vectors")
        print(f"  Transcription: {modality_counts['transcription']} vectors")
        print(f"  Total:         {sum(modality_counts.values())} vectors")
        print()
        print(f"Unique videos: {len(video_ids)}")
        print()

        # Show first 5 sample records
        print("Sample records:")
        for i, vector in enumerate(vectors[:5]):
            key = vector.get("key", "")
            metadata = vector.get("metadata", {})
            print(f"\n  Record {i+1}:")
            print(f"    key: {key}")
            print(f"    video_id: {metadata.get('video_id')}")
            print(f"    segment_id: {metadata.get('segment_id')}")
            print(f"    modality_type: {metadata.get('modality_type')}")
            print(f"    s3_uri: {metadata.get('s3_uri', '')[:60] if metadata.get('s3_uri') else 'None'}...")

        print()
        print("=" * 80)

        if modality_counts['visual'] == 0 or modality_counts['audio'] == 0:
            print("❌ PROBLEM: Missing visual or audio vectors in unified index!")
            print("   The import script may have failed to import all modalities.")
            print("   You need to re-import using import_s3_to_s3vectors.py")
        else:
            print("✅ All modalities present in unified-embeddings index")

    except Exception as e:
        print(f"❌ Error querying S3 Vectors: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
