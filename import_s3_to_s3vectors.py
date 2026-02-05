#!/usr/bin/env python3
"""
Import embeddings from S3 (tl-brice-media/embeddings/) to S3 Vectors unified-embeddings index.
"""

import sys
import os
import json
import boto3
from botocore.exceptions import ClientError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from s3_vectors_client import S3VectorsClient


def get_s3_embedding_folders(bucket: str, prefix: str = "embeddings/"):
    """List all video embedding folders in S3."""
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    folders = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for prefix_info in page.get("CommonPrefixes", []):
            folders.append(prefix_info["Prefix"])

    return folders


def load_embeddings_from_s3(bucket: str, video_folder: str):
    """Load embeddings JSON from S3 for a video."""
    s3 = boto3.client("s3")

    # List objects in the video folder to find the output.json
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=video_folder):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith("/output.json"):
                # Found the output.json
                response = s3.get_object(Bucket=bucket, Key=obj["Key"])
                content = response["Body"].read().decode("utf-8")
                return json.loads(content)

    return None


def extract_video_id_from_path(video_folder: str):
    """Extract video ID from S3 path."""
    # Example: embeddings/WBD_project_Videos_Ready_Friends.1994.S05E01.The.One.After.Ross.Says.Rachel.mp4/
    parts = video_folder.rstrip("/").split("/")
    if len(parts) >= 2:
        filename = parts[-1]
        # Use last 8 chars of filename as video_id (matches MongoDB)
        import hashlib
        return hashlib.md5(filename.encode()).hexdigest()[:8]
    return "unknown"


def import_video_to_unified_index(s3v_client: S3VectorsClient, bucket: str, video_folder: str):
    """Import a single video's embeddings to unified-embeddings index."""
    print(f"\nProcessing: {video_folder}")

    # Load embeddings from S3
    data = load_embeddings_from_s3(bucket, video_folder)
    if not data:
        print(f"  ❌ No output.json found")
        return False

    embeddings_data = data.get("data", [])
    if not embeddings_data:
        print(f"  ❌ No embeddings in output.json")
        return False

    # Extract video ID
    video_id = extract_video_id_from_path(video_folder)

    # Group by segment and modality
    segments = {}
    for item in embeddings_data:
        start_sec = item["startSec"]
        end_sec = item["endSec"]
        modality = item["embeddingOption"]
        embedding = item["embedding"]

        # Create segment key
        segment_key = f"{start_sec}_{end_sec}"

        if segment_key not in segments:
            segments[segment_key] = {
                "start_time": start_sec,
                "end_time": end_sec,
                "embeddings": {}
            }

        segments[segment_key]["embeddings"][modality] = embedding

    # Convert to list with segment IDs
    segment_list = []
    for i, (seg_key, seg_data) in enumerate(sorted(segments.items())):
        # Extract just the filename: WBD_project_Videos_Ready_Friends...mp4 → Friends...mp4
        folder_name = video_folder.split('/')[-2]  # Get folder name with .mp4
        filename = folder_name.replace('WBD_project_Videos_Ready_', '').replace('WBD_project_Videos_', '')

        segment_list.append({
            "segment_id": i,
            "start_time": seg_data["start_time"],
            "end_time": seg_data["end_time"],
            "s3_uri": f"s3://{bucket}/WBD_project/Videos/proxy/{filename}",  # No extra .mp4!
            "embeddings": seg_data["embeddings"]
        })

    print(f"  Found {len(segment_list)} segments")

    # Prepare vectors for unified index
    unified_vectors = []
    for segment in segment_list:
        segment_id = segment["segment_id"]
        s3_uri = segment["s3_uri"]
        start_time = segment["start_time"]
        end_time = segment["end_time"]
        embeddings = segment.get("embeddings", {})

        for modality in ["visual", "audio", "transcription"]:
            if modality not in embeddings or not embeddings[modality]:
                continue

            unified_key = f"{video_id}_{segment_id}_{modality}"
            unified_vectors.append({
                "key": unified_key,
                "data": {"float32": embeddings[modality]},
                "metadata": {
                    "video_id": video_id,
                    "segment_id": str(segment_id),
                    "modality_type": modality,
                    "s3_uri": s3_uri,
                    "start_time": str(start_time),
                    "end_time": str(end_time)
                }
            })

    print(f"  Total vectors to write: {len(unified_vectors)}")

    # Write to unified index in batches of 100
    batch_size = 100
    success_count = 0

    for i in range(0, len(unified_vectors), batch_size):
        batch = unified_vectors[i:i + batch_size]
        try:
            s3v_client.client.put_vectors(
                vectorBucketName=s3v_client.bucket_name,
                indexName=s3v_client.UNIFIED_INDEX_NAME,
                vectors=batch
            )
            success_count += len(batch)
            print(f"  ✅ Batch {i//batch_size + 1}: {len(batch)} vectors written")
        except Exception as e:
            print(f"  ❌ Batch {i//batch_size + 1} failed: {e}")

    print(f"  ✅ Total: {success_count}/{len(unified_vectors)} vectors written")
    return success_count > 0


def main():
    # Configuration
    source_bucket = "tl-brice-media"
    source_prefix = "embeddings/"

    # Initialize S3 Vectors client
    s3v_client = S3VectorsClient(
        bucket_name="brice-video-search-multimodal",
        region="us-east-1"
    )

    print("=" * 80)
    print("S3 to S3 Vectors Unified Index Import")
    print("=" * 80)
    print(f"Source: s3://{source_bucket}/{source_prefix}")
    print(f"Target: S3 Vectors unified-embeddings index")
    print()

    # Get all video folders
    video_folders = get_s3_embedding_folders(source_bucket, source_prefix)
    print(f"Found {len(video_folders)} video folders to import")
    print()

    # Import each video
    success_count = 0
    for video_folder in video_folders:
        if import_video_to_unified_index(s3v_client, source_bucket, video_folder):
            success_count += 1

    print()
    print("=" * 80)
    print(f"Import complete: {success_count}/{len(video_folders)} videos imported")
    print("=" * 80)


if __name__ == "__main__":
    main()
