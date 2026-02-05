#!/usr/bin/env python3
"""
Automated Video Processing Pipeline

Processes videos through the full pipeline:
1. Generate embeddings via Bedrock Marengo 3.0
2. Store in MongoDB Atlas
3. Store in S3 Vectors (multi-index)
4. Move video from Ready/ to proxy/ folder

Usage:
    python process_video_pipeline.py s3://bucket/path/to/video.mp4
    python process_video_pipeline.py --watch  # Watch Ready folder
"""

import os
import sys
import time
import argparse
from urllib.parse import urlparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import boto3
from pymongo import MongoClient
from bedrock_client import BedrockMarengoClient
from s3_vectors_client import S3VectorsClient

# Configuration
MONGODB_URI = os.environ.get("MONGODB_URI")
S3_BUCKET = "tl-brice-media"
READY_PREFIX = "WBD_project/Videos/Ready/"
PROXY_PREFIX = "WBD_project/Videos/proxy/"
AWS_REGION = "us-east-1"
AWS_PROFILE = "TlFullDevelopmentAccess-026090552520"


class VideoProcessingPipeline:
    """Automated video processing pipeline."""

    def __init__(self):
        """Initialize the pipeline with all required clients."""
        # AWS clients
        session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
        self.s3_client = session.client("s3")

        # Bedrock Marengo client
        self.bedrock = BedrockMarengoClient(
            region=AWS_REGION,
            output_bucket=S3_BUCKET
        )

        # S3 Vectors client
        self.s3_vectors = S3VectorsClient(
            bucket_name="brice-video-search-multimodal",
            region=AWS_REGION,
            profile_name=AWS_PROFILE
        )

        # MongoDB client
        if not MONGODB_URI:
            raise ValueError("MONGODB_URI environment variable required")
        self.mongo_client = MongoClient(MONGODB_URI)
        self.db = self.mongo_client["video_search"]
        self.collection = self.db["video_embeddings"]

    def process_video(self, s3_uri: str) -> dict:
        """
        Process a video through the complete pipeline.

        Args:
            s3_uri: S3 URI of the video (s3://bucket/key)

        Returns:
            Dict with processing results
        """
        print(f"\n{'='*60}")
        print(f"Processing: {s3_uri}")
        print("="*60)

        # Parse S3 URI
        parsed = urlparse(s3_uri)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")

        # Extract video ID from filename
        filename = os.path.basename(key)
        video_id = os.path.splitext(filename)[0]

        results = {
            "video_id": video_id,
            "s3_uri": s3_uri,
            "status": "processing",
            "steps": {}
        }

        try:
            # Step 1: Generate embeddings via Bedrock Marengo
            print("\n[1/4] Generating embeddings via Bedrock Marengo 3.0...")
            embeddings_data = self.bedrock.get_video_embeddings(
                bucket=bucket,
                s3_key=key,
                embedding_types=["visual", "audio", "transcription"],
                segment_length_sec=6
            )

            segments = embeddings_data["segments"]
            print(f"✓ Generated embeddings for {len(segments)} segments")
            results["steps"]["bedrock"] = {
                "status": "success",
                "segments": len(segments)
            }

            # Determine proxy S3 URI (where video will be after moving)
            if key.startswith(READY_PREFIX):
                proxy_key = key.replace(READY_PREFIX, PROXY_PREFIX)
                proxy_s3_uri = f"s3://{bucket}/{proxy_key}"
            else:
                proxy_s3_uri = s3_uri

            # Step 2: Store in MongoDB Atlas
            print("\n[2/4] Storing embeddings in MongoDB Atlas...")
            mongo_docs = []
            for segment in segments:
                # Store each modality separately in MongoDB
                # Use proxy S3 URI so app serves from proxy location
                for modality in ["visual", "audio", "transcription"]:
                    if modality in segment["embeddings"]:
                        doc = {
                            "video_id": video_id,
                            "segment_id": segment["segment_id"],
                            "s3_uri": proxy_s3_uri,  # Store proxy path, not Ready path
                            "start_time": segment["start_time"],
                            "end_time": segment["end_time"],
                            "modality_type": modality,
                            "embedding": segment["embeddings"][modality]
                        }
                        mongo_docs.append(doc)

            if mongo_docs:
                self.collection.insert_many(mongo_docs)
                print(f"✓ Stored {len(mongo_docs)} embeddings in MongoDB")
                results["steps"]["mongodb"] = {
                    "status": "success",
                    "documents": len(mongo_docs)
                }

            # Step 3: Store in S3 Vectors (multi-index)
            print("\n[3/4] Storing embeddings in S3 Vectors...")
            s3v_result = self.s3_vectors.store_all_segments(video_id, segments)
            print(f"✓ Stored in S3 Vectors:")
            print(f"  - Visual: {s3v_result['visual_stored']} segments")
            print(f"  - Audio: {s3v_result['audio_stored']} segments")
            print(f"  - Transcription: {s3v_result['transcription_stored']} segments")
            results["steps"]["s3_vectors"] = s3v_result

            # Step 4: Move video from Ready/ to proxy/
            print("\n[4/4] Moving video to proxy folder...")
            if key.startswith(READY_PREFIX):
                # Destination key in proxy folder
                proxy_key = key.replace(READY_PREFIX, PROXY_PREFIX)

                # Copy to proxy location
                copy_source = {"Bucket": bucket, "Key": key}
                self.s3_client.copy_object(
                    CopySource=copy_source,
                    Bucket=bucket,
                    Key=proxy_key
                )

                # Delete from Ready location
                self.s3_client.delete_object(Bucket=bucket, Key=key)

                print(f"✓ Moved to: s3://{bucket}/{proxy_key}")
                results["steps"]["move"] = {
                    "status": "success",
                    "from": key,
                    "to": proxy_key
                }
                results["proxy_s3_uri"] = f"s3://{bucket}/{proxy_key}"
            else:
                print(f"⚠ Video not in Ready folder, skipping move")
                results["steps"]["move"] = {
                    "status": "skipped",
                    "reason": "not in Ready folder"
                }

            results["status"] = "completed"
            print(f"\n{'='*60}")
            print(f"✓ Processing complete for {video_id}")
            print("="*60)

        except Exception as e:
            print(f"\n✗ Error during processing: {e}")
            import traceback
            traceback.print_exc()
            results["status"] = "failed"
            results["error"] = str(e)

        return results

    def watch_folder(self, interval: int = 60):
        """
        Watch the Ready folder for new videos and process them.

        Args:
            interval: Check interval in seconds (default: 60)
        """
        print(f"Watching s3://{S3_BUCKET}/{READY_PREFIX} every {interval}s...")
        print("Press Ctrl+C to stop")

        processed_keys = set()

        try:
            while True:
                # List objects in Ready folder
                response = self.s3_client.list_objects_v2(
                    Bucket=S3_BUCKET,
                    Prefix=READY_PREFIX
                )

                if "Contents" in response:
                    for obj in response["Contents"]:
                        key = obj["Key"]

                        # Skip if already processed or not a video file
                        if key in processed_keys or not self._is_video_file(key):
                            continue

                        # Process the video
                        s3_uri = f"s3://{S3_BUCKET}/{key}"
                        print(f"\nNew video detected: {key}")

                        result = self.process_video(s3_uri)

                        if result["status"] == "completed":
                            processed_keys.add(key)

                # Wait before next check
                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n\nStopping watcher...")

    def _is_video_file(self, key: str) -> bool:
        """Check if the file is a video."""
        video_extensions = [".mp4", ".mov", ".avi", ".mkv", ".webm"]
        return any(key.lower().endswith(ext) for ext in video_extensions)

    def close(self):
        """Close connections."""
        self.mongo_client.close()


def main():
    parser = argparse.ArgumentParser(
        description="Automated video processing pipeline"
    )
    parser.add_argument(
        "video_uri",
        nargs="?",
        help="S3 URI of video to process (s3://bucket/key)"
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch Ready folder for new videos"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Watch interval in seconds (default: 60)"
    )

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = VideoProcessingPipeline()

    try:
        if args.watch:
            # Watch mode
            pipeline.watch_folder(interval=args.interval)
        elif args.video_uri:
            # Process single video
            result = pipeline.process_video(args.video_uri)

            # Print summary
            print("\nProcessing Summary:")
            print(f"Status: {result['status']}")
            if result["status"] == "failed":
                print(f"Error: {result.get('error', 'Unknown')}")
        else:
            parser.print_help()
            sys.exit(1)

    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
