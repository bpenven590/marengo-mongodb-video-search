#!/usr/bin/env python3
"""
Delete the S3 Vectors unified-embeddings index.
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
    print("Delete S3 Vectors unified-embeddings Index")
    print("=" * 80)
    print()

    try:
        print(f"Deleting index: {client.UNIFIED_INDEX_NAME}")
        print()

        response = client.client.delete_index(
            vectorBucketName=client.bucket_name,
            indexName=client.UNIFIED_INDEX_NAME
        )

        print("✅ Index deleted successfully!")
        print()
        print(f"Response: {response}")
        print()
        print("=" * 80)

    except Exception as e:
        if "ResourceNotFoundException" in str(e):
            print("⚠️  Index already deleted or doesn't exist")
        else:
            print(f"❌ Error deleting index: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
