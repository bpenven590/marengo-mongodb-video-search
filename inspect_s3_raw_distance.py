#!/usr/bin/env python3
"""
Inspect raw distance values returned by S3 Vectors API.
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

    query = "Ross says I take thee Rachel at a wedding"
    result = bedrock_client.get_text_query_embedding(query)
    query_embedding = result["embedding"]

    print("Raw S3 Vectors API Response")
    print("=" * 80)
    print()

    for modality in ["transcription"]:  # Just check one
        print(f"Modality: {modality}")
        print("-" * 80)

        # Call S3 Vectors API directly to see raw response
        index_name = s3v_client.INDEX_NAMES[modality]
        response = s3v_client.client.query_vectors(
            vectorBucketName=s3v_client.bucket_name,
            indexName=index_name,
            queryVector={"float32": query_embedding},
            topK=5,
            returnMetadata=True,
            returnDistance=True
        )

        vectors = response.get("vectors", [])
        print(f"Retrieved {len(vectors)} vectors")
        print()

        for i, vector in enumerate(vectors[:5], 1):
            distance = vector.get("distance", 0)
            converted_score = 1 - distance

            print(f"Result {i}:")
            print(f"  Raw distance: {distance:.6f}")
            print(f"  Our conversion (1 - distance): {converted_score:.6f}")
            print(f"  Expected (from MongoDB): ~0.89 for top result")
            print()

if __name__ == "__main__":
    main()
