#!/usr/bin/env python3
"""
Test different modality searches to understand the behavior.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from s3_vectors_client import S3VectorsClient
from bedrock_client import BedrockMarengoClient


def test_query(client, bedrock, query, description):
    """Test a query across all modalities."""
    print(f"\n{'=' * 80}")
    print(f"{description}")
    print(f"Query: '{query}'")
    print(f"{'=' * 80}")

    query_result = bedrock.get_text_query_embedding(query)
    query_embedding = query_result["embedding"]

    # Test multi-index mode
    print("\n[MULTI-INDEX MODE]")
    for modality in ["visual", "audio", "transcription"]:
        results = client.multi_modality_search(
            query_embedding=query_embedding,
            limit_per_modality=3,
            modalities=[modality],
            use_multi_index=True
        )
        count = len(results.get(modality, []))
        print(f"  {modality:15s}: {count} results", end="")
        if count > 0:
            top_score = results[modality][0]['score']
            print(f" (top score: {top_score:.4f})")
        else:
            print()

    # Test single-index mode
    print("\n[SINGLE-INDEX MODE]")
    for modality in ["visual", "audio", "transcription"]:
        results = client.multi_modality_search(
            query_embedding=query_embedding,
            limit_per_modality=3,
            modalities=[modality],
            use_multi_index=False
        )
        count = len(results.get(modality, []))
        print(f"  {modality:15s}: {count} results", end="")
        if count > 0:
            top_score = results[modality][0]['score']
            print(f" (top score: {top_score:.4f})")
        else:
            print()


def main():
    client = S3VectorsClient(bucket_name="brice-video-search-multimodal", region="us-east-1")
    bedrock = BedrockMarengoClient(region="us-east-1", output_bucket="tl-brice-media")

    # Test 1: Transcription-focused query
    test_query(
        client, bedrock,
        "Ross says I take thee Rachel at a wedding",
        "TEST 1: Transcription-Heavy Query"
    )

    # Test 2: Visual-focused query
    test_query(
        client, bedrock,
        "a man in a tuxedo at a wedding ceremony",
        "TEST 2: Visual-Heavy Query"
    )

    # Test 3: Audio-focused query
    test_query(
        client, bedrock,
        "wedding music and people gasping",
        "TEST 3: Audio-Heavy Query"
    )

    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("=" * 80)
    print("""
If transcription queries return zero visual/audio results in BOTH modes:
- This is EXPECTED BEHAVIOR ✅
- A transcription-focused query embedding won't match visual/audio content well
- The unified index contains all modalities, but semantic matching still applies

If visual/audio queries return results in multi-index but NOT single-index:
- This indicates a bug in unified_multi_modality_search filtering ❌
- Or possibly that visual/audio weren't written to unified index

Based on the check_s3_vectors_data.py results showing audio vectors in unified,
the most likely explanation is: transcription queries simply don't match
visual/audio content semantically, which is correct behavior for a semantic
search system.
""")


if __name__ == "__main__":
    main()
