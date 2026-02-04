#!/usr/bin/env python3
"""
Migrate embeddings from single collection to modality-specific collections.

Creates three new collections:
- visual_embeddings
- audio_embeddings
- transcription_embeddings

Each with their own vector index for comparison testing.
"""

import os
import sys
from pymongo import MongoClient

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def migrate_to_multi_index(mongodb_uri: str, database_name: str = "video_search"):
    """Split video_embeddings into modality-specific collections."""

    client = MongoClient(mongodb_uri)
    db = client[database_name]

    source_collection = db["video_embeddings"]

    # Target collections
    collections = {
        "visual": db["visual_embeddings"],
        "audio": db["audio_embeddings"],
        "transcription": db["transcription_embeddings"]
    }

    # Check source count
    total = source_collection.count_documents({})
    print(f"Source collection has {total} documents")

    for modality, target_collection in collections.items():
        # Check if already migrated
        existing = target_collection.count_documents({})
        if existing > 0:
            print(f"  {modality}_embeddings already has {existing} documents, skipping...")
            continue

        # Count documents for this modality
        count = source_collection.count_documents({"modality_type": modality})
        print(f"\nMigrating {count} {modality} embeddings...")

        # Copy documents (excluding modality_type field since it's implicit)
        pipeline = [
            {"$match": {"modality_type": modality}},
            {"$project": {
                "video_id": 1,
                "segment_id": 1,
                "s3_uri": 1,
                "embedding": 1,
                "start_time": 1,
                "end_time": 1,
                "created_at": 1
                # modality_type excluded - implicit in collection name
            }}
        ]

        docs = list(source_collection.aggregate(pipeline))

        if docs:
            # Remove _id to let MongoDB generate new ones
            for doc in docs:
                doc.pop("_id", None)

            result = target_collection.insert_many(docs)
            print(f"  Inserted {len(result.inserted_ids)} documents into {modality}_embeddings")
        else:
            print(f"  No documents found for modality: {modality}")

    # Summary
    print("\n=== Migration Summary ===")
    for modality in collections:
        count = collections[modality].count_documents({})
        print(f"  {modality}_embeddings: {count} documents")

    client.close()
    print("\nMigration complete!")
    print("\nNext step: Create vector indexes on each collection using MongoDB Atlas UI or CLI")
    print("See scripts/mongodb_setup.md for index definitions")


def create_indexes_info():
    """Print the index definitions needed for each collection."""

    index_template = '''
{{
  "fields": [
    {{ "type": "vector", "path": "embedding", "numDimensions": 512, "similarity": "cosine" }},
    {{ "type": "filter", "path": "video_id" }}
  ]
}}
'''

    print("\n=== Vector Index Definitions ===")
    print("Create these indexes in MongoDB Atlas UI (Search Indexes):")
    print()

    for modality in ["visual", "audio", "transcription"]:
        print(f"Collection: {modality}_embeddings")
        print(f"Index name: vector_index")
        print(f"Definition:{index_template}")


if __name__ == "__main__":
    mongodb_uri = os.environ.get("MONGODB_URI")

    if not mongodb_uri:
        print("Error: MONGODB_URI environment variable not set")
        print("Usage: MONGODB_URI='mongodb+srv://...' python migrate_to_multi_index.py")
        sys.exit(1)

    migrate_to_multi_index(mongodb_uri)
    create_indexes_info()
