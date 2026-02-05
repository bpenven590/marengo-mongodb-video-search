#!/usr/bin/env python3
"""
Migrate MongoDB from single-index to multi-index mode.

Copies documents from unified-embeddings collection into modality-specific
collections (visual_embeddings, audio_embeddings, transcription_embeddings).

IMPORTANT: After running this script, you MUST create vector search indexes
in MongoDB Atlas for each new collection:
- visual_embeddings → visual_embeddings_vector_index
- audio_embeddings → audio_embeddings_vector_index
- transcription_embeddings → transcription_embeddings_vector_index

Index configuration:
- Type: Vector Search
- Path: embedding
- Similarity: cosine
- Dimensions: 512
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from mongodb_client import MongoDBEmbeddingClient


def main():
    print("=" * 80)
    print("MongoDB Multi-Index Migration")
    print("=" * 80)
    print()

    # Initialize client
    mongodb_uri = os.environ.get("MONGODB_URI")
    if not mongodb_uri:
        print("❌ MONGODB_URI not set")
        return 1

    client = MongoDBEmbeddingClient(connection_string=mongodb_uri)

    # Get current stats
    print("Current unified-embeddings collection stats:")
    stats = client.get_collection_stats()
    print(f"  Total documents: {stats['total_documents']}")
    print(f"  Visual:          {stats['by_modality']['visual']}")
    print(f"  Audio:           {stats['by_modality']['audio']}")
    print(f"  Transcription:   {stats['by_modality']['transcription']}")
    print()

    if stats['total_documents'] == 0:
        print("❌ No documents to migrate!")
        return 1

    # Confirm migration
    print("This will copy documents from unified-embeddings into:")
    print("  - visual_embeddings")
    print("  - audio_embeddings")
    print("  - transcription_embeddings")
    print()
    print("⚠️  The unified-embeddings collection will remain unchanged (safe migration)")
    print()

    response = input("Continue? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Migration cancelled")
        return 0

    print()
    print("-" * 80)
    print("Starting migration...")
    print("-" * 80)
    print()

    # Migrate each modality
    migration_stats = {}

    for modality in ["visual", "audio", "transcription"]:
        print(f"Migrating {modality} embeddings...")

        # Get source collection and target collection
        source_collection = client.collection
        target_collection_name = client.MODALITY_COLLECTIONS[modality]
        target_collection = client.db[target_collection_name]

        # Find all documents for this modality
        cursor = source_collection.find({"modality_type": modality})

        # Copy documents (remove _id to let MongoDB generate new ones)
        documents_to_insert = []
        for doc in cursor:
            # Remove _id and modality_type (not needed in modality-specific collection)
            doc_copy = {k: v for k, v in doc.items() if k not in ['_id', 'modality_type']}
            documents_to_insert.append(doc_copy)

        # Bulk insert
        if documents_to_insert:
            result = target_collection.insert_many(documents_to_insert)
            inserted_count = len(result.inserted_ids)
            migration_stats[modality] = inserted_count
            print(f"  ✅ Inserted {inserted_count} documents into {target_collection_name}")
        else:
            migration_stats[modality] = 0
            print(f"  ⚠️  No documents found for {modality}")

        print()

    # Summary
    print("=" * 80)
    print("Migration Summary")
    print("=" * 80)
    print()
    print(f"Visual embeddings:        {migration_stats.get('visual', 0)} documents")
    print(f"Audio embeddings:         {migration_stats.get('audio', 0)} documents")
    print(f"Transcription embeddings: {migration_stats.get('transcription', 0)} documents")
    print(f"Total migrated:           {sum(migration_stats.values())} documents")
    print()

    # Verify migration
    print("-" * 80)
    print("Verification")
    print("-" * 80)
    print()

    for modality in ["visual", "audio", "transcription"]:
        collection_name = client.MODALITY_COLLECTIONS[modality]
        collection = client.db[collection_name]
        count = collection.count_documents({})
        expected = migration_stats.get(modality, 0)

        if count == expected:
            print(f"✅ {collection_name}: {count} documents")
        else:
            print(f"❌ {collection_name}: {count} documents (expected {expected})")

    print()
    print("=" * 80)
    print("⚠️  IMPORTANT: Next Steps")
    print("=" * 80)
    print()
    print("You MUST create vector search indexes in MongoDB Atlas:")
    print()
    print("1. Go to MongoDB Atlas → Database → Search → Create Index")
    print()
    print("2. Create these 3 vector search indexes:")
    print()
    print("   Collection: visual_embeddings")
    print("   Index Name: visual_embeddings_vector_index")
    print("   Field Path: embedding")
    print("   Type: vectorSearch")
    print("   Similarity: cosine")
    print("   Dimensions: 512")
    print()
    print("   Collection: audio_embeddings")
    print("   Index Name: audio_embeddings_vector_index")
    print("   Field Path: embedding")
    print("   Type: vectorSearch")
    print("   Similarity: cosine")
    print("   Dimensions: 512")
    print()
    print("   Collection: transcription_embeddings")
    print("   Index Name: transcription_embeddings_vector_index")
    print("   Field Path: embedding")
    print("   Type: vectorSearch")
    print("   Similarity: cosine")
    print("   Dimensions: 512")
    print()
    print("3. Wait for indexes to build (may take several minutes)")
    print()
    print("4. Test multi-index mode by setting use_multi_index=True in search requests")
    print()
    print("=" * 80)

    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
