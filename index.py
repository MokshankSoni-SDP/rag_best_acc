import json
import os
from qdrant_client import QdrantClient
from qdrant_client.http import models

# --- CONFIGURATION ---
INPUT_FILE = "semantic_chunks.json"
COLLECTION_NAME = "rag_collection_demo"  # Name of your database table
# ---------------------

def index_data():
    print("🚀 Starting Indexing Pipeline...")

    # 1. Load the Chunks
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: '{INPUT_FILE}' not found. Run chunking.py first.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    print(f"   Loaded {len(chunks)} semantic chunks.")

    # 2. Initialize Qdrant (Local Mode)
    # This creates a folder named 'qdrant_db' in your project to store data.
    client = QdrantClient(path="qdrant_db") 
    
    print("   ✅ Qdrant Client initialized (Local mode).")

    # 3. Create Collection (The "Table")
    # We configure it for Hybrid Search (Dense + Sparse)
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        print(f"   Creating collection '{COLLECTION_NAME}'...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=client.get_fastembed_vector_params(), # Auto-configure for BGE-M3
            sparse_vectors_config=client.get_fastembed_sparse_vector_params(), # Auto-configure for Sparse
        )
    else:
        print(f"   Collection '{COLLECTION_NAME}' already exists. Appending data...")

    # 4. Generate Embeddings & Upsert
    # Qdrant's 'add' method with FastEmbed does the heavy lifting:
    # It automatically downloads BGE-M3, embeds the text, and uploads it.
    
    documents = [chunk["content"] for chunk in chunks]
    metadata = [chunk for chunk in chunks] # We store the full chunk info as metadata
    
    print("   🧠 Generating BGE-M3 vectors (Dense + Sparse)... (This may take time on first run)")
    
    client.add(
        collection_name=COLLECTION_NAME,
        documents=documents,
        metadata=metadata,
        ids=list(range(len(chunks))) # Simple ID 0, 1, 2...
    )

    print(f"🎉 SUCCESS! Indexed {len(chunks)} documents into Qdrant.")
    print("   Data is stored in the 'qdrant_db' folder.")

if __name__ == "__main__":
    index_data()