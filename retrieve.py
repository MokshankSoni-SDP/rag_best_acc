from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

# --- CONFIGURATION ---
COLLECTION_NAME = "rag_collection_demo"
# We use the 'base' model. It is lighter than 'large' but smarter than FlashRank.
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base" 
# ---------------------

def search_and_rerank(query_text):
    print(f"\n🔎 User Query: '{query_text}'")
    
    # 1. Initialize Client
    client = QdrantClient(path="qdrant_db")
    
    # 2. HYBRID SEARCH (Retrieve Candidates)
    print("   ...Retrieving top 25 candidates (Hybrid Search)...")
    
    # We ask for MORE candidates (25) because the Reranker will filter the bad ones.
    search_results = client.query(
        collection_name=COLLECTION_NAME,
        query_text=query_text,
        limit=25 
    )
    
    if not search_results:
        print("   ❌ No documents found.")
        return []
    
    # 3. PREPARE FOR RERANKER
    # The CrossEncoder needs pairs: [ [Query, Text1], [Query, Text2], ... ]
    # It compares them directly to see how closely they match.
    passages = []
    cross_encoder_inputs = []
    
    for hit in search_results:
        text = hit.metadata["content"]
        passages.append(hit) # Keep the original object so we can return it later
        cross_encoder_inputs.append([query_text, text]) # The pair for the AI

    # 4. RERANKING (The Accuracy Step)
    print(f"   ...Reranking {len(passages)} candidates with {RERANKER_MODEL_NAME}...")
    
    # Load the model (downloads automatically on first run)
    model = CrossEncoder(RERANKER_MODEL_NAME)
    
    # Predict scores (higher is better)
    scores = model.predict(cross_encoder_inputs)
    
    # 5. SORT & FILTER
    # We zip the scores with the passages and sort them by score (descending)
    ranked_results = sorted(zip(scores, passages), key=lambda x: x[0], reverse=True)
    
    # Keep Top 5
    final_top_8 = []
    for score, hit in ranked_results[:8]:
        final_top_8.append({
            "score": float(score), # Convert numpy float to python float
            "text": hit.metadata["content"],
            "meta": hit.metadata
        })

    print(f"   ✅ Returning top {len(final_top_8)} highly relevant chunks.\n")
    return final_top_8

if __name__ == "__main__":
    # Test Query
    test_query = "What is the economic impact of AI in 2026?"
    
    results = search_and_rerank(test_query)
    
    for i, res in enumerate(results):
        print(f"Rank {i+1} (Score: {res['score']:.4f}):")
        print(f"   {res['text'][:150]}...")
        print("-" * 40)