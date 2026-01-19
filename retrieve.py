from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
COLLECTION_NAME = "rag_collection_demo" # Using the existing collection as per user state
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base" 
API_KEY = os.getenv("GROQ_API_KEY")

# Check if API Key is available
if not API_KEY and "gsk_" not in str(os.getenv("GROQ_API_KEY", "")):
    # Fallback or strict warning. Assuming user has .env or will set it.
    print("⚠️ WARNING: GROQ_API_KEY not found in environment.")
# ---------------------

def generate_query_variations(query):
    """
    Uses LLM to generate 3 variations of the user query for better coverage.
    """
    print(f"   ✨ Generating query variations for: '{query}'")
    
    if not API_KEY:
        print("   ⚠️ No API Key, skipping expansion.")
        return [query]

    client = Groq(api_key=API_KEY)
    
    prompt = f"""
    You are an AI assistant. Your task is to generate 3 different search queries based on the user's question.
    These queries should be optimized for a Vector Search engine (semantic similarity).
    
    User Question: "{query}"
    
    Return ONLY the 3 queries, one per line. Do not include numbering or bullets.
    """
    
    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
        )
        content = completion.choices[0].message.content
        variations = [line.strip() for line in content.split("\n") if line.strip()]
        # Always include the original query!
        variations.insert(0, query)
        return variations[:4] # Cap at 4 total
    except Exception as e:
        print(f"   ⚠️ Query expansion failed: {e}. using original query only.")
        return [query]

def search_and_rerank(query_text):
    print(f"\n🔎 User Query: '{query_text}'")
    
    # 1. Initialize Client
    client = QdrantClient(path="qdrant_db")
    
    # 2. QUERY EXPANSION
    queries = generate_query_variations(query_text)
    print(f"   🔍 Searching for: {queries}")

    all_results = []
    
    # 3. HYBRID SEARCH (Loop through variations)
    for q in queries:
        # We rely on client.query doing the embedding automatically via FastEmbed
        # (Assuming the collection was created with FastEmbed support as per index.py)
        try:
            search_results = client.query(
                collection_name=COLLECTION_NAME,
                query_text=q,
                limit=25 # Grab 25 per variation
            )
            all_results.extend(search_results)
        except Exception as e:
            print(f"   ⚠️ Search failed for query '{q}': {e}")
            
    print(f"   ...Collected {len(all_results)} raw candidates...")

    # 4. DEDUPLICATION
    # We remove duplicates based on the ID
    unique_results = {}
    for hit in all_results:
        # Use database ID as unique key
        if hit.id not in unique_results:
            unique_results[hit.id] = hit
    
    candidates = list(unique_results.values())
    print(f"   ...Deduplicated to {len(candidates)} unique candidates...")
    
    if not candidates:
        print("   ❌ No documents found.")
        return []
    
    # 5. PREPARE FOR RERANKER
    passages = []
    cross_encoder_inputs = []
    
    for hit in candidates:
        # Handle metadata safely
        text = hit.metadata.get("content", "") if hasattr(hit, "metadata") else hit.payload.get("content", "")
        if not text:
             continue
             
        passages.append(hit) 
        # Compare Query vs Text. We use the ORIGINAL query for reranking, 
        # because that is the user's true intent.
        cross_encoder_inputs.append([query_text, text]) 

    # 6. RERANKING
    print(f"   ...Reranking {len(candidates)} candidates with {RERANKER_MODEL_NAME}...")
    
    try:
        model = CrossEncoder(RERANKER_MODEL_NAME)
        scores = model.predict(cross_encoder_inputs)
        
        # 7. SORT & FILTER
        ranked_results = sorted(zip(scores, passages), key=lambda x: x[0], reverse=True)
        
        # Keep Top 8
        final_top_8 = []
        for score, hit in ranked_results[:8]:
            # Handle metadata access again
            content = hit.metadata.get("content", "") if hasattr(hit, "metadata") else hit.payload.get("content", "")
            meta = hit.metadata if hasattr(hit, "metadata") else hit.payload
            
            final_top_8.append({
                "score": float(score),
                "text": content,
                "meta": meta
            })

        print(f"   ✅ Returning top {len(final_top_8)} highly relevant chunks.\n")
        return final_top_8

    except Exception as e:
        print(f"   ❌ Reranking failed: {e}. Returning unranked top results.")
        # Fallback: just return the top search results without reranking logic if model fails
        fallback = []
        for hit in candidates[:5]:
             content = hit.metadata.get("content", "") if hasattr(hit, "metadata") else hit.payload.get("content", "")
             meta = hit.metadata if hasattr(hit, "metadata") else hit.payload
             fallback.append({"score": 0.0, "text": content, "meta": meta})
        return fallback

if __name__ == "__main__":
    # Test Query
    test_query = "What is the detailed syllabus for the hardware workshop?"
    results = search_and_rerank(test_query)
    
    for i, res in enumerate(results):
        print(f"Rank {i+1} (Score: {res['score']:.4f}):")
        print(f"   {res['text'][:150]}...")
        print("-" * 40)