import os
from groq import Groq
from retrieve import search_and_rerank # Import Step 4
from dotenv import load_dotenv

load_dotenv()
# --- CONFIGURATION ---
# Replace with your actual key!
API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.3-70b-versatile" # The smartest model on Groq
# ---------------------

def generate_answer(query):
    print(f"🤖 RAG System processing: '{query}'")
    
    # 1. RETRIEVE (Step 4)
    # Get the top 5 chunks using our Hybrid + Reranking engine
    retrieved_chunks = search_and_rerank(query)
    
    if not retrieved_chunks:
        return "Sorry, I couldn't find any information in the documents."

    # 2. CONTEXT PREPARATION
    # We stitch the chunks together into one big text block.
    # We also include the source metadata so the LLM can cite it.
    context_text = ""
    for i, chunk in enumerate(retrieved_chunks):
        context_text += f"\n[Source {i+1}]: {chunk['text']}"

    # 3. PROMPT ENGINEERING (The System Prompt)
    # This instructs the AI specifically on HOW to answer.
    system_prompt = f"""
    You are a careful and factual assistant.
    
    Your task is to answer the user's question using ONLY the provided context.
    
    CRITICAL INSTRUCTION:
    Before answering, you must THINK STEP BY STEP to analyze the context.
    
    Follow these rules strictly:
    
    1. Use ONLY the information present in the context.
    2. If the answer is not explicitly stated in the context, respond with:
       "I don't know based on the provided documents."
       This rule has the highest priority.
    3. Do NOT use outside knowledge or assumptions.
    4. Do NOT rephrase facts in a misleading way.
    5. Cite the source number (e.g., [Source 1]) after each factual statement.
    6. Keep the answer concise, precise, and professional.

    Context:
    {context_text}
    """

    # 4. CALL LLM API (Groq)
    print("   ...Synthesizing answer with Llama 3 (Groq)...")
    
    client = Groq(api_key=API_KEY)
    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": query,
            }
        ],
        model=MODEL_NAME,
        temperature=0.0, # Keep it factual (0 creativity)
    )

    # 5. OUTPUT
    answer = chat_completion.choices[0].message.content
    return answer

if __name__ == "__main__":
    # Test Question
    user_query = "What are the new job roles mentioned in the report?"
    
    final_response = generate_answer(user_query)
    
    print("\n" + "="*50)
    print("FINAL ANSWER:")
    print("="*50)
    print(final_response)
    print("="*50)