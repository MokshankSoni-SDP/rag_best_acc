import streamlit as st
import os
import shutil
import json
from groq import Groq

# --- IMPORT YOUR BACKEND SCRIPTS ---
# We import specific functions from your existing files
from ingest import load_and_structure_file
from chunking import create_semantic_chunks
from index import index_data
from retrieve import search_and_rerank

# --- CONFIGURATION ---
# IMPORTANT: Put your Groq API Key here or use os.environ
API_KEY = os.getenv("GROQ_API_KEY") 
MODEL_NAME = "llama-3.3-70b-versatile"
from dotenv import load_dotenv

load_dotenv()

# Setup Page
st.set_page_config(page_title="RAG Knowledge Base", layout="wide")

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

# --- SIDEBAR: FILE UPLOAD & PROCESSING ---
with st.sidebar:
    st.header("📂 Knowledge Base")
    st.write("Upload a PDF, DOCX, TXT, or Image (PNG/JPG) to chat.")
    
    uploaded_file = st.file_uploader("Upload Document", type=["pdf", "txt", "docx", "png", "jpg", "jpeg"])
    
    if uploaded_file and uploaded_file.name != st.session_state.processed_file:
        with st.spinner("🚀 Processing file... (Ingesting -> Chunking -> Indexing)"):
            
            # 1. SAVE FILE LOCALLY
            # Streamlit keeps files in RAM, we must save it to disk for ingest.py to read
            save_path = uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.info(f"File saved: {save_path}")

            # 2. RUN INGESTION (Step 1)
            # We call your function directly!
            try:
                raw_blocks = load_and_structure_file(save_path)
                # Save to JSON for the next step (mimicking your pipeline)
                with open("raw_data.json", "w", encoding="utf-8") as f:
                    json.dump(raw_blocks, f, indent=2, ensure_ascii=False)
                st.success(f"Ingestion complete: {len(raw_blocks)} raw blocks extracted.")
                
                # 3. RUN CHUNKING (Step 2)
                semantic_chunks = create_semantic_chunks(raw_blocks)
                with open("semantic_chunks.json", "w", encoding="utf-8") as f:
                    json.dump(semantic_chunks, f, indent=2, ensure_ascii=False)
                st.success(f"Chunking complete: {len(semantic_chunks)} semantic chunks created.")

                # 4. RUN INDEXING (Step 3)
                # We need to ensure index.py reads the correct file
                # Since index.py reads 'semantic_chunks.json' by default, we are good.
                index_data() 
                st.success("Indexing complete: Data stored in Qdrant!")
                
                # Mark as done
                st.session_state.processed_file = uploaded_file.name
                st.balloons()
                
            except Exception as e:
                st.error(f"Error during processing: {e}")

    st.markdown("---")
    st.markdown("**Status:** " + ("✅ Ready" if st.session_state.processed_file else "⚠️ Waiting for file"))

# --- MAIN CHAT INTERFACE ---
st.title("🤖 Chat with your Documents")
st.caption("Powered by Llama 3, Qdrant & BGE-Reranker")

# Display History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask a question about your document..."):
    # 1. User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Assistant Logic
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("Thinking..."):
            try:
                # A. RETRIEVAL (Step 4)
                # We call search_and_rerank directly to get the chunks
                retrieved_chunks = search_and_rerank(prompt)
                
                # B. DISPLAY CHUNKS (Your Requirement)
                # We put this in an expander so it looks clean
                with st.expander("🔍 Verified Sources (Top 5 Chunks)", expanded=False):
                    if not retrieved_chunks:
                        st.warning("No relevant information found.")
                    else:
                        for i, chunk in enumerate(retrieved_chunks):
                            st.markdown(f"**Rank {i+1} (Score: {chunk['score']:.4f})**")
                            st.info(f"📄 *Page {chunk['meta'].get('page', '?')}*: {chunk['text']}")
                            st.markdown("---")

                # C. GENERATION (Step 5)
                # We construct the prompt here to ensure we use the chunks we just displayed
                if not retrieved_chunks:
                     response_text = "I couldn't find any relevant information in the documents to answer your question."
                else:
                    context_text = ""
                    for i, chunk in enumerate(retrieved_chunks):
                        context_text += f"\n[Source {i+1}]: {chunk['text']}"
                    
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
                    # Call Groq
                    client = Groq(api_key=API_KEY)
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        model=MODEL_NAME,
                        temperature=0.0,
                    )
                    response_text = chat_completion.choices[0].message.content

                # D. Display Output
                message_placeholder.markdown(response_text)
                
                # Save to history
                st.session_state.messages.append({"role": "assistant", "content": response_text})

            except Exception as e:
                st.error(f"An error occurred: {e}")