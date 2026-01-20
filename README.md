# 🚀 High-Accuracy RAG System

This project implements a Retrieval-Augmented Generation (RAG) system optimized for high accuracy using Hybrid Search, Context-Aware Chunking, and Re-ranking. It is designed to process complex documents (like syllabi or technical reports) and provide precise, cited answers.

## 🏗️ Architecture Overview

The pipeline follows a standard RAG workflow with enhanced intermediate steps:
1.  **Ingestion**: Extracts text and tables from PDF/images using `unstructured` and `pytesseract`.
2.  **Chunking**: "Context-Aware Semantic Chunking" that injects global context into local chunks.
3.  **Embedding**: Generates Dense and Sparse vectors using FastEmbed (BGE models) in Qdrant.
4.  **Retrieval**: Hybrid Search (Keywords + Semantic) followed by Cross-Encoder Reranking.
5.  **Generation**: Llama 3.3 70B (via Groq) synthesizes the answer with strict adherence to facts.

---

## 🧩 Chunking Strategy

We employ a custom **Context-Aware Semantic Chunking** strategy (found in `chunking.py`) to address the "Lost in the Middle" problem where chunks lose their parent context.

*   **Logic**:
    *   **Context Injection**: Every chunk is prefixed with its "Subject" context. The system detects "SUBJECT: ..." lines to switch the global context.
        *   *Example*: A chunk about "Course Outcome 1" will be rewritten as: `Subject: HARDWARE WORKSHOP - Course Outcome 1...`.
    *   **Header Detection**: Regex-based detection of numbered headers (e.g., `1.1`, `[1]`) to break chunks at logical boundaries.
    *   **Dynamic Sizing**: Chunks grow until a new header or subject is found, preventing arbitrary cut-offs mid-sentence.

## 🧠 Embedding & Retrieval

### Embedding Choice
*   **Vector Database**: [Qdrant](https://qdrant.tech/) (Local mode)
*   **Model**: **FastEmbed (BGE Series)**.
    *   **Dense Vectors**: Captures semantic meaning.
    *   **Sparse Vectors**: Captures exact keyword matches (BM25-style).
*   **Why?**: Using `client.get_fastembed_vector_params()` automatically selects optimized quantization-friendly models (typically `BAAI/bge-small-en-v1.5` or `BAAI/bge-m3`) for high performance with low latency.

### Retrieval Pipeline
1.  **Query Expansion**: The LLM generates 3 variations of the user's query to capture different phrasings.
2.  **Hybrid Search**: We query Qdrant using both dense and sparse vectors to retrieve the top 25 candidates per variation.
3.  **Deduplication**: Results from variations are merged and deduplicated by ID.
4.  **Reranking**:
    *   **Model**: `BAAI/bge-reranker-base`.
    *   **Method**: A Cross-Encoder model scores the relevance of the (Query, Document) pair.
    *   **Selection**: The top 12 highest-scored chunks are passed to the Generator.

## 🛡️ Confidence & Hallucination Prevention

To ensure reliability, we enforce strict constraints at the Prompt Engineering level (`generate.py`):

1.  **"I Don't Know" Priority**: The system prompt explicitly instructs the model: *"If the answer is not explicitly stated in the context, respond with 'I don't know based on the provided documents.'"* This is the highest priority rule.
2.  **Citation Requirement**: Every factual statement must be immediately followed by a citation like `[Source 1]`.
3.  **Step-by-Step Reasoning**: The model is forced to `THINK STEP BY STEP` before generating the final response, which reduces logical leaps.
4.  **Zero Temperature**: We use `temperature=0.0` to minimize creativity and hallucination.

## 🤖 Model Choice

### LLM: Llama 3.3 70B Versatile
*   **Provider**: [Groq](https://groq.com/)
*   **Why Selection?**:
    *   **70B Parameter Size**: Essential for complex reasoning and synthesizing information from 12+ chunks without getting confused.
    *   **Groq LPU Inference**: Provides near-instant tokens-per-second, allowing us to use a massive 70B model with the latency of a small model. This enables the heavy "Re-ranking + Reasoning" pipeline to remain interactive.

## ⚠️ Limitations

1.  **OCR Dependency**: The system relies on `Tesseract-OCR` for processing images or scanned PDFs. Performance is directly tied to OCR quality.
2.  **External API**: Requires a valid Groq API Key (`GROQ_API_KEY`).
3.  **Structure Assumptions**: The "Context-Aware" chunking relies on specific regex patterns (e.g., "SUBJECT:", numbered lists). Documents with radically different formatting might require `chunking.py` adjustments.
4.  **Stateful Processing**: The `retrieve.py` and `generate.py` scripts are stateless, but the indexing is stateful (stored in `./qdrant_db`). Re-running `index.py` appends data unless the DB is cleared.

---

## 🛠️ Setup & Usage

1.  **Install Requirements**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Install Tesseract**:
    *   Windows: Install from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki).
    *   Linux: `apt-get install tesseract-ocr`.
3.  **Set API Key**:
    ```bash
    export GROQ_API_KEY="gsk_..."
    ```
4.  **Run Pipeline**:
    *   `python ingest.py` (Process PDF to JSON)
    *   `python chunking.py` (Semantic Chunking)
    *   `python index.py` (Embed & Index)
    *   `python app.py` (Launch Streamlit UI)
