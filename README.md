# DocInsight: Compliance Multi-Document RAG Platform

**Compliance Multi-Document RAG** — zero external data leakage, fully auditable answers across your PDF / DOCX / TXT files.
Ingest documents, semantically chunk them, index into ChromaDB, then query via **hybrid search (BM25 + dense vectors)** + **cross-encoder re-ranking**. The LLM runs through HuggingFace Inference API (free tier); embeddings and re-ranking run entirely **on your machine**.

---

## Key Features

### RAG Pipeline
- **Document Ingestion**: Load PDFs, DOCX, and TXT files with automatic format detection.
- **Semantic Chunking**: RecursiveCharacter + semantic merge splitting for context-aware chunks.
- **Hybrid Search**: Weighted fusion of BM25 (sparse) and dense vector retrieval.
- **Cross-Encoder Re-Ranking**: ms-marco-MiniLM for precision-focused result re-ordering.

### Source Attribution
- **Grounded Answers**: Every response cites the source document and relevance score.
- **Deduplication**: Unique source attribution across overlapping chunks.
- **Excerpt Preview**: Inline excerpts from matched documents for quick validation.

### Data Locality
- **Local Embeddings**: SentenceTransformers (all-MiniLM-L6-v2) runs on your machine.
- **Local Vector Store**: ChromaDB persistent storage on disk.
- **Local Re-Ranking**: Cross-Encoder inference without external calls.
- **API-Only LLM**: Only the generation step uses HuggingFace Inference API.

---

## Installation & Setup

```bash
cd DocInsight
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your HuggingFace API key
```

### Run the App
```bash
streamlit run app.py
```
Or on Windows, double-click `start.bat`.

The app opens at **http://localhost:8501**.

---

## Architecture

```
 User query
      |
      v
 [EmbeddingService] ──> dense vector
 [BM25 (sparse)]    ──> keyword scores
      |
      v
 [HybridRetriever]  ──> weighted fusion (configurable)
      |
      v
 [Reranker]         ──> CrossEncoder (ms-marco-MiniLM)
      |
      v
 [LlmService]       ──> HuggingFace Inference API (Phi-3)
      |
      v
 Answer + source attribution
```

---

### Components

| Layer | Technology | Data locality |
|-------|-----------|---------------|
| File loading | PyPDFLoader, Docx2txt, TextLoader | **Local** |
| Chunking | RecursiveCharacter + semantic merge | **Local** |
| Dense embeddings | SentenceTransformers (all-MiniLM-L6-v2) | **Local** |
| Sparse search | BM25 (rank-bm25) | **Local** |
| Vector store | ChromaDB (persistent, on disk) | **Local** |
| Re-ranker | Cross-Encoder (ms-marco-MiniLM-L-6-v2) | **Local** |
| LLM | HuggingFace Inference API (Phi-3 / any HF model) | API (HuggingFace) |
| UI | Streamlit (chat interface) | **Local** |

**Only the LLM call** leaves your machine — to HuggingFace's inference endpoint. Embeddings, search, and re-ranking all run locally.

---

*Source-Grounded Compliance Intelligence*