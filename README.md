# DocInsight

**Compliance Multi-Document RAG** — zero external data leakage, fully auditable answers across your PDF / DOCX / TXT files.
Ingest documents, semantically chunk them, index into ChromaDB, then query via **hybrid search (BM25 + dense vectors)** + **cross-encoder re-ranking**. The LLM runs through HuggingFace Inference API (free tier); embeddings and re-ranking run entirely **on your machine**.

---

### Setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt

# Configure
cp .env.example .env

# Run
python main.py
```

Opens at `http://localhost:8501`.

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