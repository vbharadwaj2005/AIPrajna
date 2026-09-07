"""
AIPrajna — Configuration and Environment Settings.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("aiprajna")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "documents"
CHROMA_DIR = BASE_DIR / "storage" / "chroma"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL", "microsoft/Phi-3-mini-4k-instruct")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
SEMANTIC_THRESHOLD = float(os.getenv("SEMANTIC_THRESHOLD", "0.75"))
TOP_K_INITIAL = int(os.getenv("TOP_K_INITIAL", "20"))
TOP_K_RERANKED = int(os.getenv("TOP_K_RERANKED", "5"))
HYBRID_WEIGHT_DENSE = float(os.getenv("HYBRID_WEIGHT_DENSE", "0.5"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "aiprajna")
MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", "4096"))

if not 0 <= HYBRID_WEIGHT_DENSE <= 1:
    logger.warning("HYBRID_WEIGHT_DENSE must be between 0 and 1, resetting to 0.5")
    HYBRID_WEIGHT_DENSE = 0.5
