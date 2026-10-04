"""Shared configuration for Lab 18."""

import os

from dotenv import load_dotenv

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT_DIR, ".env"))

# --- API Keys ---
OFFLINE = os.getenv("LAB_OFFLINE", "0") == "1"
OPENAI_API_KEY = "" if OFFLINE else os.getenv("OPENAI_API_KEY", "")

# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
SEMANTIC_MODEL = os.getenv("SEMANTIC_MODEL", "all-MiniLM-L6-v2")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
MODEL_DTYPE = os.getenv("MODEL_DTYPE", "float32")
MODEL_CACHE_DIR = os.path.join(ROOT_DIR, ".model-cache")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or None
EVAL_EMBEDDING_MODEL = os.getenv(
    "EVAL_EMBEDDING_MODEL",
    "openai/text-embedding-3-small"
    if OPENAI_BASE_URL and "openrouter.ai" in OPENAI_BASE_URL
    else "text-embedding-3-small",
)
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
EMBEDDING_DIM = 1024

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
