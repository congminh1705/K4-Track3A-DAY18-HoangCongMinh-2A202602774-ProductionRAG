from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BM25_TOP_K, COLLECTION_NAME, DENSE_TOP_K, EMBEDDING_MODEL, HYBRID_TOP_K, QDRANT_HOST, QDRANT_PORT


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    import unicodedata

    from underthesea import word_tokenize

    return word_tokenize(unicodedata.normalize("NFC", text).lower(), format="text").replace("_", " ")


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        from rank_bm25 import BM25Okapi

        self.documents = list(chunks)
        self.corpus_tokens = [segment_vietnamese(c["text"]).split() or ["__empty__"] for c in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens) if chunks else None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or top_k <= 0:
            return []
        scores = self.bm25.get_scores(segment_vietnamese(query).split())
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [
            SearchResult(self.documents[i]["text"], float(scores[i]), self.documents[i].get("metadata", {}), "bm25")
            for i in indices[:top_k]
            if scores[i] > 0
        ]


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient

        try:
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2)
            self.client.get_collections()
        except Exception:
            self.client = QdrantClient(":memory:")
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            self._encoder = _cached_dense_encoder(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import Distance, PointStruct, VectorParams

        if not chunks:
            raise ValueError("Cannot build a dense index from an empty corpus")
        vectors = self._get_encoder().encode(
            [c["text"] for c in chunks], normalize_embeddings=True, batch_size=4, show_progress_bar=True
        )
        if self.client.collection_exists(collection):
            self.client.delete_collection(collection)
        self.client.create_collection(
            collection, vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE)
        )
        for start in range(0, len(chunks), 64):
            points = [
                PointStruct(
                    id=i,
                    vector=vectors[i].tolist(),
                    payload={"text": chunks[i]["text"], "metadata": chunks[i].get("metadata", {})},
                )
                for i in range(start, min(start + 64, len(chunks)))
            ]
            self.client.upsert(collection, points, wait=True)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if top_k <= 0 or not self.client.collection_exists(collection):
            return []
        vector = self._get_encoder().encode(query, normalize_embeddings=True).tolist()
        response = self.client.query_points(collection, query=vector, limit=top_k)
        return [
            SearchResult(pt.payload["text"], float(pt.score), pt.payload.get("metadata", {}), "dense")
            for pt in response.points
        ]


from functools import lru_cache


@lru_cache(maxsize=1)
def _cached_dense_encoder(name):
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(4)

    from config import MODEL_CACHE_DIR, MODEL_DTYPE, OFFLINE

    encoder = SentenceTransformer(
        name, cache_folder=MODEL_CACHE_DIR, local_files_only=OFFLINE, model_kwargs={"torch_dtype": MODEL_DTYPE}
    )
    encoder.max_seq_length = 512
    return encoder


def reciprocal_rank_fusion(
    results_list: list[list[SearchResult]], k: int = 60, top_k: int = HYBRID_TOP_K
) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    if k < 0:
        raise ValueError("k must be nonnegative")
    scores, representatives = {}, {}
    for results in results_list:
        seen = set()
        for rank, result in enumerate(results):
            key = (result.metadata.get("source", ""), result.text)
            if key in seen:
                continue
            seen.add(key)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            representatives.setdefault(key, result)
    ordered = sorted(scores, key=scores.get, reverse=True)[: max(0, top_k)]
    return [
        SearchResult(representatives[key].text, scores[key], representatives[key].metadata, "hybrid") for key in ordered
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""

    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print("Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
