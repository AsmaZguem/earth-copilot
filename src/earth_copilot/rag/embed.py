"""Dense embeddings (meaning) from Ollama and sparse BM25 vectors (exact words) from fastembed."""

from functools import lru_cache

import ollama
from fastembed import SparseTextEmbedding
from qdrant_client import models

from earth_copilot.config import settings

# nomic-embed-text was trained with these task prefixes; leaving them out lowers retrieval quality
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


@lru_cache
def _ollama() -> ollama.Client:
    return ollama.Client(host=settings.ollama_host)


@lru_cache
def _bm25() -> SparseTextEmbedding:
    return SparseTextEmbedding("Qdrant/bm25", cache_dir="data/models")


def dense(texts: list[str], *, query: bool = False) -> list[list[float]]:
    prefix = QUERY_PREFIX if query else DOC_PREFIX
    response = _ollama().embed(model=settings.embed_model, input=[prefix + t for t in texts])
    return response.embeddings


def sparse(texts: list[str], *, query: bool = False) -> list[models.SparseVector]:
    model = _bm25()
    vectors = model.query_embed(texts) if query else model.embed(texts)
    return [
        models.SparseVector(indices=v.indices.tolist(), values=v.values.tolist()) for v in vectors
    ]
