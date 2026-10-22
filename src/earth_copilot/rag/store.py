import atexit
from functools import lru_cache

from qdrant_client import QdrantClient, models

from earth_copilot.config import settings

COLLECTION = "spectral_indices"
DENSE = "dense"
SPARSE = "bm25"


@lru_cache
def get_client() -> QdrantClient:
    location = settings.qdrant_location
    if location.startswith("http"):
        client = QdrantClient(url=location)
    else:
        client = QdrantClient(path=location)
    # Embedded mode must close before interpreter shutdown, or it prints a spurious error
    atexit.register(client.close)
    return client


def recreate_collection(client: QdrantClient, dense_size: int) -> None:
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        COLLECTION,
        vectors_config={DENSE: models.VectorParams(size=dense_size, distance=models.Distance.COSINE)},
        # IDF weighting is what turns raw term counts into BM25 scores
        sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
