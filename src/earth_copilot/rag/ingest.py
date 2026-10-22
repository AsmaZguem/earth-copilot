from qdrant_client import models

from earth_copilot.rag import embed
from earth_copilot.rag.catalogue import load_catalogue
from earth_copilot.rag.store import COLLECTION, DENSE, SPARSE, get_client, recreate_collection


def ingest(batch_size: int = 32) -> int:
    """Rebuild the collection from the catalogue. Returns the number of indexed documents."""
    docs = load_catalogue()
    texts = [doc.text() for doc in docs]

    dense_vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        dense_vectors += embed.dense(texts[start : start + batch_size])
    sparse_vectors = embed.sparse(texts)

    client = get_client()
    recreate_collection(client, dense_size=len(dense_vectors[0]))
    client.upsert(
        COLLECTION,
        points=[
            models.PointStruct(id=i, vector={DENSE: d, SPARSE: s}, payload=doc.payload())
            for i, (doc, d, s) in enumerate(zip(docs, dense_vectors, sparse_vectors))
        ],
    )
    return len(docs)
