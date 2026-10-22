"""Search the catalogue by meaning (dense), exact words (sparse/BM25), or both fused (hybrid)."""

from dataclasses import dataclass
from typing import Literal

from qdrant_client import models

from earth_copilot.rag import embed
from earth_copilot.rag.store import COLLECTION, DENSE, SPARSE, get_client

Mode = Literal["dense", "sparse", "hybrid"]


@dataclass
class Hit:
    short_name: str
    long_name: str
    domain: str
    formula: str
    bands: list[str]
    reference: str
    text: str
    score: float


def search(
    query: str, k: int = 5, mode: Mode = "hybrid", sentinel2_only: bool = True
) -> list[Hit]:
    client = get_client()
    flt = (
        models.Filter(
            must=[models.FieldCondition(key="sentinel2", match=models.MatchValue(value=True))]
        )
        if sentinel2_only
        else None
    )

    if mode == "dense":
        result = client.query_points(
            COLLECTION, query=embed.dense([query], query=True)[0], using=DENSE,
            query_filter=flt, limit=k, with_payload=True,
        )
    elif mode == "sparse":
        result = client.query_points(
            COLLECTION, query=embed.sparse([query], query=True)[0], using=SPARSE,
            query_filter=flt, limit=k, with_payload=True,
        )
    else:
        # Each retriever proposes candidates, then Reciprocal Rank Fusion merges the two rankings
        result = client.query_points(
            COLLECTION,
            prefetch=[
                models.Prefetch(
                    query=embed.dense([query], query=True)[0], using=DENSE, filter=flt, limit=k * 4
                ),
                models.Prefetch(
                    query=embed.sparse([query], query=True)[0], using=SPARSE, filter=flt,
                    limit=k * 4,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=k,
            with_payload=True,
        )

    return [
        Hit(
            short_name=p.payload["short_name"],
            long_name=p.payload["long_name"],
            domain=p.payload["domain"],
            formula=p.payload["formula"],
            bands=p.payload["bands"],
            reference=p.payload["reference"],
            text=p.payload["text"],
            score=p.score,
        )
        for p in result.points
    ]
