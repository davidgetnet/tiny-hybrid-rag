"""Retrieve evidence from Chroma using our own SentenceTransformer vectors."""

from pathlib import Path
from typing import Any

from embeddings import embed_query, load_embedding_model
from vector_store import (
    CHROMA_DIRECTORY,
    VectorSearchResult,
    get_collection,
    open_client,
    query_by_embedding,
)


def retrieve(
    query: str,
    top_k: int = 3,
    where: dict[str, Any] | None = None,
    *,
    persist_directory: Path | str = CHROMA_DIRECTORY,
    model=None,
) -> list[VectorSearchResult]:
    """Embed a query ourselves and retrieve the nearest persistent records."""

    embedding_model = model or load_embedding_model()
    query_vector = embed_query(query, embedding_model)
    collection = get_collection(open_client(persist_directory))
    return query_by_embedding(collection, query_vector, top_k=top_k, where=where)
