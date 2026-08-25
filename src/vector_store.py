"""Persist our precomputed chunk embeddings in a small local Chroma store."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import chromadb
from chromadb.api.models.Collection import Collection
from numpy.typing import NDArray

from embeddings import EmbeddedChunk, MODEL_NAME


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIRECTORY = PROJECT_ROOT / "data" / "chroma"
COLLECTION_NAME = "acorn_chunks"
EMBEDDING_VERSION = MODEL_NAME
DISTANCE_METRIC = "cosine"


@dataclass(frozen=True)
class VectorSearchResult:
    """One stored record returned by Chroma, including its cosine distance."""

    id: str
    text: str
    source: str
    chunk_id: int
    embedding_model: str
    distance: float


def open_client(persist_directory: Path | str = CHROMA_DIRECTORY):
    """Open a local database whose records survive the current Python process."""

    return chromadb.PersistentClient(path=str(persist_directory))


def get_collection(client) -> Collection:
    """Create or reopen our collection, explicitly configured for cosine distance."""

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        configuration={"hnsw": {"space": DISTANCE_METRIC}},
        embedding_function=None,
    )
    actual_metric = collection.configuration["hnsw"]["space"]
    if actual_metric != DISTANCE_METRIC:
        raise ValueError(
            f"Collection {COLLECTION_NAME!r} uses {actual_metric!r}, "
            f"not {DISTANCE_METRIC!r}"
        )
    return collection


def stable_id(metadata: dict[str, Any]) -> str:
    """Derive record identity from the logical source and per-source chunk number."""

    return f"{metadata['source']}:{metadata['chunk_id']}"


def upsert_chunks(
    collection: Collection, embedded_chunks: Sequence[EmbeddedChunk]
) -> None:
    """Insert new records or replace records that have the same stable IDs."""

    collection.upsert(
        ids=[stable_id(chunk.metadata) for chunk in embedded_chunks],
        documents=[chunk.content for chunk in embedded_chunks],
        metadatas=[
            {
                **chunk.metadata,
                "embedding_model": EMBEDDING_VERSION,
            }
            for chunk in embedded_chunks
        ],
        embeddings=[chunk.vector.tolist() for chunk in embedded_chunks],
    )


def query_by_embedding(
    collection: Collection,
    query_embedding: NDArray,
    top_k: int = 3,
    where: dict[str, Any] | None = None,
) -> list[VectorSearchResult]:
    """Ask Chroma for the nearest stored records to one supplied query vector."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    raw = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    ids = raw["ids"][0]
    documents = raw["documents"][0]
    metadatas = raw["metadatas"][0]
    distances = raw["distances"][0]

    return [
        VectorSearchResult(
            id=record_id,
            text=document,
            source=metadata["source"],
            chunk_id=int(metadata["chunk_id"]),
            embedding_model=metadata["embedding_model"],
            distance=float(distance),
        )
        for record_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=True
        )
    ]
