"""Create embeddings and compare them without a vector database."""

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from documents import Chunk


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class EmbeddedChunk:
    """A chunk kept together with its identity, text, and semantic vector."""

    content: str
    metadata: dict[str, Any]
    vector: NDArray[np.float32]


@dataclass(frozen=True)
class ScoredChunk:
    """One embedded chunk and its similarity to a query."""

    score: float
    chunk: EmbeddedChunk


def load_embedding_model() -> SentenceTransformer:
    """Load the one model used for both chunk and query embeddings."""

    return SentenceTransformer(MODEL_NAME)


def embed_chunks(
    chunks: Sequence[Chunk], model: SentenceTransformer
) -> tuple[list[EmbeddedChunk], NDArray[np.float32]]:
    """Encode all chunk text and preserve which vector belongs to which chunk."""

    matrix = model.encode(
        [chunk.content for chunk in chunks],
        convert_to_numpy=True,
    )
    embedded_chunks = [
        EmbeddedChunk(
            content=chunk.content,
            metadata=chunk.metadata.copy(),
            vector=vector,
        )
        for chunk, vector in zip(chunks, matrix, strict=True)
    ]
    return embedded_chunks, matrix


def embed_query(text: str, model: SentenceTransformer) -> NDArray[np.float32]:
    """Place a query in the same embedding space as the chunks."""

    return model.encode(text, convert_to_numpy=True)


def cosine_similarity(vector_a: NDArray, vector_b: NDArray) -> float:
    """Measure how closely two vectors point in the same direction."""

    denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if denominator == 0:
        raise ValueError("Cosine similarity is undefined for a zero vector")
    return float(np.dot(vector_a, vector_b) / denominator)


def rank_chunks(
    query_vector: NDArray, embedded_chunks: Sequence[EmbeddedChunk]
) -> list[ScoredChunk]:
    """Compare one query with every chunk, then return highest scores first."""

    scored = [
        ScoredChunk(
            score=cosine_similarity(query_vector, embedded_chunk.vector),
            chunk=embedded_chunk,
        )
        for embedded_chunk in embedded_chunks
    ]
    return sorted(scored, key=lambda result: result.score, reverse=True)
