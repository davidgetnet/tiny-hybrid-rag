"""Exercise Tiny RAG's real HTTP boundary against a running Chroma server."""

import os
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vector_store import open_client, query_by_embedding  # noqa: E402


def test_http_client_can_store_and_query_known_vectors() -> None:
    if not os.getenv("CHROMA_HOST"):
        pytest.fail(
            "CHROMA_HOST is required; start Chroma and configure the HTTP boundary"
        )

    client = open_client()
    try:
        client.heartbeat()
    except Exception as error:
        pytest.fail(f"Chroma server is not reachable: {error}")

    collection_name = f"ci_known_vectors_{uuid4().hex}"
    collection = client.create_collection(
        name=collection_name,
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=None,
    )

    try:
        collection.add(
            ids=["record-a", "record-b", "record-c"],
            documents=["axis x", "axis y", "axis z"],
            metadatas=[
                {"source": "fixture", "chunk_id": 0, "embedding_model": "known"},
                {"source": "fixture", "chunk_id": 1, "embedding_model": "known"},
                {"source": "fixture", "chunk_id": 2, "embedding_model": "known"},
            ],
            embeddings=[
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        )

        results = query_by_embedding(
            collection,
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            top_k=3,
        )

        assert collection.count() == 3
        assert results[0].id == "record-a"
        assert results[0].text == "axis x"
        assert results[0].distance == pytest.approx(0.0)
    finally:
        client.delete_collection(collection_name)
