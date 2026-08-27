"""Fast contracts for vector-store behavior that needs no Chroma server."""

import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vector_store import query_by_embedding, stable_id  # noqa: E402


def test_stable_id_combines_source_and_chunk_number() -> None:
    metadata = {"source": "handbook.md", "chunk_id": 2}

    assert stable_id(metadata) == "handbook.md:2"


def test_query_rejects_non_positive_result_count_before_calling_chroma() -> None:
    collection = Mock()
    query_vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    with pytest.raises(ValueError, match="top_k must be at least 1"):
        query_by_embedding(collection, query_vector, top_k=0)

    collection.query.assert_not_called()
