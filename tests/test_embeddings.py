"""Structural tests for embeddings without brittle exact-score assertions."""

import math
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chunk_documents import chunk_documents  # noqa: E402
from embeddings import (  # noqa: E402
    embed_chunks,
    embed_query,
    load_embedding_model,
    rank_chunks,
)
from load_documents import load_documents  # noqa: E402


class EmbeddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = load_embedding_model()
        cls.chunks = chunk_documents(load_documents())
        cls.embedded_chunks, cls.matrix = embed_chunks(cls.chunks, cls.model)
        cls.query_vector = embed_query(
            "What technology does the Backend team primarily use?", cls.model
        )

    def test_exactly_eight_chunks_are_embedded(self) -> None:
        self.assertEqual(len(self.embedded_chunks), 8)
        self.assertEqual(self.matrix.shape[0], 8)

    def test_chunk_embeddings_have_expected_dimensions(self) -> None:
        self.assertEqual(self.matrix.shape, (8, 384))
        for embedded_chunk in self.embedded_chunks:
            self.assertEqual(embedded_chunk.vector.shape, (384,))
            self.assertGreater(embedded_chunk.vector.size, 0)

    def test_query_and_chunks_share_dimensions(self) -> None:
        self.assertEqual(self.query_vector.shape, self.embedded_chunks[0].vector.shape)

    def test_similarity_scores_are_finite(self) -> None:
        results = rank_chunks(self.query_vector, self.embedded_chunks)
        self.assertTrue(all(math.isfinite(result.score) for result in results))

    def test_ranking_contains_every_chunk_once(self) -> None:
        results = rank_chunks(self.query_vector, self.embedded_chunks)
        identities = [
            (result.chunk.metadata["source"], result.chunk.metadata["chunk_id"])
            for result in results
        ]
        self.assertEqual(len(identities), 8)
        self.assertEqual(len(set(identities)), 8)


if __name__ == "__main__":
    unittest.main()
