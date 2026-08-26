"""Structural tests for persistent storage and retrieval without brittle rankings."""

import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from embeddings import embed_query, load_embedding_model  # noqa: E402
from index_chunks import index_chunks  # noqa: E402
from vector_store import open_client, query_by_embedding  # noqa: E402


EXPECTED_IDS = {
    "handbook.md:0",
    "handbook.md:1",
    "handbook.md:2",
    "handbook.md:3",
    "policies.md:0",
    "policies.md:1",
    "policies.md:2",
    "policies.md:3",
}


class VectorStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary_directory = tempfile.TemporaryDirectory()
        cls.model = load_embedding_model()
        cls.collection, _, _ = index_chunks(cls.temporary_directory.name, cls.model)
        cls.query_vector = embed_query(
            "Who reviews security-sensitive deployments?", cls.model
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary_directory.cleanup()

    def test_collection_contains_exactly_eight_records(self) -> None:
        self.assertEqual(self.collection.count(), 8)

    def test_indexing_twice_is_idempotent(self) -> None:
        collection, _, _ = index_chunks(self.temporary_directory.name, self.model)
        self.assertEqual(collection.count(), 8)

    def test_expected_stable_ids_and_metadata_are_stored(self) -> None:
        stored = self.collection.get(include=["metadatas"])
        self.assertEqual(set(stored["ids"]), EXPECTED_IDS)
        for record_id, metadata in zip(
            stored["ids"], stored["metadatas"], strict=True
        ):
            source, chunk_id = record_id.rsplit(":", 1)
            self.assertEqual(metadata["source"], source)
            self.assertEqual(metadata["chunk_id"], int(chunk_id))
            self.assertEqual(
                metadata["embedding_model"],
                "sentence-transformers/all-MiniLM-L6-v2",
            )

    def test_stored_embeddings_have_384_dimensions(self) -> None:
        stored = self.collection.get(include=["embeddings"])
        self.assertEqual(len(stored["embeddings"]), 8)
        self.assertTrue(all(len(vector) == 384 for vector in stored["embeddings"]))

    def test_query_returns_requested_count_identity_metadata_and_distance(self) -> None:
        results = query_by_embedding(self.collection, self.query_vector, top_k=3)
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIn(result.id, EXPECTED_IDS)
            self.assertEqual(result.id, f"{result.source}:{result.chunk_id}")
            self.assertTrue(result.text)
            self.assertTrue(math.isfinite(result.distance))

    def test_metadata_filter_limits_candidate_source(self) -> None:
        results = query_by_embedding(
            self.collection,
            self.query_vector,
            top_k=3,
            where={"source": "handbook.md"},
        )
        self.assertEqual(len(results), 3)
        self.assertTrue(all(result.source == "handbook.md" for result in results))

    def test_collection_uses_cosine_distance(self) -> None:
        self.assertEqual(self.collection.configuration["hnsw"]["space"], "cosine")


class ClientConfigurationTests(unittest.TestCase):
    def test_environment_selects_http_client(self) -> None:
        with (
            patch.dict(
                "os.environ",
                {"CHROMA_HOST": "chroma", "CHROMA_PORT": "8123"},
                clear=True,
            ),
            patch("vector_store.chromadb.HttpClient") as http_client,
        ):
            open_client()

        http_client.assert_called_once_with(host="chroma", port=8123)

    def test_explicit_directory_keeps_tests_and_local_tools_embedded(self) -> None:
        with (
            patch.dict(
                "os.environ",
                {"CHROMA_HOST": "chroma", "CHROMA_PORT": "8000"},
                clear=True,
            ),
            patch("vector_store.chromadb.PersistentClient") as persistent_client,
        ):
            open_client("/tmp/explicit-chroma-test")

        persistent_client.assert_called_once_with(path="/tmp/explicit-chroma-test")


if __name__ == "__main__":
    unittest.main()
