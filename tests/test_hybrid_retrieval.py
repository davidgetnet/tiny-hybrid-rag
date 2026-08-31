"""Behavioral tests for deterministic hybrid retrieval orchestration."""

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hybrid_retriever import (  # noqa: E402
    QUESTION_A,
    QUESTION_B,
    QUESTION_C,
    RetrievalMode,
    UnsupportedQueryError,
    retrieve_hybrid,
)
from vector_store import VectorSearchResult  # noqa: E402


def vector_result(record_id: str, distance: float = 0.25) -> VectorSearchResult:
    source, chunk_id = record_id.rsplit(":", 1)
    return VectorSearchResult(
        id=record_id,
        text=f"Evidence from {record_id}",
        source=source,
        chunk_id=int(chunk_id),
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        distance=distance,
    )


class HybridRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calls: list[tuple[str, int]] = []

        def fake_vector_retriever(query: str, top_k: int):
            self.calls.append((query, top_k))
            return [
                vector_result("policies.md:1", 0.20),
                vector_result("handbook.md:1", 0.30),
                vector_result("policies.md:1", 0.20),
            ]

        self.vector_retriever = fake_vector_retriever

    def test_vector_only_mode_returns_only_vector_evidence(self) -> None:
        result = retrieve_hybrid(
            QUESTION_A, vector_retriever=self.vector_retriever
        )
        self.assertEqual(result.mode, RetrievalMode.VECTOR)
        self.assertTrue(result.vector_evidence)
        self.assertIsNone(result.graph_evidence)

    def test_graph_only_mode_returns_only_graph_evidence(self) -> None:
        result = retrieve_hybrid(
            QUESTION_B, vector_retriever=self.vector_retriever
        )
        self.assertEqual(result.mode, RetrievalMode.GRAPH)
        self.assertFalse(result.vector_evidence)
        self.assertIsNotNone(result.graph_evidence)
        self.assertEqual(self.calls, [])

    def test_hybrid_mode_returns_both_evidence_forms(self) -> None:
        result = retrieve_hybrid(
            QUESTION_C, vector_retriever=self.vector_retriever
        )
        self.assertEqual(result.mode, RetrievalMode.HYBRID)
        self.assertTrue(result.vector_evidence)
        self.assertIsNotNone(result.graph_evidence)

    def test_question_b_graph_path_resolves_to_alice(self) -> None:
        result = retrieve_hybrid(QUESTION_B)
        self.assertEqual(result.graph_evidence.path, ("Python", "Backend", "Alice"))

    def test_question_c_graph_evidence_contains_required_entities(self) -> None:
        result = retrieve_hybrid(
            QUESTION_C, vector_retriever=self.vector_retriever
        )
        self.assertTrue(
            {"Backend", "Alice", "Clara"}.issubset(result.graph_evidence.entities)
        )

    def test_vector_evidence_retains_identity_and_provenance(self) -> None:
        result = retrieve_hybrid(
            QUESTION_C, vector_retriever=self.vector_retriever
        )
        evidence = result.vector_evidence[0]
        self.assertEqual(evidence.id, "policies.md:1")
        self.assertEqual(evidence.source, "policies.md")
        self.assertEqual(evidence.chunk_id, 1)

    def test_graph_evidence_retains_relationship_provenance(self) -> None:
        result = retrieve_hybrid(
            QUESTION_C, vector_retriever=self.vector_retriever
        )
        for edge in result.graph_evidence.relationships:
            self.assertTrue(edge.source.endswith(".md"))
            self.assertIsInstance(edge.chunk_id, int)
            self.assertTrue(edge.relationship)

    def test_duplicate_vector_ids_are_removed_without_reordering(self) -> None:
        result = retrieve_hybrid(
            QUESTION_C, vector_retriever=self.vector_retriever
        )
        self.assertEqual(
            [evidence.id for evidence in result.vector_evidence],
            ["policies.md:1", "handbook.md:1"],
        )

    def test_unknown_query_fails_clearly(self) -> None:
        with self.assertRaisesRegex(UnsupportedQueryError, "No educational routing rule"):
            retrieve_hybrid(
                "Tell me something interesting",
                vector_retriever=self.vector_retriever,
            )

    def test_no_llm_or_vector_call_is_required_for_graph_mode(self) -> None:
        def unexpected_vector_call(query: str, top_k: int):
            raise AssertionError("graph-only routing must not call vector retrieval")

        result = retrieve_hybrid(
            QUESTION_B,
            vector_retriever=unexpected_vector_call,
        )
        self.assertEqual(result.graph_evidence.entities[-1], "Alice")


if __name__ == "__main__":
    unittest.main()
