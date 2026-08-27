"""Behavioral tests for the manually encoded Acorn Labs graph."""

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from graph_retriever import (  # noqa: E402
    find_manager_of_team,
    find_manager_of_team_using,
    find_relationship_targets,
    find_security_reviewer,
    find_service_owned_by_team_managed_by,
    find_team_using,
)
from knowledge_graph import (  # noqa: E402
    MANAGES,
    MUST_BE_RECORDED_IN,
    OWNS,
    REQUIRES_REVIEW_FROM,
    USES,
    build_knowledge_graph,
)


class KnowledgeGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = build_knowledge_graph()

    def test_expected_important_nodes_exist(self) -> None:
        expected = {
            "Alice",
            "Ben",
            "Clara",
            "Backend",
            "Frontend",
            "Python",
            "TypeScript",
            "Payments API",
            "Customer Portal",
            "Production Deployment",
            "Security-Sensitive Deployment",
            "Deployment Log",
        }
        self.assertTrue(expected.issubset(self.graph.nodes))

    def test_expected_relationships_and_types_exist(self) -> None:
        expected = {
            ("Alice", "Backend", MANAGES),
            ("Backend", "Python", USES),
            ("Backend", "Payments API", OWNS),
            (
                "Security-Sensitive Deployment",
                "Clara",
                REQUIRES_REVIEW_FROM,
            ),
            (
                "Production Deployment",
                "Deployment Log",
                MUST_BE_RECORDED_IN,
            ),
        }
        actual = {
            (source, target, attributes["relationship"])
            for source, target, attributes in self.graph.edges(data=True)
        }
        self.assertTrue(expected.issubset(actual))

    def test_important_edges_preserve_provenance(self) -> None:
        for source, target in (
            ("Alice", "Backend"),
            ("Backend", "Python"),
            ("Security-Sensitive Deployment", "Clara"),
        ):
            attributes = self.graph.edges[source, target]
            self.assertIn("source", attributes)
            self.assertIn("chunk_id", attributes)
            self.assertIsInstance(attributes["chunk_id"], int)

    def test_find_team_using_python(self) -> None:
        self.assertEqual(find_team_using("Python", self.graph), "Backend")

    def test_find_manager_of_backend(self) -> None:
        self.assertEqual(find_manager_of_team("Backend", self.graph), "Alice")

    def test_composed_python_to_manager_traversal(self) -> None:
        self.assertEqual(
            find_manager_of_team_using("Python", self.graph),
            "Alice",
        )

    def test_alice_to_team_to_owned_service_traversal(self) -> None:
        self.assertEqual(
            find_service_owned_by_team_managed_by("Alice", self.graph),
            "Payments API",
        )

    def test_security_sensitive_review_lookup(self) -> None:
        self.assertEqual(find_security_reviewer(graph=self.graph), "Clara")

    def test_unknown_queries_do_not_invent_answers(self) -> None:
        self.assertIsNone(find_team_using("Rust", self.graph))
        self.assertIsNone(find_manager_of_team("Mobile", self.graph))
        self.assertEqual(
            find_relationship_targets(
                "Emergency Deployment", MUST_BE_RECORDED_IN, self.graph
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
