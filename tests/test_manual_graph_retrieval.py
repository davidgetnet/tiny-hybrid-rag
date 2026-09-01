"""Focused tests for the explicit Lesson 5.5 NetworkX operations."""

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from knowledge_graph import build_knowledge_graph  # noqa: E402
from manual_graph_retrieval import (  # noqa: E402
    build_chunk_lookup,
    find_node,
    get_one_hop,
    get_two_hops,
    resolve_edge_source,
)


class ManualGraphRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = build_knowledge_graph()

    def test_known_node_can_be_found_with_attributes(self) -> None:
        self.assertEqual(
            find_node(self.graph, "Backend"),
            ("Backend", {"entity_type": "Team"}),
        )

    def test_one_hop_returns_expected_neighbor_and_relationship(self) -> None:
        edges = get_one_hop(self.graph, "Backend", relationship="USES")
        self.assertEqual(len(edges), 1)
        source, target, attributes = edges[0]
        self.assertEqual((source, target), ("Backend", "Python"))
        self.assertEqual(attributes["relationship"], "USES")

    def test_two_hops_reach_python_through_backend(self) -> None:
        paths = get_two_hops(self.graph, "Alice")
        relationship_paths = [
            (
                first[2]["relationship"],
                first[1],
                second[2]["relationship"],
                second[1],
            )
            for first, second in paths
        ]
        self.assertIn(("MANAGES", "Backend", "USES", "Python"), relationship_paths)

    def test_nonexistent_node_fails_cleanly(self) -> None:
        self.assertIsNone(find_node(self.graph, "Database"))
        self.assertEqual(get_one_hop(self.graph, "Database"), [])

    def test_missing_relationship_returns_no_evidence(self) -> None:
        self.assertEqual(
            get_one_hop(self.graph, "Backend", relationship="USES_DATABASE"),
            [],
        )

    def test_provenance_resolves_to_expected_source_chunk(self) -> None:
        edge = get_one_hop(self.graph, "Backend", relationship="USES")[0]
        chunk = resolve_edge_source(edge, build_chunk_lookup())
        self.assertEqual(chunk.metadata["source"], "handbook.md")
        self.assertEqual(chunk.metadata["chunk_id"], 1)
        self.assertIn("using Python", chunk.content)

    def test_traversal_output_is_deterministic(self) -> None:
        first = get_two_hops(self.graph, "Alice")
        second = get_two_hops(self.graph, "Alice")
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
