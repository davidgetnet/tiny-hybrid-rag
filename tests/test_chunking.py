"""Small checks for this increment's loading and chunking behavior."""

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chunk_documents import chunk_documents  # noqa: E402
from load_documents import load_documents  # noqa: E402


class ChunkingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = load_documents()
        self.chunks = chunk_documents(self.documents)

    def test_loads_both_sources(self) -> None:
        self.assertEqual(
            [document.source for document in self.documents],
            ["handbook.md", "policies.md"],
        )

    def test_chunks_are_non_empty_and_have_metadata(self) -> None:
        self.assertTrue(self.chunks)
        for chunk in self.chunks:
            self.assertTrue(chunk.content)
            self.assertIn("source", chunk.metadata)
            self.assertIn("chunk_id", chunk.metadata)

    def test_chunk_numbers_restart_and_are_deterministic(self) -> None:
        actual = [
            (chunk.metadata["source"], chunk.metadata["chunk_id"])
            for chunk in self.chunks
        ]
        self.assertEqual(
            actual,
            [
                ("handbook.md", 0),
                ("handbook.md", 1),
                ("handbook.md", 2),
                ("handbook.md", 3),
                ("policies.md", 0),
                ("policies.md", 1),
                ("policies.md", 2),
                ("policies.md", 3),
            ],
        )

    def test_important_facts_remain_visible(self) -> None:
        all_text = "\n".join(chunk.content for chunk in self.chunks)
        important_phrases = (
            "Alice manages the Backend team",
            "using Python",
            "owns the Payments API",
            "Ben manages the Frontend team",
            "uses TypeScript",
            "owns the Customer Portal",
            "Clara is the Security Lead",
            "Every production deployment requires approval",
            "additional review from Clara",
            "deployment log",
            "Emergency deployments still require approval",
        )
        for phrase in important_phrases:
            self.assertIn(phrase, all_text)


if __name__ == "__main__":
    unittest.main()
