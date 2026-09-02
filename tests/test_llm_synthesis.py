"""Unit tests for the inspectable, retrieval-independent synthesis boundary."""

import sys
import os
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hybrid_retriever import (  # noqa: E402
    GraphEdgeEvidence,
    GraphEvidence,
    HybridEvidence,
    RetrievalMode,
)
from llm_synthesizer import (  # noqa: E402
    PROMPT_VERSION,
    build_prompt,
    format_graph_evidence,
    format_vector_evidence,
    synthesize,
)
from vector_store import VectorSearchResult  # noqa: E402


def hybrid_evidence() -> HybridEvidence:
    vector = VectorSearchResult(
        id="policies.md:1",
        text="Alice can approve Backend production deployments.",
        source="policies.md",
        chunk_id=1,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        distance=0.23456,
    )
    graph = GraphEvidence(
        entities=("Backend", "Python", "Alice"),
        relationships=(
            GraphEdgeEvidence(
                "Backend", "USES", "Python", "handbook.md", 1
            ),
            GraphEdgeEvidence(
                "Alice",
                "CAN_APPROVE",
                "Backend Production Deployment",
                "policies.md",
                1,
            ),
        ),
        path=("Python", "Backend", "Alice"),
    )
    return HybridEvidence(
        query="Who can approve deployment for the team using Python?",
        mode=RetrievalMode.HYBRID,
        vector_evidence=(vector,),
        graph_evidence=graph,
        trace=(),
    )


def test_vector_evidence_formats_text_distance_and_reference() -> None:
    rendered = format_vector_evidence(hybrid_evidence())
    assert "[policies.md:1]" in rendered
    assert "Alice can approve" in rendered
    assert "distance: 0.2346" in rendered


def test_graph_evidence_preserves_relationship_and_provenance() -> None:
    rendered = format_graph_evidence(hybrid_evidence().graph_evidence)
    assert "Backend --USES--> Python" in rendered
    assert "source: handbook.md:1" in rendered
    assert "Alice --CAN_APPROVE--> Backend Production Deployment" in rendered


def test_prompt_is_explicit_and_accepts_hybrid_evidence() -> None:
    evidence = hybrid_evidence()
    prompt = build_prompt(evidence.query, evidence)
    assert PROMPT_VERSION in prompt
    assert "USER QUESTION" in prompt
    assert "VECTOR EVIDENCE" in prompt
    assert "GRAPH EVIDENCE" in prompt
    assert "using only the supplied evidence" in prompt
    assert "available evidence is insufficient" in prompt
    assert "explicitly acknowledge the conflict" in prompt


def test_prompt_excludes_embeddings_and_environment_secrets(monkeypatch) -> None:
    secret = "sk-test-secret-that-must-never-appear"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    prompt = build_prompt(hybrid_evidence().query, hybrid_evidence())
    assert secret not in prompt
    assert "0.027" not in prompt
    assert "384" not in prompt
    assert "sentence-transformers/all-MiniLM-L6-v2" not in prompt


def test_synthesize_uses_stub_and_retains_references_and_trace() -> None:
    calls = []

    def stub(prompt: str, model: str):
        calls.append((prompt, model))
        return "Alice can approve it [policies.md:1].", 100, 12

    result = synthesize(hybrid_evidence().query, hybrid_evidence(), model_caller=stub)
    assert result.answer.endswith("[policies.md:1].")
    assert result.evidence_references == ("handbook.md:1", "policies.md:1")
    assert result.api_called is True
    assert result.input_tokens == 100
    assert result.output_tokens == 12
    assert calls and "VECTOR EVIDENCE" in calls[0][0]
    assert result.trace[-1] == "8. answer returned"


def test_dry_run_does_not_call_model() -> None:
    def forbidden_call(prompt: str, model: str):
        raise AssertionError("model must not be called")

    result = synthesize(
        hybrid_evidence().query,
        hybrid_evidence(),
        dry_run=True,
        model_caller=forbidden_call,
    )
    assert result.api_called is False
    assert "No LLM call" in result.answer


def test_conflicting_evidence_instruction_is_sent_with_both_claims() -> None:
    evidence = hybrid_evidence()
    first = evidence.vector_evidence[0]
    conflicting = VectorSearchResult(
        id="test-conflict.md:1",
        text="Backend uses Go.",
        source="test-conflict.md",
        chunk_id=1,
        embedding_model="test-only",
        distance=0.1,
    )
    evidence = HybridEvidence(
        query="What does Backend use?",
        mode=RetrievalMode.VECTOR,
        vector_evidence=(first, conflicting),
        graph_evidence=None,
        trace=(),
    )
    prompt = build_prompt(evidence.query, evidence)
    assert "Alice can approve" in prompt
    assert "Backend uses Go" in prompt
    assert "explicitly acknowledge the conflict" in prompt


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="optional live test requires OPENAI_API_KEY"
)
def test_optional_live_openai_smoke() -> None:
    evidence = hybrid_evidence()
    result = synthesize(evidence.query, evidence)
    assert result.api_called is True
    assert result.answer
