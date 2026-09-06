"""Behavioral tests for explicit LangGraph orchestration."""

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hybrid_retriever import (  # noqa: E402
    QUESTION_A,
    QUESTION_B,
    QUESTION_C,
    retrieve_graph,
)
from langgraph_workflow import (  # noqa: E402
    WorkflowDependencies,
    WorkflowInvariantError,
    build_workflow,
    run_workflow,
    select_route,
)
from vector_store import VectorSearchResult  # noqa: E402


def vector_result(record_id: str, text: str, distance: float) -> VectorSearchResult:
    source, chunk_id = record_id.rsplit(":", 1)
    return VectorSearchResult(
        id=record_id,
        text=text,
        source=source,
        chunk_id=int(chunk_id),
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        distance=distance,
    )


@pytest.fixture
def workflow():
    def fake_vector_retriever(query: str, top_k: int):
        if query == QUESTION_A:
            return [
                vector_result(
                    "handbook.md:2", "Frontend primarily uses TypeScript.", 0.44
                ),
                vector_result("handbook.md:1", "Backend primarily uses Python.", 0.52),
            ]
        return [
            vector_result(
                "policies.md:2",
                "Security-sensitive deployments require Clara's review.",
                0.45,
            ),
            vector_result(
                "policies.md:1", "Alice can approve Backend deployments.", 0.50
            ),
        ]

    return build_workflow(
        WorkflowDependencies(
            vector_retriever=fake_vector_retriever,
            graph_retriever=retrieve_graph,
        )
    )


def test_graph_compiles_with_expected_nodes(workflow) -> None:
    nodes = set(workflow.get_graph().nodes)
    assert {
        "__start__",
        "initialize",
        "route_query",
        "retrieve_vector",
        "retrieve_graph",
        "retrieve_hybrid",
        "unsupported_route",
        "retrieval_failure",
        "build_context",
        "synthesize",
        "__end__",
    } <= nodes


def test_question_a_takes_vector_path(workflow) -> None:
    final = run_workflow(QUESTION_A, dry_run=True, workflow=workflow)
    assert final["retrieval_mode"] == "vector"
    assert "retrieve_vector" in final["trace"]
    assert "retrieve_graph" not in final["trace"]
    assert len(final["vector_evidence"]) == 2
    assert final["graph_evidence"] is None


def test_question_b_takes_graph_path_and_resolves_alice(workflow) -> None:
    final = run_workflow(QUESTION_B, dry_run=True, workflow=workflow)
    assert final["retrieval_mode"] == "graph"
    assert "retrieve_graph" in final["trace"]
    assert final["vector_evidence"] == ()
    edges = final["graph_evidence"].relationships
    assert any(
        edge.subject == "Alice" and edge.relationship == "MANAGES" for edge in edges
    )


def test_question_c_takes_hybrid_path_and_preserves_both_evidence_types(
    workflow,
) -> None:
    final = run_workflow(QUESTION_C, dry_run=True, workflow=workflow)
    assert final["retrieval_mode"] == "hybrid"
    assert "retrieve_hybrid" in final["trace"]
    assert len(final["vector_evidence"]) == 2
    graph_text = " ".join(
        f"{edge.subject} {edge.object}"
        for edge in final["graph_evidence"].relationships
    )
    assert all(name in graph_text for name in ("Backend", "Alice", "Clara"))
    assert final["overall_retrieval_status"] == "success"
    assert final["vector_retrieval_status"] == "success"
    assert final["graph_retrieval_status"] == "success"
    assert final["lexical_retrieval_status"] == "not_configured"
    assert final["retrieved_evidence"] is not None


def test_context_and_dry_run_answer_reach_final_state(workflow) -> None:
    final = run_workflow(QUESTION_C, dry_run=True, workflow=workflow)
    assert "VECTOR EVIDENCE" in final["synthesis_prompt"]
    assert "GRAPH EVIDENCE" in final["synthesis_prompt"]
    assert "No LLM call was made" in final["answer"]
    assert final["api_called"] is False
    assert final["trace"][-2:] == ["synthesize", "END"]


def test_prompt_contains_text_but_not_raw_embedding_vectors(workflow) -> None:
    final = run_workflow(QUESTION_A, dry_run=True, workflow=workflow)
    prompt = final["synthesis_prompt"]
    assert "Backend primarily uses Python" in prompt
    assert "0.027" not in prompt
    assert "384" not in prompt
    assert "sentence-transformers/all-MiniLM-L6-v2" not in prompt


def test_unsupported_question_stops_without_synthesis(workflow) -> None:
    final = run_workflow(
        "What cloud provider does Acorn Labs use?", dry_run=True, workflow=workflow
    )
    assert final["retrieval_mode"] == "unsupported"
    assert "unsupported_route (no retrieval rule matched)" in final["trace"]
    assert final["vector_evidence"] == ()
    assert final["graph_evidence"] is None
    assert final["overall_retrieval_status"] == "failed"
    assert final["synthesis_prompt"] is None
    assert final["answer"] is None
    assert final["synthesis_status"] == "not_started"
    assert final["trace"][-2:] == ["retrieval_failure", "END"]


def resilient_workflow(*, vector_retriever, graph_retriever, synthesizer=None):
    dependencies = WorkflowDependencies(
        vector_retriever=vector_retriever,
        graph_retriever=graph_retriever,
        **({"synthesizer": synthesizer} if synthesizer else {}),
    )
    return build_workflow(dependencies)


def useful_vectors(query: str, top_k: int):
    return [vector_result("policies.md:1", "Useful policy evidence.", 0.2)]


def test_vector_failure_degrades_to_graph_evidence() -> None:
    def failed_vector(query: str, top_k: int):
        raise ConnectionError("simulated vector outage")

    final = run_workflow(
        QUESTION_C,
        dry_run=True,
        workflow=resilient_workflow(
            vector_retriever=failed_vector,
            graph_retriever=retrieve_graph,
        ),
    )
    assert final["vector_retrieval_status"] == "failed"
    assert final["vector_failed"] is True
    assert final["vector_retrieval_error"] == "simulated vector outage"
    assert final["graph_retrieval_status"] == "success"
    assert final["overall_retrieval_status"] == "degraded"
    assert final["synthesis_status"] == "success"


def test_graph_failure_degrades_to_vector_evidence() -> None:
    def failed_graph(query: str):
        raise TimeoutError("simulated graph timeout")

    final = run_workflow(
        QUESTION_C,
        dry_run=True,
        workflow=resilient_workflow(
            vector_retriever=useful_vectors,
            graph_retriever=failed_graph,
        ),
    )
    assert final["vector_retrieval_status"] == "success"
    assert final["graph_retrieval_status"] == "failed"
    assert final["graph_failed"] is True
    assert final["graph_retrieval_error"] == "simulated graph timeout"
    assert final["overall_retrieval_status"] == "degraded"
    assert final["synthesis_status"] == "success"


def test_all_retrieval_paths_fail_cleanly_and_record_components() -> None:
    def failed_vector(query: str, top_k: int):
        raise ConnectionError("vector unavailable")

    def failed_graph(query: str):
        raise RuntimeError("graph unavailable")

    final = run_workflow(
        QUESTION_C,
        dry_run=True,
        workflow=resilient_workflow(
            vector_retriever=failed_vector,
            graph_retriever=failed_graph,
        ),
    )
    assert final["overall_retrieval_status"] == "failed"
    assert final["synthesis_status"] == "not_started"
    assert final["synthesis_prompt"] is None
    assert final["answer"] is None
    assert {error["component"] for error in final["errors"]} == {"vector", "graph"}
    assert final["vector_failed"] is True
    assert final["graph_failed"] is True
    assert final["lexical_failed"] is False
    assert final["last_error"]["message"] == "graph unavailable"


def test_no_usable_evidence_never_reaches_synthesis() -> None:
    synthesis_calls = []

    def empty_graph(query: str):
        return None

    def unexpected_synthesis(*args, **kwargs):
        synthesis_calls.append((args, kwargs))
        raise AssertionError("synthesis must not run without evidence")

    final = run_workflow(
        QUESTION_C,
        dry_run=True,
        workflow=resilient_workflow(
            vector_retriever=lambda query, top_k: [],
            graph_retriever=empty_graph,
            synthesizer=unexpected_synthesis,
        ),
    )
    assert final["vector_retrieval_status"] == "no_evidence"
    assert final["graph_retrieval_status"] == "no_evidence"
    assert final["overall_retrieval_status"] == "failed"
    assert synthesis_calls == []


@pytest.mark.parametrize("invalid", [None, "lexical", "VECTOR"])
def test_invalid_route_fails_clearly(invalid) -> None:
    with pytest.raises(WorkflowInvariantError, match="Invalid retrieval mode"):
        select_route({"retrieval_mode": invalid})
