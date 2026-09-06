"""Failure-aware LangGraph orchestration above the existing RAG components."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import operator
import os
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from hybrid_retriever import (
    GraphEvidence,
    HybridEvidence,
    RetrievalMode,
    UnsupportedQueryError,
    deduplicate_vector_evidence,
    retrieve_graph as retrieve_existing_graph,
    route_query as choose_retrieval_mode,
)
from llm_synthesizer import (
    LLM_MODEL,
    PROMPT_VERSION,
    SynthesisResult,
    build_prompt,
    synthesize as synthesize_existing,
)
from vector_retriever import retrieve as retrieve_existing_vector
from vector_store import VectorSearchResult


RouteName = Literal["vector", "graph", "hybrid", "unsupported"]
ComponentStatus = Literal[
    "not_requested", "not_configured", "success", "no_evidence", "failed"
]
OverallRetrievalStatus = Literal["pending", "success", "degraded", "failed"]


class RetrievalError(TypedDict):
    component: Literal["vector", "graph", "lexical", "routing"]
    error_type: str
    message: str


class WorkflowState(TypedDict, total=False):
    """Per-request state; deliberately not persisted in this learning phase."""

    query: str
    original_question: str
    retrieval_mode: RouteName | None
    vector_evidence: tuple[VectorSearchResult, ...]
    graph_evidence: GraphEvidence | None
    retrieved_evidence: HybridEvidence | None
    vector_retrieval_status: ComponentStatus
    vector_retrieval_error: str | None
    vector_failed: bool
    graph_retrieval_status: ComponentStatus
    graph_retrieval_error: str | None
    graph_failed: bool
    lexical_retrieval_status: ComponentStatus
    lexical_retrieval_error: str | None
    lexical_failed: bool
    overall_retrieval_status: OverallRetrievalStatus
    errors: tuple[RetrievalError, ...]
    last_error: RetrievalError | None
    synthesis_status: Literal["not_started", "success"]
    synthesis_prompt: str | None
    answer: str | None
    dry_run: bool
    api_called: bool
    prompt_version: str
    model: str
    trace: Annotated[list[str], operator.add]


VectorRetriever = Callable[[str, int], Sequence[VectorSearchResult]]
GraphRetriever = Callable[[str], GraphEvidence | None]
Synthesizer = Callable[..., SynthesisResult]


class WorkflowInvariantError(ValueError):
    """Raised when state would send execution through an unknown route."""


@dataclass(frozen=True)
class WorkflowDependencies:
    vector_retriever: VectorRetriever = retrieve_existing_vector
    graph_retriever: GraphRetriever = retrieve_existing_graph
    synthesizer: Synthesizer = synthesize_existing


def _as_hybrid_evidence(state: WorkflowState) -> HybridEvidence:
    mode_name = state.get("retrieval_mode")
    if mode_name not in {"vector", "graph", "hybrid", "unsupported"}:
        raise WorkflowInvariantError(f"Invalid retrieval mode: {mode_name!r}")
    mode = (
        RetrievalMode.VECTOR if mode_name == "unsupported" else RetrievalMode(mode_name)
    )
    return HybridEvidence(
        query=state["original_question"],
        mode=mode,
        vector_evidence=state.get("vector_evidence", ()),
        graph_evidence=state.get("graph_evidence"),
        trace=tuple(state.get("trace", ())),
    )


def _failure(
    component: Literal["vector", "graph", "lexical", "routing"], error: Exception
) -> RetrievalError:
    return {
        "component": component,
        "error_type": type(error).__name__,
        "message": str(error),
    }


def _has_usable_evidence(state: WorkflowState) -> bool:
    graph = state.get("graph_evidence")
    return bool(state.get("vector_evidence")) or bool(graph and graph.relationships)


def _complete_retrieval(updates: WorkflowState, state: WorkflowState) -> WorkflowState:
    combined = {**state, **updates}
    errors = tuple(combined.get("errors", ()))
    usable = _has_usable_evidence(combined)
    updates["overall_retrieval_status"] = (
        "failed" if not usable else "degraded" if errors else "success"
    )
    updates["retrieved_evidence"] = _as_hybrid_evidence(combined) if usable else None
    updates["last_error"] = errors[-1] if errors else None
    return updates


def initialize_node(state: WorkflowState) -> WorkflowState:
    question = state["query"]
    return {
        "query": question,
        "original_question": question,
        "retrieval_mode": None,
        "vector_evidence": (),
        "graph_evidence": None,
        "retrieved_evidence": None,
        "vector_retrieval_status": "not_requested",
        "vector_retrieval_error": None,
        "vector_failed": False,
        "graph_retrieval_status": "not_requested",
        "graph_retrieval_error": None,
        "graph_failed": False,
        "lexical_retrieval_status": "not_configured",
        "lexical_retrieval_error": None,
        "lexical_failed": False,
        "overall_retrieval_status": "pending",
        "errors": (),
        "last_error": None,
        "synthesis_status": "not_started",
        "synthesis_prompt": None,
        "answer": None,
        "dry_run": state.get("dry_run", True),
        "api_called": False,
        "prompt_version": PROMPT_VERSION,
        "model": LLM_MODEL,
        "trace": ["START", "initialize"],
    }


def route_query_node(state: WorkflowState) -> WorkflowState:
    try:
        mode: RouteName = choose_retrieval_mode(state["original_question"]).value
        return {"retrieval_mode": mode, "trace": ["route_query"]}
    except UnsupportedQueryError as error:
        failure = _failure("routing", error)
        return {
            "retrieval_mode": "unsupported",
            "errors": (failure,),
            "last_error": failure,
            "trace": ["route_query"],
        }


def select_route(state: WorkflowState) -> RouteName:
    mode = state.get("retrieval_mode")
    if mode not in {"vector", "graph", "hybrid", "unsupported"}:
        raise WorkflowInvariantError(f"Invalid retrieval mode: {mode!r}")
    return mode


def select_after_retrieval(state: WorkflowState) -> Literal["evidence", "failure"]:
    return "evidence" if _has_usable_evidence(state) else "failure"


def build_workflow(dependencies: WorkflowDependencies | None = None):
    """Compile the graph without persistence or retries."""

    dependencies = dependencies or WorkflowDependencies()

    def safe_vector(state: WorkflowState) -> WorkflowState:
        try:
            evidence = deduplicate_vector_evidence(
                dependencies.vector_retriever(state["original_question"], 3)
            )
            return {
                "vector_evidence": evidence,
                "vector_retrieval_status": "success" if evidence else "no_evidence",
                "vector_failed": False,
                "trace": ["retrieve_vector"],
            }
        except Exception as error:  # workflow failure boundary
            failure = _failure("vector", error)
            return {
                "vector_evidence": (),
                "vector_retrieval_status": "failed",
                "vector_retrieval_error": failure["message"],
                "vector_failed": True,
                "errors": (failure,),
                "trace": ["retrieve_vector (failed)"],
            }

    def safe_graph(state: WorkflowState) -> WorkflowState:
        try:
            evidence = dependencies.graph_retriever(state["original_question"])
            usable = bool(evidence and evidence.relationships)
            return {
                "graph_evidence": evidence if usable else None,
                "graph_retrieval_status": "success" if usable else "no_evidence",
                "graph_failed": False,
                "trace": ["retrieve_graph"],
            }
        except Exception as error:  # workflow failure boundary
            failure = _failure("graph", error)
            return {
                "graph_evidence": None,
                "graph_retrieval_status": "failed",
                "graph_retrieval_error": failure["message"],
                "graph_failed": True,
                "errors": (failure,),
                "trace": ["retrieve_graph (failed)"],
            }

    def retrieve_vector_node(state: WorkflowState) -> WorkflowState:
        return _complete_retrieval(safe_vector(state), state)

    def retrieve_graph_node(state: WorkflowState) -> WorkflowState:
        return _complete_retrieval(safe_graph(state), state)

    def retrieve_hybrid_node(state: WorkflowState) -> WorkflowState:
        vector_updates = safe_vector(state)
        graph_updates = safe_graph({**state, **vector_updates})
        errors = tuple(vector_updates.get("errors", ())) + tuple(
            graph_updates.get("errors", ())
        )
        updates: WorkflowState = {
            **vector_updates,
            **graph_updates,
            "errors": errors,
            "trace": ["retrieve_hybrid"],
        }
        return _complete_retrieval(updates, state)

    def unsupported_route_node(state: WorkflowState) -> WorkflowState:
        return _complete_retrieval(
            {"trace": ["unsupported_route (no retrieval rule matched)"]}, state
        )

    def retrieval_failure_node(state: WorkflowState) -> WorkflowState:
        return {
            "synthesis_status": "not_started",
            "trace": ["retrieval_failure", "END"],
        }

    def build_context_node(state: WorkflowState) -> WorkflowState:
        prompt = build_prompt(state["original_question"], _as_hybrid_evidence(state))
        return {"synthesis_prompt": prompt, "trace": ["build_context"]}

    def synthesize_node(state: WorkflowState) -> WorkflowState:
        result = dependencies.synthesizer(
            state["original_question"],
            _as_hybrid_evidence(state),
            model=state["model"],
            dry_run=state["dry_run"],
        )
        if result.prompt != state["synthesis_prompt"]:
            raise WorkflowInvariantError(
                "Synthesis prompt changed after context building"
            )
        return {
            "answer": result.answer,
            "api_called": result.api_called,
            "synthesis_status": "success",
            "trace": ["synthesize", "END"],
        }

    builder = StateGraph(WorkflowState)
    for name, node in (
        ("initialize", initialize_node),
        ("route_query", route_query_node),
        ("retrieve_vector", retrieve_vector_node),
        ("retrieve_graph", retrieve_graph_node),
        ("retrieve_hybrid", retrieve_hybrid_node),
        ("unsupported_route", unsupported_route_node),
        ("retrieval_failure", retrieval_failure_node),
        ("build_context", build_context_node),
        ("synthesize", synthesize_node),
    ):
        builder.add_node(name, node)
    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "route_query")
    builder.add_conditional_edges(
        "route_query",
        select_route,
        {
            "vector": "retrieve_vector",
            "graph": "retrieve_graph",
            "hybrid": "retrieve_hybrid",
            "unsupported": "unsupported_route",
        },
    )
    for node in (
        "retrieve_vector",
        "retrieve_graph",
        "retrieve_hybrid",
        "unsupported_route",
    ):
        builder.add_conditional_edges(
            node,
            select_after_retrieval,
            {
                "evidence": "build_context",
                "failure": "retrieval_failure",
            },
        )
    builder.add_edge("retrieval_failure", END)
    builder.add_edge("build_context", "synthesize")
    builder.add_edge("synthesize", END)
    return builder.compile()


WORKFLOW = build_workflow()


def initial_state(query: str, *, dry_run: bool | None = None) -> WorkflowState:
    if dry_run is None:
        dry_run = not bool(os.getenv("OPENAI_API_KEY"))
    return {"query": query, "dry_run": dry_run, "trace": []}


def run_workflow(
    query: str, *, dry_run: bool | None = None, workflow=None
) -> WorkflowState:
    return (workflow or WORKFLOW).invoke(initial_state(query, dry_run=dry_run))
