"""Explicit LangGraph orchestration above the existing RAG components."""

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
    retrieve_hybrid as retrieve_existing_hybrid,
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


class WorkflowState(TypedDict, total=False):
    """Data carried through one workflow invocation, not a knowledge base."""

    query: str
    retrieval_mode: RouteName | None
    vector_evidence: tuple[VectorSearchResult, ...]
    graph_evidence: GraphEvidence | None
    synthesis_prompt: str | None
    answer: str | None
    dry_run: bool
    api_called: bool
    prompt_version: str
    model: str
    trace: Annotated[list[str], operator.add]


VectorRetriever = Callable[[str, int], Sequence[VectorSearchResult]]
Synthesizer = Callable[..., SynthesisResult]


class WorkflowInvariantError(ValueError):
    """Raised when state would send execution through an unknown route."""


@dataclass(frozen=True)
class WorkflowDependencies:
    vector_retriever: VectorRetriever = retrieve_existing_vector
    synthesizer: Synthesizer = synthesize_existing


def _as_hybrid_evidence(state: WorkflowState) -> HybridEvidence:
    mode_name = state.get("retrieval_mode")
    if mode_name not in {"vector", "graph", "hybrid", "unsupported"}:
        raise WorkflowInvariantError(f"Invalid retrieval mode: {mode_name!r}")
    mode = RetrievalMode.VECTOR if mode_name == "unsupported" else RetrievalMode(mode_name)
    return HybridEvidence(
        query=state["query"],
        mode=mode,
        vector_evidence=state.get("vector_evidence", ()),
        graph_evidence=state.get("graph_evidence"),
        trace=tuple(state.get("trace", ())),
    )


def initialize_node(state: WorkflowState) -> WorkflowState:
    """Establish clean per-run fields without doing retrieval."""

    return {
        "query": state["query"],
        "retrieval_mode": None,
        "vector_evidence": (),
        "graph_evidence": None,
        "synthesis_prompt": None,
        "answer": None,
        "dry_run": state.get("dry_run", True),
        "api_called": False,
        "prompt_version": PROMPT_VERSION,
        "model": LLM_MODEL,
        "trace": ["START", "initialize"],
    }


def route_query_node(state: WorkflowState) -> WorkflowState:
    """Reuse deterministic routing and represent unsupported input explicitly."""

    try:
        mode: RouteName = choose_retrieval_mode(state["query"]).value
    except UnsupportedQueryError:
        mode = "unsupported"
    return {"retrieval_mode": mode, "trace": ["route_query"]}


def select_route(state: WorkflowState) -> RouteName:
    """Validate the route used by LangGraph's conditional edge."""

    mode = state.get("retrieval_mode")
    if mode not in {"vector", "graph", "hybrid", "unsupported"}:
        raise WorkflowInvariantError(f"Invalid retrieval mode: {mode!r}")
    return mode


def build_workflow(dependencies: WorkflowDependencies | None = None):
    """Define and compile the acyclic workflow without a checkpointer."""

    dependencies = dependencies or WorkflowDependencies()

    def retrieve_vector_node(state: WorkflowState) -> WorkflowState:
        evidence = deduplicate_vector_evidence(
            dependencies.vector_retriever(state["query"], 3)
        )
        return {"vector_evidence": evidence, "trace": ["retrieve_vector"]}

    def retrieve_graph_node(state: WorkflowState) -> WorkflowState:
        evidence = retrieve_existing_hybrid(
            state["query"], vector_retriever=dependencies.vector_retriever
        )
        return {
            "graph_evidence": evidence.graph_evidence,
            "trace": ["retrieve_graph"],
        }

    def retrieve_hybrid_node(state: WorkflowState) -> WorkflowState:
        evidence = retrieve_existing_hybrid(
            state["query"], vector_retriever=dependencies.vector_retriever
        )
        return {
            "vector_evidence": evidence.vector_evidence,
            "graph_evidence": evidence.graph_evidence,
            "trace": ["retrieve_hybrid"],
        }

    def unsupported_route_node(state: WorkflowState) -> WorkflowState:
        return {"trace": ["unsupported_route (no retrieval rule matched)"]}

    def build_context_node(state: WorkflowState) -> WorkflowState:
        prompt = build_prompt(state["query"], _as_hybrid_evidence(state))
        return {"synthesis_prompt": prompt, "trace": ["build_context"]}

    def synthesize_node(state: WorkflowState) -> WorkflowState:
        result = dependencies.synthesizer(
            state["query"],
            _as_hybrid_evidence(state),
            model=state["model"],
            dry_run=state["dry_run"],
        )
        if result.prompt != state["synthesis_prompt"]:
            raise WorkflowInvariantError("Synthesis prompt changed after context building")
        return {
            "answer": result.answer,
            "api_called": result.api_called,
            "trace": ["synthesize", "END"],
        }

    builder = StateGraph(WorkflowState)
    builder.add_node("initialize", initialize_node)
    builder.add_node("route_query", route_query_node)
    builder.add_node("retrieve_vector", retrieve_vector_node)
    builder.add_node("retrieve_graph", retrieve_graph_node)
    builder.add_node("retrieve_hybrid", retrieve_hybrid_node)
    builder.add_node("unsupported_route", unsupported_route_node)
    builder.add_node("build_context", build_context_node)
    builder.add_node("synthesize", synthesize_node)

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
    for retrieval_node in (
        "retrieve_vector",
        "retrieve_graph",
        "retrieve_hybrid",
        "unsupported_route",
    ):
        builder.add_edge(retrieval_node, "build_context")
    builder.add_edge("build_context", "synthesize")
    builder.add_edge("synthesize", END)
    return builder.compile()


WORKFLOW = build_workflow()


def initial_state(query: str, *, dry_run: bool | None = None) -> WorkflowState:
    """Create the minimal input; initialize_node establishes all other fields."""

    if dry_run is None:
        dry_run = not bool(os.getenv("OPENAI_API_KEY"))
    return {"query": query, "dry_run": dry_run, "trace": []}


def run_workflow(
    query: str,
    *,
    dry_run: bool | None = None,
    workflow=None,
) -> WorkflowState:
    """Invoke the compiled graph and expose its final state."""

    return (workflow or WORKFLOW).invoke(initial_state(query, dry_run=dry_run))
