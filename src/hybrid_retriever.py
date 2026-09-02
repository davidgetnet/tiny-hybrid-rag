"""Orchestrate vector and graph retrieval while preserving both evidence types."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum

import networkx as nx

from graph_retriever import (
    DEFAULT_GRAPH,
    find_deployment_approver_for_team,
    find_manager_of_team,
    find_security_reviewer,
    find_team_using,
)
from knowledge_graph import (
    CAN_APPROVE,
    MANAGES,
    REQUIRES_REVIEW_FROM,
    USES,
)
from vector_retriever import retrieve as retrieve_vector
from vector_store import VectorSearchResult


QUESTION_A = "What technology does the Backend team primarily use?"
QUESTION_B = "Who manages the team that uses Python?"
QUESTION_C = (
    "Who can approve a production deployment for the team that uses Python, "
    "and what additional requirement applies if the deployment is security-sensitive?"
)
EMERGENCY_LOGGING_QUESTION = (
    "What special logging timing is allowed for emergency deployments?"
)


class RetrievalMode(str, Enum):
    VECTOR = "vector"
    GRAPH = "graph"
    HYBRID = "hybrid"


class UnsupportedQueryError(ValueError):
    """Raised when the deliberately small educational router has no rule."""


@dataclass(frozen=True)
class GraphEdgeEvidence:
    """One explicit domain relationship with its source provenance."""

    subject: str
    relationship: str
    object: str
    source: str
    chunk_id: int


@dataclass(frozen=True)
class GraphEvidence:
    """Ordered entities and relationships produced by graph traversal."""

    entities: tuple[str, ...]
    relationships: tuple[GraphEdgeEvidence, ...]
    path: tuple[str, ...]


@dataclass(frozen=True)
class HybridEvidence:
    """Combined retrieval output; this is evidence rather than an answer."""

    query: str
    mode: RetrievalMode
    vector_evidence: tuple[VectorSearchResult, ...]
    graph_evidence: GraphEvidence | None
    trace: tuple[str, ...]


VectorRetriever = Callable[[str, int], Sequence[VectorSearchResult]]


def route_query(query: str) -> RetrievalMode:
    """Apply explicit learning rules, not a production natural-language router."""

    normalized = " ".join(query.lower().split())
    mentions_python_team = "team" in normalized and "python" in normalized

    if (
        mentions_python_team
        and "approv" in normalized
        and "security-sensitive" in normalized
    ):
        return RetrievalMode.HYBRID
    if mentions_python_team and ("manag" in normalized or "manager" in normalized):
        return RetrievalMode.GRAPH
    if "backend" in normalized and "technology" in normalized:
        return RetrievalMode.VECTOR
    if "emergency" in normalized and ("log" in normalized or "logging" in normalized):
        return RetrievalMode.VECTOR
    raise UnsupportedQueryError(
        "No educational routing rule matched this query. "
        "Use one of the documented learning scenarios."
    )


def deduplicate_vector_evidence(
    results: Sequence[VectorSearchResult],
) -> tuple[VectorSearchResult, ...]:
    """Remove duplicate stable record IDs without disturbing Chroma ranking."""

    seen: set[str] = set()
    unique: list[VectorSearchResult] = []
    for result in results:
        if result.id not in seen:
            seen.add(result.id)
            unique.append(result)
    return tuple(unique)


def _edge_evidence(
    graph: nx.DiGraph,
    subject: str,
    relationship: str,
    object_: str,
) -> GraphEdgeEvidence:
    attributes = graph.edges[subject, object_]
    if attributes["relationship"] != relationship:
        raise ValueError(
            f"Expected {subject} --{relationship}--> {object_}, "
            f"found {attributes['relationship']}"
        )
    return GraphEdgeEvidence(
        subject=subject,
        relationship=relationship,
        object=object_,
        source=attributes["source"],
        chunk_id=int(attributes["chunk_id"]),
    )


def _question_b_graph_evidence(graph: nx.DiGraph) -> GraphEvidence:
    technology = "Python"
    team = find_team_using(technology, graph)
    if team is None:
        raise LookupError(f"No team has a USES relationship to {technology}")
    manager = find_manager_of_team(team, graph)
    if manager is None:
        raise LookupError(f"No manager has a MANAGES relationship to {team}")

    return GraphEvidence(
        entities=(technology, team, manager),
        relationships=(
            _edge_evidence(graph, team, USES, technology),
            _edge_evidence(graph, manager, MANAGES, team),
        ),
        path=(technology, team, manager),
    )


def _question_c_graph_evidence(graph: nx.DiGraph) -> GraphEvidence:
    technology = "Python"
    team = find_team_using(technology, graph)
    if team is None:
        raise LookupError(f"No team has a USES relationship to {technology}")

    deployment = f"{team} Production Deployment"
    approver = find_deployment_approver_for_team(team, graph)
    if approver is None:
        raise LookupError(f"No approver is represented for {deployment}")

    security_deployment = "Security-Sensitive Deployment"
    reviewer = find_security_reviewer(security_deployment, graph)
    if reviewer is None:
        raise LookupError(f"No reviewer is represented for {security_deployment}")

    return GraphEvidence(
        entities=(technology, team, deployment, approver, security_deployment, reviewer),
        relationships=(
            _edge_evidence(graph, team, USES, technology),
            _edge_evidence(graph, approver, CAN_APPROVE, deployment),
            _edge_evidence(
                graph,
                security_deployment,
                REQUIRES_REVIEW_FROM,
                reviewer,
            ),
        ),
        path=(technology, team, deployment, approver, security_deployment, reviewer),
    )


def retrieve_hybrid(
    query: str,
    top_k: int = 3,
    *,
    vector_retriever: VectorRetriever = retrieve_vector,
    graph: nx.DiGraph = DEFAULT_GRAPH,
) -> HybridEvidence:
    """Route a learning query and preserve vector and graph evidence separately."""

    mode = route_query(query)
    trace = ["1. received query", f"2. routing mode = {mode.value}"]
    vector_evidence: tuple[VectorSearchResult, ...] = ()
    graph_evidence: GraphEvidence | None = None

    if mode in (RetrievalMode.VECTOR, RetrievalMode.HYBRID):
        trace.append("3. vector retriever called")
        vector_evidence = deduplicate_vector_evidence(
            vector_retriever(query, top_k)
        )
        trace.append(f"4. vector records returned = {len(vector_evidence)}")

    if mode == RetrievalMode.GRAPH:
        trace.append("3. graph traversal started from Python")
        graph_evidence = _question_b_graph_evidence(graph)
        trace.append(f"4. resolved team = {graph_evidence.entities[1]}")
        trace.append(f"5. resolved manager = {graph_evidence.entities[2]}")
    elif mode == RetrievalMode.HYBRID:
        trace.append("5. graph traversal started from Python")
        graph_evidence = _question_c_graph_evidence(graph)
        trace.append(f"6. resolved team = {graph_evidence.entities[1]}")
        trace.append(f"7. resolved approver = {graph_evidence.entities[3]}")
        trace.append(f"8. resolved security reviewer = {graph_evidence.entities[5]}")
        trace.append("9. combined evidence created")

    return HybridEvidence(
        query=query,
        mode=mode,
        vector_evidence=vector_evidence,
        graph_evidence=graph_evidence,
        trace=tuple(trace),
    )
