"""Manually encode the small Acorn Labs knowledge graph."""

from typing import Any

import networkx as nx


RELATIONSHIP = "relationship"
SOURCE = "source"
CHUNK_ID = "chunk_id"

MANAGES = "MANAGES"
USES = "USES"
OWNS = "OWNS"
CAN_APPROVE = "CAN_APPROVE"
REQUIRES_REVIEW_FROM = "REQUIRES_REVIEW_FROM"
MUST_BE_RECORDED_IN = "MUST_BE_RECORDED_IN"


def _add_fact(
    graph: nx.DiGraph,
    subject: str,
    relationship: str,
    object_: str,
    *,
    source: str,
    chunk_id: int,
) -> None:
    """Add one directed domain fact with its document provenance."""

    graph.add_edge(
        subject,
        object_,
        relationship=relationship,
        source=source,
        chunk_id=chunk_id,
    )


def build_knowledge_graph() -> nx.DiGraph:
    """Build the graph explicitly; no text or relationship extraction occurs here."""

    graph = nx.DiGraph(name="Acorn Labs Knowledge Graph")
    graph.add_nodes_from(
        [
            ("Alice", {"entity_type": "Person"}),
            ("Ben", {"entity_type": "Person"}),
            ("Clara", {"entity_type": "Person"}),
            ("Backend", {"entity_type": "Team"}),
            ("Frontend", {"entity_type": "Team"}),
            ("Python", {"entity_type": "Technology"}),
            ("TypeScript", {"entity_type": "Technology"}),
            ("Payments API", {"entity_type": "Service"}),
            ("Customer Portal", {"entity_type": "Service"}),
            (
                "Production Deployment",
                {"entity_type": "Deployment Type"},
            ),
            (
                "Backend Production Deployment",
                {"entity_type": "Deployment Type"},
            ),
            (
                "Frontend Production Deployment",
                {"entity_type": "Deployment Type"},
            ),
            (
                "Security-Sensitive Deployment",
                {"entity_type": "Deployment Type"},
            ),
            ("Deployment Log", {"entity_type": "Artifact"}),
        ]
    )

    _add_fact(graph, "Alice", MANAGES, "Backend", source="handbook.md", chunk_id=1)
    _add_fact(graph, "Backend", USES, "Python", source="handbook.md", chunk_id=1)
    _add_fact(
        graph,
        "Backend",
        OWNS,
        "Payments API",
        source="handbook.md",
        chunk_id=1,
    )
    _add_fact(graph, "Ben", MANAGES, "Frontend", source="handbook.md", chunk_id=2)
    _add_fact(
        graph,
        "Frontend",
        USES,
        "TypeScript",
        source="handbook.md",
        chunk_id=2,
    )
    _add_fact(
        graph,
        "Frontend",
        OWNS,
        "Customer Portal",
        source="handbook.md",
        chunk_id=2,
    )
    _add_fact(
        graph,
        "Alice",
        CAN_APPROVE,
        "Backend Production Deployment",
        source="policies.md",
        chunk_id=1,
    )
    _add_fact(
        graph,
        "Ben",
        CAN_APPROVE,
        "Frontend Production Deployment",
        source="policies.md",
        chunk_id=1,
    )
    _add_fact(
        graph,
        "Security-Sensitive Deployment",
        REQUIRES_REVIEW_FROM,
        "Clara",
        source="policies.md",
        chunk_id=2,
    )
    _add_fact(
        graph,
        "Production Deployment",
        MUST_BE_RECORDED_IN,
        "Deployment Log",
        source="policies.md",
        chunk_id=2,
    )

    return graph


def relationship_types(graph: nx.DiGraph) -> set[str]:
    """Return the explicit relationship vocabulary used by a graph."""

    return {
        attributes[RELATIONSHIP]
        for _, _, attributes in graph.edges(data=True)
    }


def fact_attributes(graph: nx.DiGraph, subject: str, object_: str) -> dict[str, Any]:
    """Return a copy of one fact's relationship and provenance attributes."""

    return dict(graph.edges[subject, object_])
