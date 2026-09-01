"""Small, explicit NetworkX operations for the Lesson 5.5 walkthrough."""

from collections.abc import Sequence
from typing import Any

import networkx as nx

from chunk_documents import chunk_documents
from documents import Chunk
from load_documents import load_documents


GraphEdge = tuple[str, str, dict[str, Any]]
TwoHopPath = tuple[GraphEdge, GraphEdge]


def find_node(graph: nx.DiGraph, entity: str) -> tuple[str, dict[str, Any]] | None:
    """Perform a literal NetworkX node-membership lookup."""

    if entity not in graph:
        return None
    return entity, dict(graph.nodes[entity])


def get_one_hop(
    graph: nx.DiGraph,
    entity: str,
    relationship: str | None = None,
) -> list[GraphEdge]:
    """Return outgoing edges reached by following exactly one stored edge."""

    if entity not in graph:
        return []

    # This is the physical NetworkX traversal operation: obtain every outgoing
    # edge from the start node together with its edge-attribute dictionary.
    raw_edges = list(graph.out_edges(entity, data=True))
    matching_edges = [
        (source, target, dict(attributes))
        for source, target, attributes in raw_edges
        if relationship is None or attributes["relationship"] == relationship
    ]
    return sorted(
        matching_edges,
        key=lambda edge: (edge[2]["relationship"], edge[1]),
    )


def get_two_hops(graph: nx.DiGraph, entity: str) -> list[TwoHopPath]:
    """Follow one outgoing edge, then one outgoing edge from each reached node."""

    paths: list[TwoHopPath] = []
    for first_edge in get_one_hop(graph, entity):
        intermediate = first_edge[1]
        for second_edge in get_one_hop(graph, intermediate):
            # Avoid immediately walking back to the start if a reverse edge is
            # introduced later. This is not general cycle detection.
            if second_edge[1] != entity:
                paths.append((first_edge, second_edge))
    return sorted(
        paths,
        key=lambda path: (
            path[0][2]["relationship"],
            path[0][1],
            path[1][2]["relationship"],
            path[1][1],
        ),
    )


def node_source_chunk_ids(graph: nx.DiGraph, entity: str) -> list[str]:
    """Derive node provenance from incoming and outgoing edge provenance."""

    if entity not in graph:
        return []
    incident_edges = list(graph.in_edges(entity, data=True)) + list(
        graph.out_edges(entity, data=True)
    )
    return sorted(
        {
            f"{attributes['source']}:{attributes['chunk_id']}"
            for _, _, attributes in incident_edges
        }
    )


def build_chunk_lookup(chunks: Sequence[Chunk] | None = None) -> dict[str, Chunk]:
    """Map the graph's source/chunk provenance keys back to original chunks."""

    available_chunks = (
        list(chunks) if chunks is not None else chunk_documents(load_documents())
    )
    return {
        f"{chunk.metadata['source']}:{chunk.metadata['chunk_id']}": chunk
        for chunk in available_chunks
    }


def resolve_edge_source(
    edge: GraphEdge,
    chunk_lookup: dict[str, Chunk] | None = None,
) -> Chunk | None:
    """Use an edge's source and chunk_id to recover its source text."""

    attributes = edge[2]
    provenance_id = f"{attributes['source']}:{attributes['chunk_id']}"
    lookup = chunk_lookup or build_chunk_lookup()
    return lookup.get(provenance_id)
