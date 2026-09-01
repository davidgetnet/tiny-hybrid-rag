"""Deterministic relationship traversals over the Acorn Labs graph."""

from collections.abc import Iterable

import networkx as nx

from knowledge_graph import (
    MANAGES,
    OWNS,
    RELATIONSHIP,
    REQUIRES_REVIEW_FROM,
    USES,
    build_knowledge_graph,
)


DEFAULT_GRAPH = build_knowledge_graph()


def _one_or_none(values: Iterable[str], description: str) -> str | None:
    """Return one result, no result, or reject ambiguous graph knowledge."""

    matches = list(values)
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Expected one {description}, found {matches}")
    return matches[0]


def find_relationship_targets(
    entity: str,
    relationship: str,
    graph: nx.DiGraph = DEFAULT_GRAPH,
) -> list[str]:
    """Follow outgoing edges of one named relationship type."""

    if entity not in graph:
        return []
    return sorted(
        target
        for _, target, attributes in graph.out_edges(entity, data=True)
        if attributes[RELATIONSHIP] == relationship
    )


def find_relationship_sources(
    entity: str,
    relationship: str,
    graph: nx.DiGraph = DEFAULT_GRAPH,
) -> list[str]:
    """Follow incoming edges of one named relationship type backward."""

    if entity not in graph:
        return []
    return sorted(
        source
        for source, _, attributes in graph.in_edges(entity, data=True)
        if attributes[RELATIONSHIP] == relationship
    )


def find_team_using(
    technology: str, graph: nx.DiGraph = DEFAULT_GRAPH
) -> str | None:
    """Find the team at the incoming end of a USES edge."""

    return _one_or_none(
        find_relationship_sources(technology, USES, graph),
        f"team using {technology}",
    )


def find_manager_of_team(
    team: str, graph: nx.DiGraph = DEFAULT_GRAPH
) -> str | None:
    """Find the person at the incoming end of a MANAGES edge."""

    return _one_or_none(
        find_relationship_sources(team, MANAGES, graph),
        f"manager of {team}",
    )


def find_manager_of_team_using(
    technology: str, graph: nx.DiGraph = DEFAULT_GRAPH
) -> str | None:
    """Compose USES and MANAGES traversals without hard-coding an answer."""

    team = find_team_using(technology, graph)
    return None if team is None else find_manager_of_team(team, graph)


def find_service_owned_by_team_managed_by(
    manager: str, graph: nx.DiGraph = DEFAULT_GRAPH
) -> str | None:
    """Traverse manager --MANAGES--> team --OWNS--> service."""

    team = _one_or_none(
        find_relationship_targets(manager, MANAGES, graph),
        f"team managed by {manager}",
    )
    if team is None:
        return None
    return _one_or_none(
        find_relationship_targets(team, OWNS, graph),
        f"service owned by {team}",
    )


def find_security_reviewer(
    deployment_type: str = "Security-Sensitive Deployment",
    graph: nx.DiGraph = DEFAULT_GRAPH,
) -> str | None:
    """Follow the explicit deployment review relationship."""

    return _one_or_none(
        find_relationship_targets(deployment_type, REQUIRES_REVIEW_FROM, graph),
        f"reviewer for {deployment_type}",
    )
