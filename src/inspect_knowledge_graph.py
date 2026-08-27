"""Print the complete graph and demonstrate deterministic traversals."""

from graph_retriever import (
    find_manager_of_team,
    find_manager_of_team_using,
    find_relationship_targets,
    find_security_reviewer,
    find_service_owned_by_team_managed_by,
    find_team_using,
)
from knowledge_graph import MUST_BE_RECORDED_IN, build_knowledge_graph


QUESTION_B = "Who manages the team that uses Python?"
UNANSWERABLE_QUESTION = (
    "What wording does Acorn Labs use to describe emergency deployment logging?"
)


def main() -> None:
    graph = build_knowledge_graph()

    print("ACORN LABS KNOWLEDGE GRAPH")
    print(f"NUMBER OF NODES: {graph.number_of_nodes()}")
    print(f"NUMBER OF EDGES: {graph.number_of_edges()}")

    print("\nNODES")
    for node, attributes in sorted(graph.nodes(data=True)):
        print(f"- {node} [{attributes['entity_type']}]")

    print("\nRELATIONSHIPS")
    relationships = sorted(
        graph.edges(data=True),
        key=lambda edge: (edge[0], edge[2]["relationship"], edge[1]),
    )
    for source, target, attributes in relationships:
        print(f"{source} --{attributes['relationship']}--> {target}")
        print(f"  source: {attributes['source']}")
        print(f"  chunk_id: {attributes['chunk_id']}")

    technology = "Python"
    team = find_team_using(technology, graph)
    manager = None if team is None else find_manager_of_team(team, graph)
    print(f"\nQUESTION B\n{QUESTION_B}")
    print(f"Start: {technology}")
    print(f"Incoming USES: {team} --USES--> {technology}")
    print(f"Incoming MANAGES: {manager} --MANAGES--> {team}")
    print(f"Result: {find_manager_of_team_using(technology, graph)}")

    service = find_service_owned_by_team_managed_by("Alice", graph)
    print("\nALICE TO OWNED SERVICE")
    print("Alice --MANAGES--> Backend")
    print(f"Backend --OWNS--> {service}")
    print(f"Result: {service}")

    reviewer = find_security_reviewer(graph=graph)
    print("\nSECURITY REVIEW")
    print(
        "Security-Sensitive Deployment "
        f"--REQUIRES_REVIEW_FROM--> {reviewer}"
    )
    print(f"Result: {reviewer}")

    emergency_logging = find_relationship_targets(
        "Emergency Deployment", MUST_BE_RECORDED_IN, graph
    )
    print(f"\nINTENTIONALLY UNANSWERABLE\n{UNANSWERABLE_QUESTION}")
    print(f"Structured graph matches: {emergency_logging}")
    print(
        "Result: no answer. The graph does not encode the emergency-deployment "
        "timing or its original wording."
    )

    print("\nVECTOR SEARCH VERSUS GRAPH SEARCH")
    print("Vector: embed question -> rank semantically nearby stored chunks")
    print("Graph: Python <-USES- Backend <-MANAGES- Alice")
    print("The vector and graph retrieval systems remain independent.")


if __name__ == "__main__":
    main()
