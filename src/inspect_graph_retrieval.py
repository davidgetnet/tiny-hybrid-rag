"""Make graph storage, traversal, and provenance visible from the terminal."""

from knowledge_graph import build_knowledge_graph
from manual_graph_retrieval import (
    build_chunk_lookup,
    find_node,
    get_one_hop,
    get_two_hops,
    node_source_chunk_ids,
    resolve_edge_source,
)


def print_edge(edge, chunk_lookup, *, indent: str = "") -> None:
    source, target, attributes = edge
    print(f"{indent}{source} --{attributes['relationship']}--> {target}")
    print(
        f"{indent}  provenance: "
        f"{attributes['source']}:{attributes['chunk_id']}"
    )
    chunk = resolve_edge_source(edge, chunk_lookup)
    print(f"{indent}  source text: {chunk.content if chunk else '(not found)'}")


def main() -> None:
    graph = build_knowledge_graph()
    chunk_lookup = build_chunk_lookup()

    print("GRAPH SUMMARY")
    print("-------------")
    print(f"Graph Python type: {type(graph)}")
    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")

    print("\nNODES")
    print("-----")
    for entity in sorted(graph.nodes):
        print(entity)
        print(f"  raw attributes: {dict(graph.nodes[entity])}")
        print(f"  source_chunk_ids: {node_source_chunk_ids(graph, entity)}")
    print(
        "Node source_chunk_ids are derived from incident edge provenance; "
        "Lesson 5 stored entity_type directly on nodes."
    )

    print("\nEDGES")
    print("-----")
    all_edges = sorted(
        graph.edges(data=True),
        key=lambda edge: (edge[0], edge[2]["relationship"], edge[1]),
    )
    for edge in all_edges:
        print(f"RAW EDGE OBJECT: {edge}")
        print_edge(edge, chunk_lookup)

    query = "What technology does the Backend team use?"
    start_entity = "Backend"
    print("\nQUERY")
    print("-----")
    print(query)
    print("\nMANUALLY IDENTIFIED START ENTITY")
    print("--------------------------------")
    print(start_entity)
    print("\nWHY MANUAL?")
    print("-----------")
    print(
        "Lesson 5.5 isolates traversal. NetworkX does not interpret the question; "
        "we supplied the start entity."
    )

    node = find_node(graph, start_entity)
    print("\nNODE LOOKUP")
    print("-----------")
    print(f"Expression: {start_entity!r} in graph")
    print(f"Raw lookup result: {node}")

    print("\nONE-HOP RETRIEVAL")
    print("-----------------")
    raw_neighbors = list(graph.successors(start_entity))
    raw_out_edges = list(graph.out_edges(start_entity, data=True))
    print(f"NetworkX call: list(graph.successors({start_entity!r}))")
    print(f"RAW NEIGHBORS: {raw_neighbors}")
    print(f"NetworkX call: list(graph.out_edges({start_entity!r}, data=True))")
    print(f"RAW OUT EDGES: {raw_out_edges}")
    print("FILTERED RELATIONSHIP: USES")
    for edge in get_one_hop(graph, start_entity, relationship="USES"):
        print_edge(edge, chunk_lookup)

    print("\nTWO-HOP TRACE")
    print("-------------")
    two_hop_start = "Alice"
    print(f"Hop 0: {two_hop_start}")
    paths = get_two_hops(graph, two_hop_start)
    print(f"Raw Python paths: {paths}")
    for path_number, (first_edge, second_edge) in enumerate(paths, start=1):
        print(f"Path {path_number}")
        print("  Hop 1:")
        print_edge(first_edge, chunk_lookup, indent="    ")
        print("  Hop 2:")
        print_edge(second_edge, chunk_lookup, indent="    ")

    failure_query = "What database does the Backend team use?"
    print("\nFAILURE EXPERIMENT")
    print("------------------")
    print(f"Question: {failure_query}")
    print(f"Start entity found: {find_node(graph, 'Backend') is not None}")
    missing = get_one_hop(graph, "Backend", relationship="USES_DATABASE")
    print("Required relationship: USES_DATABASE")
    print(f"Raw matching edges: {missing}")
    print("Graph evidence: none")
    print("The graph can only traverse relationships that were stored.")

    print("\nMECHANICAL COMPARISON")
    print("---------------------")
    print("VECTOR: question -> query embedding -> vector distance -> ranked chunks")
    print(
        "GRAPH: question -> manually identify entity -> node lookup -> "
        "stored edges -> connected nodes -> source chunks"
    )
    print("No embedding or Chroma call was made by this script.")


if __name__ == "__main__":
    main()
