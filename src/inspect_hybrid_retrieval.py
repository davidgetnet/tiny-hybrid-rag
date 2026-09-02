"""Inspect vector, graph, and hybrid evidence without generating answers."""

from functools import partial

from embeddings import load_embedding_model
from graph_retriever import find_relationship_targets
from hybrid_retriever import (
    EMERGENCY_LOGGING_QUESTION,
    QUESTION_A,
    QUESTION_B,
    QUESTION_C,
    GraphEvidence,
    HybridEvidence,
    retrieve_hybrid,
)
from inspect_embeddings import preview
from knowledge_graph import MUST_BE_RECORDED_IN
from vector_retriever import retrieve as retrieve_vector


def print_vector_evidence(results) -> None:
    print("VECTOR EVIDENCE")
    if not results:
        print("(none)")
        return
    for rank, result in enumerate(results, start=1):
        print(f"{rank}. {result.id} | cosine distance={result.distance:.4f}")
        print(f"   source={result.source} chunk_id={result.chunk_id}")
        print(f"   {preview(result.text, length=100)}")


def print_graph_evidence(evidence: GraphEvidence | None) -> None:
    print("GRAPH EVIDENCE")
    if evidence is None:
        print("(none)")
        return
    print(f"Traversal order: {' -> '.join(evidence.path)}")
    for edge in evidence.relationships:
        print(f"{edge.subject} --{edge.relationship}--> {edge.object}")
        print(f"  source={edge.source} chunk_id={edge.chunk_id}")


def print_result(result: HybridEvidence) -> None:
    print("=" * 108)
    print(f"QUERY\n{result.query}")
    print(f"\nROUTE / RETRIEVAL MODE\n{result.mode.value}")
    if result.vector_evidence:
        print()
        print_vector_evidence(result.vector_evidence)
    if result.graph_evidence is not None:
        print()
        print_graph_evidence(result.graph_evidence)
    print("\nTRACE")
    for step in result.trace:
        print(step)
    print("\nOUTPUT: retrieved evidence, not a generated answer")


def main() -> None:
    model = load_embedding_model()
    vector_retriever = partial(retrieve_vector, model=model)

    for query in (QUESTION_A, QUESTION_B, QUESTION_C):
        result = retrieve_hybrid(query, vector_retriever=vector_retriever)
        print_result(result)

        if query == QUESTION_B:
            print("\nQUESTION B THROUGH VECTOR SEARCH ALONE")
            print_vector_evidence(vector_retriever(query, 3))
            print(
                "Vector retrieval found text; it did not execute the explicit "
                "Python <-USES- Backend <-MANAGES- Alice path."
            )
        print()

    print("=" * 108)
    print(f"GRAPH-ONLY LIMITATION\n{EMERGENCY_LOGGING_QUESTION}")
    graph_matches = find_relationship_targets(
        "Emergency Deployment", MUST_BE_RECORDED_IN
    )
    print(f"Graph matches: {graph_matches}")
    print(
        "The graph omitted the policy qualifier that the log may be completed "
        "immediately after deployment."
    )
    print("\nVECTOR RETRIEVAL FOR THE SAME QUESTION")
    print_vector_evidence(vector_retriever(EMERGENCY_LOGGING_QUESTION, 3))


if __name__ == "__main__":
    main()
