"""Print LangGraph paths and final state without exposing secrets or embeddings."""

from __future__ import annotations

import argparse

from hybrid_retriever import QUESTION_A, QUESTION_B, QUESTION_C
from inspect_llm_synthesis import UNSUPPORTED_QUESTION
from hybrid_retriever import retrieve_graph
from langgraph_workflow import (
    WorkflowDependencies,
    build_workflow,
    initial_state,
    run_workflow,
)


def inspect(question: str) -> None:
    starting = initial_state(question)
    final = run_workflow(question, dry_run=starting["dry_run"])
    graph_count = (
        len(final["graph_evidence"].relationships) if final.get("graph_evidence") else 0
    )
    line = "=" * 80
    print(line, "QUESTION", line, question, sep="\n")
    print("INITIAL STATE")
    print({"query": starting["query"], "dry_run": starting["dry_run"]})
    print("NODE TRANSITIONS")
    print(" → ".join(final["trace"]))
    print("FINAL STATE SUMMARY")
    print(f"retrieval mode: {final['retrieval_mode']}")
    print(f"vector evidence count: {len(final['vector_evidence'])}")
    print(f"graph evidence count: {graph_count}")
    print(f"prompt constructed: {bool(final['synthesis_prompt'])}")
    print(f"answer: {final['answer']}")
    print(f"execution: {'live' if final['api_called'] else 'dry-run'}")
    print(f"prompt version: {final['prompt_version']}")
    print(f"model: {final['model']}")


def inspect_degraded_example() -> None:
    """Use a deterministic fake outage so partial failure is easy to inspect."""

    def simulated_vector_failure(query: str, top_k: int):
        raise ConnectionError("simulated failure")

    workflow = build_workflow(
        WorkflowDependencies(
            vector_retriever=simulated_vector_failure,
            graph_retriever=retrieve_graph,
        )
    )
    final = run_workflow(QUESTION_C, dry_run=True, workflow=workflow)
    print("=" * 80, "DEGRADED EXAMPLE", sep="\n")
    print("QUESTION")
    print(final["original_question"])
    print(f"VECTOR RETRIEVAL: {final['vector_retrieval_status'].upper()}")
    print(f"Reason: {final['vector_retrieval_error']}")
    print(f"GRAPH RETRIEVAL: {final['graph_retrieval_status'].upper()}")
    print(f"LEXICAL RETRIEVAL: {final['lexical_retrieval_status'].upper()}")
    print(f"OVERALL RETRIEVAL: {final['overall_retrieval_status'].upper()}")
    print(f"SYNTHESIS: {final['synthesis_status'].upper()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--degraded",
        action="store_true",
        help="run only the deterministic degraded-retrieval example",
    )
    args = parser.parse_args()
    if args.degraded:
        inspect_degraded_example()
        return

    print("WORKFLOW TOPOLOGY")
    print(
        "START → initialize → route_query → "
        "{retrieve_vector | retrieve_graph | retrieve_hybrid | unsupported_route} "
        "→ build_context → synthesize → END"
    )
    for question in (QUESTION_A, QUESTION_B, QUESTION_C, UNSUPPORTED_QUESTION):
        inspect(question)
    inspect_degraded_example()


if __name__ == "__main__":
    main()
