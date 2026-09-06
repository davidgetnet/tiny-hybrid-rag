"""Print LangGraph paths and final state without exposing secrets or embeddings."""

from __future__ import annotations

from hybrid_retriever import QUESTION_A, QUESTION_B, QUESTION_C
from inspect_llm_synthesis import UNSUPPORTED_QUESTION
from langgraph_workflow import initial_state, run_workflow


def inspect(question: str) -> None:
    starting = initial_state(question)
    final = run_workflow(question, dry_run=starting["dry_run"])
    graph_count = (
        len(final["graph_evidence"].relationships)
        if final.get("graph_evidence")
        else 0
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


def main() -> None:
    print("WORKFLOW TOPOLOGY")
    print(
        "START → initialize → route_query → "
        "{retrieve_vector | retrieve_graph | retrieve_hybrid | unsupported_route} "
        "→ build_context → synthesize → END"
    )
    for question in (QUESTION_A, QUESTION_B, QUESTION_C, UNSUPPORTED_QUESTION):
        inspect(question)


if __name__ == "__main__":
    main()
