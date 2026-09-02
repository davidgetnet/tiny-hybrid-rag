"""Inspect retrieval, prompt construction, and optional live synthesis."""

from __future__ import annotations

import argparse
import os

from hybrid_retriever import (
    QUESTION_A,
    QUESTION_B,
    QUESTION_C,
    HybridEvidence,
    RetrievalMode,
    retrieve_hybrid,
)
from llm_synthesizer import (
    LLM_MODEL,
    format_graph_evidence,
    format_vector_evidence,
    synthesize,
)
from vector_retriever import retrieve as retrieve_vector
from vector_store import VectorSearchResult


UNSUPPORTED_QUESTION = "What cloud provider does Acorn Labs use?"
CONFLICT_QUESTION = "What technology does the Backend team use?"


def conflicting_test_evidence() -> HybridEvidence:
    def record(record_id: str, text: str) -> VectorSearchResult:
        source, chunk = record_id.rsplit(":", 1)
        return VectorSearchResult(
            id=record_id,
            text=text,
            source=source,
            chunk_id=int(chunk),
            embedding_model="test-only",
            distance=0.1,
        )

    return HybridEvidence(
        query=CONFLICT_QUESTION,
        mode=RetrievalMode.VECTOR,
        vector_evidence=(
            record("test-a.md:1", "Backend uses Python."),
            record("test-b.md:1", "Backend uses Go."),
        ),
        graph_evidence=None,
        trace=("test-only conflicting evidence constructed",),
    )


def retrieve_for_inspection(question: str) -> HybridEvidence:
    if question == CONFLICT_QUESTION:
        return conflicting_test_evidence()
    if question == UNSUPPORTED_QUESTION:
        return HybridEvidence(
            query=question,
            mode=RetrievalMode.VECTOR,
            vector_evidence=tuple(retrieve_vector(question, top_k=3)),
            graph_evidence=None,
            trace=("1. received query", "2. unsupported experiment uses vector retrieval"),
        )
    return retrieve_hybrid(question)


def print_case(question: str, *, live: bool) -> None:
    evidence = retrieve_for_inspection(question)
    result = synthesize(question, evidence, dry_run=not live)
    line = "=" * 80
    print(line, "QUESTION", line, question, sep="\n")
    print("RETRIEVAL MODE", evidence.mode.value, sep="\n")
    print("VECTOR EVIDENCE", format_vector_evidence(evidence), sep="\n")
    print("GRAPH EVIDENCE", format_graph_evidence(evidence.graph_evidence), sep="\n")
    if question == QUESTION_C:
        print("BEFORE LLM", "structured evidence shown above", sep="\n")
    print("FINAL PROMPT / CONTEXT SUMMARY", result.prompt, sep="\n")
    print("LLM ANSWER", result.answer, sep="\n")
    if question == QUESTION_C:
        print("AFTER LLM", result.answer, sep="\n")
    print("EVIDENCE REFERENCES", ", ".join(result.evidence_references) or "(none)", sep="\n")
    print("TRACE", *result.trace, sep="\n")
    print(f"model: {result.model}")
    if result.input_tokens is not None:
        print(f"tokens: input={result.input_tokens}, output={result.output_tokens}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live", action="store_true", help="call the API (requires OPENAI_API_KEY)"
    )
    args = parser.parse_args()
    if args.live and not os.getenv("OPENAI_API_KEY"):
        parser.error("--live requires OPENAI_API_KEY")
    print(f"SYNTHESIS MODEL: {LLM_MODEL}")
    for question in (
        QUESTION_A,
        QUESTION_B,
        QUESTION_C,
        UNSUPPORTED_QUESTION,
        CONFLICT_QUESTION,
    ):
        print_case(question, live=args.live)


if __name__ == "__main__":
    main()
