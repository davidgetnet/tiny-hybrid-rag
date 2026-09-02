"""Turn retrieved evidence into a grounded answer without doing retrieval itself."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os
from typing import Any

from hybrid_retriever import GraphEvidence, HybridEvidence


LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5-mini")
PROMPT_VERSION = "llm-synthesis-v1"
INSUFFICIENT_EVIDENCE_MESSAGE = (
    "The available evidence is insufficient to answer the question."
)


@dataclass(frozen=True)
class SynthesisResult:
    answer: str
    model: str
    prompt_version: str
    prompt: str
    evidence_references: tuple[str, ...]
    trace: tuple[str, ...]
    input_tokens: int | None = None
    output_tokens: int | None = None
    api_called: bool = False


ModelCaller = Callable[[str, str], tuple[str, int | None, int | None]]


def format_vector_evidence(evidence: HybridEvidence) -> str:
    """Render retrieved text and metadata, never embedding coordinates."""

    if not evidence.vector_evidence:
        return "(none)"
    blocks = []
    for item in evidence.vector_evidence:
        blocks.append(f"[{item.id}]\n{item.text}\ndistance: {item.distance:.4f}")
    return "\n\n".join(blocks)


def format_graph_evidence(graph_evidence: GraphEvidence | None) -> str:
    """Keep relationship direction, type, and provenance visible."""

    if graph_evidence is None or not graph_evidence.relationships:
        return "(none)"
    return "\n\n".join(
        f"{edge.subject} --{edge.relationship}--> {edge.object}\n"
        f"source: {edge.source}:{edge.chunk_id}"
        for edge in graph_evidence.relationships
    )


def evidence_references(evidence: HybridEvidence) -> tuple[str, ...]:
    references = {item.id for item in evidence.vector_evidence}
    if evidence.graph_evidence:
        references.update(
            f"{edge.source}:{edge.chunk_id}"
            for edge in evidence.graph_evidence.relationships
        )
    return tuple(sorted(references))


def build_prompt(question: str, evidence: HybridEvidence) -> str:
    """Build the complete, inspectable context supplied to the model."""

    return f"""PROMPT VERSION
{PROMPT_VERSION}

USER QUESTION
{question}

VECTOR EVIDENCE
{format_vector_evidence(evidence)}

GRAPH EVIDENCE
{format_graph_evidence(evidence.graph_evidence)}

INSTRUCTION
Answer the user's question using only the supplied evidence.
If the evidence is insufficient, say that the available evidence is insufficient.
Do not invent Acorn Labs facts or use outside knowledge.
If supplied evidence conflicts, explicitly acknowledge the conflict; do not silently choose one claim.
Distinguish explicit graph facts from supporting document text when useful.
Keep the answer concise.
Cite claims with the supplied [source:chunk] identifiers.
"""


def _openai_call(prompt: str, model: str) -> tuple[str, int | None, int | None]:
    """Call the official SDK lazily so prompt inspection needs no SDK or key."""

    from openai import OpenAI

    response = OpenAI().responses.create(model=model, input=prompt)
    usage: Any = getattr(response, "usage", None)
    return (
        response.output_text.strip(),
        getattr(usage, "input_tokens", None),
        getattr(usage, "output_tokens", None),
    )


def synthesize(
    question: str,
    evidence: HybridEvidence,
    *,
    model: str = LLM_MODEL,
    dry_run: bool = False,
    model_caller: ModelCaller | None = None,
) -> SynthesisResult:
    """Construct context and optionally ask one model to compose the answer."""

    prompt = build_prompt(question, evidence)
    trace = [
        "1. query received",
        f"2. retrieval mode selected = {evidence.mode.value}",
        f"3. vector retrieval completed = {len(evidence.vector_evidence)} record(s)",
        "4. graph retrieval completed = "
        + ("yes" if evidence.graph_evidence else "not used"),
        "5. evidence context constructed",
        f"6. synthesis prompt created = {PROMPT_VERSION}",
    ]
    if dry_run:
        trace.append("7. LLM call skipped (dry-run prompt inspection)")
        return SynthesisResult(
            answer="[DRY RUN] No LLM call was made; inspect the prompt above.",
            model=model,
            prompt_version=PROMPT_VERSION,
            prompt=prompt,
            evidence_references=evidence_references(evidence),
            trace=tuple(trace),
        )

    caller = model_caller or _openai_call
    answer, input_tokens, output_tokens = caller(prompt, model)
    trace.extend((f"7. LLM called = {model}", "8. answer returned"))
    return SynthesisResult(
        answer=answer,
        model=model,
        prompt_version=PROMPT_VERSION,
        prompt=prompt,
        evidence_references=evidence_references(evidence),
        trace=tuple(trace),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        api_called=True,
    )
