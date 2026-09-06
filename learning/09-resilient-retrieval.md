# 09 — Resilient retrieval

## Failure boundaries

A failure boundary is the place where one component's exception is caught and
translated into explicit workflow state. The workflow now gives vector and
graph retrieval separate boundaries. A vector exception therefore records
`vector_failed = True`, `vector_retrieval_status = failed`, and its message without erasing graph
evidence already available (or preventing graph retrieval from being tried).

## Graceful degradation

Graceful degradation means doing a smaller but still valid job when an optional
part fails. For a hybrid query, both retrievers succeeding produces `success`.
If either vector or graph fails while the other produces usable evidence, the
overall status is `degraded` and synthesis uses only that evidence. A component
failure does not have to become a system failure when another component can
still support the critical operation.

There is no lexical retriever in this project yet, so state exposes lexical
status as `not_configured`; no imaginary fallback was added. If a lexical path
is introduced later, it can use the same status/error boundary and contribute
evidence in the same way.

## Failure without invention

Evidence is the gate to synthesis. If every requested path fails or returns no
usable evidence, overall retrieval becomes `failed`, the graph follows the
`retrieval_failure` node, and execution ends with no prompt or answer. The LLM
is never asked to fill the gap.

## Fallback is not retry

A fallback uses a different available capability—for example, graph evidence
after vector retrieval fails. A retry calls the failed capability again,
usually with timing, limits, and backoff. This phase adds fallback behavior but
no retry loop, queue, or database persistence. Those mechanisms need separate
operational decisions and would obscure the small lesson about failure
boundaries and state transitions.

## State makes partial failure visible

LangGraph state retains the original question, combined retrieved evidence,
per-component status and error text, overall status, structured errors, and the
last error. That makes `success`, `degraded`, and `failed` executions visibly
different while keeping retrieval, context building, and synthesis as separate
nodes.

The same pattern appears in an offline point-of-sale system. Receipt analytics
or loyalty lookup may be unavailable, but a locally authorized sale can still
complete. The optional subsystem's failure must remain observable; it simply
should not destroy the critical workflow. If authorization and every valid
offline fallback are unavailable, the sale stops explicitly instead of
pretending it succeeded.
