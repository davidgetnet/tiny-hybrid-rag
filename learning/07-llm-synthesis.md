# 07 — Grounded LLM synthesis

## 1. Where we were

The previous phase ended here:

```text
question
   ↓
hybrid retrieval
   ↓
evidence
```

This phase adds one boundary:

```text
evidence → context builder → LLM → grounded answer
```

The retriever still finds knowledge. The synthesizer reads only that returned
knowledge and composes an answer. It cannot call Chroma, traverse NetworkX,
search files, use tools, or search the web.

## 2. What the LLM adds—and does not add

The LLM adds interpretation and natural-language composition. For a multi-hop
question, it can turn several facts into one concise explanation and attach the
facts' internal references.

It does not make the evidence trustworthy. It does not repair missing facts,
validate stale documents, or magically reconcile conflicts. Acorn Labs facts
must enter through `HybridEvidence`, not through the model's pretrained memory.

## 3. Our first synthesis prompt

`src/llm_synthesizer.py` builds the entire prompt visibly. Its shape is:

```text
PROMPT VERSION
llm-synthesis-v1

USER QUESTION
...

VECTOR EVIDENCE
...

GRAPH EVIDENCE
...

INSTRUCTION
Answer the user's question using only the supplied evidence.
If the evidence is insufficient, say so.
Do not invent Acorn Labs facts or use outside knowledge.
Explicitly acknowledge conflicting evidence.
Cite claims with [source:chunk] identifiers.
```

No API key or environment dump is interpolated into this string.

## 4. Context construction

Context construction is distinct from retrieval. Retrieval returns typed
records. The context builder selects their human-readable fields and lays them
out for the model:

```text
instructions + question + vector text + graph relationships
```

Vector evidence retains chunk text, stable ID, and distance. Graph evidence
retains subject, relationship type, object, and provenance. The two forms remain
visibly separate so we can inspect exactly what knowledge enters generation.

## 5. Question A: misleading ranking remains visible

Question:

> What technology does the Backend team primarily use?

Actual Chroma evidence from the 2026-09-02 dry run:

```text
[handbook.md:2] Frontend primarily uses TypeScript ... distance: 0.4411
[handbook.md:1] Backend primarily develops using Python ... distance: 0.5194
[handbook.md:3] Clara is Security Lead ... distance: 0.6094
```

The top result is wrong for the question, while the correct answer is in the
second result. We deliberately did not repair ranking. A capable synthesizer
has enough evidence to answer “Python [handbook.md:1],” but a live call was not
executed because `OPENAI_API_KEY` was absent. The actual answer field was:

```text
[DRY RUN] No LLM call was made; inspect the prompt above.
```

Therefore this run cannot claim that a model recovered from the bad ranking.
It proves only that the context makes both the ranking failure and the correct
support visible.

## 6. Question B: structurally strong graph evidence

Question:

> Who manages the team that uses Python?

Actual evidence:

```text
Backend --USES--> Python
source: handbook.md:1

Alice --MANAGES--> Backend
source: handbook.md:1
```

The deterministic traversal already resolved the join. Generation needs only
to verbalize it as “Alice [handbook.md:1].” No vector evidence is present and no
live answer was generated in this environment.

## 7. Question C: the centerpiece

Question:

> Who can approve a production deployment for the team that uses Python, and
> what additional requirement applies if the deployment is security-sensitive?

Actual vector evidence:

```text
[policies.md:2] Security-sensitive deployments require an additional review
from Clara ... distance: 0.4459

[policies.md:1] Alice can approve production deployments for Backend services
... distance: 0.5028

[handbook.md:3] Clara is the Security Lead ... distance: 0.5609
```

Actual graph evidence:

```text
Backend --USES--> Python                         [handbook.md:1]
Alice --CAN_APPROVE--> Backend Production Deployment [policies.md:1]
Security-Sensitive Deployment --REQUIRES_REVIEW_FROM--> Clara [policies.md:2]
```

Before the LLM, these are structured facts and supporting passages. A grounded
answer should compose them into: Alice can approve the Backend deployment
`[policies.md:1]`; security-sensitive deployments additionally require Clara's
review `[policies.md:2]`. That sentence is an expected interpretation, not an
observed live output. The observed dry-run output explicitly says no call was
made.

The LLM would not discover Alice or Clara. Both were already present before the
call. Its contribution is turning the retrieved evidence into one readable
answer.

## 8. Unsupported question

Question:

> What cloud provider does Acorn Labs use?

Actual retrieval returned two document headings and the Backend/Python chunk;
none names a cloud provider. The prompt therefore contains irrelevant evidence
but no answer. Desired grounded behavior is to say that the available evidence
does not specify a provider. Guessing AWS would be unsupported generation.
Again, no live behavioral claim is made: the environment had no API key.

## 9. Conflicting evidence

The inspection script constructs test-only evidence without changing the real
documents:

```text
[test-a.md:1] Backend uses Python.
[test-b.md:1] Backend uses Go.
```

The prompt says to acknowledge the conflict rather than silently choose a
claim. Synthesis cannot establish which source is correct; source authority,
freshness, and conflict policy are system-design concerns. Dry-run inspection
verifies that both claims and the conflict instruction reach the model.

## 10. Grounding

Grounding means constraining an answer to available evidence. It reduces
hallucination risk but does not guarantee correctness. Failures remain possible
when retrieval is wrong, documents are stale, graph facts are incorrect, the
prompt is weak, the model misreads context, or sources conflict.

## 11. Retrieval failure versus hallucination

- If retrieval omits the Python chunk and returns only TypeScript, the primary
  problem is retrieval failure.
- If no cloud-provider fact is supplied and generation says AWS, that is
  hallucination or unsupported generation.
- If Python and Go are both supplied and generation silently selects one, that
  is evidence-handling failure.

Not every incorrect final answer has the same cause, so debugging must inspect
evidence before inspecting prose.

## 12. Embeddings never entered the prompt

An embedding such as `[0.027, -0.091, ...]` is useful to retrieval
infrastructure for numerical similarity. It is not normally the evidence a
language model needs. Our path is:

```text
query
  ↓
query embedding
  ↓
Chroma similarity search
  ↓
record IDs
  ↓
original chunk TEXT
  ↓
LLM
```

The test suite asserts that raw coordinates, the 384 dimension count, and the
embedding-model metadata do not appear in the synthesis prompt.

## 13. Evidence versus answer

```text
RETRIEVAL                         GENERATION
input: question                   input: question + evidence
output: evidence                  output: answer
deterministic project logic       probabilistic model behavior
```

For Question C, “before LLM” is the vector passages plus graph edges. “After
LLM” would be a coherent cited sentence. The dry run intentionally stops at the
boundary and prints the complete prompt.

## 14. What RAG means now

The document path is now a minimal Retrieval-Augmented Generation pipeline:

```text
Retrieval → Augmented context → Generation
```

Graph relations add a second, structured retrieval source, so this is hybrid
retrieval feeding generation. It is not a claim that this tiny project embodies
every production GraphRAG architecture.

## 15. Provenance

References such as `[policies.md:1]` let a reader trace an answer claim back to
retrieved evidence. Graph edges carry the same source/chunk metadata. This is
useful now for inspection and later for richer tracing, but no observability
framework was added.

## 16. Prompt and model versioning

Identical question and evidence can produce a different answer after a prompt
change. A model change can also alter interpretation, phrasing, and conflict
handling. This phase records `prompt_version=llm-synthesis-v1` and the configured
model (`gpt-5-mini`) in each synthesis result. A production system might later
record prompt, model, embedding, and graph versions together.

## 17. Failure boundaries

| Layer | Example failure |
|---|---|
| Documents | Fact missing or stale |
| Chunking | Related fact split poorly |
| Embedding | Weak semantic representation |
| Vector retrieval | Wrong chunks ranked |
| Knowledge graph | Missing or wrong edge |
| Hybrid retrieval | Wrong routing or evidence assembly |
| LLM synthesis | Hallucination or misinterpretation |

Question A demonstrates a vector-ranking problem. The cloud question is designed
to reveal unsupported generation. The conflict case tests evidence handling.

## 18. Architecture now

```text
                           QUESTION
                              ↓
                       Hybrid Retriever
                        /            \
                       ↓              ↓
                  Vector            Graph
                 Retrieval         Retrieval
                       ↓              ↓
                  text evidence   relation evidence
                        \            /
                         ↓          ↓
                       Context Builder
                              ↓
                             LLM
                              ↓
                       Grounded Answer
```

## 19. Why LangGraph comes next—not now

Ordinary Python functions currently coordinate routing, retrieval, context
construction, the model call, and the returned result. This is already a
workflow. A later phase can represent execution and state transitions
explicitly. This phase intentionally adds no LangGraph, agents, loops, approval
nodes, or persistent workflow state.

## 20. Reproduce the experiment

```bash
python src/index_chunks.py
python src/inspect_llm_synthesis.py
OPENAI_API_KEY=... python src/inspect_llm_synthesis.py --live
```

The first inspection command makes no network model call. The last command is
optional and is the only one that generates live model answers.
