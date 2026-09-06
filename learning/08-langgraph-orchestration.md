# 08 — LangGraph orchestration

## 1. Where we were

Before this phase, ordinary Python calls and conditionals controlled the journey:

```python
mode = route(query)
if mode == "vector":
    evidence = retrieve_vector(query)
elif mode == "graph":
    evidence = retrieve_graph(query)
else:
    evidence = retrieve_hybrid(query)
prompt = build_context(query, evidence)
answer = synthesize(query, evidence)
```

The components worked, but execution order existed implicitly in the function.

## 2. The new problem

This phase does not improve retrieval or generation. It makes workflow control
explicit: what data is carried, which computation runs, and where execution can
go next. LangGraph sits above the existing Chroma, NetworkX, hybrid retrieval,
and synthesis modules.

## 3. What LangGraph adds

Our implementation uses LangGraph 1.2.11 and its `StateGraph`, `START`, `END`,
nodes, ordinary edges, and conditional edges. It adds a declared execution
model—not new Acorn Labs knowledge.

```text
START → initialize → route_query
                       ├─ vector → retrieve_vector ─┐
                       ├─ graph  → retrieve_graph  ─┤
                       ├─ hybrid → retrieve_hybrid ─┤
                       └─ unsupported → unsupported_route
                                                   ↓
                                build_context → synthesize → END
```

The builder is the workflow definition. `builder.compile()` produces an
executable graph. It is compiled without a checkpointer.

## 4. Our first state

The actual `WorkflowState` schema is:

```text
query: str
retrieval_mode: vector | graph | hybrid | unsupported | None
vector_evidence: tuple[VectorSearchResult, ...]
graph_evidence: GraphEvidence | None
synthesis_prompt: str | None
answer: str | None
dry_run: bool
api_called: bool
prompt_version: str
model: str
trace: list[str] (append reducer)
```

`query` is the input. Routing writes `retrieval_mode`. Retrieval nodes fill one
or both evidence fields. Context construction fills `synthesis_prompt`.
Synthesis fills `answer` and `api_called`. Configuration fields keep the model
boundary inspectable. The trace records node transitions.

Each node reads shared state, performs one job, and returns only its updates.
LangGraph combines those updates into the state carried forward.

## 5. State is not the knowledge graph

We now have two graphs with different meanings:

```text
Knowledge Graph = what the domain data says
LangGraph       = what the program does
```

The NetworkX knowledge graph contains persistent in-memory domain relationships
such as `Backend --USES--> Python`. LangGraph state contains runtime data for
one invocation. `state["graph_evidence"]` may temporarily carry that fact, but
the state is not the source knowledge graph.

## 6. Our nodes

- `initialize` creates clean fields; it does no retrieval.
- `route_query` reuses the deterministic educational router.
- `retrieve_vector` calls the existing vector retriever only.
- `retrieve_graph` reuses existing graph retrieval only.
- `retrieve_hybrid` reuses existing hybrid retrieval and preserves both forms.
- `unsupported_route` records that no deterministic rule matched.
- `build_context` calls the existing prompt builder but not the model.
- `synthesize` calls the existing synthesizer in dry-run or live mode.

A node doing all of these jobs would hide control flow again and defeat much of
the educational value of explicit orchestration.

## 7. Our edges

A node represents work. An edge represents where execution goes afterward.
`initialize → route_query` means routing follows initialization. All retrieval
branches converge on `build_context`; context always precedes synthesis; the
synthesis edge reaches `END`.

`START` and `END` are execution boundaries. They are not Acorn Labs entities.

## 8. Conditional routing

The `route_query` node writes a mode. LangGraph then calls `select_route` for a
conditional edge:

```text
vector      → retrieve_vector
graph       → retrieve_graph
hybrid      → retrieve_hybrid
unsupported → unsupported_route
```

The route therefore exists in topology instead of one giant retrieval node.
Routing remains deterministic; no LLM classification was introduced.

## 9. Question A execution

Actual path on 2026-09-02:

```text
START → initialize → route_query → retrieve_vector
      → build_context → synthesize → END
```

Final state summary:

```text
retrieval_mode: vector
vector evidence count: 3
graph evidence count: 0
prompt constructed: True
answer: [DRY RUN] No LLM call was made; inspect the prompt above.
execution: dry-run
```

The graph did not improve the known TypeScript-before-Python ranking. It merely
made the chosen execution path explicit.

## 10. Question B execution

Actual path:

```text
START → initialize → route_query → retrieve_graph
      → build_context → synthesize → END
```

Final state contains zero vector records and two graph relationships:
`Backend --USES--> Python` and `Alice --MANAGES--> Backend`. The synthesis node
completed in dry-run mode.

## 11. Question C execution

Actual centerpiece path:

```text
question
  ↓
route_query (hybrid)
  ↓
retrieve_hybrid
  ├─ 3 vector records
  └─ 3 graph relationships (Backend, Alice, Clara)
  ↓
build_context
  ↓
synthesize (dry-run)
  ↓
END
```

Actual transition trace:

```text
START → initialize → route_query → retrieve_hybrid
      → build_context → synthesize → END
```

The prompt was constructed and the final answer was the honest dry-run marker.
No live result was invented because `OPENAI_API_KEY` was absent.

## 12. Unsupported question

For “What cloud provider does Acorn Labs use?” the existing educational router
has no rule. The routing node converts that known condition into a safe explicit
path:

```text
START → initialize → route_query → unsupported_route
      → build_context → synthesize → END
```

Both evidence collections remain empty, so the prompt visibly contains
`(none)`. An invalid internal mode is different: `select_route` raises
`WorkflowInvariantError` instead of silently continuing with corrupt state.

## 13. Node versus edge

`retrieve_graph` is a node because it performs work. The arrow
`retrieve_graph → build_context` is an edge because it declares the next step.
Separating the two lets topology communicate the workflow.

## 14. Knowledge Graph node versus LangGraph node

`Backend` is a knowledge-graph node: a domain entity. `retrieve_graph` is a
LangGraph node: a computation step. They share the word “node,” but nothing else
about their role.

## 15. Knowledge edge versus execution edge

```text
Backend --USES--> Python
```

is a domain claim. By contrast:

```text
retrieve_graph → build_context
```

means context construction executes after graph retrieval. One describes
knowledge; the other describes program control.

## 16. State transitions for Question C

```text
Initial input:
  query and dry_run set

After initialize:
  evidence empty; prompt and answer unset; versions configured

After route_query:
  retrieval_mode = hybrid

After retrieve_hybrid:
  vector_evidence = 3 records
  graph_evidence = 3 relationships

After build_context:
  synthesis_prompt populated

After synthesize:
  answer populated; api_called = False
```

The initializer does not retrieve, the context node does not generate, and the
synthesis node does not retrieve.

## 17. Deterministic and probabilistic nodes

Initialization, rule-based routing, graph traversal, vector search for a fixed
index/model, hybrid assembly, and prompt construction are operationally
deterministic for fixed inputs and dependencies. A live synthesis node is
probabilistic model computation. In this run it took the deterministic dry-run
path.

LangGraph can coordinate both kinds without turning the workflow into an agent.

## 18. Why not just use if/else?

For this tiny acyclic workflow, ordinary Python is shorter and arguably easier.
LangGraph becomes more useful as branches, cycles, interruptions, persistence,
human decisions, retries, multiple tools, or long-running work appear. We use it
now to learn the execution model, not because eight nodes require a framework.

## 19. Why no agent yet?

This is a developer-defined workflow. Every possible route and transition is
declared. An agent would allow a model to participate in action selection. That
is a separate idea and is deliberately absent.

## 20. Why no loop yet?

The graph is acyclic. Loops could later represent retry, review, correction, or
re-retrieval, but they also require termination rules and more failure handling.
They are outside this phase.

## 21. Why no checkpoint yet?

One `invoke` carries state from START to END in memory. Durable checkpointing is
useful for resumable or long-running workflows, but would obscure the first
lesson. No `MemorySaver`, SQLite, or Postgres checkpointer is configured.

## 22. Failure boundaries

| Layer | Example failure |
|---|---|
| Documents | Fact missing |
| Chunking | Poor boundary |
| Embeddings | Semantic mismatch |
| Vector retrieval | Wrong ranking |
| Knowledge graph | Missing edge |
| Hybrid retrieval | Bad evidence combination |
| LLM synthesis | Hallucination |
| LangGraph orchestration | Wrong route, transition, or missing state |

An incorrect Question C answer now requires checking the execution path as well
as the retrieval evidence and generated answer.

## 23. Architecture now

```text
                           LangGraph
                              │
                             START
                              ↓
                         initialize
                              ↓
                           routing
                      /        |        \
                     ↓         ↓         ↓
                  vector     graph     hybrid
                  Chroma    NetworkX    both
                     \         |         /
                      \        ↓        /
                        build context
                              ↓
                    OpenAI SDK / dry-run
                              ↓
                             END
```

## 24. What LangGraph did not replace

It did not replace documents, the embedding model, Chroma, NetworkX, the hybrid
retriever, or the LLM synthesizer. It orchestrates those existing components.

## 25. What comes next

Explicit execution lets a future observability phase ask which nodes ran, which
route was chosen, what evidence entered state, which prompt/model versions were
used, and what answer emerged. This phase records a teaching trace only; it adds
no LangSmith, OpenTelemetry, or other observability framework.
