# Tiny Hybrid RAG

Tiny Hybrid RAG is a deliberately small retrieval-and-synthesis system. It indexes eight paragraph-level chunks from two fictional Acorn Labs documents, combines Chroma text evidence with explicit NetworkX facts, and can ask one LLM to compose a grounded answer from that evidence.

The current implementation keeps the corpus small enough to compare database retrieval directly with brute-force cosine ranking. This makes retrieval quality, persistence, provenance, and failure modes observable without hiding them behind a framework.

## Architecture

```text
                         user query
                             ↓
                    deterministic router
                    /        |        \
              vector       graph     hybrid
                 ↓            ↓       /   \
              Chroma       NetworkX  /     \
                 ↓            ↓     ↓       ↓
            ranked text     paths   text + structured relationships
                    \        |        /
                     retrieval evidence
```

The vector pipeline uses deterministic paragraph chunks and 384-dimensional `sentence-transformers/all-MiniLM-L6-v2` embeddings. The graph pipeline uses manually encoded Acorn Labs entities and directed relationships. The hybrid layer orchestrates them without collapsing their distinct evidence into one score.

No lexical retriever, reranker, LangChain, or LangGraph orchestration is implemented. Retrieval stays deterministic; the optional LLM receives only the constructed evidence context.

## Implemented components

- Markdown loading with source provenance
- Deterministic paragraph chunking and per-document chunk IDs
- SentenceTransformer embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Manual cosine-similarity ranking for baseline comparison
- Persistent local Chroma storage with an explicitly configured cosine index
- Stable record IDs such as `handbook.md:1`
- Metadata-aware top-k retrieval and source filtering
- Repeatable indexing through upsert semantics
- Structural tests covering chunks, embeddings, persistence, IDs, metadata, distances, filtering, and idempotent indexing
- Reproducible Docker execution that creates the index before inspecting retrieval
- A manually encoded NetworkX knowledge graph with typed entities, named directed relationships, and source-chunk provenance
- Deterministic graph traversals for explicit one-hop and multi-hop questions
- Explicit vector, graph, and hybrid retrieval modes that preserve provenance
- Inspectable grounded synthesis prompt with dry-run and optional OpenAI API modes
- Deterministic vector, graph, and hybrid routing with structured evidence, provenance, stable-ID deduplication, and a trace-like execution record
- Explicit LangGraph state, nodes, edges, and conditional retrieval branches

## Engineering decisions

### Application-owned embeddings

Both documents and queries are embedded by the same application code and model. Chroma receives precomputed 384-dimensional vectors and is not allowed to select an implicit embedding function. This keeps stored and query vectors in one known semantic space.

### Stable identity and provenance

Each Chroma record combines its vector with the original text and metadata:

```text
ID: handbook.md:1
source: handbook.md
chunk_id: 1
embedding_model: sentence-transformers/all-MiniLM-L6-v2
```

Stable IDs make repeated indexing idempotent for the current frozen chunking strategy. Source and chunk metadata allow retrieved evidence to be traced back to the input documents.

### Local persistence without generated state in Git

Chroma persists under `data/chroma/`, but that generated database is excluded from Git and Docker build context. Indexing recreates the collection from version-controlled source documents and declared embedding code.

## Observed retrieval behavior

The database and brute-force cosine implementations produced identical top-three ordering for all three inspection queries. This confirms that Chroma searches the supplied representation rather than improving its semantic quality.

| Query | Chroma winner | Cosine distance | Relevant observation |
|---|---|---:|---|
| Backend technology | `handbook.md:2` | 0.4411 | The Frontend/TypeScript chunk outranks Backend/Python because its phrasing more closely matches the query. |
| Server-side programming language | `handbook.md:0` | 0.5275 | A sparse heading containing “Acorn Labs” outranks the Backend/Python paragraph. |
| Security-sensitive deployment review | `policies.md:2` | 0.3201 | The directly relevant Clara review paragraph ranks first. |

These results are intentionally retained. They demonstrate that persistent vector infrastructure does not correct weak chunk boundaries or embedding-model relevance errors.

## Repository layout

```text
.
├── data/
│   ├── handbook.md
│   └── policies.md
├── src/
│   ├── load_documents.py
│   ├── chunk_documents.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── vector_retriever.py
│   ├── index_chunks.py
│   ├── inspect_vector_retrieval.py
│   ├── knowledge_graph.py
│   ├── graph_retriever.py
│   └── inspect_knowledge_graph.py
├── tests/
├── Dockerfile
└── requirements.txt
```

## Run locally

Python 3.12 is the current development target.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/index_chunks.py
python src/inspect_vector_retrieval.py
python src/inspect_llm_synthesis.py        # prompt inspection, no API call
python src/inspect_llm_synthesis.py --live # requires OPENAI_API_KEY
python src/inspect_langgraph_workflow.py   # workflow paths and final state
```

Indexing is safely repeatable for the current corpus: the same logical chunks receive the same IDs and are upserted rather than duplicated.

Run all tests:

```bash
python -m unittest discover -s tests -v
```

Inspect the complete knowledge graph and its deterministic traversals:

```bash
python src/inspect_knowledge_graph.py
```

## Run with Docker

```bash
docker build -t tiny-hybrid-rag .
docker run --rm tiny-hybrid-rag
```

The container installs the declared dependencies, creates a fresh Chroma index from the version-controlled sample documents, and prints retrieval inspection results. No host database is required.

## Optional multi-container development environment

The infrastructure learning setup keeps the Phase 4 runtime image but moves Chroma behind HTTP and adds a browser UI:

```bash
docker compose up -d chroma
docker compose up -d --build
docker compose logs app
```

The app uses `chroma:8000` on the Compose network. Chroma is available to the host at `http://localhost:8000`, and its development UI is available at `http://localhost:8090`. In the UI, connect to `http://localhost:8000` with the default tenant and database. Server data persists separately in `data/chroma-server/`; the earlier embedded database at `data/chroma/` is not migrated or modified.

See `learning/infra/01-docker-compose-chroma-service.md` for the infrastructure walkthrough. This is Infrastructure Increment 01, not another RAG phase.

## CI checks

The GitHub Actions workflow separates three questions into independent jobs:

```text
unit-tests         → does known Python behavior still hold?
integration-tests  → can our client use Chroma through HTTP?
docker-build       → can the runtime image still be constructed?
```

Run the equivalent commands locally:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests/test_chunking.py tests/test_embeddings.py tests/test_vector_store.py tests/unit

docker compose up -d chroma
CHROMA_HOST=localhost CHROMA_PORT=8000 python -m pytest tests/integration
docker compose down

docker build -t tiny-rag:ci .
```

See `learning/infra/02-ci-testing-foundation.md` for the testing and CI progression. This is a CI/testing infrastructure increment, not another RAG phase.

## Local Git precheck

Install the repository-local alias once per clone, then inspect each intended commit before pushing:

```bash
scripts/setup-git-alias.sh
git add .
git precheck
```

The precheck reports branch context, staged change categories, CI-alignment hints, and merged local branch candidates. It is a fast warning system, not a replacement for tests or GitHub Actions. See `learning/infra/03-git-precheck-and-branch-hygiene.md` for the complete workflow.

## Current limitations

- The corpus contains only eight paragraph chunks.
- Chunk synchronization is intentionally simple; a changed chunking strategy should rebuild or explicitly reconcile obsolete IDs.
- The small general-purpose embedding model produces known relevance failures for two inspection queries.
- The educational query router recognizes only the documented scenarios.
- The knowledge graph is manually encoded in memory and contains only selected facts from the sample documents.
- Live answer generation requires `OPENAI_API_KEY`; dry-run inspection does not.
- There is no production service API.

## Project status

- [x] Plain documents
- [x] Loading
- [x] Chunking
- [x] Embeddings
- [x] Vector retrieval
- [x] Knowledge graph
- [x] Graph retrieval
- [x] Hybrid retrieval
- [x] LLM synthesis
- [x] LangGraph orchestration
- [ ] Observability

## Direction

The next learning increment may make execution observable across the explicit workflow. No observability framework is included yet.
