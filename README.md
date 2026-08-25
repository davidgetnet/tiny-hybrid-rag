# Tiny Hybrid RAG

Tiny Hybrid RAG is a deliberately small retrieval system for examining vector search behavior before adding graph retrieval and LLM synthesis. It indexes eight paragraph-level chunks from two fictional Acorn Labs documents, persists their embeddings in Chroma, and exposes retrieval results with stable identity and source metadata.

The current implementation keeps the corpus small enough to compare database retrieval directly with brute-force cosine ranking. This makes retrieval quality, persistence, provenance, and failure modes observable without hiding them behind a framework.

## Architecture

```text
Markdown documents
        ↓
deterministic paragraph chunking
        ↓
sentence-transformers/all-MiniLM-L6-v2
        ↓
384-dimensional float32 embeddings
        ↓
local persistent Chroma collection (cosine distance)
        ↓
top-k records with text, stable ID, source, and chunk metadata
```

No LLM, knowledge graph, lexical retriever, reranker, LangChain, or LangGraph orchestration is currently implemented. The system retrieves evidence; it does not synthesize answers.

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
│   └── inspect_vector_retrieval.py
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
```

Indexing is safely repeatable for the current corpus: the same logical chunks receive the same IDs and are upserted rather than duplicated.

Run all tests:

```bash
python -m unittest discover -s tests -v
```

## Run with Docker

```bash
docker build -t tiny-hybrid-rag .
docker run --rm tiny-hybrid-rag
```

The container installs the declared dependencies, creates a fresh Chroma index from the version-controlled sample documents, and prints retrieval inspection results. No host database is required.

## Current limitations

- The corpus contains only eight paragraph chunks.
- Chunk synchronization is intentionally simple; a changed chunking strategy should rebuild or explicitly reconcile obsolete IDs.
- The small general-purpose embedding model produces known relevance failures for two inspection queries.
- Retrieval currently uses vectors and metadata filters only.
- There is no answer generation, relationship traversal, or production service API.

## Direction

The next retrieval increments will represent explicit domain relationships, add graph retrieval, and then compare or combine graph and vector evidence. Later work may add LLM synthesis, LangGraph orchestration, automated tests in CI, Docker validation, branch protection, image publishing, and deployment. These capabilities are not claimed by the current implementation.
