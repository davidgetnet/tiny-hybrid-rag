# Infrastructure Increment 01: Docker Compose, Persistence, and Service Boundaries

This increment changes where Chroma runs, not what Tiny RAG represents or retrieves. The same eight chunks, MiniLM model, 384-dimensional embeddings, cosine distance, queries, metadata filter, and manual-cosine comparison remain intact.

The teaching question is: what changes when Chroma stops being a library opened by the Python process and becomes an independent service that multiple clients can use?

## Before and after

Phase 4 embedded mode remains the default outside Compose:

```text
Python process
    ↓ PersistentClient(path="data/chroma")
local Chroma files
```

Compose selects server mode through environment variables:

```text
app container                 browser on the Mac
    │                              │
    │ HttpClient                   │ Chroma v2 HTTP API
    │ host=chroma, port=8000       │ http://localhost:8000
    ▼                              ▼
              Chroma server container
                         │
                         ▼
              ./data/chroma-server/
```

The UI is an infrastructure and development inspection tool. It is not a RAG application component and does not answer questions for Tiny RAG.

## The three services

`app` reuses `tiny-rag:phase4`. It bind-mounts `./src` at `/app/src` and `./data` at `/app/data`, then runs the existing indexing and retrieval inspection programs. `CHROMA_HOST=chroma` and `CHROMA_PORT=8000` select `chromadb.HttpClient`.

`chroma` uses the server version matching the Python client, `chromadb/chroma:1.5.9`. Port `8000` is published to the Mac. `./data/chroma-server` is mounted at `/data`, the server persistence path. Its configuration permits browser requests from `http://localhost:8090`.

`chroma-ui` builds a pinned revision of the maintained community ChromaDB UI. It serves a static browser application on host port `8090`. In the connection screen, use `http://localhost:8000`, `default_tenant`, and `default_database`. The browser—not the UI container—calls Chroma, which explains both the host URL and the CORS setting.

## Image, container, bind mount, and persistent directory

An **image** is a stable runtime template. `tiny-rag:phase4` contains Linux, Python 3.12, CPU PyTorch, sentence-transformers, the Chroma client, grpcio, and the cached model/runtime layers.

A **container** is a running or stopped instance of an image with its own writable layer and process state. The app container is intentionally a batch job: it indexes, inspects, prints results, and exits successfully. Chroma and the UI keep running.

A **bind mount** projects a real host path into a container. `/app/src` is not the older source snapshot from the image while Compose runs; the host `src/` hides that location and becomes immediately visible there. The same is true for `/app/data`.

The **persistent Chroma directory** is the host path `data/chroma-server/`. Its SQLite database and HNSW segment files remain after the Chroma container is stopped or deleted. This is separate from `data/chroma/`, the earlier embedded Phase 4 database. No automatic migration occurs between them.

## Why source edits do not require an app rebuild

```text
edit src/*.py on the Mac
        ↓
the /app/src bind mount exposes the edit immediately
        ↓
rerun: docker compose run --rm app
```

The validation probe printed version 1, was edited on the host, and printed version 2 from a new app container without building an image.

Rebuild the app image when its stable runtime changes—for example the Dockerfile, Python version, OS packages, or `requirements.txt`. Do not rebuild it for an ordinary source or sample-data edit in this Compose workflow.

The UI is different: this repository does not own a prebuilt UI image. Compose builds its pinned upstream source revision once, and Docker reuses the cached image afterward.

## Docker networking and `localhost`

Compose creates a private network and DNS entries for service names. From inside `app`, `chroma` resolves to the Chroma container, so the correct endpoint is `chroma:8000`.

Inside a container, `localhost` means that same container. If the app used `localhost:8000`, it would look for Chroma inside the app container and fail.

The Mac sees published ports instead. The browser therefore uses `http://localhost:8000`, and the UI page is served at `http://localhost:8090`.

## Multiple clients

The Python app and the browser UI speak the same Chroma HTTP API. They can inspect the same collection because identity, documents, metadata, embeddings, and index files are owned by the independent Chroma server rather than either client process.

```text
Tiny RAG Python client ─┐
                       ├─ HTTP → one Chroma server → one persisted database
browser UI client ─────┘
```

This is the practical reason to introduce a service boundary: multiple programs can share one database without opening the same embedded files directly.

## Startup ordering

`depends_on` uses `condition: service_healthy`. The Chroma 1.5.9 image does not contain `curl`, so the health check uses its bundled `chroma browse` command to call the server heartbeat indirectly and waits for a valid response. The app and UI start only after Chroma is ready, avoiding a race between container startup and HTTP readiness.

## Persistence lifecycle

- Deleting the app container loses only its process and writable layer. Chroma data is unaffected.
- Deleting the Chroma container loses that server instance, but not `data/chroma-server/` on the Mac.
- `docker compose stop` stops processes and preserves containers and host data.
- `docker compose down` removes Compose containers and its network, but this bind-mounted directory remains.
- Deleting `data/chroma-server/` would delete the new server database. Do not confuse it with `data/chroma/`.

Validation deleted and recreated the Chroma container, then read the collection before re-indexing. The server still reported eight records.

## Run and inspect

Start the database alone first:

```bash
docker compose up -d chroma
docker compose ps
```

Start all services, building only the UI when needed:

```bash
docker compose up -d --build
docker compose logs app
```

The app service has no `build:` section, so this command does not rebuild `tiny-rag:phase4`.

Rerun the application after a source edit:

```bash
docker compose run --rm app
```

Inspect server health, persistence, and logs:

```bash
curl http://localhost:8000/api/v2/heartbeat
find data/chroma-server -maxdepth 2 -type f
docker compose logs chroma
```

Open `http://localhost:8090`, connect to `http://localhost:8000`, and select `acorn_chunks`. The validated UI displayed all eight records with stable IDs, documents, metadata, and vector inspection controls.

Stop the environment without deleting host data:

```bash
docker compose down
```

## Observed parity

The server-backed experiment stored eight records and reproduced the Phase 4 top-three order for every query:

| Query | Top three IDs |
|---|---|
| Backend technology | `handbook.md:2`, `handbook.md:1`, `handbook.md:3` |
| Server-side programming language | `handbook.md:0`, `policies.md:0`, `handbook.md:1` |
| Security-sensitive review | `policies.md:2`, `policies.md:3`, `handbook.md:3` |

Every Chroma ordering matched manual cosine ordering. Metadata filtering still limited results by source, and complete-record inspection still included ID, text, source, chunk ID, embedding model, and cosine distance.

## Infrastructure versus RAG logic

RAG development concerns chunks, representations, relevance, ranking, hybrid retrieval, and answer generation. Infrastructure concerns containers, networking, persistence, service boundaries, reproducible runtimes, and future CI/CD foundations.

This increment changes only the second category. A server boundary can improve sharing and observability, but it does not improve retrieval quality. The known ranking behavior remains intentionally unchanged.
