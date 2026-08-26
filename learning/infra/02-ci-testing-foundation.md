# Infrastructure Increment 02: CI and Testing Foundations

This increment does not change Tiny RAG's chunks, embeddings, retrieval ranking, or learning-phase numbering. It adds repeatable evidence around behavior that already matters.

## 1. The question that caused this increment

The concern was reasonable: GitHub Actions can be green while the system still contains errors, so how does CI reduce developer responsibility?

CI does not prove the entire product is correct. It automates known checks:

```text
developer discovers important behavior
          ↓
expresses it as a test or build check
          ↓
CI executes that check on every relevant change
```

The developer still decides which expectations matter. Automation changes those expectations from something a person must remember into executable repository knowledge.

For this repository, we now ask three separate questions:

```text
CI
├── unit-tests
├── integration-tests
└── docker-build
```

## 2. Tests versus GitHub Actions

The Python files under `tests/` define what must be true. For example, `tests/unit/test_vector_store_contracts.py` asserts how stable IDs are constructed and that invalid query sizes fail before Chroma is called. `tests/integration/test_chroma_http.py` asserts that a known vector can travel through the real HTTP boundary and return the expected record.

`.github/workflows/ci.yml` does not contain those expectations. It defines triggers, clean Ubuntu machines, Python 3.12, dependency installation, Chroma startup, and commands to execute.

GitHub Actions supplies the clean runner and automation:

```text
tests/*.py
    ↓ define assertions and expected behavior
.github/workflows/ci.yml
    ↓ defines when and where commands run
GitHub Actions
    ↓ supplies a fresh runner and executes them
```

If a ranking assertion belongs anywhere, it belongs in Python test code—not as shell logic embedded in YAML. This keeps the same test runnable on a laptop, in an IDE, or in CI.

## 3. Unit test

The new unit module checks two small vector-store contracts.

Stable identity:

```text
stable_id function
      ↓
controlled metadata {source: handbook.md, chunk_id: 2}
      ↓
expected output handbook.md:2
      ↓
assert equality
```

Query validation:

```text
query_by_embedding
      ↓
mock collection + top_k=0
      ↓
expected ValueError
      ↓
assert collection.query was never called
```

The second assertion is valuable because it protects both behavior and ordering: our code rejects invalid input before crossing the database boundary.

Docker and a Chroma server are deliberately absent. A mock controls the collaborator, the vectors are fixed, and the tests finish quickly. Failure points directly at Python behavior rather than networking or service startup.

The existing suite is not uniformly unit-level. `test_chunking.py` is fast and deterministic. `test_embeddings.py` loads the real MiniLM model, while most of `test_vector_store.py` combines the model with embedded Chroma. Those are useful component checks, so this increment preserves them in the non-server job rather than reorganizing good tests merely for a perfect directory taxonomy.

## 4. Integration test

The integration test deliberately crosses the service boundary configured in the previous infrastructure increment:

```text
test code
   ↓ open_client()
Chroma Python client
   ↓ HTTP
Chroma 1.5.9 server
   ↓
unique temporary collection
   ↓ add three known 3D vectors
query [1, 0, 0]
   ↓
assert record-a is first
```

The records are intentionally simple:

```text
record-a → [1, 0, 0]
record-b → [0, 1, 0]
record-c → [0, 0, 1]
```

There is no MiniLM call or model download. The test is about service interaction, not embedding quality. It verifies heartbeat access, collection creation, insertion, record count, cosine querying, result conversion through `query_by_embedding`, and collection deletion.

Each run uses a UUID-based collection name and deletes the collection in `finally`, so reruns do not inherit another run's records.

This boundary can catch failures a unit test cannot:

- wrong host or port;
- an unreachable Chroma server;
- incompatible client/server behavior;
- invalid collection-creation assumptions;
- HTTP insert or query failures;
- incorrect assumptions about Chroma's query response.

It is not trying to prove Chroma itself is correct. It proves our installed client, connection abstraction, and query wrapper work with the server version we operate.

## 5. Why CI can still be green while a bug exists

CI only knows what we encoded.

```text
no test for behavior X
        ↓
behavior X breaks
        ↓
all encoded checks still pass
        ↓
CI remains green
```

Green means “the checks we chose passed,” not “no bug exists anywhere.” This limitation creates a feedback loop rather than making tests useless:

```text
bug discovered
    ↓
reproduce it with a test
    ↓
test fails
    ↓
fix the code
    ↓
test passes
    ↓
CI protects that behavior from returning unnoticed
```

A **regression test** is the test retained after a bug is fixed. It converts one discovered failure into a permanent automated expectation.

## 6. How tests grow with development

Tiny RAG's checks follow its real evolution:

```text
chunking feature          → chunk boundaries and ID tests
embeddings                → vector shape and structure tests
vector retrieval          → identity, metadata, distance, and filter tests
Chroma becomes a service  → HTTP integration test
Docker runtime            → clean Docker image build
```

Not every line needs a test. Tests are most useful around observable behavior, contracts between components, edge cases, bug regressions, and failure modes that would be expensive to rediscover.

The eight Acorn chunks, MiniLM model, 384-dimensional representations, queries, and Phase 4 rankings remain unchanged by this increment.

## 7. Separate CI jobs

One giant job answers only “something failed.” Three jobs localize the category:

```text
unit-tests         ✅
integration-tests  ❌
docker-build       ✅
```

That result suggests Python expectations and image construction are healthy while the service boundary needs investigation. A failed `docker-build` with passing tests instead points toward the Dockerfile, dependency installation, or build context.

The jobs are independent, so GitHub can run them in parallel and show their results separately. They do not need to wait for one another because none produces an artifact required by another.

## 8. Fresh environment value

Local success on a Mac does not prove dependencies install or commands behave on a clean Linux host:

```text
works on Mac
    ≠
installs and runs on fresh Ubuntu
```

Each GitHub-hosted job starts from a fresh Ubuntu runner, checks out only repository content, installs declared dependencies, and executes its one category of validation. This catches hidden reliance on local caches, global packages, IDE configuration, stale database files, or operating-system differences.

This connects directly to the earlier Docker dependency issue: a local environment may already contain a package or model that a clean environment does not. Reconstructing the environment is itself a useful test.

## 9. Developer responsibility after CI

Before automation, responsibility tends to look like:

```text
remember every known check
run every known check manually
remember which services must be running
```

After automation, responsibility moves toward:

```text
design meaningful checks
maintain tests when intended behavior changes
review what CI does and does not cover
investigate failures instead of bypassing them
```

CI reduces repetitive memory work. It does not replace engineering judgment.

## 10. Local and CI reproducibility

Developer and CI commands are intentionally the same or direct equivalents.

Install runtime plus test tooling:

```bash
python -m pip install -r requirements-dev.txt
```

`requirements.txt` declares application runtime libraries. `requirements-dev.txt` includes it and adds pytest, which is developer/test tooling rather than an application runtime requirement.

Run all existing non-server checks plus the focused unit directory:

```bash
python -m pytest \
  tests/test_chunking.py \
  tests/test_embeddings.py \
  tests/test_vector_store.py \
  tests/unit
```

Run the HTTP integration test:

```bash
docker compose up -d chroma

CHROMA_HOST=localhost \
CHROMA_PORT=8000 \
python -m pytest tests/integration

docker compose down
```

CI polls `http://localhost:8000/api/v2/heartbeat` instead of assuming a fixed startup time. If readiness never arrives, it prints Chroma logs and fails clearly.

Validate image construction:

```bash
docker build -t tiny-rag:ci .
```

Observed locally for this increment:

```text
non-server job:       20 passed
HTTP integration:      1 passed
Docker image build:    successful
```

## 11. Future evolution

Possible later gates include linting, formatting, type checking, coverage thresholds, Compose smoke tests, dependency or image security scanning, branch protection, and continuous delivery.

They are not implemented merely because they exist. Each should become a gate when its signal is valuable enough to justify maintenance and runtime cost. Continuous delivery is especially separate: CI builds confidence in a change, while CD moves an accepted artifact into an environment.

The central lesson is:

> CI does not automatically know what correct software means. Developers progressively encode important expectations as tests, and CI makes those expectations repeatable on every relevant change.
