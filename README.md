# tiny-hybrid-rag

This is a small learning project for understanding hybrid retrieval-augmented generation one layer at a time. Each layer will be introduced only after the previous layer is understood:

```text
Plain Documents
      ↓
Chunking
      ↓
Embeddings
      ↓
Vector Retrieval
      ↓
Knowledge Graph
      ↓
Graph Retrieval
      ↓
Hybrid Retrieval
      ↓
LLM
      ↓
LangGraph
      ↓
Observability
```

For now, the project contains only a tiny set of natural-language documents about the fictional company Acorn Labs.

## Progress

- [x] Plain documents
- [x] Loading
- [x] Chunking
- [x] Embeddings
- [x] Vector retrieval
- [ ] Knowledge graph
- [ ] Graph retrieval
- [ ] Hybrid retrieval
- [ ] LLM synthesis
- [ ] LangGraph orchestration
- [ ] Observability

## Future Learning Questions

### Question A — Semantic retrieval

**What technology does the Backend team primarily use?**

Later, we expect vector retrieval to handle this well because the answer is directly described in the documentation.

### Question B — Relationship reasoning

**Who manages the team that uses Python?**

This requires connecting `Python ← USES ← Backend ← MANAGES ← Alice`. We are not implementing that connection yet.

### Question C — Hybrid question

**Who can approve a production deployment for the team that uses Python, and what additional requirement applies if the deployment is security-sensitive?**

Later, this question will help us understand why semantic document retrieval and explicit relationship traversal can complement one another.
