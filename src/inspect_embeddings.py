"""Inspect real embeddings and manually compare queries with every chunk."""

import re

from chunk_documents import chunk_documents
from embeddings import (
    MODEL_NAME,
    embed_chunks,
    embed_query,
    load_embedding_model,
    rank_chunks,
)
from load_documents import load_documents


QUERIES = (
    ("Query 1 — close wording", "What technology does the Backend team primarily use?"),
    (
        "Query 2 — different wording, same general meaning",
        "Which programming language powers Acorn Labs' server-side services?",
    ),
    ("Query 3 — unrelated topic", "Who reviews security-sensitive deployments?"),
)


def preview(text: str, length: int = 72) -> str:
    """Keep ranking output compact without changing stored chunk content."""

    one_line = " ".join(text.split())
    return one_line if len(one_line) <= length else f"{one_line[: length - 3]}..."


def exact_words(text: str) -> set[str]:
    """Return lowercase words for a deliberately tiny lexical comparison."""

    return set(re.findall(r"[a-z]+", text.lower()))


def print_ranking(results, limit: int) -> None:
    print("RANK | SCORE  | SOURCE       | CHUNK | PREVIEW")
    print("-" * 100)
    for rank, result in enumerate(results[:limit], start=1):
        metadata = result.chunk.metadata
        print(
            f"{rank:>4} | {result.score:.4f} | "
            f"{metadata['source']:<12} | {metadata['chunk_id']:>5} | "
            f"{preview(result.chunk.content)}"
        )


def main() -> None:
    documents = load_documents()
    chunks = chunk_documents(documents)
    model = load_embedding_model()
    embedded_chunks, matrix = embed_chunks(chunks, model)

    example = embedded_chunks[1]
    print("EMBEDDED CHUNKS")
    print(f"Chunks embedded: {len(embedded_chunks)}")
    print(f"Embedding model: {MODEL_NAME}")
    print(f"Embedding dimensionality: {matrix.shape[1]}")
    print(f"Embedding matrix type: {type(matrix)}")
    print(f"Embedding matrix dtype: {matrix.dtype}")
    print(f"Embedding matrix shape: {matrix.shape}")
    print(f"\nExample chunk ({example.metadata['source']}:{example.metadata['chunk_id']}):")
    print(example.content)
    print("First 8 values (inspection only):")
    print(example.vector[:8])
    print("These values are not separately named features such as 'Python' or 'Backend'.")

    all_rankings = []
    for index, (label, query) in enumerate(QUERIES):
        query_vector = embed_query(query, model)
        results = rank_chunks(query_vector, embedded_chunks)
        all_rankings.append(results)

        print(f"\n{'=' * 100}\n{label}\nQUERY: {query}")
        print(f"Query vector shape: {query_vector.shape}")
        print(f"Query vector first 8 values: {query_vector[:8]}")
        print("Same model means query and chunks share one 384-dimensional space.")
        print_ranking(results, limit=len(results) if index == 0 else 3)

    query_2 = QUERIES[1][1]
    query_2_results = all_rankings[1]
    best_for_query_2 = query_2_results[0]
    python_result = next(
        result
        for result in query_2_results
        if result.chunk.metadata["source"] == "handbook.md"
        and result.chunk.metadata["chunk_id"] == 1
    )
    python_rank = query_2_results.index(python_result) + 1
    shared_words = exact_words(query_2) & exact_words(python_result.chunk.content)
    print(f"\n{'=' * 100}\nKEYWORD COMPARISON FOR QUERY 2")
    print(f"Exact words shared with the Backend/Python chunk: {sorted(shared_words)}")
    print(
        "The wording differs: the query says 'programming language', 'server-side', "
        "and 'powers'; the chunk says 'Backend', 'Python', and 'develops services'."
    )
    print(
        f"Backend/Python embedding result: rank {python_rank} at {python_result.score:.4f}."
    )
    print(
        f"Actual overall winner: {best_for_query_2.chunk.metadata['source']}:"
        f"{best_for_query_2.chunk.metadata['chunk_id']} at {best_for_query_2.score:.4f}."
    )
    print(
        "Keyword and semantic methods solve different problems; production systems "
        "often combine them, but no lexical or hybrid retriever is implemented here."
    )

    print(f"\n{'=' * 100}\nIDENTITY STAYS OUTSIDE THE VECTOR")
    print("OBJECT")
    print("id: handbook.md:1")
    print(f"text: {preview(example.content)}")
    print("source: handbook.md")
    print("chunk_id: 1")
    print("embedding: [384 numbers]")
    print("The surrounding EmbeddedChunk associates identity, original text, and vector.")
    print("The vector alone does not know its filename or chunk number.")


if __name__ == "__main__":
    main()
