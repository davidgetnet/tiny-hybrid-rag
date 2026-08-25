"""Inspect Chroma retrieval, provenance, filtering, and manual-cosine parity."""

from chunk_documents import chunk_documents
from embeddings import embed_chunks, embed_query, load_embedding_model, rank_chunks
from inspect_embeddings import QUERIES, preview
from load_documents import load_documents
from vector_store import (
    CHROMA_DIRECTORY,
    COLLECTION_NAME,
    DISTANCE_METRIC,
    get_collection,
    open_client,
    query_by_embedding,
)


def print_results(results) -> None:
    print("RANK | DISTANCE | ID             | SOURCE       | CHUNK | PREVIEW")
    print("-" * 108)
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank:>4} | {result.distance:>8.4f} | {result.id:<14} | "
            f"{result.source:<12} | {result.chunk_id:>5} | {preview(result.text)}"
        )


def main() -> None:
    model = load_embedding_model()
    chunks = chunk_documents(load_documents())
    embedded_chunks, _ = embed_chunks(chunks, model)
    collection = get_collection(open_client(CHROMA_DIRECTORY))

    print(f"Collection: {COLLECTION_NAME}")
    print(f"Stored records reopened from disk: {collection.count()}")
    print(f"Verified distance metric: {collection.configuration['hnsw']['space']}")
    print("Lower cosine distance means closer; it is not a similarity score.")

    for label, query in QUERIES:
        query_vector = embed_query(query, model)
        chroma_results = query_by_embedding(collection, query_vector, top_k=3)
        manual_results = rank_chunks(query_vector, embedded_chunks)[:3]
        chroma_ids = [result.id for result in chroma_results]
        manual_ids = [
            f"{result.chunk.metadata['source']}:{result.chunk.metadata['chunk_id']}"
            for result in manual_results
        ]

        print(f"\n{'=' * 108}\n{label}\nQUERY: {query}")
        print_results(chroma_results)
        print(f"Manual cosine top 3: {manual_ids}")
        print(f"Chroma top 3:       {chroma_ids}")
        print(f"Order identical: {chroma_ids == manual_ids}")

    filter_query = QUERIES[2][1]
    filter_vector = embed_query(filter_query, model)
    print(f"\n{'=' * 108}\nMETADATA FILTER EXPERIMENT")
    print(f"QUERY: {filter_query}")
    for source in ("handbook.md", "policies.md"):
        results = query_by_embedding(
            collection,
            filter_vector,
            top_k=3,
            where={"source": source},
        )
        print(f"\nFilter: source = {source}")
        print_results(results)

    example = query_by_embedding(collection, filter_vector, top_k=1)[0]
    print(f"\n{'=' * 108}\nCOMPLETE RETRIEVED RECORD")
    print(f"ID: {example.id}")
    print(f"TEXT: {example.text}")
    print(f"SOURCE: {example.source}")
    print(f"CHUNK ID: {example.chunk_id}")
    print(f"EMBEDDING MODEL: {example.embedding_model}")
    print(f"{DISTANCE_METRIC.upper()} DISTANCE: {example.distance:.6f}")


if __name__ == "__main__":
    main()
