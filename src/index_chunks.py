"""Build or refresh the persistent Chroma records for all eight chunks."""

from pathlib import Path

from chunk_documents import chunk_documents
from embeddings import MODEL_NAME, embed_chunks, load_embedding_model
from load_documents import load_documents
from vector_store import (
    CHROMA_DIRECTORY,
    COLLECTION_NAME,
    get_collection,
    open_client,
    stable_id,
    upsert_chunks,
)


def index_chunks(persist_directory: Path | str = CHROMA_DIRECTORY, model=None):
    """Embed and upsert the current chunks, then return the collection and records."""

    chunks = chunk_documents(load_documents())
    embedding_model = model or load_embedding_model()
    embedded_chunks, matrix = embed_chunks(chunks, embedding_model)
    collection = get_collection(open_client(persist_directory))
    upsert_chunks(collection, embedded_chunks)
    return collection, embedded_chunks, matrix


def main() -> None:
    collection, embedded_chunks, matrix = index_chunks()

    print(f"Collection: {COLLECTION_NAME}")
    print(f"Embedding model: {MODEL_NAME}")
    print(f"Embedding dimensions: {matrix.shape[1]}")
    print(f"Records stored: {collection.count()}")
    print()
    for chunk in embedded_chunks:
        print(stable_id(chunk.metadata))


if __name__ == "__main__":
    main()
