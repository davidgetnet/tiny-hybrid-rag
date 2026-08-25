"""Print every chunk so the document-to-chunk transformation is visible."""

from collections import Counter

from chunk_documents import chunk_documents
from load_documents import load_documents


SEPARATOR = "-" * 40


def main() -> None:
    documents = load_documents()
    chunks = chunk_documents(documents)

    print("DOCUMENT AND CHUNK COUNTS")
    print(f"Source documents: {len(documents)}")
    print(f"Total chunks: {len(chunks)}")

    chunks_per_source = Counter(chunk.metadata["source"] for chunk in chunks)
    for document in documents:
        print(f"Chunks from {document.source}: {chunks_per_source[document.source]}")

    print(f"\n{SEPARATOR}\nALL CHUNKS\n{SEPARATOR}")
    for chunk in chunks:
        print(f"SOURCE: {chunk.metadata['source']}")
        print(f"CHUNK: {chunk.metadata['chunk_id']}")
        print(chunk.content)
        print(SEPARATOR)

    print("\nLEARNING AID (manually identified, not retrieved)")
    print("Question A: handbook.md chunk 1 describes Backend technology.")
    print("Question B: handbook.md chunk 1 contains both the manager and Python facts.")
    print(
        "Question C: handbook.md chunk 1 identifies the Python-using team; "
        "policies.md chunks 1 and 2 contain its approver and the extra security review."
    )


if __name__ == "__main__":
    main()
