"""Split loaded documents at paragraph boundaries without a framework."""

import re

from documents import Chunk, Document


def chunk_document(document: Document) -> list[Chunk]:
    """Turn one document into non-empty, deterministically numbered chunks."""

    paragraphs = re.split(r"\n\s*\n", document.content)
    chunks = []

    for paragraph in paragraphs:
        content = paragraph.strip()
        if not content:
            continue

        chunk_id = len(chunks)
        chunks.append(
            Chunk(
                content=content,
                metadata={
                    **document.metadata,
                    "source": document.source,
                    "chunk_id": chunk_id,
                },
            )
        )

    return chunks


def chunk_documents(documents: list[Document]) -> list[Chunk]:
    """Chunk several documents, retaining per-document chunk numbering."""

    return [
        chunk
        for document in documents
        for chunk in chunk_document(document)
    ]
