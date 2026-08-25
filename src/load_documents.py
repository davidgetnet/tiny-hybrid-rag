"""Load the Acorn Labs Markdown files into small Document objects."""

from pathlib import Path

from documents import Document


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIRECTORY = PROJECT_ROOT / "data"
DOCUMENT_FILENAMES = ("handbook.md", "policies.md")


def load_documents() -> list[Document]:
    """Read each known Markdown file while preserving its source filename."""

    documents = []

    for filename in DOCUMENT_FILENAMES:
        path = DATA_DIRECTORY / filename
        text = path.read_text(encoding="utf-8")
        documents.append(
            Document(
                content=text,
                source=filename,
                metadata={"path": str(path.relative_to(PROJECT_ROOT))},
            )
        )

    return documents


if __name__ == "__main__":
    for document in load_documents():
        print(f"{document.source}: {len(document.content)} characters")
