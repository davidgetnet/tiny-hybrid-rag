"""Small data structures used while learning how documents become chunks."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    """Text loaded from one file, together with where it came from."""

    content: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """One retrieval-sized piece of a document and its identifying metadata."""

    content: str
    metadata: dict[str, Any]
