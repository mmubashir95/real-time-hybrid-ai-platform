"""Shared output and error models for document text extraction.

These models deliberately describe parsing only. Retrieval chunks, embeddings,
and storage identifiers belong to later ingestion phases.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ContentBlock:
    """One ordered unit of source text, using zero-based block indexes."""

    text: str
    block_index: int
    page: int | None = None
    section: str | None = None


@dataclass(frozen=True)
class ParsedDocument:
    """Extracted content plus unchanged source-document metadata."""

    document_id: str
    tenant_id: str
    version: str
    status: str
    title: str
    format: str
    source_path: str
    effective_date: str
    visibility: str
    synthetic: str
    topic_family: str
    sha256: str
    word_count: str
    pdf_pages: str
    section_count: str
    scope_note: str
    content: tuple[ContentBlock, ...]

    @property
    def extracted_word_count(self) -> int:
        """Return a simple whitespace-delimited count for diagnostics."""
        return sum(len(block.text.split()) for block in self.content)

    @property
    def empty_pdf_pages(self) -> int:
        """Count page blocks that contain no extractable PDF text."""
        if self.format != "pdf":
            return 0
        return sum(not block.text.strip() for block in self.content)

    def to_dict(self) -> dict[str, Any]:
        """Return a representation suitable for JSON serialization."""
        return asdict(self)


@dataclass(frozen=True)
class ParsingError:
    """An actionable failure for one validated source document."""

    document_id: str
    tenant_id: str
    version: str
    source_path: str
    error_type: str
    error_message: str


@dataclass(frozen=True)
class ParsingResult:
    """Batch parsing outcome containing successes and accumulated failures."""

    total_documents: int
    documents: tuple[ParsedDocument, ...]
    errors: tuple[ParsingError, ...]

    @property
    def passed(self) -> bool:
        """Return whether every validated input parsed successfully."""
        return not self.errors and len(self.documents) == self.total_documents

    @property
    def total_content_blocks(self) -> int:
        """Count all content blocks across successful documents."""
        return sum(len(document.content) for document in self.documents)

    @property
    def total_extracted_words(self) -> int:
        """Count extracted words across successful documents."""
        return sum(document.extracted_word_count for document in self.documents)

    @property
    def empty_documents(self) -> int:
        """Count successful documents that contain no meaningful text."""
        return sum(document.extracted_word_count == 0 for document in self.documents)

    @property
    def empty_pdf_pages(self) -> int:
        """Count empty PDF page blocks across successful documents."""
        return sum(document.empty_pdf_pages for document in self.documents)
