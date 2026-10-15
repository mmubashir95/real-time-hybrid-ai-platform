"""Dispatch validated records to parsers and accumulate batch failures."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from app.rag.ingestion.models import (
    ContentBlock,
    ParsedDocument,
    ParsingError,
    ParsingResult,
)
from app.rag.ingestion.parsers.html_parser import parse_html
from app.rag.ingestion.parsers.markdown_parser import parse_markdown
from app.rag.ingestion.parsers.pdf_parser import parse_pdf
from app.rag.ingestion.parsers.text_parser import parse_text
from app.rag.ingestion.discovery import ValidatedDocument


PARSERS: dict[str, Callable[[Path], tuple[ContentBlock, ...]]] = {
    "pdf": parse_pdf,
    "html": parse_html,
    "md": parse_markdown,
    "txt": parse_text,
}


def parse_document(document: ValidatedDocument) -> ParsedDocument:
    """Parse one Phase 1 validated record into the common output model."""
    parser = PARSERS.get(document.format)
    if parser is None:
        raise ValueError(f"Unsupported document format: {document.format}")

    content = parser(document.resolved_file_path)
    metadata = document.metadata
    # Source metadata is copied verbatim; parsing status is represented only by
    # ParsingResult so values such as status=archived keep their original meaning.
    return ParsedDocument(
        document_id=document.document_id,
        tenant_id=document.tenant_id,
        version=document.version,
        status=document.status,
        title=document.title,
        format=document.format,
        source_path=metadata["path"],
        effective_date=metadata["effective_date"],
        visibility=metadata["visibility"],
        synthetic=metadata["synthetic"],
        topic_family=metadata["topic_family"],
        sha256=document.sha256,
        word_count=metadata["word_count"],
        pdf_pages=metadata["pdf_pages"],
        section_count=metadata["section_count"],
        scope_note=metadata["scope_note"],
        content=content,
    )


def parse_documents(documents: Iterable[ValidatedDocument]) -> ParsingResult:
    """Parse every validated record while retaining successes after failures."""
    inputs = tuple(documents)
    parsed_documents: list[ParsedDocument] = []
    errors: list[ParsingError] = []

    for document in inputs:
        try:
            parsed_documents.append(parse_document(document))
        except (OSError, UnicodeError, ValueError) as error:
            # Expected document-level failures must not hide results for the rest
            # of the validated batch.
            errors.append(
                ParsingError(
                    document_id=document.document_id,
                    tenant_id=document.tenant_id,
                    version=document.version,
                    source_path=document.metadata.get(
                        "path", str(document.resolved_file_path)
                    ),
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
            )

    return ParsingResult(
        total_documents=len(inputs),
        documents=tuple(parsed_documents),
        errors=tuple(errors),
    )
