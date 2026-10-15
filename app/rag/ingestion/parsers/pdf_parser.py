"""Page-aware PDF text extraction using PyMuPDF."""

from pathlib import Path

import fitz

from app.rag.ingestion.models import ContentBlock
from app.rag.ingestion.parsers.section_markers import split_at_section_markers


def parse_pdf(file_path: Path) -> tuple[ContentBlock, ...]:
    """Extract page- and section-attributed blocks using 1-based page numbers.

    Each page is split at ``[SECTION-KEY] Title`` lines, so a block never spans
    two pages or two sections. A section that continues onto the next page keeps
    its name there. Empty pages remain represented by one empty block for
    diagnostics. A PDF is rejected only when every page lacks extractable text,
    which commonly indicates a scanned file.
    """
    blocks: list[ContentBlock] = []
    open_section: str | None = None
    try:
        with fitz.open(file_path) as pdf:
            for page_index, page in enumerate(pdf):
                page_number = page_index + 1  # Source-facing pages are 1-based.
                segments = split_at_section_markers(
                    page.get_text("text"), open_section
                )
                if not segments:
                    segments = [(open_section, "")]
                for section, text in segments:
                    blocks.append(
                        ContentBlock(
                            text=text,
                            page=page_number,
                            section=section,
                            block_index=len(blocks),
                        )
                    )
                open_section = segments[-1][0]
    except (fitz.FileDataError, RuntimeError) as error:
        raise ValueError(f"PDF could not be read: {error}") from error

    if not blocks:
        raise ValueError("PDF contains no pages")
    if not any(block.text for block in blocks):
        raise ValueError("PDF contains no extractable text; OCR is not enabled")
    return tuple(blocks)
