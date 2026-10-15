"""Page-aware PDF text extraction using PyMuPDF."""

from pathlib import Path

import fitz

from app.rag.ingestion.models import ContentBlock


def parse_pdf(file_path: Path) -> tuple[ContentBlock, ...]:
    """Extract one block per physical PDF page using 1-based page numbers.

    Empty pages remain represented for diagnostics. A PDF is rejected only when
    every page lacks extractable text, which commonly indicates a scanned file.
    """
    blocks: list[ContentBlock] = []
    try:
        with fitz.open(file_path) as pdf:
            for page_index, page in enumerate(pdf):
                text = page.get_text("text").strip()
                blocks.append(
                    ContentBlock(
                        text=text,
                        page=page_index + 1,  # Source-facing page numbers are 1-based.
                        block_index=page_index,
                    )
                )
    except (fitz.FileDataError, RuntimeError) as error:
        raise ValueError(f"PDF could not be read: {error}") from error

    if not blocks:
        raise ValueError("PDF contains no pages")
    if not any(block.text for block in blocks):
        raise ValueError("PDF contains no extractable text; OCR is not enabled")
    return tuple(blocks)
