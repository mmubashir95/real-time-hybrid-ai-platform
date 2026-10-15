"""Plain-text extraction that preserves the source's meaningful line breaks."""

from pathlib import Path

from app.rag.ingestion.models import ContentBlock
from app.rag.ingestion.parsers.section_markers import split_at_section_markers


def parse_text(file_path: Path) -> tuple[ContentBlock, ...]:
    """Read one UTF-8 text file as ordered section blocks.

    Text before the first ``[SECTION-KEY] Title`` line has no section; a file
    without markers becomes a single block. Raises UnicodeDecodeError for
    invalid UTF-8 and ValueError when the source contains no non-whitespace text.
    """
    text = file_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("Document contains no extractable text")

    return tuple(
        ContentBlock(text=segment_text, section=section, block_index=index)
        for index, (section, segment_text) in enumerate(
            split_at_section_markers(text, open_section=None)
        )
    )
