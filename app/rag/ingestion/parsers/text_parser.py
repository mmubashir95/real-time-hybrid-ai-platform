"""Plain-text extraction that preserves the source's meaningful line breaks."""

from pathlib import Path

from app.rag.ingestion.models import ContentBlock


def parse_text(file_path: Path) -> tuple[ContentBlock, ...]:
    """Read one UTF-8 text file as a single content block.

    Raises UnicodeDecodeError for invalid UTF-8 and ValueError when the source
    contains no non-whitespace text.
    """
    text = file_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("Document contains no extractable text")
    return (ContentBlock(text=text, block_index=0),)
