"""Deterministic Markdown extraction with heading-to-section association."""

import re
from pathlib import Path

from app.rag.ingestion.models import ContentBlock


HEADING_PATTERN = re.compile(r"^ {0,3}#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$")


def parse_markdown(file_path: Path) -> tuple[ContentBlock, ...]:
    """Split UTF-8 Markdown into ordered heading-associated content blocks.

    Heading lines remain in block text so extraction does not silently discard
    meaningful source content. Markdown syntax, including lists and fenced code,
    is otherwise preserved for a later normalization phase.
    """
    markdown = file_path.read_text(encoding="utf-8")
    if not markdown.strip():
        raise ValueError("Document contains no extractable text")

    blocks: list[ContentBlock] = []
    current_lines: list[str] = []
    current_section: str | None = None
    inside_fence = False

    def append_block() -> None:
        """Append the current non-empty section using stable source order."""
        text = "\n".join(current_lines).strip()
        if text:
            blocks.append(
                ContentBlock(
                    text=text,
                    section=current_section,
                    block_index=len(blocks),
                )
            )

    for line in markdown.splitlines():
        stripped_line = line.lstrip()
        if stripped_line.startswith(("```", "~~~")):
            inside_fence = not inside_fence

        heading_match = None if inside_fence else HEADING_PATTERN.match(line)
        if heading_match:
            append_block()
            current_section = heading_match.group(1).strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    append_block()
    if not blocks:
        raise ValueError("Document contains no extractable text")
    return tuple(blocks)
