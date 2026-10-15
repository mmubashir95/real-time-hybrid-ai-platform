"""Human-readable HTML extraction with headings and source order preserved."""

from pathlib import Path

from bs4 import BeautifulSoup, Tag

from app.rag.ingestion.models import ContentBlock


CONTENT_TAGS = {"p", "pre", "blockquote", "li", "dt", "dd", "tr"}
HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}


def _tag_text(tag: Tag) -> str:
    """Collapse markup whitespace while retaining readable text boundaries."""
    return " ".join(tag.stripped_strings)


def parse_html(file_path: Path) -> tuple[ContentBlock, ...]:
    """Extract ordered, non-duplicated content blocks from one UTF-8 HTML file."""
    markup = file_path.read_text(encoding="utf-8")
    if not markup.strip():
        raise ValueError("Document contains no extractable text")

    soup = BeautifulSoup(markup, "html.parser")
    for irrelevant in soup(["script", "style", "noscript", "template"]):
        irrelevant.decompose()

    root = soup.body or soup
    blocks: list[ContentBlock] = []
    current_section: str | None = None
    for element in root.find_all(HEADING_TAGS | CONTENT_TAGS):
        if element.name in HEADING_TAGS:
            current_section = _tag_text(element) or current_section
            if current_section:
                blocks.append(
                    ContentBlock(
                        text=current_section,
                        section=current_section,
                        block_index=len(blocks),
                    )
                )
            continue

        # A parent paragraph/list/table row would repeat text handled by a more
        # specific nested content element, so only the innermost useful tag wins.
        if element.find(CONTENT_TAGS):
            continue
        text = _tag_text(element)
        if text:
            blocks.append(
                ContentBlock(
                    text=text,
                    section=current_section,
                    block_index=len(blocks),
                )
            )

    if not blocks:
        # Some valid HTML fragments contain plain body text with no semantic tag.
        fallback_text = " ".join(root.stripped_strings)
        if fallback_text:
            blocks.append(ContentBlock(text=fallback_text, block_index=0))

    if not blocks:
        raise ValueError("Document contains no extractable text")
    return tuple(blocks)
