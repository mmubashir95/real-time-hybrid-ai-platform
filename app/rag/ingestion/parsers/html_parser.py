"""Human-readable HTML extraction with headings and source order preserved."""

from pathlib import Path

from bs4 import BeautifulSoup
from bs4.element import PreformattedString, Tag

from app.rag.ingestion.models import ContentBlock


HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
CONTENT_TAGS = {"p", "pre", "blockquote", "li", "dt", "dd", "tr"}
CONTAINER_TAGS = {
    "address",
    "article",
    "aside",
    "body",
    "details",
    "div",
    "dl",
    "fieldset",
    "figcaption",
    "figure",
    "footer",
    "form",
    "header",
    "main",
    "nav",
    "ol",
    "section",
    "summary",
    "table",
    "ul",
}
# Table cells are deliberately absent so a row's cells stay together in one
# block; inline tags such as <strong> are absent so they join their sentence.
STRUCTURAL_TAGS = HEADING_TAGS | CONTENT_TAGS | CONTAINER_TAGS
NON_CONTENT_TAGS = ["head", "script", "style", "noscript", "template"]


def _text_runs(root: Tag) -> list[tuple[Tag | None, list[str]]]:
    """Group visible strings by their nearest structural ancestor, in order.

    Every string belongs to exactly one run, so text is neither dropped (for
    example loose text in a <div> or a list item's text before a nested list)
    nor duplicated by also extracting an enclosing parent element.
    """
    runs: list[tuple[Tag | None, list[str]]] = []
    for string in root.find_all(string=True):
        # Comments, doctypes, and CDATA are markup, not readable content.
        if isinstance(string, PreformattedString):
            continue
        text = string.strip()
        if not text:
            continue
        owner = string.find_parent(STRUCTURAL_TAGS)
        if runs and runs[-1][0] is owner:
            runs[-1][1].append(text)
        else:
            runs.append((owner, [text]))
    return runs


def parse_html(file_path: Path) -> tuple[ContentBlock, ...]:
    """Extract ordered, non-duplicated content blocks from one UTF-8 HTML file.

    Each heading becomes its own block and names the section of the blocks that
    follow it. Raises ValueError when no readable text remains.
    """
    markup = file_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(markup, "html.parser")
    for irrelevant in soup(NON_CONTENT_TAGS):
        irrelevant.decompose()

    blocks: list[ContentBlock] = []
    current_section: str | None = None
    for owner, texts in _text_runs(soup.body or soup):
        text = " ".join(texts)
        if owner is not None and owner.name in HEADING_TAGS:
            current_section = text
        blocks.append(
            ContentBlock(text=text, section=current_section, block_index=len(blocks))
        )

    if not blocks:
        raise ValueError("Document contains no extractable text")
    return tuple(blocks)
