"""Normalize parsed text without reopening sources or changing provenance.

The rules in this module intentionally stay narrow. They standardize Unicode
and whitespace, remove bounded Markdown metadata, and remove PDF banner lines
that can be verified against the document's own metadata. Chunking and semantic
rewriting are explicitly outside this module's boundary.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import fields, replace
import re
import unicodedata

from app.rag.ingestion.models import (
    ContentBlock,
    NormalizationError,
    NormalizationResult,
    NormalizationStats,
    ParsedDocument,
)


FRONT_MATTER_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*\s*:")
PDF_PAGE_LABEL = re.compile(r"^Page\s+(\d+)$", re.IGNORECASE)
PDF_PAGE_OF_TOTAL = re.compile(r"^Page\s+(\d+)\s+of\s+\d+$", re.IGNORECASE)
MARKDOWN_INDENTED_STRUCTURE = re.compile(r"^\s+(?:[-+*]\s|\d+[.)]\s|>\s?)")
FENCE_START = re.compile(r"^\s*(```|~~~)")
LARGE_REDUCTION_RATIO = 0.25
LARGE_REDUCTION_MINIMUM_WORDS = 10

# Every field except content is immutable source metadata. Keeping this list
# derived from the dataclass prevents a newly added metadata field from being
# omitted from the integrity check by accident.
METADATA_FIELDS = tuple(
    field.name for field in fields(ParsedDocument) if field.name != "content"
)


def _strip_markdown_front_matter(text: str, document_id: str) -> tuple[str, bool]:
    """Remove a bounded leading metadata block that names this document."""
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return text, False

    closing_index = next(
        (
            index
            for index, line in enumerate(lines[1:], start=1)
            if line.strip() == "---"
        ),
        None,
    )
    if closing_index is None:
        return text, False

    metadata_lines = [line.strip() for line in lines[1:closing_index] if line.strip()]
    # A delimiter pair alone may be a horizontal-rule example. Requiring all
    # non-empty interior lines to resemble YAML keys keeps removal conservative.
    if not metadata_lines or not all(
        FRONT_MATTER_KEY.match(line) for line in metadata_lines
    ):
        return text, False

    # Key-shaped prose such as "Note: refunds are NOT permitted" between two
    # horizontal rules would pass the shape check, so also require the block
    # to identify this document, as the PDF banner rule does.
    declared_ids = {
        value.strip().strip("\"'")
        for key, _, value in (line.partition(":") for line in metadata_lines)
        if key.strip() == "document_id"
    }
    if declared_ids != {document_id}:
        return text, False

    return "".join(lines[closing_index + 1 :]).lstrip("\r\n"), True


def _strip_pdf_artifacts(
    text: str,
    document: ParsedDocument,
    block: ContentBlock,
) -> tuple[str, int, int, int]:
    """Remove only metadata-verifiable corpus banners and page labels."""
    expected_header = (
        f"SYNTHETIC | {document.document_id} v{document.version} | "
        f"{document.tenant_id} | {document.status}"
    )
    kept_lines: list[str] = []
    headers_removed = 0
    footers_removed = 0
    page_labels_removed = 0

    for line in text.splitlines(keepends=True):
        label = line.strip()
        if label == expected_header:
            headers_removed += 1
            continue

        page_match = PDF_PAGE_LABEL.fullmatch(label)
        if page_match and block.page == int(page_match.group(1)):
            page_labels_removed += 1
            continue

        footer_match = PDF_PAGE_OF_TOTAL.fullmatch(label)
        if footer_match and block.page == int(footer_match.group(1)):
            footers_removed += 1
            continue

        kept_lines.append(line)

    return "".join(kept_lines), headers_removed, footers_removed, page_labels_removed


def _normalize_line(line: str, preserve_leading_space: bool) -> str:
    """Normalize horizontal spacing while retaining meaningful indentation."""
    if preserve_leading_space:
        leading_length = len(line) - len(line.lstrip(" \t"))
        leading = line[:leading_length]
        body = line[leading_length:]
        return leading + re.sub(r"[ \t]+", " ", body).rstrip()
    return re.sub(r"[ \t]+", " ", line).strip()


def _normalize_whitespace(text: str, format_name: str) -> str:
    """Standardize line endings and spacing without flattening paragraphs."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    normalized_lines: list[str] = []
    inside_markdown_fence = False

    for line in lines:
        is_markdown = format_name == "md"
        fence_match = FENCE_START.match(line) if is_markdown else None
        if inside_markdown_fence and not fence_match:
            # Code examples may depend on internal spacing and blank lines, so
            # only remove trailing extraction whitespace inside a fenced block.
            normalized_lines.append(line.rstrip())
            continue

        if is_markdown and MARKDOWN_INDENTED_STRUCTURE.match(line):
            normalized_line = _normalize_line(line, preserve_leading_space=True)
        elif is_markdown and line.startswith(("    ", "\t")):
            # Indented code blocks are as spacing-sensitive as fenced ones.
            normalized_line = line.rstrip()
        else:
            normalized_line = _normalize_line(line, preserve_leading_space=False)

        # More than one empty line carries no extra paragraph information for
        # the downstream chunker, while one empty line preserves paragraphs.
        if not normalized_line and normalized_lines and not normalized_lines[-1]:
            continue
        normalized_lines.append(normalized_line)
        if fence_match:
            inside_markdown_fence = not inside_markdown_fence

    # Strip only surrounding blank lines; a full strip() would remove the
    # indentation of a leading Markdown code line.
    return "\n".join(normalized_lines).strip("\n")


def _prepare_block_text(
    document: ParsedDocument,
    block: ContentBlock,
    is_first_block: bool,
) -> tuple[str, dict[str, int]]:
    """Apply format-specific removal and common normalization to one block."""
    original_text = block.text
    bom_removed = int(original_text.startswith("\ufeff"))
    text = original_text[1:] if bom_removed else original_text
    text = unicodedata.normalize("NFC", text).replace("\u00a0", " ")

    front_matter_removed = False
    if document.format == "md" and is_first_block:
        text, front_matter_removed = _strip_markdown_front_matter(
            text, document.document_id
        )

    headers_removed = footers_removed = page_labels_removed = 0
    if document.format == "pdf":
        text, headers_removed, footers_removed, page_labels_removed = (
            _strip_pdf_artifacts(text, document, block)
        )

    # Content preservation is verified by line-level tests, not a runtime regex:
    # whitespace cleanup cannot delete words, and a value regex could neither
    # see artifact removal nor cover every currency or negation form.
    normalized_text = _normalize_whitespace(text, document.format)

    return normalized_text, {
        "markdown_front_matters_removed": int(front_matter_removed),
        "pdf_headers_removed": headers_removed,
        "pdf_footers_removed": footers_removed,
        "pdf_page_labels_removed": page_labels_removed,
        "bom_artifacts_removed": bom_removed,
        "whitespace_adjustments": int(normalized_text != text),
    }


def _normalize_with_stats(
    document: ParsedDocument,
) -> tuple[ParsedDocument, NormalizationStats, str | None]:
    """Normalize one document and return its measured changes and warning."""
    if not isinstance(document, ParsedDocument):
        raise TypeError("normalize_document expects a ParsedDocument")

    normalized_blocks: list[ContentBlock] = []
    counters: Counter[str] = Counter()
    for position, block in enumerate(document.content):
        normalized_text, block_counters = _prepare_block_text(
            document, block, is_first_block=position == 0
        )
        counters.update(block_counters)
        # replace() preserves page, section, and the original block index. Empty
        # artifact-only blocks remain as provenance anchors instead of being
        # silently dropped or causing later blocks to be renumbered.
        normalized_blocks.append(replace(block, text=normalized_text))

    normalized_document = replace(document, content=tuple(normalized_blocks))
    input_words = document.extracted_word_count
    output_words = normalized_document.extracted_word_count
    if input_words > 0 and output_words == 0:
        raise ValueError("Normalization removed all meaningful document text")

    # Enforced invariant rather than a reported metric: any provenance change
    # fails the document, so it appears as a normalization failure.
    metadata_changed = any(
        getattr(document, name) != getattr(normalized_document, name)
        for name in METADATA_FIELDS
    )
    block_provenance_changed = any(
        (source.block_index, source.page, source.section)
        != (output.block_index, output.page, output.section)
        for source, output in zip(document.content, normalized_document.content)
    )
    if metadata_changed or block_provenance_changed:
        raise ValueError("Normalization changed document or block provenance")

    def duplicate_excess(blocks: tuple[ContentBlock, ...]) -> int:
        """Count repeated non-empty block texts beyond their first occurrence."""
        text_counts = Counter(block.text for block in blocks if block.text)
        return sum(count - 1 for count in text_counts.values() if count > 1)

    duplicate_blocks_introduced = max(
        0,
        duplicate_excess(normalized_document.content)
        - duplicate_excess(document.content),
    )

    large_reduction = int(
        input_words >= LARGE_REDUCTION_MINIMUM_WORDS
        and input_words - output_words > input_words * LARGE_REDUCTION_RATIO
    )
    warning = None
    if large_reduction:
        warning = (
            f"{document.tenant_id}/{document.document_id}/{document.version}: "
            f"word count fell from {input_words} to {output_words}"
        )

    stats = NormalizationStats(
        input_content_blocks=len(document.content),
        output_content_blocks=len(normalized_document.content),
        input_words=input_words,
        output_words=output_words,
        large_word_reductions=large_reduction,
        duplicate_blocks_introduced=duplicate_blocks_introduced,
        **counters,
    )
    return normalized_document, stats, warning


def normalize_document(document: ParsedDocument) -> ParsedDocument:
    """Return a normalized copy of one parsed document without mutating input."""
    normalized_document, _, _ = _normalize_with_stats(document)
    return normalized_document


def _sum_stats(items: Iterable[NormalizationStats]) -> NormalizationStats:
    """Add numeric fields from document-level normalization statistics."""
    totals = {
        field.name: sum(getattr(item, field.name) for item in items)
        for field in fields(NormalizationStats)
    }
    return NormalizationStats(**totals)


def normalize_documents(documents: Iterable[ParsedDocument]) -> NormalizationResult:
    """Normalize a batch in order while accumulating document-level failures."""
    inputs = tuple(documents)
    normalized_documents: list[ParsedDocument] = []
    errors: list[NormalizationError] = []
    document_stats: list[NormalizationStats] = []
    warnings: list[str] = []

    for document in inputs:
        try:
            normalized, stats, warning = _normalize_with_stats(document)
        except Exception as error:  # noqa: BLE001
            # Batch ingestion must retain later successes even for an unexpected
            # document-local defect. The error type and message remain explicit
            # so callers can distinguish malformed input from programming bugs.
            errors.append(
                NormalizationError(
                    document_id=getattr(document, "document_id", "unknown"),
                    tenant_id=getattr(document, "tenant_id", "unknown"),
                    version=getattr(document, "version", "unknown"),
                    source_path=getattr(document, "source_path", "unknown"),
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
            )
            continue

        normalized_documents.append(normalized)
        document_stats.append(stats)
        if warning:
            warnings.append(warning)

    return NormalizationResult(
        total_documents=len(inputs),
        documents=tuple(normalized_documents),
        errors=tuple(errors),
        stats=_sum_stats(document_stats),
        warnings=tuple(warnings),
    )
