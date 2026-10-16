"""Create deterministic, source-traceable chunks from normalized documents.

The chunker consumes the existing immutable ``ParsedDocument`` representation.
It never opens source files, reparses markup, mutates normalized blocks, or
performs embedding and retrieval work. Splitting proceeds from trustworthy
section metadata to blocks, paragraphs, sentences, and finally model tokens.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, fields, replace
import hashlib
from importlib.metadata import version as package_version
import json
import re

import tiktoken

from app.rag.ingestion.models import (
    ChunkingError,
    ChunkingResult,
    ChunkingStats,
    ContentBlock,
    DocumentChunk,
    ParsedDocument,
)


DEFAULT_TARGET_TOKENS = 400
DEFAULT_MAX_TOKENS = 500
DEFAULT_OVERLAP_TOKENS = 50
TOKENIZER_NAME = "o200k_base"
TOKENIZER_VERSION = package_version("tiktoken")
CHUNKING_CONFIG_VERSION = "phase4-v1"
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
FENCE_START = re.compile(r"^\s*(```|~~~)")
ENCODING = tiktoken.get_encoding(TOKENIZER_NAME)


@dataclass(frozen=True)
class ChunkingConfig:
    """Validated token budgets and reproducibility labels for chunking."""

    target_tokens: int = DEFAULT_TARGET_TOKENS
    max_tokens: int = DEFAULT_MAX_TOKENS
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS
    tokenizer_name: str = TOKENIZER_NAME
    tokenizer_version: str = TOKENIZER_VERSION
    config_version: str = CHUNKING_CONFIG_VERSION

    def __post_init__(self) -> None:
        """Fail early so invalid settings cannot produce partial batch work."""
        _validate_config(self)

    def to_dict(self) -> dict[str, int | str]:
        """Return the canonical configuration recorded in IDs and reports."""
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
        }


@dataclass(frozen=True)
class _SourceUnit:
    """A boundary-aware text unit tied to exact normalized source blocks."""

    text: str
    separator: str
    block_ids: tuple[int, ...]
    pages: tuple[int, ...]


@dataclass(frozen=True)
class _LogicalSection:
    """Consecutive normalized units sharing one trustworthy heading path."""

    heading_path: tuple[str, ...]
    units: tuple[_SourceUnit, ...]


@dataclass(frozen=True)
class _DraftChunk:
    """An internal chunk whose source units have not yet received metadata."""

    heading_path: tuple[str, ...]
    units: tuple[_SourceUnit, ...]


@dataclass(frozen=True)
class _DocumentDiagnostics:
    """Internal per-document counters returned with successful chunks."""

    normalized_blocks_processed: int
    empty_blocks_skipped: int
    oversized_paragraphs: int
    token_fallback_splits: int
    duplicated_overlap_tokens: int


def count_tokens(text: str) -> int:
    """Return the actual ``o200k_base`` token count for text."""
    return len(ENCODING.encode(text))


def _validate_config(config: ChunkingConfig) -> None:
    """Reject unsupported tokenizers and non-progressing token budgets."""
    if config.target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    if config.max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if config.target_tokens > config.max_tokens:
        raise ValueError("target_tokens must not exceed max_tokens")
    if config.overlap_tokens < 0:
        raise ValueError("overlap_tokens must not be negative")
    if config.overlap_tokens >= config.max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")
    if config.tokenizer_name != TOKENIZER_NAME:
        raise ValueError(f"Unsupported tokenizer: {config.tokenizer_name}")
    if config.tokenizer_version != TOKENIZER_VERSION:
        raise ValueError(
            f"Tokenizer version must be {TOKENIZER_VERSION} for reproducibility"
        )
    if not config.config_version.strip():
        raise ValueError("config_version must not be empty")


def _heading_path(block: ContentBlock) -> tuple[str, ...]:
    """Use parser hierarchy when available, otherwise its flat section label."""
    if block.heading_path:
        return block.heading_path
    return (block.section,) if block.section else ()


def _split_block_paragraphs(text: str) -> list[str]:
    """Split paragraphs while preserving blank lines inside Markdown fences."""
    paragraphs: list[str] = []
    current_lines: list[str] = []
    inside_fence = False

    def save_paragraph() -> None:
        """Append accumulated meaningful lines as one paragraph."""
        paragraph = "\n".join(current_lines).strip("\n")
        if paragraph.strip():
            paragraphs.append(paragraph)
        current_lines.clear()

    for line in text.splitlines():
        fence_match = FENCE_START.match(line)
        if not inside_fence and not line.strip():
            save_paragraph()
            continue
        current_lines.append(line)
        if fence_match:
            inside_fence = not inside_fence

    save_paragraph()
    return paragraphs


def _logical_sections(document: ParsedDocument) -> tuple[list[_LogicalSection], int]:
    """Convert ordered nonempty blocks into consecutive logical sections."""
    sections: list[_LogicalSection] = []
    current_path: tuple[str, ...] | None = None
    current_units: list[_SourceUnit] = []
    empty_blocks_skipped = 0

    def save_section() -> None:
        """Store accumulated source units without crossing a section boundary."""
        if current_path is not None and current_units:
            sections.append(_LogicalSection(current_path, tuple(current_units)))

    for block in document.content:
        if not block.text.strip():
            empty_blocks_skipped += 1
            continue

        path = _heading_path(block)
        if current_path is not None and path != current_path:
            save_section()
            current_units = []
        current_path = path

        paragraphs = _split_block_paragraphs(block.text)
        for paragraph_index, paragraph in enumerate(paragraphs):
            separator = "\n\n" if current_units or paragraph_index else ""
            pages = (block.page,) if block.page is not None else ()
            current_units.append(
                _SourceUnit(
                    text=paragraph,
                    separator=separator,
                    block_ids=(block.block_index,),
                    pages=pages,
                )
            )

    save_section()
    return sections, empty_blocks_skipped


def _render_units(units: Iterable[_SourceUnit]) -> str:
    """Render units with their original meaningful separators."""
    text = ""
    for unit in units:
        separator = unit.separator if text else ""
        text += separator + unit.text
    return text


def _split_sentences(text: str) -> list[str]:
    """Split normalized prose after basic sentence-ending punctuation."""
    return [
        sentence.strip()
        for sentence in SENTENCE_BOUNDARY.split(text)
        if sentence.strip()
    ]


def _decode_exact(token_ids: list[int]) -> str | None:
    """Decode only token slices that end on a complete UTF-8 character."""
    try:
        return ENCODING.decode_bytes(token_ids).decode("utf-8")
    except UnicodeDecodeError:
        return None


def _token_fallback(unit: _SourceUnit, max_tokens: int) -> list[_SourceUnit]:
    """Split an oversized sentence into lossless, token-bounded source units."""
    token_ids = ENCODING.encode(unit.text)
    pieces: list[_SourceUnit] = []
    start = 0

    while start < len(token_ids):
        end = min(len(token_ids), start + max_tokens)
        decoded: str | None = None
        while end > start:
            decoded = _decode_exact(token_ids[start:end])
            if decoded is not None and count_tokens(decoded) <= max_tokens:
                break
            end -= 1
        if end == start or decoded is None:
            raise ValueError("max_tokens cannot hold one UTF-8 token sequence")

        fitted_end = end
        fitted_text = decoded
        # Prefer a nearby word boundary. If none exists, the fitted token slice
        # remains the final fallback for an uninterrupted value.
        while end > start + 1 and end < len(token_ids):
            next_text = ENCODING.decode([token_ids[end]])
            candidate = _decode_exact(token_ids[start:end])
            if candidate is not None and next_text[:1].isspace():
                fitted_text = candidate
                fitted_end = end
                break
            end -= 1

        pieces.append(
            _SourceUnit(
                text=fitted_text,
                separator=unit.separator if not pieces else "",
                block_ids=unit.block_ids,
                pages=unit.pages,
            )
        )
        start = fitted_end

    return pieces


def _recursive_units(
    unit: _SourceUnit,
    max_tokens: int,
) -> tuple[list[_SourceUnit], bool, int]:
    """Split an oversized paragraph at sentences, then model tokens."""
    if count_tokens(unit.text) <= max_tokens:
        return [unit], False, 0

    result: list[_SourceUnit] = []
    fallback_splits = 0
    for sentence_index, sentence in enumerate(_split_sentences(unit.text)):
        sentence_unit = replace(
            unit,
            text=sentence,
            separator=unit.separator if sentence_index == 0 else " ",
        )
        if count_tokens(sentence) <= max_tokens:
            result.append(sentence_unit)
            continue
        pieces = _token_fallback(sentence_unit, max_tokens)
        fallback_splits += len(pieces)
        result.extend(pieces)

    return result, True, fallback_splits


def _pack_units(units: list[_SourceUnit], target_tokens: int) -> list[_DraftChunk]:
    """Group recursive units to the preferred target without empty chunks."""
    drafts: list[_DraftChunk] = []
    current_units: list[_SourceUnit] = []

    for unit in units:
        candidate = [*current_units, unit]
        if current_units and count_tokens(_render_units(candidate)) > target_tokens:
            drafts.append(_DraftChunk((), tuple(current_units)))
            current_units = [replace(unit, separator="")]
        else:
            current_units = candidate

    if current_units:
        drafts.append(_DraftChunk((), tuple(current_units)))
    return drafts


def _overlap_units(
    previous: _DraftChunk,
    current: _DraftChunk,
    config: ChunkingConfig,
) -> tuple[tuple[_SourceUnit, ...], int]:
    """Select a bounded suffix from the same logical section for context."""
    if config.overlap_tokens == 0:
        return (), 0

    selected: list[_SourceUnit] = []
    for unit in reversed(previous.units):
        candidate = [unit, *selected]
        candidate_text = _render_units(candidate)
        combined_units = [*candidate, replace(current.units[0], separator="\n\n")]
        combined_units.extend(current.units[1:])
        if (
            count_tokens(candidate_text) <= config.overlap_tokens
            and count_tokens(_render_units(combined_units)) <= config.max_tokens
        ):
            selected = candidate
        else:
            break

    if selected:
        overlap_text = _render_units(selected)
        return tuple(selected), count_tokens(overlap_text)

    last_unit = previous.units[-1]
    token_ids = ENCODING.encode(last_unit.text)
    take = min(config.overlap_tokens, len(token_ids))
    while take > 0:
        overlap_text = _decode_exact(token_ids[-take:])
        if overlap_text:
            overlap_unit = replace(last_unit, text=overlap_text, separator="")
            combined = (
                overlap_unit,
                replace(current.units[0], separator="\n\n"),
                *current.units[1:],
            )
            if count_tokens(_render_units(combined)) <= config.max_tokens:
                return (overlap_unit,), count_tokens(overlap_text)
        take -= 1
    return (), 0


def _draft_document(
    document: ParsedDocument,
    config: ChunkingConfig,
) -> tuple[list[_DraftChunk], _DocumentDiagnostics]:
    """Build bounded drafts and diagnostics from one normalized document."""
    sections, empty_blocks_skipped = _logical_sections(document)
    drafts: list[_DraftChunk] = []
    oversized_paragraphs = 0
    token_fallback_splits = 0
    duplicated_overlap_tokens = 0

    for section in sections:
        recursive: list[_SourceUnit] = []
        for unit in section.units:
            pieces, oversized, fallback_count = _recursive_units(
                unit, config.max_tokens
            )
            recursive.extend(pieces)
            oversized_paragraphs += int(oversized)
            token_fallback_splits += fallback_count

        section_drafts = [
            replace(draft, heading_path=section.heading_path)
            for draft in _pack_units(recursive, config.target_tokens)
        ]
        for draft_index, draft in enumerate(section_drafts):
            if draft_index == 0:
                drafts.append(draft)
                continue
            previous_source_draft = section_drafts[draft_index - 1]
            overlap, overlap_count = _overlap_units(
                previous_source_draft, draft, config
            )
            if overlap:
                current_units = (
                    replace(draft.units[0], separator="\n\n"),
                    *draft.units[1:],
                )
                draft = replace(draft, units=(*overlap, *current_units))
                duplicated_overlap_tokens += overlap_count
            drafts.append(draft)

    diagnostics = _DocumentDiagnostics(
        normalized_blocks_processed=len(document.content),
        empty_blocks_skipped=empty_blocks_skipped,
        oversized_paragraphs=oversized_paragraphs,
        token_fallback_splits=token_fallback_splits,
        duplicated_overlap_tokens=duplicated_overlap_tokens,
    )
    return drafts, diagnostics


def _ordered_unique(values: Iterable[int]) -> tuple[int, ...]:
    """Return first-seen integer values without changing their order."""
    return tuple(dict.fromkeys(values))


def _chunk_id(
    document: ParsedDocument,
    chunk_index: int,
    text: str,
    source_block_ids: tuple[int, ...],
    config: ChunkingConfig,
) -> str:
    """Generate a stable SHA-256 ID from canonical identity and configuration."""
    identity = {
        "tenant_id": document.tenant_id,
        "document_id": document.document_id,
        "document_version": document.version,
        "chunk_index": chunk_index,
        "text": text,
        "source_block_ids": source_block_ids,
        "chunking_config": config.to_dict(),
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return "chk_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _chunk_document_with_stats(
    document: ParsedDocument,
    config: ChunkingConfig,
) -> tuple[tuple[DocumentChunk, ...], _DocumentDiagnostics]:
    """Chunk one normalized document and retain its processing diagnostics."""
    if not isinstance(document, ParsedDocument):
        raise TypeError("chunk_document expects a ParsedDocument")
    _validate_config(config)
    drafts, diagnostics = _draft_document(document, config)
    chunks: list[DocumentChunk] = []

    for chunk_index, draft in enumerate(drafts):
        text = _render_units(draft.units)
        if not text.strip():
            raise ValueError("Chunking produced an empty chunk")
        actual_tokens = count_tokens(text)
        if actual_tokens > config.max_tokens:
            raise ValueError("Chunking produced a chunk above max_tokens")

        block_ids = _ordered_unique(
            block_id for unit in draft.units for block_id in unit.block_ids
        )
        pages = _ordered_unique(page for unit in draft.units for page in unit.pages)
        chunks.append(
            DocumentChunk(
                chunk_id=_chunk_id(
                    document, chunk_index, text, block_ids, config
                ),
                document_id=document.document_id,
                document_version=document.version,
                tenant_id=document.tenant_id,
                document_status=document.status,
                effective_date=document.effective_date,
                chunk_index=chunk_index,
                heading_path=draft.heading_path,
                text=text,
                token_count=actual_tokens,
                source_file=document.source_path,
                source_type=document.format,
                start_page=min(pages) if pages else None,
                end_page=max(pages) if pages else None,
                source_block_ids=block_ids,
                tokenizer_name=config.tokenizer_name,
                tokenizer_version=config.tokenizer_version,
                chunking_config_version=config.config_version,
                access_scope=document.visibility,
            )
        )

    return tuple(chunks), diagnostics


def chunk_document(
    document: ParsedDocument,
    config: ChunkingConfig = ChunkingConfig(),
) -> tuple[DocumentChunk, ...]:
    """Return deterministic chunks for one normalized document."""
    chunks, _ = _chunk_document_with_stats(document, config)
    return chunks


def _sum_stats(items: Iterable[_DocumentDiagnostics]) -> ChunkingStats:
    """Combine document diagnostics into one batch summary."""
    diagnostics = tuple(items)
    return ChunkingStats(
        normalized_blocks_processed=sum(
            item.normalized_blocks_processed for item in diagnostics
        ),
        empty_blocks_skipped=sum(item.empty_blocks_skipped for item in diagnostics),
        oversized_paragraphs=sum(
            item.oversized_paragraphs for item in diagnostics
        ),
        token_fallback_splits=sum(
            item.token_fallback_splits for item in diagnostics
        ),
        duplicated_overlap_tokens=sum(
            item.duplicated_overlap_tokens for item in diagnostics
        ),
    )


def chunk_documents(
    documents: Iterable[ParsedDocument],
    config: ChunkingConfig = ChunkingConfig(),
) -> ChunkingResult:
    """Chunk documents in order while accumulating document-level failures."""
    inputs = tuple(documents)
    chunks: list[DocumentChunk] = []
    errors: list[ChunkingError] = []
    diagnostics: list[_DocumentDiagnostics] = []
    zero_chunk_documents: list[str] = []
    processed_documents = 0
    documents_with_chunks = 0

    for document in inputs:
        try:
            document_chunks, document_diagnostics = _chunk_document_with_stats(
                document, config
            )
        except Exception as error:  # noqa: BLE001
            # A batch must preserve later document results even for an
            # unexpected document-local defect, while exposing its exact type.
            errors.append(
                ChunkingError(
                    document_id=getattr(document, "document_id", "unknown"),
                    tenant_id=getattr(document, "tenant_id", "unknown"),
                    version=getattr(document, "version", "unknown"),
                    source_path=getattr(document, "source_path", "unknown"),
                    processing_stage="chunking",
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
            )
            continue

        processed_documents += 1
        diagnostics.append(document_diagnostics)
        if document_chunks:
            documents_with_chunks += 1
            chunks.extend(document_chunks)
        else:
            zero_chunk_documents.append(
                f"{document.tenant_id}/{document.document_id}/{document.version}"
            )

    return ChunkingResult(
        total_documents=len(inputs),
        processed_documents=processed_documents,
        documents_with_chunks=documents_with_chunks,
        zero_chunk_documents=tuple(zero_chunk_documents),
        chunks=tuple(chunks),
        errors=tuple(errors),
        stats=_sum_stats(diagnostics),
    )
