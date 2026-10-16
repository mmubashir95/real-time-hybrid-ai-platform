"""Teach structure-aware recursive chunking without touching production RAG.

The exercise progresses through three visible stages: Markdown section and
paragraph handling, recursive model-token splitting, then overlap and source
metadata. It uses ``tiktoken``'s ``o200k_base`` encoding, which is appropriate
for the project's GPT-4o-family baseline. The implementation is educational;
it is not imported by the production ingestion pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re

import tiktoken

try:
    from experiments.chunking.simple_chunking import chunk_text
except ModuleNotFoundError:
    # Direct execution puts this directory, rather than the repository root, on
    # sys.path. The local import keeps the documented standalone command usable.
    from simple_chunking import chunk_text


TARGET_CHUNK_TOKENS = 400
MAX_CHUNK_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
TOKEN_ENCODING_NAME = "o200k_base"

SAMPLE_DOCUMENT = """\
# Refund Policy

Customers can request refunds within 30 days.

Refund requests must include an order number.

## Refund Processing

Approved refunds are processed within 5 business days.

Refunds are returned to the original payment method.

# Subscription Policy

Subscriptions renew automatically every month.

Customers can cancel at any time.

# Shipping Policy

Standard shipping takes 5 to 7 business days.
"""

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
ENCODING = tiktoken.get_encoding(TOKEN_ENCODING_NAME)


@dataclass(frozen=True)
class Section:
    """One Markdown section with its hierarchy and ordered paragraphs."""

    heading: str | None
    heading_path: tuple[str, ...]
    heading_lines: tuple[str, ...]
    paragraphs: tuple[str, ...]


@dataclass(frozen=True)
class ChunkingConfig:
    """Token targets used for recursive splitting and optional overlap."""

    target_tokens: int = TARGET_CHUNK_TOKENS
    max_tokens: int = MAX_CHUNK_TOKENS
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS


@dataclass(frozen=True)
class DocumentChunk:
    """One ordered educational chunk with deterministic source metadata."""

    chunk_id: str
    document_id: str
    document_version: str
    heading_path: tuple[str, ...]
    chunk_index: int
    text: str
    token_count: int
    source_reference: str


@dataclass(frozen=True)
class ChunkingResult:
    """Generated chunks and measured diagnostics for one document."""

    chunks: tuple[DocumentChunk, ...]
    oversized_paragraphs: int
    token_fallback_splits: int
    duplicated_overlap_tokens: int
    section_boundaries_preserved: bool


@dataclass(frozen=True)
class _TextUnit:
    """A recursive text piece and the separator that originally preceded it."""

    text: str
    separator: str


def _build_long_experiment_document() -> str:
    """Create a deterministic synthetic fixture large enough to require splits."""
    sections: list[str] = [SAMPLE_DOCUMENT.strip()]
    topics = (
        ("Synthetic Account Operations", "account review", "specialist approval"),
        ("Synthetic Billing Operations", "billing review", "ledger confirmation"),
        ("Synthetic Delivery Operations", "delivery review", "carrier confirmation"),
    )
    for heading, review_name, evidence_name in topics:
        paragraphs = [f"# {heading}"]
        for paragraph_number in range(1, 17):
            paragraphs.append(
                f"Synthetic example {paragraph_number} explains the {review_name} "
                "workflow for training. The operator records the request, checks "
                f"the applicable conditions, and obtains {evidence_name} before "
                "reporting a result. This paragraph is deliberately repetitive "
                "enough to exercise chunk boundaries without representing a real "
                "company policy."
            )
        sections.append("\n\n".join(paragraphs))
    return "\n\n".join(sections) + "\n"


LONG_EXPERIMENT_DOCUMENT = _build_long_experiment_document()
EXPERIMENT_CONFIGS = (
    ("A", ChunkingConfig(250, 350, 30)),
    ("B", ChunkingConfig(400, 500, 50)),
    ("C", ChunkingConfig(600, 700, 75)),
)


# ---------------------------------------------------------------------------
# Stage 3A — section and paragraph-aware chunking
# ---------------------------------------------------------------------------


def identify_sections(text: str) -> list[Section]:
    """Parse Markdown headings and attach paragraphs to their heading paths."""
    sections: list[Section] = []
    heading_stack: list[tuple[int, str, str]] = []
    current_heading: str | None = None
    current_path: tuple[str, ...] = ()
    current_heading_lines: tuple[str, ...] = ()
    current_paragraphs: list[str] = []
    paragraph_lines: list[str] = []

    def save_section() -> None:
        """Store the current heading when it owns text or is an explicit heading."""
        if current_heading is not None or current_paragraphs:
            sections.append(
                Section(
                    heading=current_heading,
                    heading_path=current_path,
                    heading_lines=current_heading_lines,
                    paragraphs=tuple(current_paragraphs),
                )
            )

    def save_paragraph() -> None:
        """Store a non-empty paragraph accumulated between structural lines."""
        paragraph = "\n".join(paragraph_lines).strip()
        if paragraph:
            current_paragraphs.append(paragraph)
        paragraph_lines.clear()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        heading_match = HEADING_PATTERN.fullmatch(line)
        if heading_match:
            save_paragraph()
            save_section()
            level = len(heading_match.group(1))
            heading = heading_match.group(2).strip()
            heading_stack = [item for item in heading_stack if item[0] < level]
            heading_stack.append((level, heading, line))
            current_heading = heading
            current_path = tuple(item[1] for item in heading_stack)
            current_heading_lines = tuple(item[2] for item in heading_stack)
            current_paragraphs = []
            continue

        if not line:
            save_paragraph()
        else:
            paragraph_lines.append(raw_line.strip())

    save_paragraph()
    save_section()
    return sections


def group_section_paragraphs(section: Section, word_limit: int) -> list[str]:
    """Group intact section paragraphs up to a simple Stage 3A word limit."""
    if word_limit <= 0:
        raise ValueError("word_limit must be positive")

    chunks: list[str] = []
    current_paragraphs: list[str] = []
    current_words = 0

    for paragraph in section.paragraphs:
        paragraph_words = len(paragraph.split())
        if current_paragraphs and current_words + paragraph_words > word_limit:
            chunks.append("\n\n".join(current_paragraphs))
            current_paragraphs = []
            current_words = 0

        # Stage 3A deliberately emits an oversized paragraph intact; Stage 3B
        # introduces sentence and token fallbacks rather than dropping it.
        current_paragraphs.append(paragraph)
        current_words += paragraph_words

    if current_paragraphs:
        chunks.append("\n\n".join(current_paragraphs))
    return chunks


def stage_3a_chunks(text: str, word_limit: int = 40) -> list[tuple[Section, str]]:
    """Return section-associated paragraph groups for the Stage 3A demo."""
    grouped: list[tuple[Section, str]] = []
    for section in identify_sections(text):
        for chunk in group_section_paragraphs(section, word_limit):
            grouped.append((section, chunk))
    return grouped


def print_stage_3a() -> None:
    """Print computed section-aware chunks from the short sample document."""
    print("STAGE 3A — SECTION-AWARE CHUNKING")
    for chunk_index, (section, text) in enumerate(stage_3a_chunks(SAMPLE_DOCUMENT), 1):
        print(f"\nSection: {section.heading or '(preamble)'}")
        print(f"Heading path: {' > '.join(section.heading_path) or '(none)'}")
        print(f"\nChunk {chunk_index}:\n{text}")


# ---------------------------------------------------------------------------
# Stage 3B — recursive splitting and model-token limits
# ---------------------------------------------------------------------------


def count_tokens(text: str) -> int:
    """Count GPT-4o-family tokens with the ``o200k_base`` encoding."""
    return len(ENCODING.encode(text))


def _validate_config(config: ChunkingConfig) -> None:
    """Reject limits that cannot produce bounded, progressing chunks."""
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


def _render_chunk(heading_lines: tuple[str, ...], body: str) -> str:
    """Render final text so token limits include headings and separators."""
    heading_text = "\n".join(heading_lines)
    if heading_text and body:
        return f"{heading_text}\n\n{body}"
    return heading_text or body


def _split_sentences(paragraph: str) -> list[str]:
    """Split a normalized paragraph after basic sentence-ending punctuation."""
    return [
        sentence.strip()
        for sentence in SENTENCE_BOUNDARY.split(paragraph)
        if sentence.strip()
    ]


def _token_fallback(
    text: str,
    heading_lines: tuple[str, ...],
    max_tokens: int,
) -> list[str]:
    """Split an oversized sentence into decodable pieces within the hard limit."""
    token_ids = ENCODING.encode(text)
    pieces: list[str] = []
    start = 0

    while start < len(token_ids):
        end = min(len(token_ids), start + max_tokens)
        # Reduce the token slice until its fully rendered chunk—including
        # headings—fits the hard maximum.
        while end > start:
            candidate = ENCODING.decode(token_ids[start:end])
            if count_tokens(_render_chunk(heading_lines, candidate)) <= max_tokens:
                break
            end -= 1
        if end == start:
            raise ValueError("max_tokens is too small for the section heading")

        # BPE tokens can divide a word into multiple pieces. When whitespace is
        # available nearby, end before the token that starts the next word so a
        # chunk boundary does not unnecessarily turn one word into two strings.
        fitted_end = end
        while end > start + 1 and end < len(token_ids):
            next_token_text = ENCODING.decode([token_ids[end]])
            if next_token_text[:1].isspace():
                break
            end -= 1
        if end == start + 1 and fitted_end > end:
            # A long uninterrupted value has no meaningful word boundary, so
            # token-level splitting remains the required final fallback.
            end = fitted_end

        pieces.append(ENCODING.decode(token_ids[start:end]))
        start = end

    return pieces


def _paragraph_units(
    paragraph: str,
    heading_lines: tuple[str, ...],
    max_tokens: int,
) -> tuple[list[_TextUnit], bool, int]:
    """Recursively turn one paragraph into paragraph, sentence, or token units."""
    if count_tokens(_render_chunk(heading_lines, paragraph)) <= max_tokens:
        return [_TextUnit(paragraph, "\n\n")], False, 0

    units: list[_TextUnit] = []
    fallback_pieces = 0
    sentences = _split_sentences(paragraph)
    for sentence_index, sentence in enumerate(sentences):
        separator = "\n\n" if sentence_index == 0 else " "
        if count_tokens(_render_chunk(heading_lines, sentence)) <= max_tokens:
            units.append(_TextUnit(sentence, separator))
            continue

        pieces = _token_fallback(sentence, heading_lines, max_tokens)
        fallback_pieces += len(pieces)
        for piece_index, piece in enumerate(pieces):
            piece_separator = separator if piece_index == 0 else ""
            units.append(_TextUnit(piece, piece_separator))

    return units, True, fallback_pieces


def _pack_units(
    units: list[_TextUnit],
    heading_lines: tuple[str, ...],
    target_tokens: int,
) -> list[str]:
    """Pack recursive units to the preferred target without exceeding it."""
    bodies: list[str] = []
    current_body = ""

    for unit in units:
        separator = unit.separator if current_body else ""
        candidate = current_body + separator + unit.text
        exceeds_target = (
            count_tokens(_render_chunk(heading_lines, candidate)) > target_tokens
        )
        if current_body and exceeds_target:
            bodies.append(current_body)
            current_body = unit.text
        else:
            current_body = candidate

    if current_body:
        bodies.append(current_body)
    return bodies


def _split_section(
    section: Section,
    config: ChunkingConfig,
) -> tuple[list[str], int, int]:
    """Recursively split one section and report fallback diagnostics."""
    if not section.paragraphs:
        return [], 0, 0
    if count_tokens(_render_chunk(section.heading_lines, "")) >= config.max_tokens:
        raise ValueError("max_tokens is too small for the section heading")

    units: list[_TextUnit] = []
    oversized_paragraphs = 0
    token_fallback_splits = 0
    for paragraph in section.paragraphs:
        paragraph_units, oversized, fallback_count = _paragraph_units(
            paragraph, section.heading_lines, config.max_tokens
        )
        units.extend(paragraph_units)
        oversized_paragraphs += int(oversized)
        token_fallback_splits += fallback_count

    bodies = _pack_units(units, section.heading_lines, config.target_tokens)
    return bodies, oversized_paragraphs, token_fallback_splits


# ---------------------------------------------------------------------------
# Stage 3C — controlled overlap, metadata, and experiments
# ---------------------------------------------------------------------------


def _select_overlap(
    previous_body: str,
    current_body: str,
    heading_lines: tuple[str, ...],
    config: ChunkingConfig,
) -> tuple[str, int]:
    """Choose a same-section overlap, preferring paragraphs then sentences."""
    if config.overlap_tokens == 0:
        return "", 0

    candidates = previous_body.split("\n\n")
    last_paragraph = candidates[-1]
    sentence_candidates = _split_sentences(last_paragraph)
    boundary_candidates = candidates + sentence_candidates

    overlap_text = ""
    for candidate in reversed(boundary_candidates):
        candidate_tokens = count_tokens(candidate)
        combined = f"{candidate}\n\n{current_body}"
        if (
            candidate_tokens <= config.overlap_tokens
            and count_tokens(_render_chunk(heading_lines, combined))
            <= config.max_tokens
        ):
            overlap_text = candidate
            break

    if not overlap_text:
        previous_tokens = ENCODING.encode(previous_body)
        take = min(config.overlap_tokens, len(previous_tokens))
        while take > 0:
            candidate = ENCODING.decode(previous_tokens[-take:]).strip()
            combined = f"{candidate}\n\n{current_body}" if candidate else current_body
            combined_fits = (
                count_tokens(_render_chunk(heading_lines, combined))
                <= config.max_tokens
            )
            if candidate and combined_fits:
                overlap_text = candidate
                break
            take -= 1

    return overlap_text, count_tokens(overlap_text) if overlap_text else 0


def _make_chunk_id(
    document_id: str,
    document_version: str,
    chunk_index: int,
    text: str,
    config: ChunkingConfig,
) -> str:
    """Create a stable SHA-256 ID from identity, text, order, and configuration."""
    identity = "\0".join(
        (
            document_id,
            document_version,
            str(chunk_index),
            text,
            str(config.target_tokens),
            str(config.max_tokens),
            str(config.overlap_tokens),
            TOKEN_ENCODING_NAME,
        )
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    return f"{document_id}:{document_version}:{digest}"


def chunk_document(
    text: str,
    document_id: str,
    document_version: str,
    source_reference: str,
    config: ChunkingConfig = ChunkingConfig(),
) -> ChunkingResult:
    """Create bounded, ordered, metadata-rich chunks from one Markdown document."""
    _validate_config(config)
    if not text.strip():
        return ChunkingResult((), 0, 0, 0, True)
    if not document_id.strip() or not document_version.strip():
        raise ValueError("document_id and document_version must not be empty")

    pending_chunks: list[tuple[Section, str]] = []
    oversized_paragraphs = 0
    token_fallback_splits = 0
    duplicated_overlap_tokens = 0

    for section in identify_sections(text):
        bodies, oversized_count, fallback_count = _split_section(section, config)
        oversized_paragraphs += oversized_count
        token_fallback_splits += fallback_count

        previous_body = ""
        for body_index, body in enumerate(bodies):
            source_body = body
            if body_index > 0:
                overlap_text, overlap_count = _select_overlap(
                    previous_body, body, section.heading_lines, config
                )
                if overlap_text:
                    body = f"{overlap_text}\n\n{body}"
                    duplicated_overlap_tokens += overlap_count
            pending_chunks.append((section, body))
            # Future overlap comes from unique source content, not overlap that
            # was already copied into this chunk, which prevents cascading.
            previous_body = source_body

    chunks: list[DocumentChunk] = []
    for chunk_index, (section, body) in enumerate(pending_chunks):
        rendered_text = _render_chunk(section.heading_lines, body)
        actual_token_count = count_tokens(rendered_text)
        if actual_token_count > config.max_tokens:
            raise RuntimeError("internal error: emitted chunk exceeds max_tokens")
        chunks.append(
            DocumentChunk(
                chunk_id=_make_chunk_id(
                    document_id,
                    document_version,
                    chunk_index,
                    rendered_text,
                    config,
                ),
                document_id=document_id,
                document_version=document_version,
                heading_path=section.heading_path,
                chunk_index=chunk_index,
                text=rendered_text,
                token_count=actual_token_count,
                source_reference=source_reference,
            )
        )

    section_boundaries_preserved = all(
        chunk.heading_path == section.heading_path
        for chunk, (section, _) in zip(chunks, pending_chunks)
    )
    return ChunkingResult(
        chunks=tuple(chunks),
        oversized_paragraphs=oversized_paragraphs,
        token_fallback_splits=token_fallback_splits,
        duplicated_overlap_tokens=duplicated_overlap_tokens,
        section_boundaries_preserved=section_boundaries_preserved,
    )


def calculate_experiment_results() -> list[tuple[str, ChunkingConfig, ChunkingResult]]:
    """Execute the three configurations and return their measured results."""
    results: list[tuple[str, ChunkingConfig, ChunkingResult]] = []
    for name, config in EXPERIMENT_CONFIGS:
        result = chunk_document(
            LONG_EXPERIMENT_DOCUMENT,
            document_id="LEARNING-STRUCTURE-001",
            document_version="1.0",
            source_reference="synthetic-learning-sample",
            config=config,
        )
        results.append((name, config, result))
    return results


def print_experiment_results() -> None:
    """Print full chunks, diagnostics, and a calculated comparison table."""
    results = calculate_experiment_results()
    for name, config, result in results:
        counts = [chunk.token_count for chunk in result.chunks]
        print(f"\nEXPERIMENT {name}")
        print(f"Target tokens: {config.target_tokens}")
        print(f"Maximum tokens: {config.max_tokens}")
        print(f"Overlap tokens: {config.overlap_tokens}")
        print(f"Total chunks: {len(result.chunks)}")
        print(f"Minimum chunk tokens: {min(counts, default=0)}")
        print(f"Maximum chunk tokens: {max(counts, default=0)}")
        print(f"Average chunk tokens: {sum(counts) / len(counts) if counts else 0:.1f}")
        print(f"Oversized paragraphs: {result.oversized_paragraphs}")
        print(f"Token fallback pieces: {result.token_fallback_splits}")
        print(f"Duplicated overlap tokens: {result.duplicated_overlap_tokens}")
        print(f"Section boundaries preserved: {result.section_boundaries_preserved}")

        for chunk in result.chunks:
            print(f"\nChunk {chunk.chunk_index + 1}")
            print(f"Document: {chunk.document_id} v{chunk.document_version}")
            print(f"Section: {' > '.join(chunk.heading_path) or '(preamble)'}")
            print(f"Token count: {chunk.token_count}")
            print(f"Text:\n{chunk.text}")

    print("\nEXPERIMENT COMPARISON")
    print("Experiment | Target | Maximum | Overlap | Chunks | Avg Tokens | Max Tokens")
    print("-" * 76)
    for name, config, result in results:
        counts = [chunk.token_count for chunk in result.chunks]
        average = sum(counts) / len(counts) if counts else 0
        print(
            f"{name:<10} | {config.target_tokens:<6} | {config.max_tokens:<7} | "
            f"{config.overlap_tokens:<7} | {len(result.chunks):<6} | "
            f"{average:<10.1f} | {max(counts, default=0)}"
        )


def print_exercise_2_comparison() -> None:
    """Show an actual fixed-size boundary beside structure-aware behavior."""
    fixed_chunks = chunk_text(SAMPLE_DOCUMENT, chunk_size=12)
    mixed_chunk = next(
        chunk for chunk in fixed_chunks if "Refund Processing" in chunk
    )
    structured = chunk_document(
        SAMPLE_DOCUMENT,
        "LEARNING-COMPARE-001",
        "1.0",
        "synthetic-learning-sample",
        ChunkingConfig(80, 100, 0),
    )

    print("\nEXERCISE 2 COMPARISON")
    print("A fixed-size chunk that crosses a structural boundary:")
    print(mixed_chunk)
    print("\nStructure-aware section paths:")
    for chunk in structured.chunks:
        print(f"- {' > '.join(chunk.heading_path) or '(preamble)'}")
    print("\nBehavior                       | Fixed-size | Structure-aware recursive")
    print("Preserves headings             | No         | Yes")
    print("Preserves paragraphs           | No         | When possible")
    print("Uses section boundaries        | No         | Yes")
    print("Counts model tokens            | No         | Yes")
    print("Supports overlap               | Yes        | Yes")
    print("Preserves section metadata     | No         | Yes")
    print("Retrieval accuracy evaluated   | No         | No")


def main() -> None:
    """Run all three learning stages and their calculated comparisons."""
    print_stage_3a()
    print("\n\nSTAGES 3B/3C — RECURSIVE TOKEN CHUNKING")
    print_experiment_results()
    print_exercise_2_comparison()


if __name__ == "__main__":
    main()


# Learning questions — intentionally unanswered:
# 1. Why do we identify sections before splitting text?
# 2. What happens when a section contains fewer than 400 tokens?
# 3. Why should a 900-token section be split at paragraph boundaries before
#    token boundaries?
# 4. What happens when a single sentence exceeds the maximum chunk size?
# 5. Why shouldn't overlap cross unrelated section boundaries?
# 6. Why must token counts include headings and separators?
# 7. Why do we need deterministic chunk IDs?
# 8. What is the difference between target chunk size and maximum chunk size?
