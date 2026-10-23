"""Build bounded LLM context from already-authorized reranked passages."""

from __future__ import annotations

from collections.abc import Sequence

import tiktoken

from app.rag.generation.models import ContextBuildResult
from app.rag.reranking.models import RerankedPassage


TOKENIZER_NAME = "o200k_base"
ENCODING = tiktoken.get_encoding(TOKENIZER_NAME)


class ContextBuildError(ValueError):
    """Report an invalid or unusable evidence token budget."""


def count_generation_tokens(text: str) -> int:
    """Return deterministic local token accounting for Phase 9 prompt budgets."""
    return len(ENCODING.encode(text))


def _page_label(passage: RerankedPassage) -> str:
    """Format available page provenance without fabricating absent values."""
    source = passage.passage
    if source.start_page is None and source.end_page is None:
        return "not available"
    if source.start_page == source.end_page or source.end_page is None:
        return str(source.start_page)
    return f"{source.start_page}-{source.end_page}"


def _evidence_header(passage: RerankedPassage, evidence_number: int) -> str:
    """Render stable source identity ahead of one untrusted passage body."""
    source = passage.passage
    section = " > ".join(source.section) if source.section else "not available"
    return (
        f"[E{evidence_number}]\n"
        f"Reranked position: {passage.rerank_rank}\n"
        f"Document ID: {source.document_id}\n"
        f"Document version: {source.document_version}\n"
        f"Document title: {source.title}\n"
        f"Section: {section}\n"
        f"Page: {_page_label(passage)}\n"
        f"Source reference: {source.source_reference}\n"
        f"Chunk ID: {source.chunk_id}\n"
        "Passage (untrusted reference text):\n"
    )


def _render_complete_block(passage: RerankedPassage, evidence_number: int) -> str:
    """Format one complete passage with explicit start and end boundaries."""
    return (
        _evidence_header(passage, evidence_number)
        + passage.passage.text
        + f"\n[/E{evidence_number}]"
    )


def _truncate_first_passage(
    passage: RerankedPassage,
    token_budget: int,
) -> str | None:
    """Fit the highest-ranked passage tail-first when no complete block fits."""
    header = _evidence_header(passage, 1)
    footer = "\n[/E1]"
    fixed_tokens = count_generation_tokens(header + footer)
    available_text_tokens = token_budget - fixed_tokens
    if available_text_tokens <= 0:
        return None
    passage_tokens = ENCODING.encode(passage.passage.text)
    # Token boundaries can merge when independently encoded strings are joined.
    # Recheck the rendered block and trim until the complete envelope fits.
    while available_text_tokens > 0:
        truncated_text = ENCODING.decode(passage_tokens[:available_text_tokens])
        rendered = header + truncated_text + footer
        if count_generation_tokens(rendered) <= token_budget:
            return rendered
        available_text_tokens -= 1
    return None


def build_context(
    passages: Sequence[RerankedPassage],
    token_budget: int,
) -> ContextBuildResult:
    """Select complete passages in reranked order under a deterministic budget.

    If no complete passage fits, the highest-ranked passage is truncated at the
    tail so its source identity and leading policy text remain available. Every
    omission and truncation is returned as a diagnostic rather than hidden.
    """
    if isinstance(token_budget, bool) or not isinstance(token_budget, int):
        raise TypeError("token_budget must be an integer")
    if token_budget <= 0:
        raise ContextBuildError("token_budget must be positive")
    if any(not isinstance(passage, RerankedPassage) for passage in passages):
        raise TypeError("passages must contain RerankedPassage values")
    if not passages:
        return ContextBuildResult("", (), (), (), 0, token_budget)

    # Rerank ranks are part of the traceability contract; rejecting disorder is
    # safer than silently changing the evidence order seen by the model.
    ranks = [passage.rerank_rank for passage in passages]
    if ranks != sorted(ranks) or len(ranks) != len(set(ranks)):
        raise ContextBuildError("passages must be in unique reranking order")

    blocks: list[str] = []
    included: list[RerankedPassage] = []
    excluded: list[str] = []
    used_tokens = 0
    for passage in passages:
        block = _render_complete_block(passage, len(included) + 1)
        separator = "\n\n" if blocks else ""
        block_tokens = count_generation_tokens(separator + block)
        if used_tokens + block_tokens <= token_budget:
            blocks.append(block)
            included.append(passage)
            used_tokens += block_tokens
        else:
            excluded.append(passage.passage.chunk_id)

    truncated: tuple[str, ...] = ()
    if not included:
        truncated_block = _truncate_first_passage(passages[0], token_budget)
        if truncated_block is None:
            raise ContextBuildError("Context budget cannot fit evidence metadata")
        blocks = [truncated_block]
        included = [passages[0]]
        used_tokens = count_generation_tokens(truncated_block)
        truncated = (passages[0].passage.chunk_id,)
        excluded = [
            passage.passage.chunk_id for passage in passages[1:]
        ]

    rendered_context = "\n\n".join(blocks)
    return ContextBuildResult(
        text=rendered_context,
        included=tuple(included),
        excluded_chunk_ids=tuple(excluded),
        truncated_chunk_ids=truncated,
        # Count the final joined string because tokenizer boundaries may merge.
        estimated_tokens=count_generation_tokens(rendered_context),
        available_tokens=token_budget,
    )
