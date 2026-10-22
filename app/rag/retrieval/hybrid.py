"""Fuse independent lexical and semantic rankings with reciprocal rank fusion."""

from __future__ import annotations

from dataclasses import dataclass

from app.rag.retrieval.keyword import KeywordMatch
from app.rag.retrieval.models import DenseSearchResult


DEFAULT_RRF_CONSTANT = 60


@dataclass(frozen=True)
class FusedCandidate:
    """Preserve both branch records and their rank-based fusion score."""

    chunk_id: str
    keyword: KeywordMatch | None
    semantic: DenseSearchResult | None
    semantic_rank: int | None
    fusion_score: float


def reciprocal_rank_fusion(
    keyword_results: tuple[KeywordMatch, ...],
    semantic_results: tuple[DenseSearchResult, ...],
    *,
    limit: int,
    constant: int = DEFAULT_RRF_CONSTANT,
) -> tuple[FusedCandidate, ...]:
    """Deduplicate by chunk ID and rank candidates using standard RRF."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    if constant <= 0:
        raise ValueError("RRF constant must be positive")
    candidates: dict[str, dict[str, object]] = {}
    for match in keyword_results:
        if match.chunk.chunk_id in candidates:
            raise ValueError("Keyword results contain duplicate chunk IDs")
        candidates[match.chunk.chunk_id] = {
            "keyword": match,
            "semantic": None,
            "semantic_rank": None,
            "score": 1.0 / (constant + match.rank),
        }
    for rank, result in enumerate(semantic_results, start=1):
        candidate = candidates.setdefault(
            result.chunk_id,
            {"keyword": None, "semantic": None, "semantic_rank": None, "score": 0.0},
        )
        if candidate["semantic"] is not None:
            raise ValueError("Semantic results contain duplicate chunk IDs")
        candidate["semantic"] = result
        candidate["semantic_rank"] = rank
        candidate["score"] = float(candidate["score"]) + 1.0 / (constant + rank)
        keyword = candidate["keyword"]
        if keyword is not None:
            # A shared stable identity must refer to identical source ownership;
            # otherwise fusion could hide corrupt or cross-tenant metadata.
            assert isinstance(keyword, KeywordMatch)
            lexical = keyword.chunk
            if (
                lexical.tenant_id, lexical.document_id, lexical.document_version,
                lexical.heading_path, lexical.source_path,
            ) != (
                result.tenant_id, result.document_id, result.document_version,
                result.heading_path, result.source_path,
            ):
                raise ValueError("Conflicting provenance for fused chunk ID")
    fused = tuple(
        FusedCandidate(
            chunk_id=chunk_id,
            keyword=values["keyword"],  # type: ignore[arg-type]
            semantic=values["semantic"],  # type: ignore[arg-type]
            semantic_rank=values["semantic_rank"],  # type: ignore[arg-type]
            fusion_score=float(values["score"]),
        )
        for chunk_id, values in candidates.items()
    )
    # Branch ranks provide deterministic secondary ordering before immutable ID.
    return tuple(sorted(
        fused,
        key=lambda item: (
            -item.fusion_score,
            item.keyword.rank if item.keyword else 10**9,
            item.semantic_rank if item.semantic_rank else 10**9,
            item.chunk_id,
        ),
    )[:limit])
