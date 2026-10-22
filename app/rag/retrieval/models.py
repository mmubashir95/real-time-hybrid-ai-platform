"""Typed, framework-independent request and result contracts for dense search."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Any

from app.rag.persistence.lifecycle import EmbeddingProfile


ALLOWED_VISIBILITIES = frozenset({"public", "internal"})
DEFAULT_TOP_K = 5
DEFAULT_MAX_TOP_K = 100


class SearchContextOrigin(str, Enum):
    """Identify the trusted boundary that established retrieval permissions."""

    AUTHENTICATED = "authenticated"
    INTERNAL = "internal"


@dataclass(frozen=True)
class HistoricalVersionScope:
    """Select one exact version after separate historical authorization."""

    document_id: str
    version: str

    def __post_init__(self) -> None:
        """Reject broad or blank historical identities."""
        if not isinstance(self.document_id, str) or not isinstance(self.version, str):
            raise TypeError("Historical document_id and version must be strings")
        if not self.document_id.strip() or not self.version.strip():
            raise ValueError("Historical scope requires document_id and version")
        has_outer_whitespace = (
            self.document_id != self.document_id.strip()
            or self.version != self.version.strip()
        )
        if has_outer_whitespace:
            raise ValueError("Historical document_id and version must not have outer whitespace")


@dataclass(frozen=True)
class TrustedSearchContext:
    """Carry server-established tenant, visibility, and version authorization.

    This object is an internal trust boundary. A public API must construct it
    from authenticated authorization data, never directly from request fields.
    """

    tenant_id: str
    permitted_visibilities: tuple[str, ...]
    origin: SearchContextOrigin
    historical: HistoricalVersionScope | None = None
    historical_access: bool = False

    def __post_init__(self) -> None:
        """Validate that missing access data cannot broaden a search."""
        if not isinstance(self.tenant_id, str):
            raise TypeError("Trusted tenant scope must be a string")
        if not self.tenant_id.strip():
            raise ValueError("Trusted tenant scope is required")
        if self.tenant_id != self.tenant_id.strip():
            raise ValueError("Trusted tenant scope must not have outer whitespace")
        if not isinstance(self.permitted_visibilities, tuple):
            raise TypeError("permitted_visibilities must be a tuple")
        if not self.permitted_visibilities:
            raise ValueError("At least one permitted visibility is required")
        if not all(
            isinstance(visibility, str)
            for visibility in self.permitted_visibilities
        ):
            raise TypeError("Permitted visibilities must be strings")
        if len(set(self.permitted_visibilities)) != len(self.permitted_visibilities):
            raise ValueError("Permitted visibilities must be unique")
        unsupported = set(self.permitted_visibilities) - ALLOWED_VISIBILITIES
        if unsupported:
            raise ValueError(
                f"Unsupported visibility scope: {', '.join(sorted(unsupported))}"
            )
        if not isinstance(self.origin, SearchContextOrigin):
            raise TypeError("origin must be a SearchContextOrigin")
        if self.historical is not None and not isinstance(
            self.historical, HistoricalVersionScope
        ):
            raise TypeError("historical must be a HistoricalVersionScope")
        if not isinstance(self.historical_access, bool):
            raise TypeError("historical_access must be a boolean")
        # Historical permission is useful only with an exact document/version;
        # rejecting mismatched flags prevents a future caller from broadening it.
        if (self.historical is None) != (not self.historical_access):
            raise ValueError(
                "Historical scope and explicit historical access must be provided together"
            )


@dataclass(frozen=True)
class DenseSearchRequest:
    """Describe one validated internal dense-search operation."""

    context: TrustedSearchContext
    profile: EmbeddingProfile
    question: str | None = None
    query_vector: Any | None = None
    top_k: int = DEFAULT_TOP_K
    max_top_k: int = DEFAULT_MAX_TOP_K

    def __post_init__(self) -> None:
        """Require exactly one query input and enforce a bounded result count."""
        if not isinstance(self.context, TrustedSearchContext):
            raise TypeError("context must be a TrustedSearchContext")
        if not isinstance(self.profile, EmbeddingProfile):
            raise TypeError("profile must be an EmbeddingProfile")
        has_question = self.question is not None
        has_vector = self.query_vector is not None
        if has_question == has_vector:
            raise ValueError("Provide exactly one of question or query_vector")
        if self.question is not None:
            if not isinstance(self.question, str):
                raise TypeError("Question must be a string")
            if not self.question.strip():
                raise ValueError("Question must not be empty")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int):
            raise TypeError("top_k must be an integer")
        if self.top_k <= 0 or self.top_k > self.max_top_k:
            raise ValueError(f"top_k must be between 1 and {self.max_top_k}")
        if self.max_top_k <= 0:
            raise ValueError("max_top_k must be positive")


@dataclass(frozen=True)
class DenseSearchResult:
    """Preserve one ranked chunk and its database source provenance."""

    chunk_id: str
    chunk_text: str
    similarity: float
    cosine_distance: float
    tenant_id: str
    document_id: str
    document_version: str
    document_title: str
    heading_path: tuple[str, ...]
    start_page: int | None
    end_page: int | None
    source_block_ids: tuple[int, ...]
    source_path: str
    source_format: str
    visibility: str
    embedding_model_id: str
    embedding_model_revision: str
    embedding_config_fingerprint: str

    def __post_init__(self) -> None:
        """Reject invalid scores returned across the database boundary."""
        if not isfinite(self.similarity) or not isfinite(self.cosine_distance):
            raise ValueError("Retrieval scores must be finite")


@dataclass(frozen=True)
class DenseSearchResponse:
    """Return ranked matches and measured retrieval diagnostics."""

    results: tuple[DenseSearchResult, ...]
    candidate_count: int
    query_encoding_seconds: float
    database_seconds: float
    total_seconds: float
