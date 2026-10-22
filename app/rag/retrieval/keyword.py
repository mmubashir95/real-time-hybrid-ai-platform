"""Build and query a scoped, in-memory BM25 lexical index."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import logging
from math import log
import re
from time import perf_counter

import asyncpg

from app.rag.retrieval.dense import RetrievalError, _quote_identifier
from app.rag.retrieval.models import SearchFilters, TrustedSearchContext


logger = logging.getLogger(__name__)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
DEFAULT_BM25_K1 = 1.5
DEFAULT_BM25_B = 0.75


def tokenize_lexical(text: str) -> tuple[str, ...]:
    """Create deterministic lowercase terms while retaining exact hyphenated IDs."""
    tokens: list[str] = []
    for match in TOKEN_PATTERN.finditer(text.casefold()):
        token = match.group(0)
        tokens.append(token)
        # Index both the exact identifier and its components so AST-REF-001
        # remains searchable without making ordinary punctuation significant.
        if "-" in token:
            tokens.extend(part for part in token.split("-") if part)
    return tuple(tokens)


@dataclass(frozen=True)
class LexicalChunk:
    """Represent one authorized stored chunk and its immutable provenance."""

    chunk_id: str
    text: str
    tenant_id: str
    document_id: str
    document_version: str
    title: str
    heading_path: tuple[str, ...]
    start_page: int | None
    end_page: int | None
    source_block_ids: tuple[int, ...]
    source_path: str
    source_format: str
    visibility: str
    status: str
    effective_date: str


@dataclass(frozen=True)
class KeywordMatch:
    """Pair one lexical chunk with its genuine BM25 score and rank."""

    chunk: LexicalChunk
    score: float
    rank: int


@dataclass(frozen=True)
class KeywordSearchResponse:
    """Return keyword matches and measured scoped-index diagnostics."""

    results: tuple[KeywordMatch, ...]
    candidate_count: int
    total_seconds: float


class BM25Index:
    """Calculate Okapi BM25 scores over a fixed authorized chunk snapshot."""

    def __init__(
        self,
        chunks: tuple[LexicalChunk, ...],
        *,
        k1: float = DEFAULT_BM25_K1,
        b: float = DEFAULT_BM25_B,
    ) -> None:
        """Validate chunk identity and precompute term statistics."""
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 > 0 and 0 <= b <= 1")
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("BM25 index cannot contain duplicate chunk IDs")
        if any(not chunk.text.strip() for chunk in chunks):
            raise ValueError("BM25 index cannot contain empty chunks")
        self._chunks = chunks
        self._k1 = k1
        self._b = b
        self._term_frequencies = tuple(Counter(tokenize_lexical(item.text)) for item in chunks)
        self._lengths = tuple(sum(counts.values()) for counts in self._term_frequencies)
        self._average_length = sum(self._lengths) / len(chunks) if chunks else 0.0
        self._document_frequencies = Counter(
            term for frequencies in self._term_frequencies for term in frequencies
        )

    def search(self, query: str, limit: int) -> tuple[KeywordMatch, ...]:
        """Rank positive-scoring chunks with chunk ID as the stable tie-breaker."""
        if limit <= 0:
            raise ValueError("limit must be positive")
        query_terms = tuple(dict.fromkeys(tokenize_lexical(query)))
        if not query_terms or not self._chunks:
            return ()
        scored: list[tuple[float, LexicalChunk]] = []
        corpus_size = len(self._chunks)
        for chunk, frequencies, length in zip(
            self._chunks, self._term_frequencies, self._lengths, strict=True
        ):
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                document_frequency = self._document_frequencies[term]
                # Robertson's non-negative IDF variant is stable on small corpora.
                inverse_document_frequency = log(
                    1 + (corpus_size - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                length_ratio = length / self._average_length if self._average_length else 0.0
                denominator = frequency + self._k1 * (1 - self._b + self._b * length_ratio)
                score += inverse_document_frequency * frequency * (self._k1 + 1) / denominator
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return tuple(
            KeywordMatch(chunk=chunk, score=score, rank=rank)
            for rank, (score, chunk) in enumerate(scored[:limit], start=1)
        )


class KeywordRetriever:
    """Load only authorized chunks from PostgreSQL and rank them with BM25."""

    def __init__(self, connection: asyncpg.Connection, *, schema: str = "public") -> None:
        """Bind the database connection used to materialize scoped indexes."""
        self._connection = connection
        self._quoted_schema = _quote_identifier(schema)

    async def search(
        self,
        query: str,
        context: TrustedSearchContext,
        filters: SearchFilters,
        limit: int,
    ) -> KeywordSearchResponse:
        """Build an authorization-first snapshot and return its BM25 matches."""
        started = perf_counter()
        try:
            rows = await self._load_authorized_chunks(context, filters)
        except asyncpg.PostgresError as error:
            logger.exception("keyword_retrieval_database_failed tenant_id=%s", context.tenant_id)
            raise RetrievalError("Keyword retrieval database query failed") from error
        chunks = tuple(self._chunk_from_row(row) for row in rows)
        results = BM25Index(chunks).search(query, limit)
        elapsed = perf_counter() - started
        logger.info(
            "keyword_retrieval_complete tenant_id=%s candidates=%d results=%d seconds=%.6f",
            context.tenant_id, len(chunks), len(results), elapsed,
        )
        return KeywordSearchResponse(results, len(chunks), elapsed)

    async def _load_authorized_chunks(
        self, context: TrustedSearchContext, filters: SearchFilters
    ) -> list[asyncpg.Record]:
        """Apply access and lifecycle restrictions before text leaves PostgreSQL."""
        historical = context.historical
        historical_document = historical.document_id if historical else None
        historical_version = historical.version if historical else None
        async with self._connection.transaction():
            await self._connection.execute(f"SET LOCAL search_path TO {self._quoted_schema}, public")
            return await self._connection.fetch(
                """
                SELECT c.chunk_id, c.text, c.heading_path, c.start_page, c.end_page,
                       c.source_block_ids, d.tenant_id, d.document_id,
                       v.version AS document_version, v.title, v.source_path,
                       v.source_format, v.visibility, v.status, v.effective_date
                FROM documents d
                JOIN document_versions v ON v.document_fk=d.id
                JOIN document_chunks c ON c.document_version_fk=v.id
                WHERE d.tenant_id=$1
                  AND v.visibility=ANY($2::text[])
                  AND (($3::text IS NULL AND v.status='active') OR
                       ($3::text IS NOT NULL AND v.status='archived'
                        AND d.document_id=$3 AND v.version=$4))
                  AND ($5::text IS NULL OR d.document_id=$5)
                  AND ($6::text IS NULL OR EXISTS (
                      SELECT 1 FROM unnest(c.heading_path) AS heading
                      WHERE heading=$6 OR heading LIKE ('[' || $6 || ']%')
                  ))
                  AND ($7::date IS NULL OR v.effective_date >= $7::date)
                  AND ($8::date IS NULL OR v.effective_date <= $8::date)
                ORDER BY c.chunk_id
                """,
                context.tenant_id,
                list(context.permitted_visibilities),
                historical_document,
                historical_version,
                filters.document_id,
                filters.section,
                filters.effective_date_from,
                filters.effective_date_to,
            )

    @staticmethod
    def _chunk_from_row(row: asyncpg.Record) -> LexicalChunk:
        """Map one authorized row into the immutable lexical contract."""
        return LexicalChunk(
            chunk_id=row["chunk_id"], text=row["text"], tenant_id=row["tenant_id"],
            document_id=row["document_id"], document_version=row["document_version"],
            title=row["title"], heading_path=tuple(row["heading_path"]),
            start_page=row["start_page"], end_page=row["end_page"],
            source_block_ids=tuple(row["source_block_ids"]), source_path=row["source_path"],
            source_format=row["source_format"], visibility=row["visibility"],
            status=row["status"], effective_date=row["effective_date"].isoformat(),
        )
