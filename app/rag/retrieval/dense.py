"""Exact PostgreSQL/pgvector dense retrieval with authorization-first filters."""

from __future__ import annotations

import logging
import re
from time import perf_counter
from typing import Protocol

import asyncpg
import numpy as np

from app.rag.embeddings.models import DEFAULT_DIMENSIONS
from app.rag.retrieval.models import (
    DenseSearchRequest,
    DenseSearchResponse,
    DenseSearchResult,
)


logger = logging.getLogger(__name__)
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class QueryEncoder(Protocol):
    """Represent the existing Phase 5 query encoder dependency."""

    config: object

    def encode_query(self, query: str) -> object:
        """Encode one already validated natural-language query."""


class RetrievalError(RuntimeError):
    """Report a safe dense-retrieval failure to internal callers."""


def _quote_identifier(identifier: str) -> str:
    """Quote a schema name only after strict identifier validation."""
    if not IDENTIFIER_PATTERN.fullmatch(identifier):
        raise ValueError("Schema must be an unquoted PostgreSQL identifier")
    return f'"{identifier}"'


def _validated_vector(vector: object) -> np.ndarray:
    """Return one finite, nonzero 384-dimensional float32 query vector."""
    try:
        values = np.asarray(vector, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("Query vector must contain numeric values") from error
    if values.shape != (DEFAULT_DIMENSIONS,):
        raise ValueError(f"Query vector must have {DEFAULT_DIMENSIONS} dimensions")
    if not np.isfinite(values).all():
        raise ValueError("Query vector must contain only finite values")
    if float(np.linalg.norm(values)) == 0.0:
        raise ValueError("Query vector must not be zero")
    return values


def _vector_text(vector: np.ndarray) -> str:
    """Serialize a validated vector for an explicitly cast pgvector parameter."""
    return "[" + ",".join(format(float(value), ".9g") for value in vector) + "]"


class DenseRetriever:
    """Encode questions and execute exact, scoped cosine search in PostgreSQL."""

    def __init__(
        self,
        connection: asyncpg.Connection,
        encoder: QueryEncoder | None,
        *,
        schema: str = "public",
    ) -> None:
        """Bind one database connection and reusable encoder to the service."""
        self._connection = connection
        self._encoder = encoder
        self._quoted_schema = _quote_identifier(schema)

    async def search(self, request: DenseSearchRequest) -> DenseSearchResponse:
        """Return exact cosine-ranked chunks within the trusted SQL scope."""
        total_started = perf_counter()
        encoding_started = perf_counter()
        if request.query_vector is not None:
            query_vector = _validated_vector(request.query_vector)
        else:
            if self._encoder is None:
                raise RetrievalError("A query encoder is required for question search")
            self._validate_encoder_profile(request)
            try:
                query_vector = _validated_vector(
                    self._encoder.encode_query(request.question or "")
                )
            except ValueError:
                raise
            except Exception as error:
                raise RetrievalError("Query embedding generation failed") from error
        encoding_seconds = perf_counter() - encoding_started
        database_started = perf_counter()
        try:
            rows = await self._search_database(request, _vector_text(query_vector))
        except asyncpg.PostgresError as error:
            logger.exception(
                "dense_retrieval_database_failed tenant_id=%s top_k=%d",
                request.context.tenant_id,
                request.top_k,
            )
            raise RetrievalError("Dense retrieval database query failed") from error
        database_seconds = perf_counter() - database_started
        results = tuple(self._result_from_row(row) for row in rows)
        candidate_count = int(rows[0]["candidate_count"]) if rows else 0
        total_seconds = perf_counter() - total_started
        logger.info(
            "dense_retrieval_complete tenant_id=%s model_id=%s top_k=%d "
            "candidates=%d results=%d database_seconds=%.6f",
            request.context.tenant_id,
            request.profile.model_id,
            request.top_k,
            candidate_count,
            len(results),
            database_seconds,
        )
        return DenseSearchResponse(
            results=results,
            candidate_count=candidate_count,
            query_encoding_seconds=encoding_seconds,
            database_seconds=database_seconds,
            total_seconds=total_seconds,
        )

    def _validate_encoder_profile(self, request: DenseSearchRequest) -> None:
        """Prevent an encoder from querying a different persisted vector space."""
        config = getattr(self._encoder, "config", None)
        expected = (
            request.profile.model_id,
            request.profile.model_revision,
            request.profile.normalize_embeddings,
            request.profile.encoding_config_version,
            request.profile.query_instruction,
        )
        actual = (
            getattr(config, "model_id", None),
            getattr(config, "model_revision", None),
            getattr(config, "normalize_embeddings", None),
            getattr(config, "encoding_config_version", None),
            getattr(config, "query_instruction", None),
        )
        if actual != expected:
            raise ValueError("Query encoder is incompatible with the selected profile")

    async def _search_database(
        self, request: DenseSearchRequest, query_vector: str
    ) -> list[asyncpg.Record]:
        """Execute filtering and ranking in one parameterized SQL statement."""
        historical = request.context.historical
        # Current policy permits active content. Historical permission is
        # intentionally narrower: it permits only one exact archived version,
        # never staged or withdrawn content.
        document_id = historical.document_id if historical else None
        document_version = historical.version if historical else None
        async with self._connection.transaction():
            await self._connection.execute(
                f"SET LOCAL search_path TO {self._quoted_schema}, public"
            )
            return await self._connection.fetch(
                """
                WITH eligible AS MATERIALIZED (
                    SELECT
                        c.chunk_id, c.text AS chunk_text, c.heading_path,
                        c.start_page, c.end_page, c.source_block_ids,
                        d.tenant_id, d.document_id, v.version AS document_version,
                        v.title AS document_title, v.source_path, v.source_format,
                        v.visibility, v.status, v.effective_date, e.embedding, e.embedding_model_id,
                        e.embedding_model_revision, e.embedding_config_fingerprint
                    FROM documents d
                    JOIN document_versions v ON v.document_fk=d.id
                    JOIN document_chunks c ON c.document_version_fk=v.id
                    JOIN chunk_embeddings e ON e.chunk_id=c.chunk_id
                    WHERE d.tenant_id=$1
                      AND v.visibility=ANY($2::text[])
                      AND (($8::text IS NULL AND v.status='active') OR
                           ($8::text IS NOT NULL AND v.status='archived'
                            AND d.document_id=$8 AND v.version=$9))
                      AND e.embedding_model_id=$3
                      AND e.embedding_model_revision=$4
                      AND e.embedding_config_fingerprint=$5
                      AND e.normalize_embeddings=$6
                      AND e.encoding_config_version=$7
                      AND e.query_instruction=$12
                      AND public.vector_dims(e.embedding)=384
                      AND ($13::text IS NULL OR d.document_id=$13)
                      AND ($14::text IS NULL OR EXISTS (
                          SELECT 1 FROM unnest(c.heading_path) AS heading
                          WHERE heading=$14 OR heading LIKE ('[' || $14 || ']%')
                      ))
                      AND ($15::date IS NULL OR v.effective_date >= $15::date)
                      AND ($16::date IS NULL OR v.effective_date <= $16::date)
                )
                -- List result columns explicitly so the stored vector is used
                -- for ranking but never sent back with each result row.
                SELECT eligible.chunk_id, eligible.chunk_text, eligible.heading_path,
                       eligible.start_page, eligible.end_page,
                       eligible.source_block_ids, eligible.tenant_id,
                       eligible.document_id, eligible.document_version,
                       eligible.document_title, eligible.source_path,
                       eligible.source_format, eligible.visibility, eligible.status,
                       eligible.effective_date,
                       eligible.embedding_model_id, eligible.embedding_model_revision,
                       eligible.embedding_config_fingerprint,
                       eligible.embedding <=> $10::public.vector AS cosine_distance,
                       1 - (eligible.embedding <=> $10::public.vector) AS similarity,
                       count(*) OVER () AS candidate_count
                FROM eligible
                ORDER BY eligible.embedding <=> $10::public.vector, chunk_id
                LIMIT $11
                """,
                request.context.tenant_id,
                list(request.context.permitted_visibilities),
                request.profile.model_id,
                request.profile.model_revision,
                request.profile.config_fingerprint,
                request.profile.normalize_embeddings,
                request.profile.encoding_config_version,
                document_id,
                document_version,
                query_vector,
                request.top_k,
                request.profile.query_instruction,
                request.filters.document_id,
                request.filters.section,
                request.filters.effective_date_from,
                request.filters.effective_date_to,
            )

    @staticmethod
    def _result_from_row(row: asyncpg.Record) -> DenseSearchResult:
        """Map one trusted database row without inventing provenance fields."""
        return DenseSearchResult(
            chunk_id=row["chunk_id"],
            chunk_text=row["chunk_text"],
            similarity=float(row["similarity"]),
            cosine_distance=float(row["cosine_distance"]),
            tenant_id=row["tenant_id"],
            document_id=row["document_id"],
            document_version=row["document_version"],
            document_title=row["document_title"],
            heading_path=tuple(row["heading_path"]),
            start_page=row["start_page"],
            end_page=row["end_page"],
            source_block_ids=tuple(row["source_block_ids"]),
            source_path=row["source_path"],
            source_format=row["source_format"],
            visibility=row["visibility"],
            status=row["status"],
            effective_date=row["effective_date"].isoformat(),
            embedding_model_id=row["embedding_model_id"],
            embedding_model_revision=row["embedding_model_revision"],
            embedding_config_fingerprint=row["embedding_config_fingerprint"],
        )

