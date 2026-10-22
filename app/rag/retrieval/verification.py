"""Reconcile dense artifacts with PostgreSQL and compare exact retrieval results.

This module is deliberately read-only. Mutation guarantees remain covered by
the existing importer integration tests, while this boundary verifies a chosen
immutable artifact run against a populated database and evaluates rankings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path
from platform import platform, python_version
from statistics import median
from time import perf_counter
from typing import Sequence

import asyncpg
import numpy as np

from app.rag.embeddings.models import EmbeddingRecord
from app.rag.persistence.artifacts import ImportPlan
from app.rag.persistence.lifecycle import EmbeddingProfile
from app.rag.retrieval.dense import DenseRetriever, _quote_identifier
from app.rag.retrieval.models import DenseSearchRequest, TrustedSearchContext


@dataclass(frozen=True)
class IntegritySummary:
    """Report exact source-to-database reconciliation counts and status."""

    source_embeddings_expected: int
    matching_database_embeddings: int
    missing_embeddings: tuple[str, ...]
    duplicate_identities: tuple[str, ...]
    dimension_mismatches: tuple[str, ...]
    vector_mismatches: tuple[str, ...]
    metadata_mismatches: tuple[str, ...]
    relationship_mismatches: tuple[str, ...]
    status: str

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible representation."""
        return asdict(self)


@dataclass(frozen=True)
class RankingComparison:
    """Describe one paired PostgreSQL and NumPy exact-search result."""

    database_chunk_ids: tuple[str, ...]
    numpy_chunk_ids: tuple[str, ...]
    exact_order_match: bool
    difference_classification: str
    maximum_score_difference: float
    candidate_count: int
    query_encoding_seconds: float
    database_seconds: float
    total_seconds: float

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible representation."""
        return asdict(self)


def percentile_95(samples: Sequence[float]) -> float | None:
    """Return the nearest-rank p95, or no value when no samples exist."""
    if not samples:
        return None
    ordered = sorted(samples)
    return ordered[max(0, ceil(0.95 * len(ordered)) - 1)]


def latency_summary(samples: Sequence[float]) -> dict[str, float | int | None]:
    """Summarize measured durations without implying unmeasured performance."""
    if not samples:
        return {"count": 0, "median_seconds": None, "p95_seconds": None,
                "minimum_seconds": None, "maximum_seconds": None}
    return {
        "count": len(samples),
        "median_seconds": median(samples),
        "p95_seconds": percentile_95(samples),
        "minimum_seconds": min(samples),
        "maximum_seconds": max(samples),
    }


def runtime_environment() -> dict[str, str]:
    """Capture portable runtime details available without external commands."""
    return {"python_version": python_version(), "platform": platform()}


async def reconcile_import_plan(
    connection: asyncpg.Connection,
    plan: ImportPlan,
    *,
    schema: str = "public",
    vector_absolute_tolerance: float = 1e-6,
) -> IntegritySummary:
    """Compare every expected imported identity, vector, and provenance field.

    The query is constrained to the exact artifact identities and embedding
    profile, so unrelated legitimate database rows do not create false errors.
    """
    if vector_absolute_tolerance <= 0:
        raise ValueError("vector_absolute_tolerance must be positive")
    quoted_schema = _quote_identifier(schema)
    expected = {item.chunk_id: item for item in plan.embeddings}
    chunk_metadata = {item.chunk_id: item for item in plan.chunks}
    version_metadata = {
        (item.tenant_id, item.document_id, item.version): item
        for item in plan.versions
    }
    if not expected:
        raise ValueError("Import plan contains no embeddings")

    async with connection.transaction():
        await connection.execute(f"SET LOCAL search_path TO {quoted_schema}, public")
        rows = await connection.fetch(
            """
            SELECT e.chunk_id, e.embedding::text AS embedding,
                   public.vector_dims(e.embedding) AS dimensions,
                   e.embedding_model_id, e.embedding_model_revision,
                   e.embedding_config_fingerprint, e.input_fingerprint,
                   e.document_instruction, e.query_instruction,
                   e.normalize_embeddings, e.encoding_config_version,
                   c.chunk_index, c.text, c.token_count, c.heading_path,
                   c.start_page, c.end_page, c.source_block_ids,
                   c.tokenizer_name, c.tokenizer_version,
                   c.chunking_config_version, c.chunking_config_fingerprint,
                   d.tenant_id, d.document_id, v.version, v.title,
                   v.status, v.visibility, v.source_sha256, v.source_path,
                   v.source_format
            FROM chunk_embeddings e
            JOIN document_chunks c ON c.chunk_id=e.chunk_id
            JOIN document_versions v ON v.id=c.document_version_fk
            JOIN documents d ON d.id=v.document_fk
            WHERE e.chunk_id=ANY($1::text[])
              AND e.embedding_model_id=$2
              AND e.embedding_model_revision=$3
              AND e.embedding_config_fingerprint=$4
            ORDER BY e.chunk_id
            """,
            sorted(expected),
            plan.model_id,
            plan.model_revision,
            plan.embeddings[0].embedding_config_fingerprint,
        )

    occurrences: dict[str, int] = {}
    for row in rows:
        occurrences[row["chunk_id"]] = occurrences.get(row["chunk_id"], 0) + 1
    duplicate_identities = tuple(
        sorted(chunk_id for chunk_id, count in occurrences.items() if count > 1)
    )
    missing = tuple(sorted(set(expected) - set(occurrences)))
    dimensions: list[str] = []
    vectors: list[str] = []
    metadata: list[str] = []
    relationships: list[str] = []
    matching = 0

    for row in rows:
        chunk_id = row["chunk_id"]
        source_embedding = expected[chunk_id]
        source_chunk = chunk_metadata[chunk_id]
        source_version = version_metadata[
            (source_chunk.tenant_id, source_chunk.document_id,
             source_chunk.document_version)
        ]
        row_has_error = False
        if row["dimensions"] != plan.dimensions:
            dimensions.append(chunk_id)
            row_has_error = True
        stored_vector = np.fromstring(row["embedding"].strip("[]"), sep=",")
        source_vector = np.asarray(source_embedding.vector, dtype=np.float64)
        if (
            stored_vector.shape != source_vector.shape
            or not np.isfinite(stored_vector).all()
            or not np.allclose(stored_vector, source_vector, rtol=0.0,
                               atol=vector_absolute_tolerance)
        ):
            vectors.append(chunk_id)
            row_has_error = True
        embedding_values = (
            row["embedding_model_id"], row["embedding_model_revision"],
            row["embedding_config_fingerprint"], row["input_fingerprint"],
            row["document_instruction"], row["query_instruction"],
            row["normalize_embeddings"], row["encoding_config_version"],
        )
        expected_embedding_values = (
            source_embedding.model_id, source_embedding.model_revision,
            source_embedding.embedding_config_fingerprint,
            source_embedding.input_fingerprint,
            source_embedding.document_instruction, source_embedding.query_instruction,
            source_embedding.normalize_embeddings,
            source_embedding.encoding_config_version,
        )
        chunk_values = (
            row["chunk_index"], row["text"], row["token_count"],
            tuple(row["heading_path"]), row["start_page"], row["end_page"],
            tuple(row["source_block_ids"]), row["tokenizer_name"],
            row["tokenizer_version"], row["chunking_config_version"],
            row["chunking_config_fingerprint"],
        )
        expected_chunk_values = (
            source_chunk.chunk_index, source_chunk.text, source_chunk.token_count,
            source_chunk.heading_path, source_chunk.start_page, source_chunk.end_page,
            source_chunk.source_block_ids, source_chunk.tokenizer_name,
            source_chunk.tokenizer_version, source_chunk.chunking_config_version,
            source_chunk.chunking_config_fingerprint,
        )
        if embedding_values != expected_embedding_values or chunk_values != expected_chunk_values:
            metadata.append(chunk_id)
            row_has_error = True
        relationship_values = (
            row["tenant_id"], row["document_id"], row["version"], row["title"],
            row["status"], row["visibility"], row["source_sha256"],
            row["source_path"], row["source_format"],
        )
        expected_relationship_values = (
            source_version.tenant_id, source_version.document_id,
            source_version.version, source_version.title, source_version.status,
            source_version.visibility, source_version.source_sha256,
            source_version.source_path, source_version.source_format,
        )
        if relationship_values != expected_relationship_values:
            relationships.append(chunk_id)
            row_has_error = True
        if not row_has_error:
            matching += 1

    failures = (
        missing, duplicate_identities, dimensions, vectors, metadata, relationships
    )
    return IntegritySummary(
        source_embeddings_expected=len(expected),
        matching_database_embeddings=matching,
        missing_embeddings=missing,
        duplicate_identities=duplicate_identities,
        dimension_mismatches=tuple(sorted(set(dimensions))),
        vector_mismatches=tuple(sorted(set(vectors))),
        metadata_mismatches=tuple(sorted(set(metadata))),
        relationship_mismatches=tuple(sorted(set(relationships))),
        status="PASS" if not any(failures) and matching == len(expected) else "FAIL",
    )


def exact_numpy_search(
    query_vector: np.ndarray,
    vectors: np.ndarray,
    records: Sequence[EmbeddingRecord],
    context: TrustedSearchContext,
    *,
    top_k: int,
) -> list[tuple[str, float]]:
    """Rank the database-equivalent authorized candidate set in NumPy."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    query = np.asarray(query_vector, dtype=np.float64)
    if query.ndim != 1 or not np.isfinite(query).all() or np.linalg.norm(query) == 0:
        raise ValueError("query_vector must be a finite nonzero vector")
    historical = context.historical
    candidate_indexes = [
        index for index, record in enumerate(records)
        if record.tenant_id == context.tenant_id
        and record.access_scope in context.permitted_visibilities
        and (
            (historical is None and record.document_status == "active")
            or (
                historical is not None
                and record.document_status == "archived"
                and record.document_id == historical.document_id
                and record.document_version == historical.version
            )
        )
    ]
    if not candidate_indexes:
        return []
    candidate_vectors = np.asarray(vectors[candidate_indexes], dtype=np.float64)
    similarities = (candidate_vectors @ query) / (
        np.linalg.norm(candidate_vectors, axis=1) * np.linalg.norm(query)
    )
    # PostgreSQL uses ascending distance then chunk_id. Since distance is
    # 1-similarity, NumPy must use descending similarity and the same ID key.
    ranked = sorted(
        range(len(candidate_indexes)),
        key=lambda position: (
            -float(similarities[position]),
            records[candidate_indexes[position]].chunk_id,
        ),
    )[:top_k]
    return [
        (records[candidate_indexes[position]].chunk_id,
         float(similarities[position]))
        for position in ranked
    ]


async def compare_exact_ranking(
    retriever: DenseRetriever,
    *,
    query_vector: np.ndarray,
    vectors: np.ndarray,
    records: Sequence[EmbeddingRecord],
    context: TrustedSearchContext,
    profile: EmbeddingProfile,
    top_k: int,
    score_absolute_tolerance: float = 1e-6,
) -> RankingComparison:
    """Reuse one query vector for paired PostgreSQL and NumPy exact search."""
    if score_absolute_tolerance <= 0:
        raise ValueError("score_absolute_tolerance must be positive")
    started = perf_counter()
    numpy_results = exact_numpy_search(
        query_vector, vectors, records, context, top_k=top_k
    )
    response = await retriever.search(
        DenseSearchRequest(
            context=context, profile=profile, query_vector=query_vector, top_k=top_k
        )
    )
    database_ids = tuple(item.chunk_id for item in response.results)
    numpy_ids = tuple(item[0] for item in numpy_results)
    numpy_scores = dict(numpy_results)
    database_scores = {item.chunk_id: item.similarity for item in response.results}
    score_differences = [
        abs(item.similarity - numpy_scores[item.chunk_id])
        for item in response.results if item.chunk_id in numpy_scores
    ]
    exact_match = database_ids == numpy_ids
    same_identity_set = set(database_ids) == set(numpy_ids)
    # Equal or near-equal scores can swap across numerical implementations.
    # Classify that explicitly rather than hiding it or calling it a SQL defect.
    mismatched_ids = {
        chunk_id
        for position, chunk_id in enumerate(database_ids)
        if position >= len(numpy_ids) or numpy_ids[position] != chunk_id
    } | {
        chunk_id
        for position, chunk_id in enumerate(numpy_ids)
        if position >= len(database_ids) or database_ids[position] != chunk_id
    }
    mismatched_scores = [
        numpy_scores[chunk_id] for chunk_id in mismatched_ids
        if chunk_id in numpy_scores
    ]
    explained_tie = (
        not exact_match
        and same_identity_set
        and mismatched_scores
        and max(mismatched_scores) - min(mismatched_scores)
        <= score_absolute_tolerance
        and all(
            abs(database_scores[chunk_id] - numpy_scores[chunk_id])
            <= score_absolute_tolerance
            for chunk_id in mismatched_ids
        )
    )
    classification = (
        "exact" if exact_match else
        "explained_numerical_tie" if explained_tie else
        "unexplained"
    )
    return RankingComparison(
        database_chunk_ids=database_ids,
        numpy_chunk_ids=numpy_ids,
        exact_order_match=exact_match,
        difference_classification=classification,
        maximum_score_difference=max(score_differences, default=0.0),
        candidate_count=response.candidate_count,
        query_encoding_seconds=response.query_encoding_seconds,
        database_seconds=response.database_seconds,
        total_seconds=perf_counter() - started,
    )


async def database_versions(connection: asyncpg.Connection) -> dict[str, str]:
    """Read PostgreSQL and pgvector versions for the evaluation report."""
    return {
        "postgresql": await connection.fetchval("SHOW server_version"),
        "pgvector": await connection.fetchval(
            "SELECT extversion FROM pg_extension WHERE extname='vector'"
        ) or "NOT INSTALLED",
    }


def artifact_run_fingerprint(run_directory: Path, report: dict[str, object]) -> str:
    """Return a stable artifact identity from its immutable checksums."""
    checksums = report.get("artifact_sha256")
    if not isinstance(checksums, dict) or not checksums:
        raise ValueError(f"Artifact run lacks checksums: {run_directory}")
    vectors_hash = checksums.get("vectors.npy")
    metadata_hash = checksums.get("metadata.jsonl")
    if not isinstance(vectors_hash, str) or not isinstance(metadata_hash, str):
        raise ValueError("Artifact report has invalid checksums")
    return f"{run_directory.name}:{vectors_hash}:{metadata_hash}"
