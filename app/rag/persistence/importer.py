"""Persist one validated RAG import plan atomically with strict idempotency.

The service accepts only prevalidated records. PostgreSQL uniqueness constraints,
per-document transaction locks, content verification, and final reconciliation
work together so retries cannot silently reuse conflicting data.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from time import perf_counter

import asyncpg
import numpy as np

from app.rag.embeddings.service import validate_vectors
from app.rag.persistence.artifacts import (
    ChunkImport,
    EmbeddingImport,
    ImportPlan,
    VersionImport,
)


IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class RagImportConflict(RuntimeError):
    """Report an existing database identity whose content differs from input."""


@dataclass(frozen=True)
class ImportCounts:
    """Separate newly inserted and safely reused rows by database entity."""

    documents_inserted: int
    documents_reused: int
    versions_inserted: int
    versions_reused: int
    chunks_inserted: int
    chunks_reused: int
    embeddings_inserted: int
    embeddings_reused: int


@dataclass(frozen=True)
class ImportResult:
    """Return committed import counts and measured database duration."""

    counts: ImportCounts
    database_seconds: float


def _quote_identifier(identifier: str) -> str:
    """Quote a schema name after validation at the importer boundary."""
    if not IDENTIFIER_PATTERN.fullmatch(identifier):
        raise ValueError("Schema must be an unquoted PostgreSQL identifier")
    return f'"{identifier}"'


def _vector_text(vector: np.ndarray) -> str:
    """Serialize float32 components with enough precision for exact round trips."""
    values = np.asarray(vector, dtype=np.float32)
    return "[" + ",".join(format(float(value), ".9g") for value in values) + "]"


def _record_matches(record: asyncpg.Record, expected: dict[str, object]) -> bool:
    """Compare selected scalar database fields with their validated source values."""
    return all(record[name] == value for name, value in expected.items())


async def _lock_documents(
    connection: asyncpg.Connection,
    plan: ImportPlan,
) -> None:
    """Serialize imports sharing document identities without a database-wide lock."""
    for document in sorted(
        plan.documents, key=lambda item: (item.tenant_id, item.document_id)
    ):
        # Sorted plan identities give concurrent multi-document imports one lock
        # order, avoiding deadlocks while allowing disjoint imports to proceed.
        await connection.execute(
            "SELECT pg_advisory_xact_lock(hashtext($1), hashtext($2))",
            document.tenant_id,
            document.document_id,
        )


def _validate_plan(plan: ImportPlan) -> None:
    """Defend the database boundary when callers construct a plan directly."""
    document_identities = {
        (document.tenant_id, document.document_id) for document in plan.documents
    }
    version_identities = {
        (version.tenant_id, version.document_id, version.version)
        for version in plan.versions
    }
    chunk_ids = [chunk.chunk_id for chunk in plan.chunks]
    embedding_ids = [embedding.chunk_id for embedding in plan.embeddings]
    if len(document_identities) != len(plan.documents):
        raise ValueError("Import plan contains duplicate document identities")
    if len(version_identities) != len(plan.versions):
        raise ValueError("Import plan contains duplicate version identities")
    if len(set(chunk_ids)) != len(chunk_ids) or len(set(embedding_ids)) != len(
        embedding_ids
    ):
        raise ValueError("Import plan contains duplicate chunk or embedding identities")
    if set(chunk_ids) != set(embedding_ids):
        raise ValueError("Import plan must contain one embedding per chunk identity")
    if any(
        (version.tenant_id, version.document_id) not in document_identities
        for version in plan.versions
    ):
        raise ValueError("Import plan version has no parent document")
    if any(
        (chunk.tenant_id, chunk.document_id, chunk.document_version)
        not in version_identities
        for chunk in plan.chunks
    ):
        raise ValueError("Import plan chunk has no parent version")
    first = plan.embeddings[0]
    contract = (
        first.model_id,
        first.model_revision,
        first.embedding_config_fingerprint,
        first.document_instruction,
        first.query_instruction,
        first.normalize_embeddings,
        first.encoding_config_version,
    )
    if any(
        (
            embedding.model_id,
            embedding.model_revision,
            embedding.embedding_config_fingerprint,
            embedding.document_instruction,
            embedding.query_instruction,
            embedding.normalize_embeddings,
            embedding.encoding_config_version,
        )
        != contract
        for embedding in plan.embeddings
    ):
        raise ValueError("Import plan mixes incompatible embedding contracts")
    if (plan.model_id, plan.model_revision) != contract[:2]:
        raise ValueError("Import plan summary conflicts with embedding rows")
    validate_vectors(
        np.stack([embedding.vector for embedding in plan.embeddings]),
        len(plan.embeddings),
        plan.dimensions,
        first.normalize_embeddings,
    )


async def _insert_or_verify_document(
    connection: asyncpg.Connection,
    tenant_id: str,
    document_id: str,
) -> tuple[int, bool]:
    """Insert a logical document or reuse its exact tenant-scoped identity."""
    inserted = await connection.fetchval(
        """
        INSERT INTO documents (tenant_id, document_id)
        VALUES ($1, $2)
        ON CONFLICT (tenant_id, document_id) DO NOTHING
        RETURNING id
        """,
        tenant_id,
        document_id,
    )
    if inserted is not None:
        return inserted, True
    existing = await connection.fetchval(
        "SELECT id FROM documents WHERE tenant_id=$1 AND document_id=$2",
        tenant_id,
        document_id,
    )
    if existing is None:
        raise RagImportConflict(f"Document identity disappeared: {tenant_id}/{document_id}")
    return existing, False


async def _insert_or_verify_version(
    connection: asyncpg.Connection,
    document_pk: int,
    version: VersionImport,
) -> tuple[int, bool]:
    """Insert version metadata or reject any conflicting historical identity."""
    values = (
        document_pk,
        version.version,
        version.title,
        version.effective_date,
        version.status,
        version.visibility,
        version.source_sha256,
        version.source_path,
        version.source_format,
        version.synthetic,
        version.topic_family,
        version.word_count,
        version.pdf_pages,
        version.section_count,
        version.scope_note,
        version.expected_chunk_count,
        version.expected_chunk_ids_fingerprint,
    )
    inserted = await connection.fetchval(
        """
        INSERT INTO document_versions (
            document_fk, version, title, effective_date, status, visibility,
            source_sha256, source_path, source_format, synthetic, topic_family,
            word_count, pdf_pages, section_count, scope_note,
            expected_chunk_count, expected_chunk_ids_fingerprint
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
            $16, $17
        )
        ON CONFLICT (document_fk, version) DO NOTHING
        RETURNING id
        """,
        *values,
    )
    if inserted is not None:
        return inserted, True
    existing = await connection.fetchrow(
        """
        SELECT id, title, effective_date, status, visibility, source_sha256,
               source_path, source_format, synthetic, topic_family, word_count,
               pdf_pages, section_count, scope_note, expected_chunk_count,
               expected_chunk_ids_fingerprint
        FROM document_versions
        WHERE document_fk=$1 AND version=$2
        """,
        document_pk,
        version.version,
    )
    if existing is None or not _record_matches(
        existing,
        {
            "title": version.title,
            "effective_date": version.effective_date,
            "status": version.status,
            "visibility": version.visibility,
            "source_sha256": version.source_sha256,
            "source_path": version.source_path,
            "source_format": version.source_format,
            "synthetic": version.synthetic,
            "topic_family": version.topic_family,
            "word_count": version.word_count,
            "pdf_pages": version.pdf_pages,
            "section_count": version.section_count,
            "scope_note": version.scope_note,
            "expected_chunk_count": version.expected_chunk_count,
            "expected_chunk_ids_fingerprint": version.expected_chunk_ids_fingerprint,
        },
    ):
        raise RagImportConflict(
            "Conflicting document version: "
            f"{version.tenant_id}/{version.document_id}/{version.version}"
        )
    return existing["id"], False


async def _insert_or_verify_chunk(
    connection: asyncpg.Connection,
    version_pk: int,
    chunk: ChunkImport,
) -> bool:
    """Insert one chunk or validate every persisted content/provenance field."""
    configuration_json = json.dumps(
        chunk.chunking_configuration, sort_keys=True, separators=(",", ":")
    )
    inserted = await connection.fetchval(
        """
        INSERT INTO document_chunks (
            chunk_id, document_version_fk, chunk_index, text, token_count,
            heading_path, start_page, end_page, source_block_ids, tokenizer_name,
            tokenizer_version, chunking_config_version, chunking_configuration,
            chunking_config_fingerprint
        ) VALUES (
            $1, $2, $3, $4, $5, $6::text[], $7, $8, $9::integer[], $10,
            $11, $12, $13::jsonb, $14
        )
        ON CONFLICT (chunk_id) DO NOTHING
        RETURNING chunk_id
        """,
        chunk.chunk_id,
        version_pk,
        chunk.chunk_index,
        chunk.text,
        chunk.token_count,
        list(chunk.heading_path),
        chunk.start_page,
        chunk.end_page,
        list(chunk.source_block_ids),
        chunk.tokenizer_name,
        chunk.tokenizer_version,
        chunk.chunking_config_version,
        configuration_json,
        chunk.chunking_config_fingerprint,
    )
    if inserted is not None:
        return True
    existing = await connection.fetchrow(
        """
        SELECT document_version_fk, chunk_index, text, token_count, heading_path,
               start_page, end_page, source_block_ids, tokenizer_name,
               tokenizer_version, chunking_config_version,
               chunking_configuration::text AS chunking_configuration,
               chunking_config_fingerprint
        FROM document_chunks WHERE chunk_id=$1
        """,
        chunk.chunk_id,
    )
    expected = {
        "document_version_fk": version_pk,
        "chunk_index": chunk.chunk_index,
        "text": chunk.text,
        "token_count": chunk.token_count,
        "heading_path": list(chunk.heading_path),
        "start_page": chunk.start_page,
        "end_page": chunk.end_page,
        "source_block_ids": list(chunk.source_block_ids),
        "tokenizer_name": chunk.tokenizer_name,
        "tokenizer_version": chunk.tokenizer_version,
        "chunking_config_version": chunk.chunking_config_version,
        "chunking_config_fingerprint": chunk.chunking_config_fingerprint,
    }
    if (
        existing is None
        or not _record_matches(existing, expected)
        or json.loads(existing["chunking_configuration"])
        != chunk.chunking_configuration
    ):
        raise RagImportConflict(f"Conflicting chunk identity: {chunk.chunk_id}")
    return False


async def _insert_or_verify_embedding(
    connection: asyncpg.Connection,
    embedding: EmbeddingImport,
) -> bool:
    """Insert one vector or verify the complete existing encoding and value."""
    vector_text = _vector_text(embedding.vector)
    inserted = await connection.fetchval(
        """
        INSERT INTO chunk_embeddings (
            chunk_id, embedding, embedding_model_id, embedding_model_revision,
            embedding_config_fingerprint, input_fingerprint,
            document_instruction, query_instruction, normalize_embeddings,
            encoding_config_version
        ) VALUES ($1, $2::public.vector, $3, $4, $5, $6, $7, $8, $9, $10)
        ON CONFLICT (
            chunk_id, embedding_model_id, embedding_model_revision,
            embedding_config_fingerprint
        ) DO NOTHING
        RETURNING id
        """,
        embedding.chunk_id,
        vector_text,
        embedding.model_id,
        embedding.model_revision,
        embedding.embedding_config_fingerprint,
        embedding.input_fingerprint,
        embedding.document_instruction,
        embedding.query_instruction,
        embedding.normalize_embeddings,
        embedding.encoding_config_version,
    )
    if inserted is not None:
        return True
    existing = await connection.fetchrow(
        """
        SELECT input_fingerprint, document_instruction, query_instruction,
               normalize_embeddings, encoding_config_version,
               embedding::text AS embedding
        FROM chunk_embeddings
        WHERE chunk_id=$1 AND embedding_model_id=$2
          AND embedding_model_revision=$3 AND embedding_config_fingerprint=$4
        """,
        embedding.chunk_id,
        embedding.model_id,
        embedding.model_revision,
        embedding.embedding_config_fingerprint,
    )
    expected = {
        "input_fingerprint": embedding.input_fingerprint,
        "document_instruction": embedding.document_instruction,
        "query_instruction": embedding.query_instruction,
        "normalize_embeddings": embedding.normalize_embeddings,
        "encoding_config_version": embedding.encoding_config_version,
    }
    if existing is None or not _record_matches(existing, expected):
        raise RagImportConflict(f"Conflicting embedding identity: {embedding.chunk_id}")
    stored = np.fromstring(existing["embedding"].strip("[]"), sep=",", dtype=np.float32)
    if not np.array_equal(stored, np.asarray(embedding.vector, dtype=np.float32)):
        raise RagImportConflict(f"Conflicting embedding vector: {embedding.chunk_id}")
    return False


async def _reconcile(connection: asyncpg.Connection, plan: ImportPlan) -> None:
    """Prove that every exact source identity exists before the transaction commits."""
    document_identities = {
        (document.tenant_id, document.document_id) for document in plan.documents
    }
    stored_documents = {
        (row["tenant_id"], row["document_id"])
        for row in await connection.fetch(
            """
            SELECT expected.tenant_id, expected.document_id
            FROM unnest($1::text[], $2::text[])
                 AS expected(tenant_id, document_id)
            JOIN documents d USING (tenant_id, document_id)
            """,
            [identity[0] for identity in sorted(document_identities)],
            [identity[1] for identity in sorted(document_identities)],
        )
    }
    if stored_documents != document_identities:
        raise RagImportConflict("Document reconciliation found missing identities")

    version_identities = {
        (version.tenant_id, version.document_id, version.version)
        for version in plan.versions
    }
    sorted_versions = sorted(version_identities)
    stored_versions = {
        (row["tenant_id"], row["document_id"], row["version"])
        for row in await connection.fetch(
            """
            SELECT expected.tenant_id, expected.document_id, expected.version
            FROM unnest($1::text[], $2::text[], $3::text[])
                 AS expected(tenant_id, document_id, version)
            JOIN documents d USING (tenant_id, document_id)
            JOIN document_versions v
              ON v.document_fk=d.id AND v.version=expected.version
            """,
            [identity[0] for identity in sorted_versions],
            [identity[1] for identity in sorted_versions],
            [identity[2] for identity in sorted_versions],
        )
    }
    if stored_versions != version_identities:
        raise RagImportConflict("Version reconciliation found missing identities")

    chunk_ids = [chunk.chunk_id for chunk in plan.chunks]
    stored_chunks = set(
        await connection.fetchval(
            "SELECT array_agg(chunk_id ORDER BY chunk_id) "
            "FROM document_chunks WHERE chunk_id = ANY($1::text[])",
            chunk_ids,
        )
        or []
    )
    if stored_chunks != set(chunk_ids):
        raise RagImportConflict("Chunk reconciliation found missing source identities")

    embedding_ids = [embedding.chunk_id for embedding in plan.embeddings]
    stored_embeddings = set(
        await connection.fetchval(
            """
            SELECT array_agg(chunk_id ORDER BY chunk_id)
            FROM chunk_embeddings
            WHERE chunk_id = ANY($1::text[])
              AND embedding_model_id=$2 AND embedding_model_revision=$3
              AND embedding_config_fingerprint=$4
            """,
            embedding_ids,
            plan.model_id,
            plan.model_revision,
            plan.embeddings[0].embedding_config_fingerprint,
        )
        or []
    )
    if stored_embeddings != set(embedding_ids):
        raise RagImportConflict("Embedding reconciliation found missing source identities")
    dimensions = await connection.fetchval(
        """
        SELECT bool_and(public.vector_dims(embedding)=$1)
        FROM chunk_embeddings
        WHERE chunk_id = ANY($2::text[]) AND embedding_model_id=$3
          AND embedding_model_revision=$4 AND embedding_config_fingerprint=$5
        """,
        plan.dimensions,
        embedding_ids,
        plan.model_id,
        plan.model_revision,
        plan.embeddings[0].embedding_config_fingerprint,
    )
    if dimensions is not True:
        raise RagImportConflict("Embedding reconciliation found incompatible dimensions")


async def import_rag_embeddings(
    connection: asyncpg.Connection,
    plan: ImportPlan,
    *,
    schema: str = "public",
) -> ImportResult:
    """Insert or verify one complete plan in a single PostgreSQL transaction."""
    if not plan.embeddings or len(plan.chunks) != len(plan.embeddings):
        raise ValueError("Import plan must contain one embedding per chunk")
    _validate_plan(plan)
    started_at = perf_counter()
    quoted_schema = _quote_identifier(schema)
    inserted = {"documents": 0, "versions": 0, "chunks": 0, "embeddings": 0}
    reused = {"documents": 0, "versions": 0, "chunks": 0, "embeddings": 0}

    async with connection.transaction():
        await connection.execute(f"SET LOCAL search_path TO {quoted_schema}, public")
        await _lock_documents(connection, plan)
        document_pks: dict[tuple[str, str], int] = {}
        for document in plan.documents:
            document_pk, was_inserted = await _insert_or_verify_document(
                connection, document.tenant_id, document.document_id
            )
            document_pks[(document.tenant_id, document.document_id)] = document_pk
            (inserted if was_inserted else reused)["documents"] += 1

        version_pks: dict[tuple[str, str, str], int] = {}
        # Existing non-staged versions are already published or retired; their
        # chunk sets may only be re-verified, never extended, by an import.
        frozen_versions: set[tuple[str, str, str]] = set()
        for version in plan.versions:
            document_pk = document_pks[(version.tenant_id, version.document_id)]
            version_pk, was_inserted = await _insert_or_verify_version(
                connection, document_pk, version
            )
            version_key = (version.tenant_id, version.document_id, version.version)
            version_pks[version_key] = version_pk
            (inserted if was_inserted else reused)["versions"] += 1
            # Reuse already proved the stored status equals version.status.
            if not was_inserted and version.status != "staged":
                frozen_versions.add(version_key)

        for chunk in plan.chunks:
            version_key = (chunk.tenant_id, chunk.document_id, chunk.document_version)
            was_inserted = await _insert_or_verify_chunk(
                connection, version_pks[version_key], chunk
            )
            if was_inserted and version_key in frozen_versions:
                # Re-chunking an unchanged source yields new chunk IDs under the
                # same version identity. Adding them here would silently change
                # searchable content outside staging and publication. Raising
                # inside the transaction rolls back every write from this plan.
                raise RagImportConflict(
                    "Cannot add chunks to existing "
                    f"{chunk.tenant_id}/{chunk.document_id}/{chunk.document_version}; "
                    "stage a new version instead"
                )
            (inserted if was_inserted else reused)["chunks"] += 1

        for embedding in plan.embeddings:
            was_inserted = await _insert_or_verify_embedding(connection, embedding)
            (inserted if was_inserted else reused)["embeddings"] += 1

        await _reconcile(connection, plan)

    counts = ImportCounts(
        documents_inserted=inserted["documents"],
        documents_reused=reused["documents"],
        versions_inserted=inserted["versions"],
        versions_reused=reused["versions"],
        chunks_inserted=inserted["chunks"],
        chunks_reused=reused["chunks"],
        embeddings_inserted=inserted["embeddings"],
        embeddings_reused=reused["embeddings"],
    )
    return ImportResult(counts=counts, database_seconds=perf_counter() - started_at)
