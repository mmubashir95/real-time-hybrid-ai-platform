"""Stage, publish, withdraw, and scope persisted RAG document versions."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import logging
import re
from typing import Literal

import asyncpg

from app.rag.embeddings.models import QUERY_INSTRUCTION
from app.rag.persistence.artifacts import ImportPlan
from app.rag.persistence.importer import ImportResult, import_rag_embeddings


logger = logging.getLogger(__name__)
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class DocumentLifecycleError(RuntimeError):
    """Base error for a rejected document lifecycle operation."""


class DocumentNotFound(DocumentLifecycleError):
    """Report a tenant-scoped logical document that does not exist."""


class VersionNotFound(DocumentLifecycleError):
    """Report a requested version that does not exist under the document."""


class LifecycleConflict(DocumentLifecycleError):
    """Report a stale request, invalid transition, or conflicting retry."""


class IncompleteVersion(DocumentLifecycleError):
    """Report a staged version whose chunks or embeddings are incomplete."""


@dataclass(frozen=True)
class EmbeddingProfile:
    """Identify one compatible embedding configuration used by retrieval."""

    model_id: str
    model_revision: str
    config_fingerprint: str
    normalize_embeddings: bool
    encoding_config_version: str
    query_instruction: str = QUERY_INSTRUCTION

    def __post_init__(self) -> None:
        """Reject incomplete or malformed embedding-space identities."""
        string_fields = (
            self.model_id,
            self.model_revision,
            self.config_fingerprint,
            self.encoding_config_version,
            self.query_instruction,
        )
        if not all(isinstance(value, str) for value in string_fields):
            raise TypeError("Embedding profile identity fields must be strings")
        if not self.model_id.strip():
            raise ValueError("Embedding profile model_id must not be empty")
        if not re.fullmatch(r"[0-9a-f]{40}", self.model_revision):
            raise ValueError("Embedding profile revision must be a commit SHA")
        if not re.fullmatch(r"[0-9a-f]{64}", self.config_fingerprint):
            raise ValueError("Embedding profile fingerprint must be SHA-256")
        if not isinstance(self.normalize_embeddings, bool):
            raise TypeError("Embedding profile normalization flag must be boolean")
        if not self.encoding_config_version.strip() or not self.query_instruction.strip():
            raise ValueError("Embedding profile encoding identity must not be empty")


@dataclass(frozen=True)
class StageResult:
    """Return the staged identity and underlying Task 6.4 import result."""

    tenant_id: str
    document_id: str
    version: str
    expected_chunk_count: int
    expected_chunk_ids_fingerprint: str
    import_result: ImportResult


@dataclass(frozen=True)
class TransitionResult:
    """Describe a committed lifecycle transition or a safe retry."""

    tenant_id: str
    document_id: str
    version: str
    outcome: Literal["published", "already_published", "withdrawn", "already_withdrawn"]


def chunk_ids_fingerprint(chunk_ids: list[str] | tuple[str, ...]) -> str:
    """Hash compact UTF-8 JSON of sorted unique chunk IDs deterministically."""
    unique_ids = sorted(set(chunk_ids))
    if len(unique_ids) != len(chunk_ids):
        raise ValueError("Expected chunk set contains duplicate chunk IDs")
    if not unique_ids:
        raise ValueError("A staged version must contain at least one chunk")
    serialized = json.dumps(unique_ids, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _quote_identifier(identifier: str) -> str:
    """Quote a validated PostgreSQL schema identifier."""
    if not IDENTIFIER_PATTERN.fullmatch(identifier):
        raise ValueError("Schema must be an unquoted PostgreSQL identifier")
    return f'"{identifier}"'


def _single_version_scope(
    plan: ImportPlan, tenant_id: str, document_id: str, version: str
) -> list[str]:
    """Validate and return the authoritative source chunks for one version."""
    versions = [
        item for item in plan.versions
        if (item.tenant_id, item.document_id, item.version)
        == (tenant_id, document_id, version)
    ]
    if len(plan.versions) != 1 or len(versions) != 1 or len(plan.documents) != 1:
        raise ValueError("Staging requires an import plan for exactly one document version")
    chunk_ids = [
        chunk.chunk_id for chunk in plan.chunks
        if (chunk.tenant_id, chunk.document_id, chunk.document_version)
        == (tenant_id, document_id, version)
    ]
    if len(chunk_ids) != len(plan.chunks):
        raise ValueError("Staging plan contains chunks outside the target version")
    return chunk_ids


async def stage_document_version(
    connection: asyncpg.Connection,
    plan: ImportPlan,
    *,
    tenant_id: str,
    document_id: str,
    version: str,
    schema: str = "public",
) -> StageResult:
    """Persist one validated source version as staged through the Task 6.4 importer."""
    chunk_ids = _single_version_scope(plan, tenant_id, document_id, version)
    expected_fingerprint = chunk_ids_fingerprint(chunk_ids)
    staged_version = replace(
        plan.versions[0],
        status="staged",
        expected_chunk_count=len(chunk_ids),
        expected_chunk_ids_fingerprint=expected_fingerprint,
    )
    staged_plan = replace(plan, versions=(staged_version,))
    import_result = await import_rag_embeddings(connection, staged_plan, schema=schema)
    logger.info(
        "document_version_staged tenant_id=%s document_id=%s version=%s chunks=%d",
        tenant_id,
        document_id,
        version,
        len(chunk_ids),
    )
    return StageResult(
        tenant_id=tenant_id,
        document_id=document_id,
        version=version,
        expected_chunk_count=len(chunk_ids),
        expected_chunk_ids_fingerprint=expected_fingerprint,
        import_result=import_result,
    )


async def _lock_document(
    connection: asyncpg.Connection, tenant_id: str, document_id: str
) -> int:
    """Resolve and row-lock a trusted tenant-scoped parent document."""
    document_pk = await connection.fetchval(
        "SELECT id FROM documents WHERE tenant_id=$1 AND document_id=$2 FOR UPDATE",
        tenant_id,
        document_id,
    )
    if document_pk is None:
        raise DocumentNotFound(f"Document not found: {tenant_id}/{document_id}")
    return document_pk


async def _active_version(
    connection: asyncpg.Connection, document_pk: int
) -> str | None:
    """Read the one current active version after its parent is locked."""
    return await connection.fetchval(
        "SELECT version FROM document_versions WHERE document_fk=$1 AND status='active'",
        document_pk,
    )


async def _verify_complete_version(
    connection: asyncpg.Connection,
    version_pk: int,
    expected_count: int | None,
    expected_fingerprint: str | None,
    profile: EmbeddingProfile,
) -> None:
    """Prove the stored chunk set and compatible embeddings match staging metadata."""
    if expected_count is None or expected_fingerprint is None:
        raise IncompleteVersion("Version has no authoritative expected chunk manifest")
    chunk_ids = list(await connection.fetchval(
        "SELECT array_agg(chunk_id ORDER BY chunk_id) FROM document_chunks "
        "WHERE document_version_fk=$1",
        version_pk,
    ) or [])
    if (
        len(chunk_ids) != expected_count
        or chunk_ids_fingerprint(chunk_ids) != expected_fingerprint
    ):
        raise IncompleteVersion("Stored chunk set differs from the staged source manifest")
    embedding_count = await connection.fetchval(
        """
        SELECT count(*)
        FROM document_chunks c
        JOIN chunk_embeddings e ON e.chunk_id=c.chunk_id
        WHERE c.document_version_fk=$1
        """,
        version_pk,
    )
    if embedding_count != expected_count:
        raise IncompleteVersion(
            "Staged version must contain exactly one embedding per expected chunk"
        )
    compatible_ids = list(await connection.fetchval(
        """
        SELECT array_agg(c.chunk_id ORDER BY c.chunk_id)
        FROM document_chunks c
        JOIN chunk_embeddings e ON e.chunk_id=c.chunk_id
        WHERE c.document_version_fk=$1
          AND e.embedding_model_id=$2
          AND e.embedding_model_revision=$3
          AND e.embedding_config_fingerprint=$4
          AND e.normalize_embeddings=$5
          AND e.encoding_config_version=$6
          AND e.query_instruction=$7
          AND e.input_fingerprint IS NOT NULL
          AND public.vector_dims(e.embedding)=384
        """,
        version_pk,
        profile.model_id,
        profile.model_revision,
        profile.config_fingerprint,
        profile.normalize_embeddings,
        profile.encoding_config_version,
        profile.query_instruction,
    ) or [])
    if compatible_ids != chunk_ids:
        raise IncompleteVersion("Required compatible embeddings are missing or incompatible")


async def publish_document_version(
    connection: asyncpg.Connection,
    *,
    tenant_id: str,
    document_id: str,
    target_version: str,
    expected_active_version: str | None,
    profile: EmbeddingProfile,
    schema: str = "public",
) -> TransitionResult:
    """Atomically archive the expected active version and publish a staged target."""
    quoted_schema = _quote_identifier(schema)
    async with connection.transaction():
        await connection.execute(f"SET LOCAL search_path TO {quoted_schema}, public")
        document_pk = await _lock_document(connection, tenant_id, document_id)
        active_version = await _active_version(connection, document_pk)
        target = await connection.fetchrow(
            "SELECT id, status, expected_chunk_count, expected_chunk_ids_fingerprint "
            "FROM document_versions WHERE document_fk=$1 AND version=$2",
            document_pk,
            target_version,
        )
        if target is None:
            raise VersionNotFound(f"Version not found: {target_version}")
        if target["status"] == "active":
            expected_was_superseded = (
                expected_active_version is None
                or await connection.fetchval(
                    "SELECT status='archived' FROM document_versions "
                    "WHERE document_fk=$1 AND version=$2",
                    document_pk,
                    expected_active_version,
                ) is True
            )
            if active_version == target_version and expected_was_superseded:
                return TransitionResult(
                    tenant_id, document_id, target_version, "already_published"
                )
        if active_version != expected_active_version:
            raise LifecycleConflict(
                f"Expected active version {expected_active_version!r}, found {active_version!r}"
            )
        if target["status"] != "staged":
            raise LifecycleConflict(
                f"Version {target_version} cannot be published from {target['status']}"
            )
        await _verify_complete_version(
            connection,
            target["id"],
            target["expected_chunk_count"],
            target["expected_chunk_ids_fingerprint"],
            profile,
        )
        if active_version is not None:
            archived = await connection.execute(
                "UPDATE document_versions SET status='archived' "
                "WHERE document_fk=$1 AND version=$2 AND status='active'",
                document_pk,
                active_version,
            )
            if archived != "UPDATE 1":
                raise LifecycleConflict("Expected active version changed during publication")
        activated = await connection.execute(
            "UPDATE document_versions SET status='active' WHERE id=$1 AND status='staged'",
            target["id"],
        )
        if (
            activated != "UPDATE 1"
            or await _active_version(connection, document_pk) != target_version
        ):
            raise LifecycleConflict("Publication did not establish the intended active version")
    logger.info(
        "document_version_published tenant_id=%s document_id=%s version=%s",
        tenant_id,
        document_id,
        target_version,
    )
    return TransitionResult(tenant_id, document_id, target_version, "published")


async def withdraw_document_version(
    connection: asyncpg.Connection,
    *,
    tenant_id: str,
    document_id: str,
    target_version: str,
    expected_active_version: str,
    schema: str = "public",
) -> TransitionResult:
    """Atomically withdraw the explicitly expected active version without deletion."""
    quoted_schema = _quote_identifier(schema)
    async with connection.transaction():
        await connection.execute(f"SET LOCAL search_path TO {quoted_schema}, public")
        document_pk = await _lock_document(connection, tenant_id, document_id)
        active_version = await _active_version(connection, document_pk)
        status = await connection.fetchval(
            "SELECT status FROM document_versions WHERE document_fk=$1 AND version=$2",
            document_pk,
            target_version,
        )
        if status is None:
            raise VersionNotFound(f"Version not found: {target_version}")
        if (
            status == "withdrawn"
            and target_version == expected_active_version
            and active_version is None
        ):
            return TransitionResult(
                tenant_id, document_id, target_version, "already_withdrawn"
            )
        if target_version != expected_active_version or active_version != expected_active_version:
            raise LifecycleConflict(
                f"Expected active version {expected_active_version!r}, found {active_version!r}"
            )
        if status != "active":
            raise LifecycleConflict(
                f"Version {target_version} cannot be withdrawn from {status}"
            )
        updated = await connection.execute(
            "UPDATE document_versions SET status='withdrawn' "
            "WHERE document_fk=$1 AND version=$2 AND status='active'",
            document_pk,
            target_version,
        )
        if updated != "UPDATE 1" or await _active_version(connection, document_pk) is not None:
            raise LifecycleConflict("Withdrawal did not remove the intended active version")
    logger.info(
        "document_version_withdrawn tenant_id=%s document_id=%s version=%s",
        tenant_id,
        document_id,
        target_version,
    )
    return TransitionResult(tenant_id, document_id, target_version, "withdrawn")


async def fetch_current_chunk_ids(
    connection: asyncpg.Connection,
    *,
    tenant_id: str,
    permitted_visibilities: tuple[str, ...],
    profile: EmbeddingProfile,
    schema: str = "public",
) -> tuple[str, ...]:
    """Return chunk IDs eligible for future current-policy dense retrieval."""
    if not permitted_visibilities:
        return ()
    quoted_schema = _quote_identifier(schema)
    async with connection.transaction():
        await connection.execute(f"SET LOCAL search_path TO {quoted_schema}, public")
        rows = await connection.fetch(
            """
            SELECT c.chunk_id
            FROM documents d
            JOIN document_versions v ON v.document_fk=d.id AND v.status='active'
            JOIN document_chunks c ON c.document_version_fk=v.id
            JOIN chunk_embeddings e ON e.chunk_id=c.chunk_id
            WHERE d.tenant_id=$1 AND v.visibility=ANY($2::text[])
              AND e.embedding_model_id=$3 AND e.embedding_model_revision=$4
              AND e.embedding_config_fingerprint=$5
              AND e.normalize_embeddings=$6 AND e.encoding_config_version=$7
              AND e.query_instruction=$8
            ORDER BY c.chunk_id
            """,
            tenant_id,
            list(permitted_visibilities),
            profile.model_id,
            profile.model_revision,
            profile.config_fingerprint,
            profile.normalize_embeddings,
            profile.encoding_config_version,
            profile.query_instruction,
        )
    return tuple(row["chunk_id"] for row in rows)
