"""Verify document lifecycle behavior, including real PostgreSQL concurrency."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import date
import hashlib
import os
from pathlib import Path
import unittest
from uuid import uuid4

import asyncpg
import numpy as np
from sqlalchemy.engine import make_url

from app.rag.persistence.artifacts import (
    ChunkImport,
    DocumentImport,
    EmbeddingImport,
    ImportPlan,
    VersionImport,
)
from app.rag.persistence.importer import RagImportConflict, import_rag_embeddings
from app.rag.persistence.lifecycle import (
    DocumentNotFound,
    EmbeddingProfile,
    IncompleteVersion,
    LifecycleConflict,
    chunk_ids_fingerprint,
    fetch_current_chunk_ids,
    publish_document_version,
    stage_document_version,
    withdraw_document_version,
)


ROOT = Path(__file__).resolve().parent.parent
INITIAL_SCHEMA_SQL = (ROOT / "schema.sql").read_text(encoding="utf-8")
RAG_MIGRATION_SQL = (ROOT / "migrations/002_create_rag_schema.sql").read_text(
    encoding="utf-8"
)
LIFECYCLE_MIGRATION_SQL = (
    ROOT / "migrations/003_add_document_lifecycle.sql"
).read_text(encoding="utf-8")
TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "")
MODEL_ID = "BAAI/bge-small-en-v1.5"
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
CONFIG_FINGERPRINT = "c" * 64
PROFILE = EmbeddingProfile(
    model_id=MODEL_ID,
    model_revision=MODEL_REVISION,
    config_fingerprint=CONFIG_FINGERPRINT,
    normalize_embeddings=True,
    encoding_config_version="phase5-v1",
)


def make_version_plan(
    version: str,
    *,
    tenant_id: str = "tenant-lifecycle",
    document_id: str = "DOC-LIFECYCLE",
    status: str = "active",
    chunk_suffixes: tuple[str, ...] = ("a",),
    visibility: str = "public",
) -> ImportPlan:
    """Build one validated importer plan without running the embedding model."""
    document = DocumentImport(tenant_id=tenant_id, document_id=document_id)
    version_row = VersionImport(
        tenant_id=tenant_id,
        document_id=document_id,
        version=version,
        title=f"Lifecycle fixture {version}",
        effective_date=date(2026, 10, 22),
        status=status,
        visibility=visibility,
        source_sha256=hashlib.sha256(version.encode()).hexdigest(),
        source_path=f"documents/{tenant_id}/{document_id}-{version}.md",
        source_format="md",
        synthetic=True,
        topic_family="testing",
        word_count=10,
        pdf_pages=None,
        section_count=1,
        scope_note="Synthetic lifecycle fixture",
    )
    chunks: list[ChunkImport] = []
    embeddings: list[EmbeddingImport] = []
    for index, suffix in enumerate(chunk_suffixes):
        digest = hashlib.sha256(
            f"{tenant_id}:{document_id}:{version}:{suffix}".encode()
        ).hexdigest()
        chunk_id = f"chk_{digest}"
        chunks.append(
            ChunkImport(
                tenant_id=tenant_id,
                document_id=document_id,
                document_version=version,
                chunk_id=chunk_id,
                chunk_index=index,
                text=f"Exact lifecycle text {version} {suffix}.",
                token_count=5,
                heading_path=("Policy",),
                start_page=None,
                end_page=None,
                source_block_ids=(index,),
                tokenizer_name="o200k_base",
                tokenizer_version="0.12.0",
                chunking_config_version="phase4-v4",
                chunking_configuration={"config_version": "phase4-v4"},
                chunking_config_fingerprint="b" * 64,
            )
        )
        vector = np.zeros(384, dtype=np.float32)
        vector[index % 384] = 1.0
        embeddings.append(
            EmbeddingImport(
                chunk_id=chunk_id,
                vector=vector,
                model_id=MODEL_ID,
                model_revision=MODEL_REVISION,
                embedding_config_fingerprint=CONFIG_FINGERPRINT,
                input_fingerprint="emb_" + digest,
                document_instruction="",
                query_instruction="Represent this sentence for searching relevant passages: ",
                normalize_embeddings=True,
                encoding_config_version="phase5-v1",
            )
        )
    return ImportPlan(
        run_directory=Path(f"run-{version}"),
        documents=(document,),
        versions=(version_row,),
        chunks=tuple(chunks),
        embeddings=tuple(embeddings),
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        dimensions=384,
    )


class ChunkManifestFingerprintTests(unittest.TestCase):
    """Check the source-manifest fingerprint contract without PostgreSQL."""

    def test_fingerprint_is_order_independent_and_unambiguous(self) -> None:
        """Sort IDs and use JSON boundaries so concatenation cannot collide."""
        self.assertEqual(
            chunk_ids_fingerprint(["chk_b", "chk_a"]),
            chunk_ids_fingerprint(["chk_a", "chk_b"]),
        )
        self.assertNotEqual(
            chunk_ids_fingerprint(["chk_ab", "chk_c"]),
            chunk_ids_fingerprint(["chk_a", "chk_bc"]),
        )

    def test_empty_and_duplicate_sets_are_rejected(self) -> None:
        """Require a nonempty unique authoritative chunk identity set."""
        with self.assertRaises(ValueError):
            chunk_ids_fingerprint([])
        with self.assertRaises(ValueError):
            chunk_ids_fingerprint(["chk_a", "chk_a"])


@unittest.skipUnless(TEST_DATABASE_URL, "TEST_DATABASE_URL is not set")
class DocumentLifecycleIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """Exercise lifecycle transitions in isolated real PostgreSQL schemas."""

    async def asyncSetUp(self) -> None:
        """Create a migrated disposable schema and one active initial version."""
        url = make_url(TEST_DATABASE_URL)
        if url.drivername != "postgresql+asyncpg" or "test" not in (
            url.database or ""
        ).lower():
            self.fail("TEST_DATABASE_URL must use asyncpg and contain 'test'")
        self.dsn = url.set(drivername="postgresql").render_as_string(
            hide_password=False
        )
        try:
            self.connection = await asyncpg.connect(self.dsn, timeout=5)
        except OSError:
            self.skipTest("Disposable PostgreSQL database is unreachable")
        self.addAsyncCleanup(self.connection.close)
        self.schema = f"rag_lifecycle_test_{uuid4().hex}"
        await self.connection.execute(f'CREATE SCHEMA "{self.schema}"')
        self.addAsyncCleanup(self._drop_schema)
        await self.connection.execute(f'SET search_path TO "{self.schema}", public')
        await self.connection.execute(INITIAL_SCHEMA_SQL)
        await self.connection.execute(RAG_MIGRATION_SQL)
        await self.connection.execute(LIFECYCLE_MIGRATION_SQL)
        await import_rag_embeddings(
            self.connection, make_version_plan("1.0"), schema=self.schema
        )

    async def _drop_schema(self) -> None:
        """Remove only the random schema owned by this test."""
        await self.connection.execute(f'DROP SCHEMA IF EXISTS "{self.schema}" CASCADE')

    async def _stage(self, version: str, suffixes: tuple[str, ...] = ("a",)) -> ImportPlan:
        """Stage and return one source-authoritative fixture plan."""
        plan = make_version_plan(version, chunk_suffixes=suffixes)
        await stage_document_version(
            self.connection,
            plan,
            tenant_id="tenant-lifecycle",
            document_id="DOC-LIFECYCLE",
            version=version,
            schema=self.schema,
        )
        return plan

    async def _statuses(self) -> dict[str, str]:
        """Read all fixture version statuses by external version identity."""
        await self.connection.execute(f'SET search_path TO "{self.schema}", public')
        rows = await self.connection.fetch(
            "SELECT version, status FROM document_versions ORDER BY version"
        )
        return {row["version"]: row["status"] for row in rows}

    async def test_stage_publish_retry_withdraw_and_visibility(self) -> None:
        """Preserve active state during staging and apply documented safe retries."""
        plan = await self._stage("2.0", ("a", "b"))
        retry = await stage_document_version(
            self.connection,
            plan,
            tenant_id="tenant-lifecycle",
            document_id="DOC-LIFECYCLE",
            version="2.0",
            schema=self.schema,
        )
        self.assertEqual(retry.import_result.counts.versions_reused, 1)
        self.assertEqual(await self._statuses(), {"1.0": "active", "2.0": "staged"})
        self.assertEqual(
            await fetch_current_chunk_ids(
                self.connection,
                tenant_id="tenant-lifecycle",
                permitted_visibilities=("public",),
                profile=PROFILE,
                schema=self.schema,
            ),
            (make_version_plan("1.0").chunks[0].chunk_id,),
        )

        published = await publish_document_version(
            self.connection,
            tenant_id="tenant-lifecycle",
            document_id="DOC-LIFECYCLE",
            target_version="2.0",
            expected_active_version="1.0",
            profile=PROFILE,
            schema=self.schema,
        )
        self.assertEqual(published.outcome, "published")
        retry_publish = await publish_document_version(
            self.connection,
            tenant_id="tenant-lifecycle",
            document_id="DOC-LIFECYCLE",
            target_version="2.0",
            expected_active_version="1.0",
            profile=PROFILE,
            schema=self.schema,
        )
        self.assertEqual(retry_publish.outcome, "already_published")
        self.assertEqual(await self._statuses(), {"1.0": "archived", "2.0": "active"})

        withdrawn = await withdraw_document_version(
            self.connection,
            tenant_id="tenant-lifecycle",
            document_id="DOC-LIFECYCLE",
            target_version="2.0",
            expected_active_version="2.0",
            schema=self.schema,
        )
        self.assertEqual(withdrawn.outcome, "withdrawn")
        repeated = await withdraw_document_version(
            self.connection,
            tenant_id="tenant-lifecycle",
            document_id="DOC-LIFECYCLE",
            target_version="2.0",
            expected_active_version="2.0",
            schema=self.schema,
        )
        self.assertEqual(repeated.outcome, "already_withdrawn")
        self.assertEqual(await fetch_current_chunk_ids(
            self.connection,
            tenant_id="tenant-lifecycle",
            permitted_visibilities=("public",),
            profile=PROFILE,
            schema=self.schema,
        ), ())

    async def test_first_publication_uses_explicit_expected_none(self) -> None:
        """Publish a new document only when the caller explicitly expects no active version."""
        plan = make_version_plan(
            "1.0", tenant_id="tenant-new", document_id="DOC-NEW"
        )
        await stage_document_version(
            self.connection,
            plan,
            tenant_id="tenant-new",
            document_id="DOC-NEW",
            version="1.0",
            schema=self.schema,
        )
        result = await publish_document_version(
            self.connection,
            tenant_id="tenant-new",
            document_id="DOC-NEW",
            target_version="1.0",
            expected_active_version=None,
            profile=PROFILE,
            schema=self.schema,
        )
        self.assertEqual(result.outcome, "published")

    async def test_wrong_tenant_and_invalid_transition_are_rejected(self) -> None:
        """Keep tenant ownership strict and never reactivate archived history."""
        await self._stage("2.0")
        with self.assertRaises(DocumentNotFound):
            await publish_document_version(
                self.connection,
                tenant_id="wrong-tenant",
                document_id="DOC-LIFECYCLE",
                target_version="2.0",
                expected_active_version="1.0",
                profile=PROFILE,
                schema=self.schema,
            )
        with self.assertRaises(LifecycleConflict):
            await publish_document_version(
                self.connection,
                tenant_id="tenant-lifecycle",
                document_id="DOC-LIFECYCLE",
                target_version="1.0",
                expected_active_version="1.0",
                profile=PROFILE,
                schema=self.schema,
            )

    async def test_missing_chunk_rolls_back_publication(self) -> None:
        """Reject a stored set smaller than the authoritative source set."""
        plan = await self._stage("2.0", ("a", "b"))
        missing_id = plan.chunks[1].chunk_id
        await self.connection.execute(f'SET search_path TO "{self.schema}", public')
        await self.connection.execute("DELETE FROM chunk_embeddings WHERE chunk_id=$1", missing_id)
        await self.connection.execute("DELETE FROM document_chunks WHERE chunk_id=$1", missing_id)
        with self.assertRaises(IncompleteVersion):
            await publish_document_version(
                self.connection,
                tenant_id="tenant-lifecycle",
                document_id="DOC-LIFECYCLE",
                target_version="2.0",
                expected_active_version="1.0",
                profile=PROFILE,
                schema=self.schema,
            )
        self.assertEqual(await self._statuses(), {"1.0": "active", "2.0": "staged"})

    async def test_activation_database_failure_rolls_back_archiving(self) -> None:
        """Restore the previous active row when activation fails after its archive update."""
        await self._stage("2.0")
        await self.connection.execute(f'SET search_path TO "{self.schema}", public')
        await self.connection.execute(
            """
            CREATE FUNCTION reject_fixture_activation() RETURNS TRIGGER
            LANGUAGE plpgsql AS $body$
            BEGIN
                IF NEW.version = '2.0' AND NEW.status = 'active' THEN
                    RAISE EXCEPTION 'fixture activation failure';
                END IF;
                RETURN NEW;
            END
            $body$;
            CREATE TRIGGER reject_fixture_activation
                BEFORE UPDATE OF status ON document_versions
                FOR EACH ROW EXECUTE FUNCTION reject_fixture_activation();
            """
        )
        with self.assertRaisesRegex(asyncpg.RaiseError, "fixture activation failure"):
            await publish_document_version(
                self.connection,
                tenant_id="tenant-lifecycle",
                document_id="DOC-LIFECYCLE",
                target_version="2.0",
                expected_active_version="1.0",
                profile=PROFILE,
                schema=self.schema,
            )
        self.assertEqual(await self._statuses(), {"1.0": "active", "2.0": "staged"})

    async def test_same_count_wrong_chunk_ids_are_rejected(self) -> None:
        """Detect identity substitution even when the stored count is unchanged."""
        plan = await self._stage("2.0", ("a", "b"))
        old_id = plan.chunks[1].chunk_id
        new_id = "chk_" + "f" * 64
        await self.connection.execute(f'SET search_path TO "{self.schema}", public')
        await self.connection.execute("DELETE FROM chunk_embeddings WHERE chunk_id=$1", old_id)
        await self.connection.execute(
            "UPDATE document_chunks SET chunk_id=$1 WHERE chunk_id=$2", new_id, old_id
        )
        with self.assertRaises(IncompleteVersion):
            await publish_document_version(
                self.connection,
                tenant_id="tenant-lifecycle",
                document_id="DOC-LIFECYCLE",
                target_version="2.0",
                expected_active_version="1.0",
                profile=PROFILE,
                schema=self.schema,
            )
        self.assertEqual((await self._statuses())["1.0"], "active")

    async def test_missing_or_incompatible_embedding_is_rejected(self) -> None:
        """Require one embedding matching the selected immutable profile per chunk."""
        plan = await self._stage("2.0")
        await self.connection.execute(f'SET search_path TO "{self.schema}", public')
        await self.connection.execute(
            "DELETE FROM chunk_embeddings WHERE chunk_id=$1", plan.chunks[0].chunk_id
        )
        with self.assertRaisesRegex(IncompleteVersion, "embedding"):
            await publish_document_version(
                self.connection,
                tenant_id="tenant-lifecycle",
                document_id="DOC-LIFECYCLE",
                target_version="2.0",
                expected_active_version="1.0",
                profile=PROFILE,
                schema=self.schema,
            )
        self.assertEqual((await self._statuses())["1.0"], "active")

        incompatible_plan = await self._stage("3.0")
        await self.connection.execute(f'SET search_path TO "{self.schema}", public')
        await self.connection.execute(
            "UPDATE chunk_embeddings SET embedding_model_revision=$1 WHERE chunk_id=$2",
            "e" * 40,
            incompatible_plan.chunks[0].chunk_id,
        )
        with self.assertRaisesRegex(IncompleteVersion, "incompatible"):
            await publish_document_version(
                self.connection,
                tenant_id="tenant-lifecycle",
                document_id="DOC-LIFECYCLE",
                target_version="3.0",
                expected_active_version="1.0",
                profile=PROFILE,
                schema=self.schema,
            )
        self.assertEqual((await self._statuses())["1.0"], "active")

    async def test_embeddings_from_another_profile_do_not_block_publication(self) -> None:
        """Publish a complete version whose chunks also have a second model's vectors."""
        plan = await self._stage("2.0", ("a", "b"))
        await self.connection.execute(f'SET search_path TO "{self.schema}", public')
        # Duplicate each staged embedding under a different model revision, as
        # a fine-tuned embedding run over the same chunks would.
        await self.connection.execute(
            """
            INSERT INTO chunk_embeddings (
                chunk_id, embedding, embedding_model_id, embedding_model_revision,
                embedding_config_fingerprint, input_fingerprint,
                document_instruction, query_instruction, normalize_embeddings,
                encoding_config_version
            )
            SELECT chunk_id, embedding, embedding_model_id, $1,
                   embedding_config_fingerprint, input_fingerprint,
                   document_instruction, query_instruction, normalize_embeddings,
                   encoding_config_version
            FROM chunk_embeddings WHERE chunk_id = ANY($2::text[])
            """,
            "e" * 40,
            [chunk.chunk_id for chunk in plan.chunks],
        )

        result = await publish_document_version(
            self.connection,
            tenant_id="tenant-lifecycle",
            document_id="DOC-LIFECYCLE",
            target_version="2.0",
            expected_active_version="1.0",
            profile=PROFILE,
            schema=self.schema,
        )

        self.assertEqual(result.outcome, "published")
        self.assertEqual(
            await self._statuses(), {"1.0": "archived", "2.0": "active"}
        )

    async def test_conflicting_staging_retry_is_rejected(self) -> None:
        """Never replace the stored expected count or fingerprint on retry."""
        await self._stage("2.0")
        conflicting = make_version_plan("2.0", chunk_suffixes=("a", "b"))
        with self.assertRaisesRegex(RagImportConflict, "Conflicting document version"):
            await stage_document_version(
                self.connection,
                conflicting,
                tenant_id="tenant-lifecycle",
                document_id="DOC-LIFECYCLE",
                version="2.0",
                schema=self.schema,
            )
        self.assertEqual((await self._statuses())["2.0"], "staged")

    async def test_concurrent_publications_allow_one_winner(self) -> None:
        """Serialize writers and reject the worker whose expected state is stale."""
        await self._stage("2.0")
        await self._stage("3.0")
        second_connection = await asyncpg.connect(self.dsn, timeout=5)
        self.addAsyncCleanup(second_connection.close)

        async def publish(connection: asyncpg.Connection, version: str) -> str:
            result = await publish_document_version(
                connection,
                tenant_id="tenant-lifecycle",
                document_id="DOC-LIFECYCLE",
                target_version=version,
                expected_active_version="1.0",
                profile=PROFILE,
                schema=self.schema,
            )
            return result.outcome

        results = await asyncio.gather(
            publish(self.connection, "2.0"),
            publish(second_connection, "3.0"),
            return_exceptions=True,
        )
        self.assertEqual(sum(result == "published" for result in results), 1)
        self.assertEqual(sum(isinstance(result, LifecycleConflict) for result in results), 1)
        await self.connection.execute(f'SET search_path TO "{self.schema}", public')
        self.assertEqual(await self.connection.fetchval(
            "SELECT count(*) FROM document_versions WHERE status='active'"
        ), 1)

    async def test_tenant_visibility_and_model_filters(self) -> None:
        """Apply all reusable current-policy boundaries to candidate chunks."""
        await self._stage("2.0")
        await publish_document_version(
            self.connection,
            tenant_id="tenant-lifecycle",
            document_id="DOC-LIFECYCLE",
            target_version="2.0",
            expected_active_version="1.0",
            profile=PROFILE,
            schema=self.schema,
        )
        self.assertEqual(await fetch_current_chunk_ids(
            self.connection,
            tenant_id="another-tenant",
            permitted_visibilities=("public",),
            profile=PROFILE,
            schema=self.schema,
        ), ())
        self.assertEqual(await fetch_current_chunk_ids(
            self.connection,
            tenant_id="tenant-lifecycle",
            permitted_visibilities=("internal",),
            profile=PROFILE,
            schema=self.schema,
        ), ())
        incompatible = replace(PROFILE, config_fingerprint="e" * 64)
        self.assertEqual(await fetch_current_chunk_ids(
            self.connection,
            tenant_id="tenant-lifecycle",
            permitted_visibilities=("public",),
            profile=incompatible,
            schema=self.schema,
        ), ())

    async def test_forward_migration_preserves_legacy_statuses(self) -> None:
        """Keep pre-003 active/archived rows and reject incomplete staged rows."""
        legacy_schema = f"rag_lifecycle_legacy_{uuid4().hex}"
        await self.connection.execute(f'CREATE SCHEMA "{legacy_schema}"')
        try:
            await self.connection.execute(
                f'SET search_path TO "{legacy_schema}", public'
            )
            await self.connection.execute(INITIAL_SCHEMA_SQL)
            await self.connection.execute(RAG_MIGRATION_SQL)
            document_pk = await self.connection.fetchval(
                "INSERT INTO documents (tenant_id, document_id) "
                "VALUES ('tenant-legacy', 'DOC-LEGACY') RETURNING id"
            )
            values = (
                document_pk,
                "Legacy",
                date(2026, 1, 1),
                "public",
                "a" * 64,
                "documents/legacy.md",
                True,
                "testing",
            )
            await self.connection.execute(
                """
                INSERT INTO document_versions (
                    document_fk, version, title, effective_date, status, visibility,
                    source_sha256, source_path, source_format, synthetic,
                    topic_family, word_count, section_count, scope_note
                ) VALUES
                    ($1, '1.0', $2, $3, 'archived', $4, $5, $6, 'md', $7, $8, 1, 1, 'Legacy'),
                    ($1, '2.0', $2, $3, 'active', $4, $5, $6, 'md', $7, $8, 1, 1, 'Legacy')
                """,
                *values,
            )
            await self.connection.execute(LIFECYCLE_MIGRATION_SQL)
            rows = await self.connection.fetch(
                "SELECT status, expected_chunk_count, expected_chunk_ids_fingerprint "
                "FROM document_versions ORDER BY version"
            )
            self.assertEqual(
                [
                    (
                        row["status"],
                        row["expected_chunk_count"],
                        row["expected_chunk_ids_fingerprint"],
                    )
                    for row in rows
                ],
                [("archived", None, None), ("active", None, None)],
            )
            with self.assertRaises(asyncpg.CheckViolationError):
                async with self.connection.transaction():
                    await self.connection.execute(
                        "UPDATE document_versions SET status='staged' WHERE version='1.0'"
                    )
        finally:
            await self.connection.execute(f'SET search_path TO "{self.schema}", public')
            await self.connection.execute(f'DROP SCHEMA "{legacy_schema}" CASCADE')


if __name__ == "__main__":
    unittest.main()
