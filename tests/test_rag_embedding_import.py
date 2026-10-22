"""Test strict RAG import behavior, including real PostgreSQL transactions.

Database cases run only when TEST_DATABASE_URL names a disposable test database.
Every test creates and removes its own schema without touching application data.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import date
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
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
    load_import_plan,
)
from app.rag.embeddings.models import EmbeddingConfig, EmbeddingRecord
from app.rag.embeddings.service import embedding_input_fingerprint
from app.rag.ingestion.discovery import ValidatedDocument, ValidationResult
from app.rag.ingestion.models import DocumentChunk
from app.rag.persistence.importer import RagImportConflict, import_rag_embeddings
from app.rag.retrieval.verification import reconcile_import_plan
from scripts.import_rag_embeddings import main


ROOT = Path(__file__).resolve().parent.parent
MIGRATION_SQL = (ROOT / "migrations" / "002_create_rag_schema.sql").read_text()
LIFECYCLE_MIGRATION_SQL = (
    ROOT / "migrations" / "003_add_document_lifecycle.sql"
).read_text()
INITIAL_SCHEMA_SQL = (ROOT / "schema.sql").read_text()
TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "")
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"


def make_plan(
    identity: str = "fixture",
    *,
    status: str = "active",
) -> ImportPlan:
    """Create one complete 384-dimensional import scope without model inference."""
    tenant_id = f"tenant-{identity}"
    document_id = f"DOC-{identity}"
    chunk_id = "chk_" + identity.encode().hex().ljust(64, "0")[:64]
    document = DocumentImport(tenant_id, document_id)
    version = VersionImport(
        tenant_id=tenant_id,
        document_id=document_id,
        version="1.0",
        title="Synthetic import fixture",
        effective_date=date(2026, 7, 1),
        status=status,
        visibility="public",
        source_sha256="a" * 64,
        source_path=f"documents/{tenant_id}/{document_id}.md",
        source_format="md",
        synthetic=True,
        topic_family="testing",
        word_count=3,
        pdf_pages=None,
        section_count=1,
        scope_note="Synthetic fixture",
    )
    configuration = {
        "config_version": "phase4-v4",
        "max_tokens": 500,
        "overlap_tokens": 50,
        "target_tokens": 400,
        "tokenizer_name": "o200k_base",
        "tokenizer_version": "0.12.0",
    }
    chunk = ChunkImport(
        tenant_id=tenant_id,
        document_id=document_id,
        document_version="1.0",
        chunk_id=chunk_id,
        chunk_index=0,
        text="Exact synthetic chunk text.",
        token_count=5,
        heading_path=("Policy",),
        start_page=None,
        end_page=None,
        source_block_ids=(0,),
        tokenizer_name="o200k_base",
        tokenizer_version="0.12.0",
        chunking_config_version="phase4-v4",
        chunking_configuration=configuration,
        chunking_config_fingerprint="b" * 64,
    )
    vector = np.zeros(384, dtype=np.float32)
    vector[0] = 1.0
    embedding = EmbeddingImport(
        chunk_id=chunk_id,
        vector=vector,
        model_id="BAAI/bge-small-en-v1.5",
        model_revision=MODEL_REVISION,
        embedding_config_fingerprint="c" * 64,
        input_fingerprint="emb_" + "d" * 64,
        document_instruction="",
        query_instruction="Represent this sentence for searching relevant passages: ",
        normalize_embeddings=True,
        encoding_config_version="phase5-v1",
    )
    return ImportPlan(
        run_directory=Path("run-fixture"),
        documents=(document,),
        versions=(version,),
        chunks=(chunk,),
        embeddings=(embedding,),
        model_id=embedding.model_id,
        model_revision=embedding.model_revision,
        dimensions=384,
    )


class EmbeddingImportCliTests(unittest.TestCase):
    """Keep validation-only CLI behavior independent of PostgreSQL."""

    @patch("scripts.import_rag_embeddings.load_import_plan")
    def test_check_only_validates_without_database_connection(
        self, mocked_load: unittest.mock.Mock
    ) -> None:
        """Report a complete plan while making no connection or write."""
        mocked_load.return_value = make_plan()
        output = io.StringIO()
        self.assertEqual(main(["--check-only"], output, io.StringIO()), 0)
        self.assertIn("Embeddings: 1", output.getvalue())
        self.assertIn("Transaction committed: no", output.getvalue())

    @patch(
        "scripts.import_rag_embeddings.load_import_plan",
        side_effect=ValueError("bad checksum"),
    )
    def test_validation_failure_has_nonzero_exit(
        self, _: unittest.mock.Mock
    ) -> None:
        """Reject invalid artifacts before attempting a database import."""
        output = io.StringIO()
        errors = io.StringIO()
        self.assertEqual(main(["--check-only"], output, errors), 1)
        self.assertIn("bad checksum", errors.getvalue())
        self.assertIn("FINAL STATUS: FAILED", output.getvalue())


class ImportPlanMappingTests(unittest.TestCase):
    """Verify source joins and fingerprints without touching PostgreSQL."""

    def _source_contracts(self) -> tuple[object, ...]:
        """Build one mutually consistent manifest, chunk, metadata, and report set."""
        config = EmbeddingConfig()
        document = ValidatedDocument(
            document_id="DOC-1",
            tenant_id="tenant-one",
            version="1.0",
            status="active",
            format="md",
            title="Fixture title",
            resolved_file_path=Path("fixture.md"),
            sha256="a" * 64,
            metadata={
                "effective_date": "2026-07-01",
                "visibility": "public",
                "synthetic": "true",
                "path": "documents/tenant-one/DOC-1.md",
                "topic_family": "testing",
                "word_count": "4",
                "pdf_pages": "",
                "section_count": "1",
                "scope_note": "Synthetic fixture",
            },
            manifest_row=2,
        )
        manifest = ValidationResult(1, (document,), ())
        chunk = DocumentChunk(
            chunk_id="chk_" + "1" * 64,
            document_id="DOC-1",
            document_version="1.0",
            tenant_id="tenant-one",
            document_status="active",
            effective_date="2026-07-01",
            chunk_index=0,
            heading_path=("Policy",),
            text="Exact fixture text.",
            token_count=4,
            source_file="documents/tenant-one/DOC-1.md",
            source_type="md",
            start_page=None,
            end_page=None,
            source_block_ids=(0,),
            tokenizer_name="o200k_base",
            tokenizer_version="0.12.0",
            chunking_config_version="phase4-v4",
            access_scope="public",
        )
        record = EmbeddingRecord(
            row_index=0,
            chunk_id=chunk.chunk_id,
            tenant_id=chunk.tenant_id,
            document_id=chunk.document_id,
            document_version=chunk.document_version,
            document_status=chunk.document_status,
            access_scope=chunk.access_scope,
            heading_path=chunk.heading_path,
            start_page=None,
            end_page=None,
            source_file=chunk.source_file,
            source_type=chunk.source_type,
            source_block_ids=chunk.source_block_ids,
            chunk_token_count=chunk.token_count,
            chunk_tokenizer_name=chunk.tokenizer_name,
            chunk_tokenizer_version=chunk.tokenizer_version,
            chunking_config_version=chunk.chunking_config_version,
            input_fingerprint=embedding_input_fingerprint(chunk.text, config),
            embedding_dimensions=384,
            embedding_model_id=config.model_id,
            embedding_model_revision=config.model_revision,
        )
        vector = np.zeros((1, 384), dtype=np.float32)
        vector[0, 0] = 1.0
        report = {
            "final_status": "PASSED",
            "input": {"chunks": 1, "corpus_sha256": "corpus"},
            "output": {"vectors": 1, "metadata_rows": 1, "invalid_vectors": 0},
            "model": {
                **config.to_dict(),
                "actual_dimensions": 384,
            },
        }
        return manifest, chunk, record, vector, report

    def test_consistent_sources_build_one_complete_plan(self) -> None:
        """Map every entity while preserving source and model identity."""
        manifest, chunk, record, vector, report = self._source_contracts()
        chunking_report = {
            "final_status": "PASSED",
            "chunks_jsonl_sha256": "corpus",
            "total_chunks": 1,
            "chunk_counts_per_document": {"tenant-one/DOC-1/1.0": 1},
            "chunking_configuration": {
                "config_version": "phase4-v4",
                "max_tokens": 500,
                "overlap_tokens": 50,
                "target_tokens": 400,
                "tokenizer_name": "o200k_base",
                "tokenizer_version": "0.12.0",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "chunking_report.json"
            report_path.write_text(json.dumps(chunking_report))
            with (
                patch(
                    "app.rag.persistence.artifacts.validate_manifest",
                    return_value=manifest,
                ),
                patch(
                    "app.rag.persistence.artifacts.load_chunks",
                    return_value=((chunk,), "corpus"),
                ),
                patch(
                    "app.rag.persistence.artifacts.resolve_embedding_run",
                    return_value=Path(directory),
                ),
                patch(
                    "app.rag.persistence.artifacts.load_embedding_artifacts",
                    return_value=(vector, (record,), report),
                ),
            ):
                plan = load_import_plan(
                    manifest_path=Path("manifest.csv"),
                    raw_directory=Path("raw"),
                    chunks_path=Path("chunks.jsonl"),
                    chunking_report_path=report_path,
                    embedding_root=Path(directory),
                )
        self.assertEqual(
            (len(plan.documents), len(plan.versions), len(plan.chunks), len(plan.embeddings)),
            (1, 1, 1, 1),
        )
        self.assertEqual(plan.chunks[0].text, chunk.text)
        self.assertEqual(plan.embeddings[0].input_fingerprint, record.input_fingerprint)

    def test_metadata_source_conflict_is_rejected(self) -> None:
        """Refuse a vector row that claims a different tenant from its chunk."""
        manifest, chunk, record, vector, report = self._source_contracts()
        bad_record = replace(record, tenant_id="another-tenant")
        chunking_report = {
            "final_status": "PASSED",
            "chunks_jsonl_sha256": "corpus",
            "total_chunks": 1,
            "chunk_counts_per_document": {"tenant-one/DOC-1/1.0": 1},
            "chunking_configuration": {
                "config_version": "phase4-v4",
                "tokenizer_name": "o200k_base",
                "tokenizer_version": "0.12.0",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "chunking_report.json"
            report_path.write_text(json.dumps(chunking_report))
            with (
                patch("app.rag.persistence.artifacts.validate_manifest", return_value=manifest),
                patch(
                    "app.rag.persistence.artifacts.load_chunks",
                    return_value=((chunk,), "corpus"),
                ),
                patch(
                    "app.rag.persistence.artifacts.resolve_embedding_run",
                    return_value=Path(directory),
                ),
                patch(
                    "app.rag.persistence.artifacts.load_embedding_artifacts",
                    return_value=(vector, (bad_record,), report),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "conflicts with chunk"):
                    load_import_plan(
                        manifest_path=Path("manifest.csv"),
                        raw_directory=Path("raw"),
                        chunks_path=Path("chunks.jsonl"),
                        chunking_report_path=report_path,
                        embedding_root=Path(directory),
                    )


@unittest.skipUnless(TEST_DATABASE_URL, "TEST_DATABASE_URL is not set")
class RagEmbeddingImportIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """Exercise commits, retries, conflicts, rollback, and concurrency."""

    async def asyncSetUp(self) -> None:
        """Install the RAG schema inside one random disposable schema."""
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
        self.schema = f"rag_import_test_{uuid4().hex}"
        await self.connection.execute(f'CREATE SCHEMA "{self.schema}"')
        self.addAsyncCleanup(self._drop_schema)
        await self.connection.execute(f'SET search_path TO "{self.schema}", public')
        await self.connection.execute(INITIAL_SCHEMA_SQL)
        await self.connection.execute(MIGRATION_SQL)
        await self.connection.execute(LIFECYCLE_MIGRATION_SQL)

    async def _drop_schema(self) -> None:
        """Remove only this test's generated schema."""
        await self.connection.execute(f'DROP SCHEMA IF EXISTS "{self.schema}" CASCADE')

    async def _count(self, table: str) -> int:
        """Count a trusted table fixture inside the isolated schema."""
        await self.connection.execute(f'SET search_path TO "{self.schema}", public')
        return await self.connection.fetchval(f"SELECT count(*) FROM {table}")

    async def test_import_and_identical_retry_preserve_relationships(self) -> None:
        """Commit once, reuse every identity, and retain archived source status."""
        plan = make_plan(status="archived")
        first = await import_rag_embeddings(
            self.connection, plan, schema=self.schema
        )
        second = await import_rag_embeddings(
            self.connection, plan, schema=self.schema
        )

        self.assertEqual(first.counts.embeddings_inserted, 1)
        self.assertEqual(second.counts.embeddings_inserted, 0)
        self.assertEqual(second.counts.embeddings_reused, 1)
        await self.connection.execute(f'SET search_path TO "{self.schema}", public')
        row = await self.connection.fetchrow(
            """
            SELECT d.tenant_id, d.document_id, v.version, v.status, c.text,
                   public.vector_dims(e.embedding) AS dimensions
            FROM chunk_embeddings e
            JOIN document_chunks c ON c.chunk_id=e.chunk_id
            JOIN document_versions v ON v.id=c.document_version_fk
            JOIN documents d ON d.id=v.document_fk
            """
        )
        self.assertEqual(
            (row["tenant_id"], row["document_id"], row["version"], row["status"]),
            ("tenant-fixture", "DOC-fixture", "1.0", "archived"),
        )
        self.assertEqual(row["text"], "Exact synthetic chunk text.")
        self.assertEqual(row["dimensions"], 384)

    async def test_exact_artifact_reconciliation_detects_provenance_drift(self) -> None:
        """Verify every source identity, then report changed version metadata."""
        plan = make_plan(status="archived")
        await import_rag_embeddings(self.connection, plan, schema=self.schema)

        verified = await reconcile_import_plan(
            self.connection, plan, schema=self.schema
        )
        self.assertEqual(verified.status, "PASS")
        self.assertEqual(verified.matching_database_embeddings, 1)
        self.assertEqual(verified.missing_embeddings, ())

        await self.connection.execute(f'SET search_path TO "{self.schema}", public')
        await self.connection.execute(
            "UPDATE document_versions SET title='Changed fixture title'"
        )
        drifted = await reconcile_import_plan(
            self.connection, plan, schema=self.schema
        )
        self.assertEqual(drifted.status, "FAIL")
        self.assertEqual(
            drifted.relationship_mismatches, (plan.chunks[0].chunk_id,)
        )

    async def test_conflicting_version_rolls_back_new_document(self) -> None:
        """Keep committed data and remove earlier writes from a failed retry."""
        original = make_plan("original")
        await import_rag_embeddings(self.connection, original, schema=self.schema)
        conflicting_version = replace(
            original.versions[0], source_sha256="e" * 64
        )
        new_plan = make_plan("new")
        conflict_plan = replace(
            original,
            documents=original.documents + new_plan.documents,
            versions=(conflicting_version,) + new_plan.versions,
            chunks=original.chunks + new_plan.chunks,
            embeddings=original.embeddings + new_plan.embeddings,
        )

        with self.assertRaisesRegex(RagImportConflict, "Conflicting document version"):
            await import_rag_embeddings(
                self.connection, conflict_plan, schema=self.schema
            )
        self.assertEqual(await self._count("documents"), 1)
        self.assertEqual(await self._count("chunk_embeddings"), 1)

    async def test_chunk_and_vector_conflicts_are_rejected(self) -> None:
        """Reject changed content and vector values under existing identities."""
        plan = make_plan()
        await import_rag_embeddings(self.connection, plan, schema=self.schema)
        changed_chunk = replace(plan.chunks[0], text="Conflicting text")
        with self.assertRaisesRegex(RagImportConflict, "Conflicting chunk"):
            await import_rag_embeddings(
                self.connection,
                replace(plan, chunks=(changed_chunk,)),
                schema=self.schema,
            )
        changed_vector = plan.embeddings[0].vector.copy()
        changed_vector[0], changed_vector[1] = 0.0, 1.0
        with self.assertRaisesRegex(RagImportConflict, "Conflicting embedding vector"):
            await import_rag_embeddings(
                self.connection,
                replace(
                    plan,
                    embeddings=(replace(plan.embeddings[0], vector=changed_vector),),
                ),
                schema=self.schema,
            )

    async def test_new_chunks_cannot_extend_existing_published_version(self) -> None:
        """Reject re-chunked content under an active version and keep it unchanged."""
        plan = make_plan()
        await import_rag_embeddings(self.connection, plan, schema=self.schema)
        # Same version metadata, but an extra chunk with a new stable ID, as
        # produced by re-chunking an unchanged source with another configuration.
        extra_chunk = replace(
            plan.chunks[0], chunk_id="chk_" + "f" * 64, chunk_index=1
        )
        extra_embedding = replace(plan.embeddings[0], chunk_id=extra_chunk.chunk_id)
        extended_plan = replace(
            plan,
            chunks=plan.chunks + (extra_chunk,),
            embeddings=plan.embeddings + (extra_embedding,),
        )

        with self.assertRaisesRegex(RagImportConflict, "stage a new version"):
            await import_rag_embeddings(
                self.connection, extended_plan, schema=self.schema
            )
        self.assertEqual(await self._count("document_chunks"), 1)
        self.assertEqual(await self._count("chunk_embeddings"), 1)
        retry = await import_rag_embeddings(self.connection, plan, schema=self.schema)
        self.assertEqual(retry.counts.chunks_reused, 1)

    async def test_concurrent_identical_imports_create_one_identity_set(self) -> None:
        """Serialize shared documents and reuse the winner's committed rows."""
        second_connection = await asyncpg.connect(self.dsn, timeout=5)
        self.addAsyncCleanup(second_connection.close)
        plan = make_plan("concurrent")
        results = await asyncio.gather(
            import_rag_embeddings(self.connection, plan, schema=self.schema),
            import_rag_embeddings(second_connection, plan, schema=self.schema),
        )
        self.assertEqual(
            sum(result.counts.embeddings_inserted for result in results), 1
        )
        self.assertEqual(
            sum(result.counts.embeddings_reused for result in results), 1
        )
        self.assertEqual(await self._count("chunk_embeddings"), 1)

    async def test_month1_rows_remain_intact(self) -> None:
        """Leave pre-existing application records unchanged after a RAG import."""
        await self.connection.execute(f'SET search_path TO "{self.schema}", public')
        analysis_id = await self.connection.fetchval(
            """
            INSERT INTO analyses
                (message, category, urgency, summary, entities, sentiment, latency_ms)
            VALUES ('keep', 'test', 'low', 'keep', '{}', 'neutral', 1)
            RETURNING id
            """
        )
        await import_rag_embeddings(
            self.connection, make_plan(), schema=self.schema
        )
        await self.connection.execute(f'SET search_path TO "{self.schema}", public')
        self.assertEqual(
            await self.connection.fetchval(
                "SELECT message FROM analyses WHERE id=$1", analysis_id
            ),
            "keep",
        )


if __name__ == "__main__":
    unittest.main()
