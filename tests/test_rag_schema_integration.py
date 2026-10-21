"""Exercise the Phase 6.2 SQL contract in disposable PostgreSQL schemas.

Run only against a database whose name contains 'test' via TEST_DATABASE_URL.
These tests never install extensions or modify existing application schemas.
"""

from collections.abc import Awaitable
from datetime import date
import os
from pathlib import Path
import unittest
from uuid import uuid4

import asyncpg
from sqlalchemy.engine import make_url


ROOT = Path(__file__).resolve().parent.parent
MIGRATION_SQL = (ROOT / "migrations/002_create_rag_schema.sql").read_text()
INITIAL_SCHEMA_SQL = (ROOT / "schema.sql").read_text()
TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "")
VECTOR = "[" + ",".join(["1"] + ["0"] * 383) + "]"


@unittest.skipUnless(TEST_DATABASE_URL, "TEST_DATABASE_URL is not set")
class RagSchemaIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """Check actual constraints and migration behavior without corpus imports."""

    async def asyncSetUp(self) -> None:
        """Connect to an explicitly disposable database and install isolated tables."""
        url = make_url(TEST_DATABASE_URL)
        if url.drivername != "postgresql+asyncpg" or "test" not in (url.database or "").lower():
            self.fail("TEST_DATABASE_URL must use asyncpg and name a disposable database containing 'test'")
        try:
            self.connection = await asyncpg.connect(
                url.set(drivername="postgresql").render_as_string(hide_password=False),
                timeout=5,
            )
        except OSError:
            self.skipTest("Disposable PostgreSQL database is unreachable")
        self.addAsyncCleanup(self.connection.close)
        self.schema = f"rag_test_{uuid4().hex}"
        await self.connection.execute(f'CREATE SCHEMA "{self.schema}"')
        self.addAsyncCleanup(self._drop_schema)
        await self.connection.execute(f'SET search_path TO "{self.schema}", public')
        # Existing Month 1 objects are installed only in this throwaway schema.
        await self.connection.execute(INITIAL_SCHEMA_SQL)
        await self.connection.execute(MIGRATION_SQL)

    async def _drop_schema(self) -> None:
        """Remove only this test's internally generated schema after success or failure."""
        await self.connection.execute(f'DROP SCHEMA "{self.schema}" CASCADE')

    async def _insert(self, table: str, values: dict[str, object]) -> int | str:
        """Insert a trusted fixture with parameterized values and return its key."""
        # Table/column identifiers come exclusively from fixtures in this file;
        # no caller-provided identifiers or values are interpolated into SQL.
        columns = ", ".join(values)
        parameters = ", ".join(f"${index}" for index in range(1, len(values) + 1))
        key = "chunk_id" if table == "document_chunks" else "id"
        identifier = await self.connection.fetchval(
            f"INSERT INTO {table} ({columns}) VALUES ({parameters}) RETURNING {key}",
            *values.values(),
        )
        return identifier

    async def _document(self, **overrides: object) -> int:
        """Create a tenant-owned document fixture with a distinct external identity."""
        identifier = await self._insert("documents", {
            "tenant_id": "asterdesk", "document_id": uuid4().hex, **overrides,
        })
        assert isinstance(identifier, int)
        return identifier

    async def _version(self, document_fk: int, **overrides: object) -> int:
        """Create complete source metadata matching the actual manifest contract."""
        identifier = await self._insert("document_versions", {
            "document_fk": document_fk, "version": "2.0", "title": "Fixture policy",
            "effective_date": date(2026, 7, 1),
            "status": "active", "visibility": "public", "source_sha256": "a" * 64,
            "source_path": "documents/asterdesk/fixture.md", "source_format": "md",
            "synthetic": True, "topic_family": "payments", "word_count": 3,
            "pdf_pages": None, "section_count": 1, "scope_note": "Synthetic fixture",
            **overrides,
        })
        assert isinstance(identifier, int)
        return identifier

    async def _chunk(self, document_version_fk: int, **overrides: object) -> str:
        """Create a stable chunk fixture with zero-based block provenance."""
        identifier = await self._insert("document_chunks", {
            "chunk_id": "chk_" + uuid4().hex * 2,
            "document_version_fk": document_version_fk, "chunk_index": 0,
            "text": "Identical source text", "token_count": 3,
            "heading_path": ["Policy"], "start_page": None, "end_page": None,
            "source_block_ids": [0, 1], "tokenizer_name": "o200k_base",
            "tokenizer_version": "0.12.0", "chunking_config_version": "phase4-v4",
            "chunking_configuration": '{"config_version":"phase4-v4"}',
            "chunking_config_fingerprint": "b" * 64, **overrides,
        })
        assert isinstance(identifier, str)
        return identifier

    async def _embedding(self, chunk_id: str, **overrides: object) -> int:
        """Create a nonzero 384-dimensional fixture without running a model."""
        identifier = await self._insert("chunk_embeddings", {
            "chunk_id": chunk_id, "embedding": VECTOR,
            "embedding_model_id": "BAAI/bge-small-en-v1.5",
            "embedding_model_revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
            "embedding_config_fingerprint": "c" * 64,
            "input_fingerprint": "emb_" + "d" * 64,
            "document_instruction": "", "query_instruction": "Represent this sentence: ",
            "normalize_embeddings": True, "encoding_config_version": "phase5-v1",
            **overrides,
        })
        assert isinstance(identifier, int)
        return identifier

    async def _assert_rejected(self, operation: Awaitable[object], sqlstate: str) -> None:
        """Roll back an expected database rejection and verify its failure category."""
        transaction = self.connection.transaction()
        await transaction.start()
        try:
            with self.assertRaises(asyncpg.PostgresError) as raised:
                await operation
            self.assertEqual(raised.exception.sqlstate, sqlstate)
        finally:
            await transaction.rollback()

    async def test_document_identity_and_tenant_separation(self) -> None:
        """Reject duplicate tenant identities while allowing the same ID elsewhere."""
        await self._document(document_id="shared")
        await self._document(tenant_id="brookhaven", document_id="shared")
        await self._assert_rejected(self._document(document_id="shared"), "23505")
        for field in ("tenant_id", "document_id"):
            for value in ("", " ", "\t", None):
                with self.subTest(field=field, value=value):
                    await self._assert_rejected(
                        self._document(**{field: value}), "23502" if value is None else "23514",
                    )

    async def test_document_identity_cannot_be_reassigned(self) -> None:
        """Keep descendants from being transferred through an identity UPDATE."""
        document = await self._document()
        for field in ("tenant_id", "document_id"):
            await self._assert_rejected(self.connection.execute(
                f"UPDATE documents SET {field} = 'changed' WHERE id = $1", document,
            ), "23514")

    async def test_version_identity_lifecycle_and_visibility(self) -> None:
        """Preserve history and reject duplicate, orphaned, or invalid versions."""
        document = await self._document()
        await self._version(document)
        await self._version(document, version="1.0", status="archived")
        await self._assert_rejected(self._version(document), "23505")
        await self._assert_rejected(self._version(document, version="3.0"), "23505")
        await self._assert_rejected(self._version(-1), "23503")
        for field, value in (("status", "deleted"), ("visibility", "private"),
                             ("version", " "), ("source_sha256", "invalid"),
                             ("word_count", -1), ("section_count", -1),
                             ("pdf_pages", 2), ("source_format", "docx")):
            with self.subTest(field=field):
                overrides = {"version": uuid4().hex, "status": "archived", field: value}
                await self._assert_rejected(self._version(document, **overrides), "23514")
        # Preserve the old edition while replacing its active lifecycle state.
        async with self.connection.transaction():
            await self.connection.execute(
                "UPDATE document_versions SET status='archived' WHERE document_fk=$1 AND status='active'",
                document,
            )
            await self._version(document, version="3.0")
        self.assertEqual(await self.connection.fetchval(
            "SELECT count(*) FROM document_versions WHERE document_fk=$1", document,
        ), 3)

    async def test_chunk_identity_configuration_and_provenance(self) -> None:
        """Retain equal text across sources and separate chunking configurations."""
        first_version = await self._version(await self._document())
        second_version = await self._version(await self._document(tenant_id="brookhaven"))
        first = await self._chunk(first_version)
        second = await self._chunk(second_version)
        await self._chunk(first_version, chunking_config_fingerprint="e" * 64)
        await self._assert_rejected(self._chunk(second_version, chunk_id=first, chunk_index=1), "23505")
        await self._assert_rejected(self._chunk(first_version), "23505")
        rows = await self.connection.fetch(
            "SELECT c.chunk_id, d.tenant_id, c.source_block_ids FROM document_chunks c "
            "JOIN document_versions v ON v.id = c.document_version_fk "
            "JOIN documents d ON d.id = v.document_fk WHERE c.chunk_id = ANY($1::text[])",
            [first, second],
        )
        self.assertEqual({row["tenant_id"] for row in rows}, {"asterdesk", "brookhaven"})
        self.assertTrue(all(row["source_block_ids"] == [0, 1] for row in rows))

    async def test_chunk_invalid_values_are_rejected(self) -> None:
        """Reject broken text, sequence, page ranges, and block provenance."""
        version = await self._version(await self._document())
        await self._assert_rejected(self._chunk(-1), "23503")
        cases = [
            {"text": " \n\t"}, {"chunk_index": -1}, {"token_count": 0},
            {"token_count": -1}, {"chunk_id": "invalid"},
            {"start_page": 1}, {"start_page": 2, "end_page": 1},
            {"start_page": 0, "end_page": 1}, {"source_block_ids": []},
            {"source_block_ids": [-1]}, {"source_block_ids": [None]},
            {"heading_path": [None]}, {"chunking_configuration": "[]"},
            {"chunking_config_fingerprint": "invalid"},
        ]
        for invalid in cases:
            with self.subTest(invalid=invalid):
                await self._assert_rejected(self._chunk(version, **invalid), "23514")

    async def test_embedding_round_trip_and_configuration_identity(self) -> None:
        """Read vectors back and distinguish independent encoding configurations."""
        chunk = await self._chunk(await self._version(await self._document()))
        embedding = await self._embedding(chunk)
        row = await self.connection.fetchrow(
            "SELECT embedding::text AS vector, public.vector_dims(embedding) AS dimensions "
            "FROM chunk_embeddings WHERE id = $1", embedding,
        )
        self.assertEqual(row["dimensions"], 384)
        self.assertEqual(row["vector"], VECTOR)
        await self._assert_rejected(self._embedding(chunk), "23505")
        await self._assert_rejected(self._embedding(chunk, input_fingerprint="emb_" + "e" * 64), "23505")
        await self._embedding(chunk, embedding_config_fingerprint="f" * 64, document_instruction="Passage: ")

    async def test_embedding_invalid_values_are_rejected(self) -> None:
        """Reject missing parents, dimensions, zero/nonfinite vectors, and metadata."""
        chunk = await self._chunk(await self._version(await self._document()))
        await self._assert_rejected(self._embedding("chk_" + "0" * 64), "23503")
        await self._assert_rejected(self._embedding(chunk, embedding="[1,2,3]"), "22000")
        await self._assert_rejected(self._embedding(chunk, embedding="[" + ",".join(["0"] * 384) + "]"), "23514")
        for nonfinite in ("NaN", "Infinity"):
            await self._assert_rejected(self._embedding(
                chunk, embedding="[" + ",".join([nonfinite] + ["0"] * 383) + "]",
            ), "22000")
        for field in ("embedding", "embedding_model_id", "embedding_model_revision",
                      "embedding_config_fingerprint", "input_fingerprint", "encoding_config_version"):
            with self.subTest(field=field):
                await self._assert_rejected(self._embedding(chunk, **{field: None}), "23502")
        for field in ("embedding_model_id", "embedding_model_revision",
                      "embedding_config_fingerprint", "input_fingerprint", "encoding_config_version"):
            await self._assert_rejected(self._embedding(chunk, **{field: ""}), "23514")

    async def test_parent_deletions_are_restricted(self) -> None:
        """Require explicit child-first deletion rather than destructive cascades."""
        document = await self._document()
        version = await self._version(document)
        chunk = await self._chunk(version)
        await self._embedding(chunk)
        for table, key, identifier in (("documents", "id", document),
                                       ("document_versions", "id", version),
                                       ("document_chunks", "chunk_id", chunk)):
            await self._assert_rejected(self.connection.execute(
                f"DELETE FROM {table} WHERE {key} = $1", identifier,
            ), "23503")

    async def test_migration_repeatability_and_month1_preservation(self) -> None:
        """Keep existing analysis data and created object identities on reapplication."""
        analysis = await self.connection.fetchval(
            "INSERT INTO analyses (message, category, urgency, summary, entities, sentiment, latency_ms) "
            "VALUES ('fixture', 'billing', 'low', 'fixture', '{}', 'neutral', 1) RETURNING id"
        )
        before = await self.connection.fetchval("SELECT 'documents'::regclass::oid")
        await self.connection.execute(MIGRATION_SQL)
        self.assertEqual(before, await self.connection.fetchval("SELECT 'documents'::regclass::oid"))
        self.assertEqual(await self.connection.fetchval("SELECT message FROM analyses WHERE id=$1", analysis), "fixture")
        self.assertEqual(await self.connection.fetchval("SELECT count(*) FROM rag_schema_migrations"), 1)
        self.assertAlmostEqual(await self.connection.fetchval(
            "SELECT 1 - ('[1,0,0]'::public.vector <=> '[1,0,0]'::public.vector)"
        ), 1.0)

    async def test_migration_fails_atomically_on_unexpected_table(self) -> None:
        """Detect a partial installation without changing pre-existing objects."""
        schema = f"rag_partial_{uuid4().hex}"
        await self.connection.execute(f'CREATE SCHEMA "{schema}"')
        try:
            await self.connection.execute(f'SET search_path TO "{schema}", public')
            await self.connection.execute("CREATE TABLE documents (sentinel INTEGER)")
            await self._assert_rejected(self.connection.execute(MIGRATION_SQL), "P0001")
            self.assertIsNone(await self.connection.fetchval(
                "SELECT to_regclass($1)", f"{schema}.rag_schema_migrations",
            ))
        finally:
            await self.connection.execute(f'SET search_path TO "{self.schema}", public')
            await self.connection.execute(f'DROP SCHEMA "{schema}" CASCADE')

    async def test_schema_objects_and_vector_type(self) -> None:
        """Inspect actual keys, trigger, relational indexes, and vector typmod."""
        tables = await self.connection.fetch(
            "SELECT tablename FROM pg_tables WHERE schemaname=$1", self.schema,
        )
        self.assertEqual({row["tablename"] for row in tables}, {
            "analyses", "documents", "document_versions", "document_chunks",
            "chunk_embeddings", "rag_schema_migrations",
        })
        # Check every approved column/type, rather than just table existence.
        expected_columns = {
            "documents": {
                "id": "bigint", "tenant_id": "text", "document_id": "text",
                "created_at": "timestamp with time zone",
            },
            "document_versions": {
                "id": "bigint", "document_fk": "bigint", "version": "text",
                "title": "text", "effective_date": "date", "status": "text",
                "visibility": "text", "source_sha256": "text", "source_path": "text",
                "source_format": "text", "synthetic": "boolean", "topic_family": "text",
                "word_count": "integer", "pdf_pages": "integer", "section_count": "integer",
                "scope_note": "text", "created_at": "timestamp with time zone",
            },
            "document_chunks": {
                "chunk_id": "text", "document_version_fk": "bigint", "chunk_index": "integer",
                "text": "text", "token_count": "integer", "heading_path": "text[]",
                "start_page": "integer", "end_page": "integer", "source_block_ids": "integer[]",
                "tokenizer_name": "text", "tokenizer_version": "text",
                "chunking_config_version": "text", "chunking_configuration": "jsonb",
                "chunking_config_fingerprint": "text", "created_at": "timestamp with time zone",
            },
            "chunk_embeddings": {
                "id": "bigint", "chunk_id": "text", "embedding": "vector(384)",
                "embedding_model_id": "text", "embedding_model_revision": "text",
                "embedding_config_fingerprint": "text", "input_fingerprint": "text",
                "document_instruction": "text", "query_instruction": "text",
                "normalize_embeddings": "boolean", "encoding_config_version": "text",
                "created_at": "timestamp with time zone",
            },
        }
        nullable = {("document_versions", "pdf_pages"),
                    ("document_chunks", "start_page"), ("document_chunks", "end_page")}
        for table, expected in expected_columns.items():
            columns = await self.connection.fetch(
                "SELECT attname, format_type(atttypid, atttypmod) AS sql_type, attnotnull "
                "FROM pg_attribute WHERE attrelid=$1::regclass AND attnum>0 AND NOT attisdropped",
                table,
            )
            self.assertEqual({row["attname"]: row["sql_type"] for row in columns}, expected)
            for column in columns:
                self.assertEqual(column["attnotnull"], (table, column["attname"]) not in nullable)
        constraints = await self.connection.fetch(
            "SELECT contype, confdeltype, confupdtype FROM pg_constraint "
            "WHERE connamespace=$1::regnamespace AND contype IN ('p','f','u')", self.schema,
        )
        foreign_keys = [row for row in constraints if row["contype"] == b"f" or row["contype"] == "f"]
        self.assertEqual(len(foreign_keys), 3)
        self.assertEqual(sum(row["contype"] in (b"p", "p") for row in constraints), 6)
        self.assertTrue(all(row["confdeltype"] in (b"r", "r") and row["confupdtype"] in (b"r", "r") for row in foreign_keys))
        self.assertEqual(await self.connection.fetchval(
            "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
            "WHERE attrelid='chunk_embeddings'::regclass AND attname='embedding'"
        ), "vector(384)")
        indexes = await self.connection.fetch(
            "SELECT indexname, indexdef FROM pg_indexes WHERE schemaname=$1", self.schema,
        )
        self.assertTrue(any(row["indexname"] == "uq_document_versions_one_active" for row in indexes))
        self.assertFalse(any("USING hnsw" in row["indexdef"] or "USING ivfflat" in row["indexdef"] for row in indexes))
        unique_constraints = await self.connection.fetch(
            "SELECT conname FROM pg_constraint WHERE connamespace=$1::regnamespace AND contype='u'",
            self.schema,
        )
        self.assertEqual({row["conname"] for row in unique_constraints}, {
            "uq_documents_tenant_document", "uq_document_versions_document_version",
            "uq_document_chunks_version_config_index", "uq_chunk_embeddings_identity",
        })
        self.assertEqual(await self.connection.fetchval(
            "SELECT count(*) FROM pg_trigger WHERE tgrelid='documents'::regclass "
            "AND tgname='documents_identity_immutable' AND NOT tgisinternal"
        ), 1)


if __name__ == "__main__":
    unittest.main()
