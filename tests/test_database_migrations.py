"""Verify the deployment migration runner against a disposable PostgreSQL database.

The class is skipped unless TEST_DATABASE_URL names a database containing "test".
Each test owns a random schema and never modifies existing application schemas.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

import asyncpg
from sqlalchemy.engine import make_url

from scripts.migrate_database import Migration, MigrationError, MigrationRunner


ROOT = Path(__file__).resolve().parent.parent
INITIAL_SCHEMA_SQL = (ROOT / "schema.sql").read_text(encoding="utf-8")
TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "")


@unittest.skipUnless(TEST_DATABASE_URL, "TEST_DATABASE_URL is not set")
class DatabaseMigrationIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """Exercise fresh, upgrade, repeat, drift, and recovery behavior."""

    async def asyncSetUp(self) -> None:
        """Create an isolated schema in an explicitly disposable database."""
        url = make_url(TEST_DATABASE_URL)
        if url.drivername != "postgresql+asyncpg" or "test" not in (
            url.database or ""
        ).lower():
            self.fail(
                "TEST_DATABASE_URL must use asyncpg and name a database containing 'test'"
            )
        dsn = url.set(drivername="postgresql").render_as_string(hide_password=False)
        try:
            self.connection = await asyncpg.connect(dsn, timeout=5)
        except OSError:
            self.skipTest("Disposable PostgreSQL database is unreachable")
        self.addAsyncCleanup(self.connection.close)
        self.schema = f"migration_test_{uuid4().hex}"
        await self.connection.execute(f'CREATE SCHEMA "{self.schema}"')
        self.addAsyncCleanup(self._drop_schema)
        self.runner = MigrationRunner(TEST_DATABASE_URL, schema=self.schema)

    async def _drop_schema(self) -> None:
        """Remove only this test's random schema."""
        await self.connection.execute(f'DROP SCHEMA IF EXISTS "{self.schema}" CASCADE')

    async def _set_test_search_path(self) -> None:
        """Select the isolated schema for direct assertions and fixtures."""
        await self.connection.execute(f'SET search_path TO "{self.schema}", public')

    async def test_fresh_upgrade_records_complete_history(self) -> None:
        """Provision Month 1, pgvector, and RAG objects through one command."""
        self.assertEqual(await self.runner.upgrade(), ["000", "001", "002"])
        await self._set_test_search_path()

        recorded = await self.connection.fetch(
            "SELECT version, name FROM schema_migrations ORDER BY version"
        )
        self.assertEqual(
            [(row["version"], row["name"]) for row in recorded],
            [
                ("000", "initial_schema"),
                ("001", "enable_pgvector"),
                ("002", "create_rag_schema"),
            ],
        )
        tables = await self.connection.fetch(
            "SELECT tablename FROM pg_tables WHERE schemaname = $1", self.schema
        )
        self.assertTrue(
            {
                "analyses",
                "documents",
                "document_versions",
                "document_chunks",
                "chunk_embeddings",
                "schema_migrations",
            }.issubset({row["tablename"] for row in tables})
        )
        extension_version = await self.connection.fetchval(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        )
        self.assertRegex(extension_version, r"^\d+\.\d+")

    async def test_existing_month1_upgrade_preserves_data(self) -> None:
        """Validate and adopt the legacy table without recreating or changing it."""
        await self._set_test_search_path()
        await self.connection.execute(INITIAL_SCHEMA_SQL)
        analysis_id = await self.connection.fetchval(
            """
            INSERT INTO analyses
                (message, category, urgency, summary, entities, sentiment, latency_ms)
            VALUES ('preserve-me', 'billing', 'low', 'fixture', '{}', 'neutral', 7)
            RETURNING id
            """
        )

        self.assertEqual(await self.runner.upgrade(), ["000", "001", "002"])
        await self._set_test_search_path()
        stored = await self.connection.fetchrow(
            "SELECT message, latency_ms FROM analyses WHERE id = $1", analysis_id
        )
        self.assertEqual(dict(stored), {"message": "preserve-me", "latency_ms": 7})
        self.assertIsNotNone(await self.connection.fetchval("SELECT to_regclass('documents')"))

    async def test_repeated_upgrade_is_a_no_op_and_status_is_current(self) -> None:
        """Avoid duplicate objects and report every revision as applied."""
        await self.runner.upgrade()
        await self._set_test_search_path()
        documents_oid = await self.connection.fetchval("SELECT 'documents'::regclass::oid")

        self.assertEqual(await self.runner.upgrade(), [])
        self.assertEqual(
            [(migration.version, state) for migration, state in await self.runner.status()],
            [("000", "applied"), ("001", "applied"), ("002", "applied")],
        )
        await self._set_test_search_path()
        self.assertEqual(
            documents_oid,
            await self.connection.fetchval("SELECT 'documents'::regclass::oid"),
        )

    async def test_failed_revision_is_not_recorded_and_can_be_retried(self) -> None:
        """Roll back failed DDL, preserve the prior revision, and allow a fixed retry."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            first = directory / "000.sql"
            second = directory / "001.sql"
            first.write_text(
                "CREATE TABLE stable_fixture (id INTEGER PRIMARY KEY);",
                encoding="utf-8",
            )
            second.write_text(
                "CREATE TABLE rolled_back_fixture (id INTEGER); SELECT missing_column;",
                encoding="utf-8",
            )
            migrations = (
                Migration("000", "stable", first, None),
                Migration("001", "recoverable", second, "000"),
            )
            runner = MigrationRunner(
                TEST_DATABASE_URL, schema=self.schema, migrations=migrations
            )

            with self.assertRaises(asyncpg.PostgresError):
                await runner.upgrade()
            await self._set_test_search_path()
            self.assertEqual(
                await self.connection.fetchval(
                    "SELECT array_agg(version ORDER BY version) FROM schema_migrations"
                ),
                ["000"],
            )
            self.assertIsNone(
                await self.connection.fetchval("SELECT to_regclass('rolled_back_fixture')")
            )

            second.write_text(
                "CREATE TABLE recovered_fixture (id INTEGER PRIMARY KEY);",
                encoding="utf-8",
            )
            self.assertEqual(await runner.upgrade(), ["001"])
            await self._set_test_search_path()
            self.assertIsNotNone(
                await self.connection.fetchval("SELECT to_regclass('recovered_fixture')")
            )

    async def test_checksum_drift_blocks_upgrade(self) -> None:
        """Refuse to continue after an applied migration file is changed."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            migration_path = Path(temporary_directory) / "000.sql"
            migration_path.write_text(
                "CREATE TABLE checksum_fixture (id INTEGER);", encoding="utf-8"
            )
            migration = Migration("000", "checksum", migration_path, None)
            runner = MigrationRunner(
                TEST_DATABASE_URL, schema=self.schema, migrations=(migration,)
            )
            await runner.upgrade()
            migration_path.write_text(
                "CREATE TABLE checksum_fixture (id BIGINT);", encoding="utf-8"
            )

            with self.assertRaisesRegex(MigrationError, "checksum differs"):
                await runner.upgrade()
            self.assertEqual((await runner.status())[0][1], "checksum-mismatch")


if __name__ == "__main__":
    unittest.main()
