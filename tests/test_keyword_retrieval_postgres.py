"""Validate keyword-retrieval authorization and filters in real PostgreSQL."""

from __future__ import annotations

import os
from pathlib import Path
import unittest
from uuid import uuid4

import asyncpg
from sqlalchemy.engine import make_url

from app.rag.retrieval import (
    HistoricalVersionScope,
    KeywordRetriever,
    SearchContextOrigin,
    SearchFilters,
    TrustedSearchContext,
)


ROOT = Path(__file__).resolve().parent.parent
PGVECTOR_SQL = (ROOT / "migrations/001_enable_pgvector.sql").read_text()
RAG_SCHEMA_SQL = (ROOT / "migrations/002_create_rag_schema.sql").read_text()
LIFECYCLE_SQL = (ROOT / "migrations/003_add_document_lifecycle.sql").read_text()
TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "")


def context(
    tenant: str = "tenant-a",
    visibilities: tuple[str, ...] = ("public",),
    historical: HistoricalVersionScope | None = None,
) -> TrustedSearchContext:
    """Build one trusted internal context for the fixture tenant."""
    return TrustedSearchContext(
        tenant, visibilities, SearchContextOrigin.INTERNAL,
        historical, historical is not None,
    )


@unittest.skipUnless(TEST_DATABASE_URL, "TEST_DATABASE_URL is not set")
class KeywordSearchPostgresTests(unittest.IsolatedAsyncioTestCase):
    """Verify the lexical SQL scope in a disposable PostgreSQL schema."""

    async def asyncSetUp(self) -> None:
        """Create an isolated schema and install the existing RAG migrations."""
        url = make_url(TEST_DATABASE_URL)
        self.connection = await asyncpg.connect(
            url.set(drivername="postgresql").render_as_string(hide_password=False)
        )
        await self.connection.execute(PGVECTOR_SQL)
        self.schema = f"keyword_{uuid4().hex}"
        await self.connection.execute(f'CREATE SCHEMA "{self.schema}"')
        await self.connection.execute(f'SET search_path TO "{self.schema}", public')
        await self.connection.execute(RAG_SCHEMA_SQL)
        await self.connection.execute(LIFECYCLE_SQL)
        await self._insert_fixture_rows()
        self.retriever = KeywordRetriever(self.connection, schema=self.schema)

    async def asyncTearDown(self) -> None:
        """Remove only the schema created by this test case."""
        await self.connection.execute('SET search_path TO public')
        await self.connection.execute(f'DROP SCHEMA "{self.schema}" CASCADE')
        await self.connection.close()

    async def _insert_chunk(
        self,
        *,
        tenant: str,
        document: str,
        version: str,
        status: str,
        visibility: str,
        suffix: str,
        heading: str = "[DECISION] Controlling rules",
        effective_date: str = "2026-01-01",
    ) -> str:
        """Insert one document/version/chunk path; keyword search needs no vector."""
        document_pk = await self.connection.fetchval(
            "INSERT INTO documents (tenant_id, document_id) VALUES ($1, $2) "
            "ON CONFLICT (tenant_id, document_id) DO UPDATE SET document_id=EXCLUDED.document_id "
            "RETURNING id",
            tenant,
            document,
        )
        version_pk = await self.connection.fetchval(
            """
            INSERT INTO document_versions (
                document_fk, version, title, effective_date, status, visibility,
                source_sha256, source_path, source_format, synthetic, topic_family,
                word_count, section_count, scope_note, expected_chunk_count,
                expected_chunk_ids_fingerprint
            ) VALUES ($1,$2,$3,$4::date,$5,$6,$7,$8,'md',true,'testing',
                      10,1,'fixture',1,$9)
            RETURNING id
            """,
            document_pk,
            version,
            f"Title {document}",
            effective_date,
            status,
            visibility,
            suffix * 64,
            f"sources/{document}-{version}.md",
            "9" * 64,
        )
        chunk_id = "chk_" + suffix * 64
        await self.connection.execute(
            """
            INSERT INTO document_chunks (
                chunk_id, document_version_fk, chunk_index, text, token_count,
                heading_path, source_block_ids, tokenizer_name, tokenizer_version,
                chunking_config_version, chunking_configuration,
                chunking_config_fingerprint
            ) VALUES ($1,$2,0,$3,4,ARRAY['Policy', $4],ARRAY[7],
                      'fixture','1','phase4-v4','{}'::jsonb,$5)
            """,
            chunk_id,
            version_pk,
            f"duplicate refund rule for {document} {version}",
            heading,
            "b" * 64,
        )
        return chunk_id

    async def _insert_fixture_rows(self) -> None:
        """Create one eligible row plus rows each excluded by one predicate."""
        self.active = await self._insert_chunk(
            tenant="tenant-a", document="DOC-A", version="2.0", status="active",
            visibility="public", suffix="a",
        )
        self.faq = await self._insert_chunk(
            tenant="tenant-a", document="DOC-B", version="1.0", status="active",
            visibility="public", suffix="b", heading="[FAQ] Support questions",
            effective_date="2025-06-01",
        )
        self.internal = await self._insert_chunk(
            tenant="tenant-a", document="DOC-C", version="1.0", status="active",
            visibility="internal", suffix="c",
        )
        self.archived = await self._insert_chunk(
            tenant="tenant-a", document="DOC-A", version="1.0", status="archived",
            visibility="public", suffix="d",
        )
        self.staged = await self._insert_chunk(
            tenant="tenant-a", document="DOC-STAGED", version="1.0", status="staged",
            visibility="public", suffix="1",
        )
        self.withdrawn = await self._insert_chunk(
            tenant="tenant-a", document="DOC-WITHDRAWN", version="1.0",
            status="withdrawn", visibility="public", suffix="2",
        )
        self.foreign = await self._insert_chunk(
            tenant="tenant-b", document="DOC-E", version="1.0", status="active",
            visibility="public", suffix="e",
        )

    async def _ids(self, scope: TrustedSearchContext, filters: SearchFilters = SearchFilters()) -> set[str]:
        """Return the chunk IDs a broad matching query retrieves in one scope."""
        response = await self.retriever.search("duplicate refund rule", scope, filters, 20)
        return {match.chunk.chunk_id for match in response.results}

    async def test_default_scope_returns_only_active_public_tenant_rows(self) -> None:
        """Exclude foreign, internal, archived, staged, and withdrawn rows."""
        response = await self.retriever.search(
            "duplicate refund rule", context(), SearchFilters(), 20
        )
        self.assertEqual({m.chunk.chunk_id for m in response.results}, {self.active, self.faq})
        self.assertEqual(response.candidate_count, 2)
        self.assertTrue(all(m.chunk.tenant_id == "tenant-a" for m in response.results))

    async def test_visibility_is_granted_only_by_trusted_context(self) -> None:
        """Return internal rows only when the trusted context permits them."""
        self.assertEqual(await self._ids(context(visibilities=("internal",))), {self.internal})
        self.assertEqual(
            await self._ids(context(visibilities=("public", "internal"))),
            {self.active, self.faq, self.internal},
        )

    async def test_foreign_document_filter_cannot_cross_tenants(self) -> None:
        """Treat a client document filter as narrowing, never as authorization."""
        self.assertEqual(await self._ids(context(), SearchFilters(document_id="DOC-E")), set())
        self.assertEqual(await self._ids(context("tenant-b")), {self.foreign})
        self.assertEqual(await self._ids(context("missing")), set())

    async def test_historical_scope_returns_only_the_exact_archived_version(self) -> None:
        """Permit one archived version while denying active, staged, and foreign rows."""
        self.assertEqual(
            await self._ids(context(historical=HistoricalVersionScope("DOC-A", "1.0"))),
            {self.archived},
        )
        for document_id, version in (
            ("DOC-A", "2.0"),
            ("DOC-STAGED", "1.0"),
            ("DOC-WITHDRAWN", "1.0"),
            ("DOC-E", "1.0"),
            ("DOC-A", "missing"),
        ):
            scope = context(historical=HistoricalVersionScope(document_id, version))
            self.assertEqual(await self._ids(scope), set(), document_id)

    async def test_section_and_effective_date_filters_narrow_results(self) -> None:
        """Match bracketed section keys and inclusive effective-date bounds."""
        self.assertEqual(await self._ids(context(), SearchFilters(section="FAQ")), {self.faq})
        self.assertEqual(
            await self._ids(context(), SearchFilters(effective_date_from="2026-01-01")),
            {self.active},
        )
        self.assertEqual(
            await self._ids(context(), SearchFilters(effective_date_to="2025-06-01")),
            {self.faq},
        )
        self.assertEqual(
            await self._ids(
                context(), SearchFilters(document_id="DOC-A", section="FAQ")
            ),
            set(),
        )


if __name__ == "__main__":
    unittest.main()
