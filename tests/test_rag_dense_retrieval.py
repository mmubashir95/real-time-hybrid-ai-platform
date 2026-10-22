"""Validate dense-search contracts and real PostgreSQL filtering/ranking."""

from __future__ import annotations

import os
import asyncio
from pathlib import Path
import unittest
from uuid import uuid4

import asyncpg
import numpy as np
from sqlalchemy.engine import make_url

from app.rag.embeddings.models import EmbeddingConfig
from app.rag.persistence.lifecycle import EmbeddingProfile
from app.rag.retrieval import (
    DenseRetriever,
    DenseSearchRequest,
    HistoricalVersionScope,
    TrustedSearchContext,
)


ROOT = Path(__file__).resolve().parent.parent
RAG_SCHEMA_SQL = (ROOT / "migrations/002_create_rag_schema.sql").read_text()
LIFECYCLE_SQL = (ROOT / "migrations/003_add_document_lifecycle.sql").read_text()
TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "")
PROFILE = EmbeddingProfile(
    model_id="BAAI/bge-small-en-v1.5",
    model_revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    config_fingerprint="c" * 64,
    normalize_embeddings=True,
    encoding_config_version="phase5-v1",
)


def unit_vector(axis: int) -> np.ndarray:
    """Create one normalized 384-dimensional integration-test vector."""
    vector = np.zeros(384, dtype=np.float32)
    vector[axis] = 1.0
    return vector


class FakeEncoder:
    """Provide deterministic query vectors without loading a transformer."""

    config = EmbeddingConfig()

    def encode_query(self, query: str) -> np.ndarray:
        """Return the fixture query vector after recording no external state."""
        return unit_vector(0)


class DenseSearchContractTests(unittest.TestCase):
    """Exercise validation before a database connection is needed."""

    def setUp(self) -> None:
        """Build the smallest trusted current-policy context."""
        self.context = TrustedSearchContext("tenant-a", ("public",))

    def test_question_scope_and_top_k_validation(self) -> None:
        """Reject blank queries, missing scope, invalid visibility, and bad Top-K."""
        with self.assertRaisesRegex(ValueError, "Question"):
            DenseSearchRequest(self.context, PROFILE, question="  ")
        with self.assertRaisesRegex(ValueError, "tenant"):
            TrustedSearchContext("", ("public",))
        with self.assertRaisesRegex(ValueError, "visibility"):
            TrustedSearchContext("tenant-a", ("private",))
        with self.assertRaisesRegex(ValueError, "top_k"):
            DenseSearchRequest(self.context, PROFILE, question="valid", top_k=101)

    def test_query_inputs_are_mutually_exclusive(self) -> None:
        """Require exactly one natural-language or precomputed-vector input."""
        with self.assertRaisesRegex(ValueError, "exactly one"):
            DenseSearchRequest(self.context, PROFILE)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            DenseSearchRequest(
                self.context,
                PROFILE,
                question="valid",
                query_vector=unit_vector(0),
            )

    def test_historical_scope_requires_document_and_version(self) -> None:
        """Prevent a historical request from spanning unrelated documents."""
        with self.assertRaisesRegex(ValueError, "document_id and version"):
            HistoricalVersionScope("DOC-1", "")

    def test_precomputed_vector_validation(self) -> None:
        """Reject wrong-dimensional, nonfinite, and zero query vectors."""
        for vector, message in (
            (np.ones(3), "384 dimensions"),
            (np.full(384, np.nan), "finite"),
            (np.zeros(384), "zero"),
        ):
            request = DenseSearchRequest(
                self.context, PROFILE, query_vector=vector
            )
            with self.assertRaisesRegex(ValueError, message):
                asyncio.run(DenseRetriever(None, None).search(request))

    def test_encoder_profile_mismatch_is_rejected_before_database_access(self) -> None:
        """Refuse query encoding in an incompatible model space."""
        incompatible = EmbeddingProfile(
            "different/model",
            PROFILE.model_revision,
            PROFILE.config_fingerprint,
            True,
            "phase5-v1",
        )
        request = DenseSearchRequest(
            self.context, incompatible, question="valid question"
        )
        with self.assertRaisesRegex(ValueError, "incompatible"):
            asyncio.run(DenseRetriever(None, FakeEncoder()).search(request))


@unittest.skipUnless(TEST_DATABASE_URL, "TEST_DATABASE_URL is not set")
class DenseSearchPostgresTests(unittest.IsolatedAsyncioTestCase):
    """Verify exact pgvector behavior in a disposable PostgreSQL schema."""

    async def asyncSetUp(self) -> None:
        """Create an isolated schema and install the existing RAG migrations."""
        url = make_url(TEST_DATABASE_URL)
        self.connection = await asyncpg.connect(
            url.set(drivername="postgresql").render_as_string(hide_password=False)
        )
        self.schema = f"dense_{uuid4().hex}"
        await self.connection.execute(f'CREATE SCHEMA "{self.schema}"')
        await self.connection.execute(f'SET search_path TO "{self.schema}", public')
        await self.connection.execute(RAG_SCHEMA_SQL)
        await self.connection.execute(LIFECYCLE_SQL)
        await self._insert_fixture_rows()

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
        vector: np.ndarray,
        profile: EmbeddingProfile = PROFILE,
    ) -> str:
        """Insert one complete document/version/chunk/embedding fixture path."""
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
            ) VALUES ($1,$2,$3,'2026-01-01',$4,$5,$6,$7,'md',true,'testing',
                      10,1,'fixture',1,$8)
            RETURNING id
            """,
            document_pk,
            version,
            f"Title {document}",
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
            ) VALUES ($1,$2,0,$3,4,ARRAY['Policy'],ARRAY[7],
                      'fixture','1','phase4-v4','{}'::jsonb,$4)
            """,
            chunk_id,
            version_pk,
            f"Text for {document} {version}",
            "b" * 64,
        )
        vector_text = "[" + ",".join(str(float(value)) for value in vector) + "]"
        await self.connection.execute(
            """
            INSERT INTO chunk_embeddings (
                chunk_id, embedding, embedding_model_id, embedding_model_revision,
                embedding_config_fingerprint, input_fingerprint,
                document_instruction, query_instruction, normalize_embeddings,
                encoding_config_version
            ) VALUES ($1,$2::public.vector,$3,$4,$5,$6,'',$7,$8,$9)
            """,
            chunk_id,
            vector_text,
            profile.model_id,
            profile.model_revision,
            profile.config_fingerprint,
            "emb_" + suffix * 64,
            profile.query_instruction,
            profile.normalize_embeddings,
            profile.encoding_config_version,
        )
        return chunk_id

    async def _insert_fixture_rows(self) -> None:
        """Create eligible and deliberately excluded rows for one search."""
        self.best = await self._insert_chunk(
            tenant="tenant-a", document="DOC-A", version="2.0", status="active",
            visibility="public", suffix="a", vector=unit_vector(0)
        )
        near = unit_vector(0)
        near[1] = 0.1
        self.second = await self._insert_chunk(
            tenant="tenant-a", document="DOC-B", version="1.0", status="active",
            visibility="public", suffix="b", vector=near
        )
        self.internal = await self._insert_chunk(
            tenant="tenant-a", document="DOC-C", version="1.0", status="active",
            visibility="internal", suffix="c", vector=unit_vector(0)
        )
        self.archived = await self._insert_chunk(
            tenant="tenant-a", document="DOC-D", version="1.0", status="archived",
            visibility="public", suffix="d", vector=unit_vector(0)
        )
        await self._insert_chunk(
            tenant="tenant-a", document="DOC-STAGED", version="1.0", status="staged",
            visibility="public", suffix="1", vector=unit_vector(0)
        )
        await self._insert_chunk(
            tenant="tenant-a", document="DOC-WITHDRAWN", version="1.0",
            status="withdrawn", visibility="public", suffix="2", vector=unit_vector(0)
        )
        await self._insert_chunk(
            tenant="tenant-b", document="DOC-E", version="1.0", status="active",
            visibility="public", suffix="e", vector=unit_vector(0)
        )
        incompatible = EmbeddingProfile(
            PROFILE.model_id, PROFILE.model_revision, "f" * 64, True, "phase5-v1"
        )
        await self._insert_chunk(
            tenant="tenant-a", document="DOC-F", version="1.0", status="active",
            visibility="public", suffix="f", vector=unit_vector(0), profile=incompatible
        )

    async def test_exact_ranking_and_all_default_filters(self) -> None:
        """Apply tenant, lifecycle, visibility, and model filters before Top-K."""
        request = DenseSearchRequest(
            TrustedSearchContext("tenant-a", ("public",)),
            PROFILE,
            query_vector=unit_vector(0),
            top_k=10,
        )
        response = await DenseRetriever(
            self.connection, None, schema=self.schema
        ).search(request)
        self.assertEqual([item.chunk_id for item in response.results], [self.best, self.second])
        self.assertEqual(response.candidate_count, 2)
        expected = float(
            np.dot(unit_vector(0), unit_vector(0))
            / (np.linalg.norm(unit_vector(0)) ** 2)
        )
        self.assertAlmostEqual(response.results[0].similarity, expected, places=7)
        self.assertEqual(response.results[0].heading_path, ("Policy",))
        self.assertEqual(response.results[0].source_block_ids, (7,))

    async def test_visibility_historical_and_wrong_tenant_scopes(self) -> None:
        """Broaden only explicitly trusted visibility or exact historical scope."""
        internal = DenseSearchRequest(
            TrustedSearchContext("tenant-a", ("internal",)), PROFILE,
            query_vector=unit_vector(0), top_k=5
        )
        internal_response = await DenseRetriever(
            self.connection, None, schema=self.schema
        ).search(internal)
        self.assertEqual([item.chunk_id for item in internal_response.results], [self.internal])

        historical = DenseSearchRequest(
            TrustedSearchContext(
                "tenant-a", ("public",), HistoricalVersionScope("DOC-D", "1.0")
            ),
            PROFILE,
            query_vector=unit_vector(0),
        )
        historical_response = await DenseRetriever(
            self.connection, None, schema=self.schema
        ).search(historical)
        self.assertEqual(
            [item.chunk_id for item in historical_response.results], [self.archived]
        )

        wrong_tenant = DenseSearchRequest(
            TrustedSearchContext("missing", ("public",)), PROFILE,
            query_vector=unit_vector(0)
        )
        empty = await DenseRetriever(
            self.connection, None, schema=self.schema
        ).search(wrong_tenant)
        self.assertEqual(empty.results, ())
        self.assertEqual(empty.candidate_count, 0)


if __name__ == "__main__":
    unittest.main()
