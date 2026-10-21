"""Focused unit, artifact, CLI, and opt-in real-model tests for Phase 5."""

from __future__ import annotations

from dataclasses import replace
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from app.rag.embeddings.models import EmbeddingConfig
from app.rag.embeddings.service import (
    embedding_input_fingerprint,
    generate_embeddings,
    load_embedding_artifacts,
    run_semantic_experiment,
    validate_vectors,
    write_embedding_artifacts,
)
from app.rag.ingestion.models import DocumentChunk
from scripts.embed_rag_documents import main


def make_chunk(index: int, **changes: object) -> DocumentChunk:
    """Create one complete immutable Phase 4 chunk fixture."""
    values: dict[str, object] = {
        "chunk_id": f"chk_{index:064d}",
        "document_id": "AST-REF-001",
        "document_version": "2.0",
        "tenant_id": "asterdesk",
        "document_status": "active",
        "effective_date": "2026-07-01",
        "chunk_index": index,
        "heading_path": ("DECISION",),
        "text": f"Policy content number {index}.",
        "token_count": 5,
        "source_file": "documents/asterdesk/AST-REF-001_v2-0.pdf",
        "source_type": "pdf",
        "start_page": 1,
        "end_page": 1,
        "source_block_ids": (index,),
        "tokenizer_name": "o200k_base",
        "tokenizer_version": "0.12.0",
        "chunking_config_version": "phase4-v4",
        "access_scope": "public",
    }
    values.update(changes)
    return DocumentChunk(**values)  # type: ignore[arg-type]


class FakeEncoder:
    """Deterministic normalized encoder used without model dependencies."""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        """Initialize measurable fake-model properties."""
        self.config = config or EmbeddingConfig(expected_dimensions=3, batch_size=2)
        self.model_load_seconds = 0.01
        self.tokenizer_name = "fake-tokenizer"
        self.model_max_length = 20
        self.batches: list[list[str]] = []

    def prepare_document(self, text: str) -> str:
        """Apply the configured document prefix."""
        return self.config.document_instruction + text

    def count_tokens(self, texts: list[str]) -> list[int]:
        """Count whitespace tokens plus two fake special tokens."""
        return [len(text.split()) + 2 for text in texts]

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        """Return distinct normalized vectors while recording batch order."""
        self.batches.append(list(texts))
        rows = []
        for text in texts:
            value = float(text.rsplit(" ", 1)[-1].rstrip(".")) + 1.0
            vector = np.array([value, 1.0, 2.0], dtype=np.float32)
            rows.append(vector / np.linalg.norm(vector))
        return np.stack(rows)

    def encode_query(self, query: str) -> np.ndarray:
        """Return one normalized query vector."""
        vector = np.array([1.0, 1.0, 2.0], dtype=np.float32)
        return vector / np.linalg.norm(vector)


class EmbeddingGenerationTests(unittest.TestCase):
    """Verify batching, identity, metadata, compatibility, and failures."""

    def test_zero_input_returns_typed_empty_matrix(self) -> None:
        """An empty accepted input should require no inference calls."""
        encoder = FakeEncoder()
        result = generate_embeddings((), encoder, "corpus")
        self.assertEqual(result.vectors.shape, (0, 3))
        self.assertEqual(result.records, ())
        self.assertEqual(encoder.batches, [])

    def test_batches_preserve_order_metadata_and_final_partial_batch(self) -> None:
        """Each chunk maps to one row across complete and partial batches."""
        chunks = tuple(make_chunk(index) for index in range(5))
        encoder = FakeEncoder()
        result = generate_embeddings(chunks, encoder, "corpus-hash")
        self.assertEqual(result.vectors.shape, (5, 3))
        self.assertEqual([len(batch) for batch in encoder.batches], [2, 2, 1])
        self.assertEqual(
            [record.chunk_id for record in result.records],
            [chunk.chunk_id for chunk in chunks],
        )
        first = result.records[0]
        self.assertEqual(
            (first.tenant_id, first.document_id, first.document_version),
            ("asterdesk", "AST-REF-001", "2.0"),
        )
        self.assertEqual(first.heading_path, ("DECISION",))

    def test_duplicate_ids_are_rejected(self) -> None:
        """Duplicate identities cannot produce an ambiguous matrix mapping."""
        duplicate = make_chunk(0, text="Policy content number 1.")
        with self.assertRaisesRegex(ValueError, "Duplicate chunk IDs"):
            generate_embeddings((make_chunk(0), duplicate), FakeEncoder(), "hash")

    def test_overlength_preflight_prevents_inference(self) -> None:
        """No batch may run after a model-token-limit violation."""
        encoder = FakeEncoder()
        encoder.model_max_length = 3
        with self.assertRaisesRegex(ValueError, "exceed model limit"):
            generate_embeddings((make_chunk(0),), encoder, "hash")
        self.assertEqual(encoder.batches, [])

    def test_batch_failure_names_affected_chunks_and_publishes_nothing(self) -> None:
        """Inference errors remain actionable and cannot look complete."""
        encoder = FakeEncoder()

        def fail(_: list[str]) -> np.ndarray:
            raise RuntimeError("synthetic failure")

        encoder.encode_documents = fail  # type: ignore[method-assign]
        with self.assertRaisesRegex(RuntimeError, make_chunk(0).chunk_id):
            generate_embeddings((make_chunk(0),), encoder, "hash")

    def test_fingerprint_is_stable_and_changes_with_input_or_model(self) -> None:
        """Relevant input/model changes must invalidate future reuse."""
        config = EmbeddingConfig()
        original = embedding_input_fingerprint("text", config)
        self.assertEqual(original, embedding_input_fingerprint("text", config))
        self.assertNotEqual(original, embedding_input_fingerprint("changed", config))
        changed = replace(config, model_revision="a" * 40)
        self.assertNotEqual(original, embedding_input_fingerprint("text", changed))


class VectorValidationTests(unittest.TestCase):
    """Reject every important malformed numerical result."""

    def test_shape_and_count_mismatches_are_rejected(self) -> None:
        """Wrong row or dimension counts cannot pass validation."""
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            validate_vectors(np.ones((1, 2)), 2, 2, False)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            validate_vectors(np.ones((2, 3)), 2, 2, False)

    def test_nonfinite_and_zero_vectors_are_rejected(self) -> None:
        """NaN, infinity, and zero vectors are never valid placeholders."""
        for invalid in (np.array([[np.nan, 1.0]]), np.array([[np.inf, 1.0]])):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "NaN or infinity"):
                    validate_vectors(invalid, 1, 2, False)
        with self.assertRaisesRegex(ValueError, "zero vector"):
            validate_vectors(np.zeros((1, 2)), 1, 2, False)

    def test_normalization_policy_is_enforced(self) -> None:
        """Normalized artifacts require approximately unit L2 norms."""
        with self.assertRaisesRegex(ValueError, "L2 norm"):
            validate_vectors(np.array([[2.0, 0.0]]), 1, 2, True)


class ArtifactAndRetrievalTests(unittest.TestCase):
    """Verify durable row alignment and authorization-aware experiments."""

    def test_write_read_round_trip_preserves_vectors_and_metadata(self) -> None:
        """Published NumPy and JSONL rows reload in exactly the same order."""
        config = EmbeddingConfig(expected_dimensions=3, batch_size=2)
        result = generate_embeddings(
            (make_chunk(0), make_chunk(1)), FakeEncoder(config), "corpus"
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = write_embedding_artifacts(Path(directory), result, config)
            loaded_vectors, loaded_records, report = load_embedding_artifacts(
                paths[0].parent
            )
            np.testing.assert_array_equal(loaded_vectors, result.vectors)
            self.assertEqual(loaded_records, result.records)
            self.assertEqual(report["final_status"], "PASSED")

    def test_retrieval_excludes_other_tenants_and_archived_versions(self) -> None:
        """Default experiments rank only active rows in the trusted tenant."""
        chunks = (
            make_chunk(0),
            make_chunk(1, tenant_id="brookhaven", document_id="BKH-REF-101"),
            make_chunk(2, document_version="1.0", document_status="archived"),
        )
        encoder = FakeEncoder()
        result = generate_embeddings(chunks, encoder, "corpus")
        ranked = run_semantic_experiment(
            "refund question", "asterdesk", encoder, result.vectors, result.records
        )
        self.assertEqual([item["chunk_id"] for item in ranked], [chunks[0].chunk_id])


class EmbeddingCliTests(unittest.TestCase):
    """Verify explicit CLI success and failure exit behavior."""

    @patch("scripts.embed_rag_documents.SentenceTransformerEncoder", FakeEncoder)
    @patch("scripts.embed_rag_documents.load_chunks")
    def test_check_only_success_exit_code(
        self, mocked_load: unittest.mock.Mock
    ) -> None:
        """Compatibility-only success returns zero without generating vectors."""
        mocked_load.return_value = ((make_chunk(0),), "hash")
        output = io.StringIO()
        code = main(["--check-only"], output=output, error_output=io.StringIO())
        self.assertEqual(code, 0)
        self.assertIn("COMPATIBILITY ONLY", output.getvalue())

    def test_missing_chunk_artifact_returns_failure_exit_code(self) -> None:
        """A missing input is reported as FAILED with a nonzero exit code."""
        output = io.StringIO()
        errors = io.StringIO()
        code = main(
            ["--chunks", "/path/that/does/not/exist.jsonl"],
            output=output,
            error_output=errors,
        )
        self.assertEqual(code, 1)
        self.assertIn("FINAL STATUS: FAILED", output.getvalue())


@unittest.skipUnless(
    os.getenv("RUN_REAL_EMBEDDING_TEST") == "1",
    "set RUN_REAL_EMBEDDING_TEST=1 when pinned model weights/network are available",
)
class RealModelSmokeTests(unittest.TestCase):
    """Opt-in integration checks for the pinned local Sentence Transformer."""

    def test_related_query_and_document_are_finite_and_compatible(self) -> None:
        """The real model should return finite 384-dimensional comparable vectors."""
        from app.rag.embeddings.encoder import SentenceTransformerEncoder

        encoder = SentenceTransformerEncoder(EmbeddingConfig())
        documents = encoder.encode_documents(
            ["A duplicate settled card payment may qualify for a refund."]
        )
        query = encoder.encode_query("Can I recover a duplicate card payment?")
        self.assertEqual(documents.shape, (1, 384))
        self.assertEqual(query.shape, (384,))
        self.assertTrue(np.isfinite(documents).all())
        self.assertGreater(float(np.linalg.norm(query)), 0.0)
        self.assertGreater(float(documents[0] @ query), 0.0)


if __name__ == "__main__":
    unittest.main()
