"""Test deterministic Task 6.8 evaluation helpers without model inference."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from app.rag.embeddings.models import EmbeddingRecord
from app.rag.retrieval.models import SearchContextOrigin, TrustedSearchContext
from app.rag.retrieval.verification import (
    artifact_run_fingerprint,
    exact_numpy_search,
    latency_summary,
    percentile_95,
)
from scripts.evaluate_rag_dense import _gold_is_reachable


def record(
    index: int,
    chunk_id: str,
    *,
    tenant: str = "asterdesk",
    status: str = "active",
    visibility: str = "public",
) -> EmbeddingRecord:
    """Build the smallest complete Phase 5 metadata row for ranking tests."""
    return EmbeddingRecord(
        row_index=index,
        chunk_id=chunk_id,
        tenant_id=tenant,
        document_id=f"DOC-{index}",
        document_version="1.0",
        document_status=status,
        access_scope=visibility,
        heading_path=("Policy",),
        start_page=None,
        end_page=None,
        source_file=f"doc-{index}.md",
        source_type="md",
        source_block_ids=(index,),
        chunk_token_count=5,
        chunk_tokenizer_name="fixture",
        chunk_tokenizer_version="1",
        chunking_config_version="phase4-v4",
        input_fingerprint="emb_" + str(index).ljust(64, "0"),
        embedding_dimensions=3,
        embedding_model_id="fixture/model",
        embedding_model_revision="a" * 40,
    )


class DenseEvaluationHelperTests(unittest.TestCase):
    """Cover numerical ordering, scope filtering, and report summaries."""

    def setUp(self) -> None:
        """Create a trusted public current-policy scope."""
        self.context = TrustedSearchContext(
            "asterdesk", ("public",), SearchContextOrigin.INTERNAL
        )

    def test_exact_numpy_search_matches_database_tie_breaking(self) -> None:
        """Break equal cosine scores by ascending stable chunk identity."""
        records = (
            record(0, "chk_" + "b" * 64),
            record(1, "chk_" + "a" * 64),
            record(2, "chk_" + "c" * 64),
        )
        vectors = np.asarray(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        ranked = exact_numpy_search(
            np.asarray([1.0, 0.0, 0.0]), vectors, records, self.context, top_k=3
        )
        self.assertEqual(
            [item[0] for item in ranked],
            ["chk_" + "a" * 64, "chk_" + "b" * 64, "chk_" + "c" * 64],
        )

    def test_exact_numpy_search_filters_before_top_k(self) -> None:
        """Exclude higher-scoring foreign, archived, and internal candidates."""
        records = (
            record(0, "chk_" + "a" * 64, tenant="brookhaven"),
            record(1, "chk_" + "b" * 64, status="archived"),
            record(2, "chk_" + "c" * 64, visibility="internal"),
            record(3, "chk_" + "d" * 64),
        )
        vectors = np.asarray(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
             [0.8, 0.2, 0.0]], dtype=np.float32,
        )
        ranked = exact_numpy_search(
            np.asarray([1.0, 0.0, 0.0]), vectors, records, self.context, top_k=1
        )
        self.assertEqual([item[0] for item in ranked], ["chk_" + "d" * 64])

    def test_gold_reachability_follows_permitted_visibility(self) -> None:
        """Count a question as reachable only when its gold version is visible."""
        gold_visibilities = {
            ("asterdesk", "DOC-PUBLIC", "1.0"): "public",
            ("asterdesk", "DOC-INTERNAL", "1.0"): "internal",
        }

        def question(document_id: str, tenant: str = "asterdesk") -> dict[str, str]:
            """Build the gold-label fields the reachability rule reads."""
            return {"tenant_id": tenant, "gold_document_id": document_id,
                    "gold_version": "1.0"}

        public_only = ("public",)
        self.assertTrue(
            _gold_is_reachable(question("DOC-PUBLIC"), gold_visibilities, public_only)
        )
        self.assertFalse(
            _gold_is_reachable(question("DOC-INTERNAL"), gold_visibilities, public_only)
        )
        self.assertTrue(_gold_is_reachable(
            question("DOC-INTERNAL"), gold_visibilities, ("public", "internal")
        ))
        # A gold label under another tenant must not borrow this tenant's scope.
        self.assertFalse(_gold_is_reachable(
            question("DOC-PUBLIC", tenant="brookhaven"), gold_visibilities, public_only
        ))

    def test_latency_summary_uses_nearest_rank_p95(self) -> None:
        """Report measured sample count, median, extrema, and p95."""
        samples = [float(value) for value in range(1, 21)]
        self.assertEqual(percentile_95(samples), 19.0)
        summary = latency_summary(samples)
        self.assertEqual(summary["count"], 20)
        self.assertEqual(summary["median_seconds"], 10.5)
        self.assertEqual(summary["minimum_seconds"], 1.0)
        self.assertEqual(summary["maximum_seconds"], 20.0)

    def test_artifact_fingerprint_requires_both_checksums(self) -> None:
        """Reject weak artifact identities and preserve immutable run name."""
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run-fixture"
            self.assertEqual(
                artifact_run_fingerprint(
                    run,
                    {"artifact_sha256": {
                        "vectors.npy": "a" * 64,
                        "metadata.jsonl": "b" * 64,
                    }},
                ),
                f"run-fixture:{'a' * 64}:{'b' * 64}",
            )
            with self.assertRaisesRegex(ValueError, "checksums"):
                artifact_run_fingerprint(run, {"artifact_sha256": {}})


if __name__ == "__main__":
    unittest.main()
