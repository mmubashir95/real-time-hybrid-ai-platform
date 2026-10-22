"""Verify source-based quality and latency metric calculations."""

from __future__ import annotations

from math import log2
import unittest

from app.rag.retrieval.metrics import (
    EvaluationExample,
    latency_percentiles,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


class RetrievalMetricTests(unittest.TestCase):
    """Check metric definitions on small hand-calculated examples."""

    def test_recall_mrr_and_binary_ndcg(self) -> None:
        """Score multiple gold sources without relying on mutable chunk IDs."""
        a = ("tenant", "DOC-A", "1.0")
        b = ("tenant", "DOC-B", "1.0")
        x = ("tenant", "DOC-X", "1.0")
        examples = (EvaluationExample((x, a, b), frozenset((a, b))),)
        self.assertEqual(recall_at_k(examples, 2), 0.5)
        self.assertEqual(mrr_at_k(examples, 10), 0.5)
        self.assertGreater(ndcg_at_k(examples, 10), 0.6)

    def test_ndcg_credits_each_relevant_source_once(self) -> None:
        """Keep NDCG within [0, 1] when several chunks share one gold anchor."""
        a = ("tenant", "DOC-A", "1.0")
        x = ("tenant", "DOC-X", "1.0")
        self.assertEqual(ndcg_at_k((EvaluationExample((a, a, a), frozenset((a,))),), 10), 1.0)
        self.assertAlmostEqual(
            ndcg_at_k((EvaluationExample((x, a, a), frozenset((a,))),), 10),
            1.0 / log2(3),
        )

    def test_no_answer_rows_are_excluded_from_relevance_metrics(self) -> None:
        """Avoid mathematically invalid recall claims for unanswerable questions."""
        a = ("tenant", "DOC-A", "1.0")
        examples = (
            EvaluationExample((a,), frozenset((a,))),
            EvaluationExample((), frozenset()),
        )
        self.assertEqual(recall_at_k(examples, 5), 1.0)

    def test_latency_percentiles(self) -> None:
        """Use median and nearest-rank p95 over actual samples."""
        p50, p95 = latency_percentiles(tuple(float(value) for value in range(1, 21)))
        self.assertEqual(p50, 10.5)
        self.assertEqual(p95, 19.0)


if __name__ == "__main__":
    unittest.main()
