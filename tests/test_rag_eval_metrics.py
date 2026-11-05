"""Verify Phase 11 retrieval, ranking, review, and abstention metrics."""

from __future__ import annotations

import unittest

from app.rag.evaluation.metrics import RankedEvidence, evaluate_ranking, rate
from app.rag.evaluation.operations import (
    OperationalSample,
    RequestOutcome,
    summarize_operations,
)
from app.rag.evaluation.reviews import (
    AbstentionOutcome,
    ClaimReview,
    HumanSupport,
    QuestionReview,
    summarize_answer_reviews,
)


A = ("tenant", "DOC-A", "1.0", "DECISION")
B = ("tenant", "DOC-B", "1.0", "SCOPE")
X = ("tenant", "DOC-X", "1.0", "FAQ")


class RankingMetricTests(unittest.TestCase):
    """Check hand-calculated ranking metrics and edge-case denominators."""

    def test_precision_recall_coverage_mrr_and_ndcg(self) -> None:
        """Score two graded gold sources with one irrelevant item first."""
        result = evaluate_ranking(
            (RankedEvidence(X), RankedEvidence(A), RankedEvidence(B)),
            {A: 3, B: 1},
            5,
        )
        self.assertEqual(result.precision_at_k, 0.4)
        self.assertEqual(result.recall_at_k, 1.0)
        self.assertTrue(result.complete_evidence_coverage_at_k)
        self.assertEqual(result.reciprocal_rank_at_k, 0.5)
        self.assertGreater(result.ndcg_at_k, 0.5)
        self.assertLess(result.ndcg_at_k, 1.0)

    def test_duplicates_do_not_inflate_metrics(self) -> None:
        """Credit an overlapping source anchor only once at every cutoff."""
        result = evaluate_ranking(
            (RankedEvidence(A), RankedEvidence(A), RankedEvidence(X)),
            {A: 1, B: 1},
            5,
        )
        self.assertEqual(result.unique_returned_at_k, 2)
        self.assertEqual(result.precision_at_k, 0.2)
        self.assertEqual(result.recall_at_k, 0.5)
        self.assertFalse(result.complete_evidence_coverage_at_k)

    def test_duplicates_do_not_stretch_window_or_compress_ranks(self) -> None:
        """Score gold at chunk rank 6 as outside K=5, and duplicates keep rank cost."""
        ranked = (RankedEvidence(X),) * 5 + (RankedEvidence(A),)
        at_5 = evaluate_ranking(ranked, {A: 1}, 5)
        self.assertEqual(at_5.recall_at_k, 0.0)
        self.assertEqual(at_5.reciprocal_rank_at_k, 0.0)
        self.assertEqual(at_5.unique_returned_at_k, 1)
        at_10 = evaluate_ranking(ranked, {A: 1}, 10)
        self.assertEqual(at_10.recall_at_k, 1.0)
        self.assertAlmostEqual(at_10.reciprocal_rank_at_k, 1 / 6)

    def test_no_answer_relevance_metrics_are_undefined(self) -> None:
        """Keep no-answer questions separate from standard recall and NDCG."""
        result = evaluate_ranking((RankedEvidence(X),), {}, 5)
        self.assertEqual(result.precision_at_k, 0.0)
        self.assertIsNone(result.recall_at_k)
        self.assertIsNone(result.complete_evidence_coverage_at_k)
        self.assertIsNone(result.reciprocal_rank_at_k)
        self.assertIsNone(result.ndcg_at_k)

    def test_fewer_than_k_uses_k_precision_denominator(self) -> None:
        """Apply the documented fixed-K convention to short result lists."""
        result = evaluate_ranking((RankedEvidence(A),), {A: 1}, 10)
        self.assertEqual(result.precision_at_k, 0.1)
        self.assertEqual(result.recall_at_k, 1.0)

    def test_rate_preserves_zero_denominator(self) -> None:
        """Represent an unreviewed population as undefined instead of zero."""
        self.assertIsNone(rate(0, 0))
        with self.assertRaises(ValueError):
            rate(2, 1)


class AnswerReviewMetricTests(unittest.TestCase):
    """Verify independent human-review aggregation and denominator counts."""

    def test_human_reviews_drive_claim_citation_and_abstention_rates(self) -> None:
        """Calculate rates without trusting the system's support label."""
        claims = (
            ClaimReview(
                question_id="Q1", claim_index=0, claim_text="Supported claim.",
                cited_evidence_ids=("E1",), system_verifier_status="supported",
                human_support=HumanSupport.SUPPORTED, answer_correct=True,
                answer_complete=True, answer_relevant=True,
                citation_relationships=(True,), reviewer_explanation="Direct support.",
                reviewed_by="reviewer", reviewed_at="2026-11-04T12:00:00Z",
            ),
            ClaimReview(
                question_id="Q1", claim_index=1, claim_text="Invented deadline.",
                cited_evidence_ids=("E1",), system_verifier_status="supported",
                human_support=HumanSupport.UNSUPPORTED, answer_correct=False,
                answer_complete=False, answer_relevant=True,
                citation_relationships=(False,), reviewer_explanation="No deadline.",
                reviewed_by="reviewer", reviewed_at="2026-11-04T12:00:00Z",
            ),
        )
        questions = (
            QuestionReview(
                question_id="Q1", abstention_outcome=AbstentionOutcome.CORRECT_ANSWER,
                reviewer_explanation="Answerable.", reviewed_by="reviewer",
                reviewed_at="2026-11-04T12:00:00Z",
            ),
            QuestionReview(
                question_id="Q2", abstention_outcome=AbstentionOutcome.CORRECT_ABSTENTION,
                reviewer_explanation="No evidence.", reviewed_by="reviewer",
                reviewed_at="2026-11-04T12:00:00Z",
            ),
        )
        summary = summarize_answer_reviews(claims, questions)
        self.assertEqual(summary.claim_support_rate, 0.5)
        self.assertEqual(summary.unsupported_claim_rate, 0.5)
        self.assertEqual(summary.citation_correctness, 0.5)
        self.assertEqual(summary.abstention_accuracy, 1.0)
        self.assertEqual(summary.reviewed_claims, 2)

    def test_citation_judgments_must_align(self) -> None:
        """Reject missing human judgments for cited claim-evidence pairs."""
        with self.assertRaisesRegex(ValueError, "align"):
            ClaimReview(
                question_id="Q1", claim_index=0, claim_text="Claim.",
                cited_evidence_ids=("E1",), system_verifier_status="supported",
                human_support=HumanSupport.SUPPORTED, answer_correct=True,
                answer_complete=True, answer_relevant=True,
                citation_relationships=(), reviewer_explanation="Missing relation.",
                reviewed_by="reviewer", reviewed_at="2026-11-04T12:00:00Z",
            )


class OperationalMetricTests(unittest.TestCase):
    """Verify measured latency, usage, cost, and reliability aggregation."""

    def test_operations_preserve_samples_totals_and_failures(self) -> None:
        """Calculate p50/p95 and totals from actual complete measurements."""
        samples = (
            OperationalSample(
                question_id="Q1", retrieval_ms=10, end_to_end_ms=100,
                generation_input_tokens=100, generation_output_tokens=20,
                verification_input_tokens=50, verification_output_tokens=10,
                generation_estimated_cost_usd=0.001,
                verification_estimated_cost_usd=0.0005,
                outcome=RequestOutcome.SUCCESS,
            ),
            OperationalSample(
                question_id="Q2", retrieval_ms=20, end_to_end_ms=200,
                generation_input_tokens=110, generation_output_tokens=30,
                verification_input_tokens=60, verification_output_tokens=10,
                generation_estimated_cost_usd=0.002,
                verification_estimated_cost_usd=0.001,
                outcome=RequestOutcome.TIMEOUT,
            ),
        )
        summary = summarize_operations(samples)
        self.assertEqual(summary.request_count, 2)
        self.assertEqual(summary.failure_counts["timeout"], 1)
        self.assertEqual(summary.success_rate, 0.5)
        self.assertEqual(summary.latency_ms["retrieval_ms"]["sample_count"], 2)
        self.assertEqual(summary.total_tokens, 390)
        self.assertAlmostEqual(summary.total_estimated_cost_usd, 0.0045)

    def test_missing_usage_remains_unmeasured(self) -> None:
        """Never report partial provider usage or cost as a complete total."""
        summary = summarize_operations((
            OperationalSample(question_id="Q1", outcome=RequestOutcome.CORRECT_ABSTENTION),
        ))
        self.assertIsNone(summary.total_tokens)
        self.assertIsNone(summary.total_estimated_cost_usd)
        self.assertIsNone(summary.latency_ms["generation_ms"])


if __name__ == "__main__":
    unittest.main()
