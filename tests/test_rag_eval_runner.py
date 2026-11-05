"""Verify Phase 11 aggregation, fixed candidates, and artifact reporting."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from app.rag.evaluation.dataset import GoldEvidence
from app.rag.evaluation.reporting import (
    EvaluationArtifact,
    RerankerRuntime,
    RunMetadata,
    load_candidate_baseline,
    read_artifact,
    render_markdown_report,
    write_artifact,
)
from app.rag.evaluation.runner import (
    CandidateRecord,
    CandidateSetMismatchError,
    EvaluationConfiguration,
    aggregate_question_results,
    assert_fixed_candidate_set,
    build_question_result,
    gold_relevance,
    match_gold_evidence,
)


def candidate(identifier: str, rank: int, *, relevant: int = 0) -> CandidateRecord:
    """Build one source-traceable candidate fixture."""
    return CandidateRecord(
        rank=rank, chunk_id=f"chk_{identifier}", tenant_id="asterdesk",
        document_id=f"DOC-{identifier}", document_version="2.0",
        section_key="DECISION", relevance_grade=relevant,
    )


class EvaluationRunnerTests(unittest.TestCase):
    """Test deterministic per-question and aggregate calculations."""

    def test_question_and_aggregate_results_keep_sample_counts(self) -> None:
        """Aggregate two questions without hiding a failed observation."""
        a = candidate("A", 1, relevant=1)
        gold = {a.evidence_key: 1}
        complete = build_question_result(
            "Q1", EvaluationConfiguration.HYBRID, (a,), gold, 10.0
        )
        failed = build_question_result(
            "Q2", EvaluationConfiguration.HYBRID, (), {}, 5.0,
            error_code="retrieval_error",
        )
        aggregate = aggregate_question_results(
            EvaluationConfiguration.HYBRID, (complete, failed)
        )
        self.assertEqual(aggregate.question_count, 2)
        self.assertEqual(aggregate.failed_question_count, 1)
        self.assertEqual(aggregate.recall_at_5, 1.0)
        self.assertEqual(aggregate.latency_p50_ms, 10.0)

    def test_equivalent_section_with_quote_is_credited_once_to_canonical_key(self) -> None:
        """Credit repeated statements of one fact, but only at its first rank."""
        evidence = (GoldEvidence(
            tenant_id="asterdesk", document_id="AST-CAN-003", document_version="2.0",
            section_key="DECISION", source_reference="documents/a.md",
            source_quote="Paid features remain   available", relevance_grade=3,
            equivalent_sections=("FAQ",),
        ),)
        passages = (
            ("chk_faq", "FAQ", "Q: when? A: paid features remain\navailable."),
            ("chk_doc", "DOC-CONTROL", "Owner and revision history."),
            ("chk_decision", "DECISION", "Paid features remain available."),
        )
        candidates = []
        for rank, (chunk_id, section, text) in enumerate(passages, start=1):
            grade, credited = match_gold_evidence(
                evidence, tenant_id="asterdesk", document_id="AST-CAN-003",
                document_version="2.0", section_key=section, passage_text=text,
            )
            candidates.append(CandidateRecord(
                rank=rank, chunk_id=chunk_id, tenant_id="asterdesk",
                document_id="AST-CAN-003", document_version="2.0",
                section_key=section, relevance_grade=grade,
                credited_section_key=credited,
            ))
        self.assertEqual(candidates[0].evidence_key[3], "DECISION")
        self.assertEqual(candidates[0].relevance_grade, 3)
        self.assertEqual(candidates[1].relevance_grade, 0)
        result = build_question_result(
            "Q1", EvaluationConfiguration.HYBRID, tuple(candidates),
            gold_relevance(evidence), 1.0,
        )
        self.assertEqual(result.reciprocal_rank_at_10, 1.0)
        self.assertEqual(result.recall_at_5, 1.0)
        # The fact counts once; the later DECISION duplicate earns no extra precision.
        self.assertEqual(result.precision_at_5, 0.2)

    def test_quote_or_section_mismatch_is_not_credited(self) -> None:
        """Reject unlisted sections, other versions, and gold sections lacking the fact."""
        evidence = (GoldEvidence(
            tenant_id="asterdesk", document_id="AST-CAN-003", document_version="2.0",
            section_key="DECISION", source_reference="documents/a.md",
            source_quote="paid-through timestamp", equivalent_sections=("FAQ",),
        ),)
        cases = (
            ("2.0", "SCOPE", "Access lasts to the paid-through timestamp."),
            ("1.0", "DECISION", "Access lasts to the paid-through timestamp."),
            ("2.0", "DECISION", "This chunk of the gold section omits the fact."),
        )
        for version, section, text in cases:
            with self.subTest(version=version, section=section):
                self.assertEqual(match_gold_evidence(
                    evidence, tenant_id="asterdesk", document_id="AST-CAN-003",
                    document_version=version, section_key=section, passage_text=text,
                ), (0, None))

    def test_reranker_must_preserve_identical_candidate_set(self) -> None:
        """Allow reordering but reject introduced or missing passages."""
        baseline = (candidate("A", 1), candidate("B", 2))
        reranked = (candidate("B", 1), candidate("A", 2))
        assert_fixed_candidate_set(baseline, reranked)
        with self.assertRaisesRegex(ValueError, "preserve"):
            assert_fixed_candidate_set(baseline, (candidate("A", 1),))

    def test_artifact_round_trip_and_missing_measurement_labels(self) -> None:
        """Persist JSON atomically and render absent results honestly."""
        artifact = EvaluationArtifact(
            metadata=RunMetadata(
                run_id="run-1", timestamp="2026-11-04T12:00:00Z",
                dataset_id="dataset", dataset_version="1.0.0",
                dataset_split="development", dataset_sha256="a" * 64,
                corpus_snapshot="manifest-sha", index_version="index-1",
                embedding_model="embedding", evaluation_configuration="E",
                candidate_limit=20, test_environment="unit-test",
            ),
            limitations=("Human review pending.",),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            write_artifact(artifact, path)
            loaded = read_artifact(path)
        report = render_markdown_report(loaded)
        self.assertEqual(loaded.metadata.run_id, "run-1")
        self.assertIn("NOT MEASURED", report)
        self.assertIn("PENDING HUMAN REVIEW", report)

    def test_reranker_runtime_round_trips_and_is_reported(self) -> None:
        """Persist reranker device, precision, truncation, and warm-up identity."""
        artifact = EvaluationArtifact(metadata=RunMetadata(
            run_id="run-2", timestamp="2026-11-05T12:00:00Z",
            dataset_id="dataset", dataset_version="1.1.0",
            dataset_split="development", dataset_sha256="a" * 64,
            corpus_snapshot="manifest-sha", index_version="index-1",
            embedding_model="embedding", reranker_model="reranker@rev",
            reranker_runtime=RerankerRuntime(
                device="mps", precision="bfloat16", batch_size=4, max_length=1024,
                model_load_seconds=3.5, truncated_inputs=2, warmup_question_id="Q1",
            ),
            evaluation_configuration="D", candidate_limit=20,
            test_environment="unit-test",
        ))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            write_artifact(artifact, path)
            loaded = read_artifact(path)
        self.assertEqual(loaded.metadata.reranker_runtime, artifact.metadata.reranker_runtime)
        report = render_markdown_report(loaded)
        self.assertIn("precision `bfloat16`", report)
        self.assertIn("Reranker truncated inputs: 2", report)
        self.assertIn("`Q1`", report)

    def test_candidate_set_mismatch_has_run_level_error_type(self) -> None:
        """Let callers abort a run instead of recording one failed question."""
        with self.assertRaises(CandidateSetMismatchError):
            assert_fixed_candidate_set((candidate("A", 1),), (candidate("B", 1),))


def _hybrid_artifact(**metadata_overrides: object) -> EvaluationArtifact:
    """Build a configuration C artifact with one successful and one failed question."""
    values: dict[str, object] = dict(
        run_id="c-run", timestamp="2026-11-05T12:00:00Z",
        dataset_id="dataset", dataset_version="1.1.0",
        dataset_split="development", dataset_sha256="a" * 64,
        corpus_snapshot="manifest-sha", index_version="index-1",
        embedding_model="embedding", evaluation_configuration="C",
        candidate_limit=20, test_environment="unit-test",
    )
    values.update(metadata_overrides)
    ok = build_question_result(
        "Q1", EvaluationConfiguration.HYBRID, (candidate("A", 1), candidate("B", 2)), {}, 1.0
    )
    failed = build_question_result(
        "Q2", EvaluationConfiguration.HYBRID, (), {}, 1.0, error_code="retrieval_error"
    )
    return EvaluationArtifact(metadata=RunMetadata(**values), per_question=(ok, failed))


class CandidateBaselineTests(unittest.TestCase):
    """Verify D can only be compared against a matching, complete C run."""

    expected = dict(
        dataset_sha256="a" * 64, dataset_split="development",
        corpus_snapshot="manifest-sha", index_version="index-1",
        embedding_model="embedding", candidate_limit=20,
    )

    def _load(self, artifact: EvaluationArtifact, question_ids: tuple[str, ...]):
        """Persist an artifact and load it as a candidate baseline."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            write_artifact(artifact, path)
            return load_candidate_baseline(path, question_ids=question_ids, **self.expected)

    def test_matching_hybrid_run_yields_candidates_and_identity(self) -> None:
        """Expose C's candidates per question plus a traceable artifact hash."""
        baseline = self._load(_hybrid_artifact(), ("Q1",))
        self.assertEqual(baseline.run_id, "c-run")
        self.assertRegex(baseline.artifact_sha256, r"^[0-9a-f]{64}$")
        self.assertEqual(
            [item.chunk_id for item in baseline.candidates_by_question["Q1"]],
            ["chk_A", "chk_B"],
        )

    def test_incomparable_or_incomplete_baselines_are_rejected(self) -> None:
        """Refuse wrong configuration, drifted settings, and unusable questions."""
        cases = (
            (_hybrid_artifact(evaluation_configuration="B"), ("Q1",), "configuration C"),
            (_hybrid_artifact(index_version="index-2"), ("Q1",), "index_version"),
            (_hybrid_artifact(corpus_snapshot="other"), ("Q1",), "corpus_snapshot"),
            (_hybrid_artifact(candidate_limit=10), ("Q1",), "candidate_limit"),
            (_hybrid_artifact(dataset_sha256="b" * 64), ("Q1",), "dataset_sha256"),
            (_hybrid_artifact(), ("Q2",), "no successful result for Q2"),
            (_hybrid_artifact(), ("Q3",), "no successful result for Q3"),
        )
        for artifact, question_ids, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self._load(artifact, question_ids)

    def test_baseline_identity_is_reported(self) -> None:
        """Show which C run a D report was verified against."""
        artifact = EvaluationArtifact(metadata=RunMetadata(
            **{**_hybrid_artifact().metadata.model_dump(), "evaluation_configuration": "D",
               "baseline_run_id": "c-run", "baseline_artifact_sha256": "c" * 64},
        ))
        report = render_markdown_report(artifact)
        self.assertIn("Candidate baseline (C) run: `c-run`", report)


if __name__ == "__main__":
    unittest.main()
