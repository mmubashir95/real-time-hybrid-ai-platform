#!/usr/bin/env python3
"""Validate and aggregate one reproducible Phase 11 evaluation run."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from time import perf_counter
from uuid import uuid4

from pydantic import ValidationError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.evaluation.dataset import load_evaluation_dataset  # noqa: E402
from app.rag.evaluation.reporting import (  # noqa: E402
    EvaluationArtifact,
    RunMetadata,
    new_timestamp,
    write_artifact,
    write_markdown_report,
)
from app.rag.evaluation.operations import (  # noqa: E402
    OperationalSample,
    RequestOutcome,
    summarize_operations,
)
from app.rag.evaluation.reviews import (  # noqa: E402
    ClaimReview,
    QuestionReview,
    summarize_answer_reviews,
)
from app.rag.evaluation.runner import (  # noqa: E402
    EvaluationConfiguration,
    QuestionResult,
    aggregate_question_results,
)


DEFAULT_MANIFEST = PROJECT_ROOT / "data/rag/evaluation/dataset_manifest_v1.json"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "reports/rag_evaluation"


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse dataset, result inputs, and immutable experiment identities."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", choices=("development", "held_out_test"), required=True)
    parser.add_argument("--configuration", choices=tuple(item.value for item in EvaluationConfiguration), required=True)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--question-results", type=Path)
    parser.add_argument("--claim-reviews", type=Path)
    parser.add_argument("--question-reviews", type=Path)
    parser.add_argument("--security-results", type=Path)
    parser.add_argument("--operational-samples", type=Path)
    parser.add_argument("--exploratory", action="store_true")
    parser.add_argument("--execute-live", action="store_true")
    parser.add_argument("--freeze-baseline", action="store_true")
    parser.add_argument("--corpus-snapshot", required=True)
    parser.add_argument("--index-version", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--reranker-model")
    parser.add_argument("--llm-model")
    parser.add_argument("--prompt-version")
    parser.add_argument("--candidate-limit", type=int, default=20)
    parser.add_argument("--test-environment", required=True)
    parser.add_argument("--database-url", default="")
    parser.add_argument("--schema", default="public")
    parser.add_argument("--config-fingerprint")
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--reranker-device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args(arguments)


def _load_json_lines(path: Path, model_type: type) -> tuple:
    """Load typed JSONL rows with precise file and line diagnostics."""
    records = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                records.append(model_type.model_validate_json(line))
            except (ValueError, ValidationError) as error:
                raise ValueError(f"Invalid record at {path}:{line_number}: {error}") from error
    return tuple(records)


def _read_security_results(path: Path | None) -> dict[str, str]:
    """Load explicit named security outcomes without interpreting missing tests."""
    if path is None:
        return {}
    content = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in content.items()
    ):
        raise ValueError("Security results must be a JSON object of string outcomes")
    return content


def _write_baseline(artifact: EvaluationArtifact, path: Path) -> None:
    """Freeze only a complete measured pretrained-reranker aggregate."""
    aggregate = artifact.retrieval_aggregate
    if aggregate is None or aggregate.failed_question_count:
        raise ValueError("A baseline requires complete measured retrieval results")
    if artifact.metadata.evaluation_configuration != EvaluationConfiguration.HYBRID_RERANKED.value:
        raise ValueError("The Phase 13 pretrained baseline must use configuration D")
    path.write_text(
        json.dumps({
            "status": "FROZEN_PRETRAINED_BASELINE",
            "metadata": artifact.metadata.model_dump(mode="json"),
            "retrieval_aggregate": aggregate.model_dump(mode="json"),
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_artifact(
    arguments: argparse.Namespace,
    *,
    live_question_results: tuple[QuestionResult, ...] | None = None,
    live_operational_samples: tuple[OperationalSample, ...] | None = None,
    live_grounded_answers: tuple[dict[str, object], ...] | None = None,
) -> EvaluationArtifact:
    """Validate all supplied evidence and assemble one immutable run artifact."""
    dataset = load_evaluation_dataset(
        arguments.manifest,
        official=not arguments.exploratory,
    )
    split_questions = tuple(
        item for item in dataset.questions if item.dataset_split.value == arguments.split
    )
    if not split_questions:
        raise ValueError(f"Dataset contains no questions in split {arguments.split!r}")

    configuration = EvaluationConfiguration(arguments.configuration)
    if configuration in {
        EvaluationConfiguration.HYBRID_RERANKED,
        EvaluationConfiguration.GROUNDED,
    } and not arguments.reranker_model:
        raise ValueError("Configurations D and E require --reranker-model metadata")
    if configuration is EvaluationConfiguration.GROUNDED and (
        not arguments.llm_model or not arguments.prompt_version
    ):
        raise ValueError("Configuration E requires --llm-model and --prompt-version metadata")
    question_results = live_question_results if live_question_results is not None else (
        _load_json_lines(arguments.question_results, QuestionResult)
        if arguments.question_results else ()
    )
    allowed_question_ids = {item.question_id for item in split_questions}
    if any(item.question_id not in allowed_question_ids for item in question_results):
        raise ValueError("Question results contain IDs outside the selected dataset split")
    if any(item.configuration is not configuration for item in question_results):
        raise ValueError("Question result configuration does not match the run")
    aggregate = (
        aggregate_question_results(configuration, question_results)
        if question_results else None
    )

    claim_reviews = (
        _load_json_lines(arguments.claim_reviews, ClaimReview)
        if arguments.claim_reviews else ()
    )
    question_reviews = (
        _load_json_lines(arguments.question_reviews, QuestionReview)
        if arguments.question_reviews else ()
    )
    answer_quality = (
        summarize_answer_reviews(claim_reviews, question_reviews)
        if claim_reviews or question_reviews else None
    )
    operational_samples = (
        live_operational_samples
        if live_operational_samples is not None
        else _load_json_lines(arguments.operational_samples, OperationalSample)
        if arguments.operational_samples else ()
    )
    operations = summarize_operations(operational_samples) if operational_samples else None
    manifest_hash = dataset.manifest.records_sha256
    limitations = []
    if aggregate is None:
        limitations.append("Retrieval/ranking results were not supplied; quality metrics are not measured.")
    if answer_quality is None:
        limitations.append("Independent human answer and citation reviews are pending.")
    if operations is None:
        limitations.append("Operational latency, usage, cost, and reliability were not supplied.")
    if arguments.exploratory:
        limitations.append("Dataset labels are unapproved; results are exploratory only.")

    return EvaluationArtifact(
        metadata=RunMetadata(
            run_id=str(uuid4()),
            timestamp=new_timestamp(),
            dataset_id=dataset.manifest.dataset_id,
            dataset_version=dataset.manifest.dataset_version,
            dataset_split=arguments.split,
            dataset_sha256=manifest_hash,
            corpus_snapshot=arguments.corpus_snapshot,
            index_version=arguments.index_version,
            embedding_model=arguments.embedding_model,
            reranker_model=arguments.reranker_model,
            llm_model=arguments.llm_model,
            prompt_version=arguments.prompt_version,
            evaluation_configuration=configuration.value,
            candidate_limit=arguments.candidate_limit,
            test_environment=arguments.test_environment,
        ),
        per_question=question_results,
        grounded_answers=live_grounded_answers or (),
        retrieval_aggregate=aggregate,
        answer_quality=answer_quality,
        operations=operations,
        security_results=_read_security_results(arguments.security_results),
        limitations=tuple(limitations),
    )


def _section_key(headings: tuple[str, ...]) -> str:
    """Extract the stable bracketed source key from passage headings."""
    for heading in reversed(headings):
        if heading.startswith("[") and "]" in heading:
            return heading[1:heading.index("]")]
    return ""


async def _execute_live(
    arguments: argparse.Namespace,
) -> tuple[
    tuple[QuestionResult, ...],
    tuple[OperationalSample, ...],
    tuple[dict[str, object], ...],
]:
    """Execute A–E through existing RAG services without duplicating their logic."""
    # Heavy database and model imports stay inside this opt-in path so dataset
    # validation and report generation remain fast and offline.
    import asyncpg
    from sqlalchemy.engine import make_url

    from app.config import load_rag_generation_settings
    from app.rag.embeddings.encoder import SentenceTransformerEncoder
    from app.rag.embeddings.models import EmbeddingConfig
    from app.rag.grounding import GroundedRagService
    from app.rag.grounding.models import GroundingStatus
    from app.rag.generation import OpenAIResponsesClient
    from app.rag.persistence.lifecycle import EmbeddingProfile
    from app.rag.reranking import (
        Qwen3Reranker, RerankerConfig, RerankingService, RetrievalRerankingService,
    )
    from app.rag.retrieval import (
        DenseRetriever, HistoricalVersionScope, KeywordRetriever, RetrievalService,
        SearchContextOrigin, SearchFilters, SearchMode, SearchRequest, TrustedSearchContext,
    )
    from app.rag.evaluation.runner import CandidateRecord, build_question_result

    if not arguments.database_url:
        raise ValueError("--database-url is required for --execute-live")
    configuration = EvaluationConfiguration(arguments.configuration)
    if configuration is not EvaluationConfiguration.KEYWORD and not arguments.config_fingerprint:
        raise ValueError("--config-fingerprint is required for semantic or hybrid execution")
    dataset = load_evaluation_dataset(arguments.manifest, official=not arguments.exploratory)
    questions = tuple(
        item for item in dataset.questions if item.dataset_split.value == arguments.split
    )
    embedding = EmbeddingConfig(device=arguments.embedding_device)
    expected_embedding_identity = f"{embedding.model_id}@{embedding.model_revision}"
    if arguments.embedding_model != expected_embedding_identity:
        raise ValueError(
            "--embedding-model must match the configured model and pinned revision: "
            f"{expected_embedding_identity}"
        )
    profile = EmbeddingProfile(
        embedding.model_id, embedding.model_revision,
        arguments.config_fingerprint or "0" * 64,
        embedding.normalize_embeddings, embedding.encoding_config_version,
        embedding.query_instruction,
    )
    encoder = (
        None if configuration is EvaluationConfiguration.KEYWORD
        else SentenceTransformerEncoder(embedding)
    )
    url = make_url(arguments.database_url)
    if url.drivername not in {"postgresql", "postgresql+asyncpg"}:
        raise ValueError("--database-url must use PostgreSQL")
    dsn = url.set(drivername="postgresql").render_as_string(hide_password=False)
    connection = await asyncpg.connect(dsn, timeout=10)
    try:
        retrieval = RetrievalService(
            KeywordRetriever(connection, schema=arguments.schema),
            DenseRetriever(connection, encoder, schema=arguments.schema),
        )
        reranker = None
        grounded_service = None
        if configuration in {
            EvaluationConfiguration.HYBRID_RERANKED,
            EvaluationConfiguration.GROUNDED,
        }:
            reranker_config = RerankerConfig(
                device=arguments.reranker_device,
                local_files_only=arguments.local_files_only,
            )
            expected_reranker_identity = (
                f"{reranker_config.model_id}@{reranker_config.model_revision}"
            )
            if arguments.reranker_model != expected_reranker_identity:
                raise ValueError(
                    "--reranker-model must match the configured model and pinned revision: "
                    f"{expected_reranker_identity}"
                )
            reranker = RerankingService(Qwen3Reranker(reranker_config))
        if configuration is EvaluationConfiguration.GROUNDED:
            settings = load_rag_generation_settings()
            if arguments.llm_model != settings.openai_model:
                raise ValueError("--llm-model must match OPENAI_MODEL")
            grounded_service = GroundedRagService(
                RetrievalRerankingService(retrieval, reranker),
                OpenAIResponsesClient(settings),
                settings,
            )

        question_results: list[QuestionResult] = []
        operation_samples: list[OperationalSample] = []
        grounded_answers: list[dict[str, object]] = []
        for question in questions:
            gold = {
                (
                    item.tenant_id, item.document_id, item.document_version,
                    item.section_key,
                ): item.relevance_grade
                for item in question.gold_evidence
            }
            historical_evidence = next(
                iter(question.gold_evidence), None
            ) if question.question_type.value == "historical_version" else None
            historical = (
                HistoricalVersionScope(
                    historical_evidence.document_id,
                    historical_evidence.document_version,
                )
                if historical_evidence else None
            )
            context = TrustedSearchContext(
                question.tenant_id, ("public",), SearchContextOrigin.INTERNAL,
                historical, historical is not None,
            )
            mode = {
                EvaluationConfiguration.DENSE: SearchMode.SEMANTIC,
                EvaluationConfiguration.KEYWORD: SearchMode.KEYWORD,
                EvaluationConfiguration.HYBRID: SearchMode.HYBRID,
                EvaluationConfiguration.HYBRID_RERANKED: SearchMode.HYBRID,
                EvaluationConfiguration.GROUNDED: SearchMode.HYBRID,
            }[configuration]
            request = SearchRequest(
                question.query, mode, SearchFilters(), arguments.candidate_limit,
                context, profile if mode is not SearchMode.KEYWORD else None,
                arguments.candidate_limit, arguments.candidate_limit,
            )
            started = perf_counter()
            try:
                if configuration is EvaluationConfiguration.GROUNDED:
                    response = await grounded_service.answer_question(
                        request, top_k=min(5, arguments.candidate_limit)
                    )
                    candidates = tuple(
                        CandidateRecord(
                            rank=rank, chunk_id=item.chunk_id,
                            tenant_id=item.tenant_id, document_id=item.document_id,
                            document_version=item.document_version,
                            section_key=_section_key(item.section),
                            relevance_grade=gold.get((
                                item.tenant_id, item.document_id,
                                item.document_version, _section_key(item.section),
                            ), 0),
                        )
                        for rank, item in enumerate(response.evidence, start=1)
                    )
                    operation_samples.append(OperationalSample(
                        question_id=question.question_id,
                        retrieval_ms=response.latency.retrieval_ms,
                        reranking_ms=response.latency.reranking_ms,
                        context_building_ms=response.latency.context_building_ms,
                        generation_ms=response.latency.generation_ms,
                        verification_ms=response.latency.verification_ms,
                        final_rendering_ms=response.latency.final_rendering_ms,
                        end_to_end_ms=response.latency.total_ms,
                        generation_input_tokens=(
                            response.generation_usage.input_tokens
                            if response.generation_usage
                            else 0 if response.request_count == 0 else None
                        ),
                        generation_output_tokens=(
                            response.generation_usage.output_tokens
                            if response.generation_usage
                            else 0 if response.request_count == 0 else None
                        ),
                        verification_input_tokens=(
                            response.verification_usage.input_tokens
                            if response.verification_usage else 0
                        ),
                        verification_output_tokens=(
                            response.verification_usage.output_tokens
                            if response.verification_usage else 0
                        ),
                        generation_estimated_cost_usd=(
                            response.generation_cost.estimated_cost_usd
                            if response.generation_cost
                            else 0 if response.request_count == 0 else None
                        ),
                        verification_estimated_cost_usd=(
                            response.verification_cost.estimated_cost_usd
                            if response.verification_cost else 0
                        ),
                        outcome=(
                            RequestOutcome.SUCCESS
                            if response.status is GroundingStatus.GROUNDED
                            else RequestOutcome.CORRECT_ABSTENTION
                            if (
                                response.status is GroundingStatus.INSUFFICIENT_EVIDENCE
                                and question.expected_behavior.value != "answer"
                            )
                            else RequestOutcome.SUCCESS
                            if response.status is GroundingStatus.INSUFFICIENT_EVIDENCE
                            else RequestOutcome.VERIFICATION_FAILURE
                        ),
                    ))
                    grounded_answers.append({
                        "question_id": question.question_id,
                        "response": response.model_dump(mode="json"),
                    })
                    latency_ms = response.latency.total_ms
                else:
                    search = await retrieval.search(request)
                    passages = search.results
                    rerank_scores: dict[str, float] = {}
                    latency_ms = search.diagnostics.total_seconds * 1_000
                    if configuration is EvaluationConfiguration.HYBRID_RERANKED:
                        original_chunk_ids = {item.chunk_id for item in passages}
                        reranked = await asyncio.to_thread(
                            reranker.rerank, question.query, passages,
                            arguments.candidate_limit,
                        )
                        if original_chunk_ids != {
                            item.passage.chunk_id for item in reranked.results
                        }:
                            raise ValueError("Reranker changed the frozen candidate set")
                        passages = tuple(item.passage for item in reranked.results)
                        rerank_scores = {
                            item.passage.chunk_id: item.rerank_score
                            for item in reranked.results
                        }
                        latency_ms += reranked.diagnostics.inference_seconds * 1_000
                    candidates = tuple(
                        CandidateRecord(
                            rank=rank, chunk_id=item.chunk_id,
                            tenant_id=item.tenant_id, document_id=item.document_id,
                            document_version=item.document_version,
                            section_key=_section_key(item.section),
                            relevance_grade=gold.get((
                                item.tenant_id, item.document_id,
                                item.document_version, _section_key(item.section),
                            ), 0),
                            score=(
                                rerank_scores[item.chunk_id]
                                if configuration is EvaluationConfiguration.HYBRID_RERANKED
                                else item.semantic_score if mode is SearchMode.SEMANTIC
                                else item.bm25_score if mode is SearchMode.KEYWORD
                                else item.fusion_score
                            ),
                        )
                        for rank, item in enumerate(passages, start=1)
                    )
                    operation_samples.append(OperationalSample(
                        question_id=question.question_id,
                        retrieval_ms=search.diagnostics.total_seconds * 1_000,
                        reranking_ms=(
                            reranked.diagnostics.inference_seconds * 1_000
                            if configuration is EvaluationConfiguration.HYBRID_RERANKED
                            else None
                        ),
                        end_to_end_ms=latency_ms,
                        outcome=RequestOutcome.SUCCESS,
                    ))
                question_results.append(build_question_result(
                    question.question_id, configuration, candidates, gold, latency_ms
                ))
            except asyncio.CancelledError:
                raise
            except Exception as error:
                # A failed question remains in denominators with a stable class;
                # raw exception text is not persisted because it may be sensitive.
                error_code = type(error).__name__
                elapsed_ms = (perf_counter() - started) * 1_000
                question_results.append(build_question_result(
                    question.question_id, configuration, (), gold, elapsed_ms,
                    error_code=error_code,
                ))
                operation_samples.append(OperationalSample(
                    question_id=question.question_id, end_to_end_ms=elapsed_ms,
                    outcome=RequestOutcome.UNHANDLED_ERROR,
                ))
        return (
            tuple(question_results),
            tuple(operation_samples),
            tuple(grounded_answers),
        )
    finally:
        await connection.close()


def main(arguments: list[str] | None = None) -> int:
    """Write JSON and Markdown artifacts, or report one controlled validation error."""
    parsed = parse_args(arguments)
    try:
        live_results = live_operations = live_answers = None
        if parsed.execute_live:
            if parsed.question_results or parsed.operational_samples:
                raise ValueError(
                    "--execute-live cannot be combined with supplied result samples"
                )
            live_results, live_operations, live_answers = asyncio.run(
                _execute_live(parsed)
            )
        artifact = build_artifact(
            parsed,
            live_question_results=live_results,
            live_operational_samples=live_operations,
            live_grounded_answers=live_answers,
        )
        run_directory = parsed.output_directory / artifact.metadata.run_id
        write_artifact(artifact, run_directory / "run.json")
        write_markdown_report(artifact, run_directory / "report.md")
        if parsed.freeze_baseline:
            if parsed.exploratory:
                raise ValueError("An exploratory run cannot freeze the official baseline")
            _write_baseline(artifact, run_directory / "baseline_manifest.json")
    except (ValueError, TypeError, OSError, json.JSONDecodeError) as error:
        print(f"RAG evaluation failed: {error}", file=sys.stderr)
        return 1
    print(f"Evaluation artifacts written to {run_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
