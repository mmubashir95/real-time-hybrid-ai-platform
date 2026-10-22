"""Compare exact database retrieval with the existing Phase 5 NumPy baseline."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from app.rag.embeddings.models import EmbeddingRecord
from app.rag.embeddings.service import (
    EmbeddingEncoder,
    question_search_scope,
    run_semantic_experiment,
)
from app.rag.persistence.lifecycle import EmbeddingProfile
from app.rag.retrieval.dense import DenseRetriever
from app.rag.retrieval.models import (
    DenseSearchRequest,
    HistoricalVersionScope,
    SearchContextOrigin,
    TrustedSearchContext,
)


@dataclass(frozen=True)
class DenseEvaluationReport:
    """Summarize Recall@K and exact baseline ordering agreement."""

    split: str
    evaluated_questions: int
    skipped_no_answer_questions: int
    recall_at_5: float
    recall_at_10: float
    exact_top_10_matches: int
    ranking_differences: tuple[str, ...]


async def evaluate_against_phase5(
    *,
    question_bank: Path,
    split: str,
    retriever: DenseRetriever,
    encoder: EmbeddingEncoder,
    profile: EmbeddingProfile,
    vectors: np.ndarray,
    records: Sequence[EmbeddingRecord],
    permitted_visibilities: tuple[str, ...] = ("public", "internal"),
) -> DenseEvaluationReport:
    """Evaluate DB recall and compare ordered IDs on equivalent candidate scopes.

    Expected-no-answer rows remain separate because recall cannot score
    abstention. The caller must supply artifacts equivalent to the imported
    database profile; any reported ordering differences remain inspectable by
    question ID.
    """
    with question_bank.open(encoding="utf-8", newline="") as csv_file:
        questions = [
            row for row in csv.DictReader(csv_file) if row["split"] == split
        ]
    if not questions:
        raise ValueError(f"No evaluation questions found for split: {split}")

    evaluated = skipped = hits_at_5 = hits_at_10 = exact_matches = 0
    differences: list[str] = []
    for question in questions:
        if question["expected_no_answer"].strip().lower() == "true":
            skipped += 1
            continue
        document_id, document_version = question_search_scope(question)
        historical = (
            HistoricalVersionScope(document_id, document_version)
            if document_id is not None and document_version is not None
            else None
        )
        response = await retriever.search(
            DenseSearchRequest(
                context=TrustedSearchContext(
                    tenant_id=question["tenant_id"],
                    permitted_visibilities=permitted_visibilities,
                    origin=SearchContextOrigin.INTERNAL,
                    historical=historical,
                    historical_access=historical is not None,
                ),
                profile=profile,
                question=question["query"],
                top_k=10,
            )
        )
        baseline = run_semantic_experiment(
            question["query"],
            question["tenant_id"],
            encoder,
            vectors,
            records,
            top_k=10,
            document_id=document_id,
            document_version=document_version,
        )
        database_ids = [result.chunk_id for result in response.results]
        baseline_ids = [item["chunk_id"] for item in baseline]
        if database_ids == baseline_ids:
            exact_matches += 1
        else:
            differences.append(question["question_id"])
        matches = [
            result.document_id == question["gold_document_id"]
            and result.document_version == question["gold_version"]
            and question["gold_section_key"] in " > ".join(result.heading_path)
            for result in response.results
        ]
        evaluated += 1
        hits_at_5 += int(any(matches[:5]))
        hits_at_10 += int(any(matches[:10]))
    return DenseEvaluationReport(
        split=split,
        evaluated_questions=evaluated,
        skipped_no_answer_questions=skipped,
        recall_at_5=hits_at_5 / evaluated if evaluated else 0.0,
        recall_at_10=hits_at_10 / evaluated if evaluated else 0.0,
        exact_top_10_matches=exact_matches,
        ranking_differences=tuple(differences),
    )
