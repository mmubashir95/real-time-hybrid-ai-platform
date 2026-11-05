"""Expose the Phase 11 golden-dataset validation boundary."""

from app.rag.evaluation.dataset import (
    DatasetManifest,
    EvaluationDataset,
    EvaluationQuestion,
    GoldEvidence,
    load_evaluation_dataset,
)
from app.rag.evaluation.runner import EvaluationConfiguration

__all__ = [
    "DatasetManifest",
    "EvaluationDataset",
    "EvaluationQuestion",
    "GoldEvidence",
    "EvaluationConfiguration",
    "load_evaluation_dataset",
]
