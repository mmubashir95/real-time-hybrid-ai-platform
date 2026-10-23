"""Public Phase 9 evidence-context and OpenAI generation interfaces."""

from app.rag.generation.context_builder import (
    ContextBuildError,
    build_context,
    count_generation_tokens,
)
from app.rag.generation.models import (
    ContextBuildResult,
    ContextDiagnostics,
    CostEstimate,
    EvidenceIdentity,
    GenerationErrorCode,
    GenerationLatency,
    GenerationPipelineError,
    GenerationStatus,
    LlmGenerationResult,
    RagAnswerResponse,
    TokenUsage,
)
from app.rag.generation.openai_client import OpenAIResponsesClient, estimate_cost
from app.rag.generation.service import RagGenerationService

__all__ = [
    "ContextBuildError",
    "ContextBuildResult",
    "ContextDiagnostics",
    "CostEstimate",
    "EvidenceIdentity",
    "GenerationErrorCode",
    "GenerationLatency",
    "GenerationPipelineError",
    "GenerationStatus",
    "LlmGenerationResult",
    "OpenAIResponsesClient",
    "RagAnswerResponse",
    "RagGenerationService",
    "TokenUsage",
    "build_context",
    "count_generation_tokens",
    "estimate_cost",
]
