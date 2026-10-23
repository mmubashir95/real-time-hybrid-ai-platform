"""Orchestrate authorized retrieval, Qwen reranking, context, and generation."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import logging
from time import perf_counter
from typing import Protocol

from app.config import RagGenerationSettings
from app.rag.generation.context_builder import (
    ContextBuildError,
    build_context,
    count_generation_tokens,
)
from app.rag.generation.models import (
    ContextDiagnostics,
    EvidenceIdentity,
    GenerationErrorCode,
    GenerationLatency,
    GenerationPipelineError,
    GenerationStatus,
    RagAnswerResponse,
)
from app.rag.generation.prompts import SYSTEM_INSTRUCTION, compose_user_message
from app.rag.reranking.qwen3 import RerankerInferenceError
from app.rag.retrieval.dense import RetrievalError
from app.rag.retrieval.models import SearchRequest


logger = logging.getLogger(__name__)
PROMPT_TOKEN_OVERHEAD = 32
NO_EVIDENCE_ANSWER = (
    "The available authorized documents do not contain enough information "
    "to answer this question."
)


class RetrievalAndReranking(Protocol):
    """Describe the existing Phase 7/8 orchestration boundary."""

    async def search(self, request: SearchRequest, *, top_k: int):
        """Return the original search response and its reranked candidates."""


class AnswerGenerator(Protocol):
    """Describe the OpenAI client boundary used by orchestration tests."""

    async def generate(
        self, system_instruction: str, user_message: str, *, request_id: str = "-"
    ):
        """Return one validated provider generation result."""


class RagGenerationService:
    """Run one bounded, authorized question-to-answer pipeline."""

    def __init__(
        self,
        retrieval_and_reranking: RetrievalAndReranking,
        generator: AnswerGenerator,
        settings: RagGenerationSettings,
    ) -> None:
        """Bind the existing pipeline and one reusable asynchronous LLM client."""
        self._retrieval_and_reranking = retrieval_and_reranking
        self._generator = generator
        self._settings = settings

    async def answer_question(
        self,
        request: SearchRequest,
        *,
        top_k: int = 5,
        request_id: str = "-",
    ) -> RagAnswerResponse:
        """Return a generated answer or deterministic no-evidence response."""
        started = perf_counter()
        try:
            request = self._restrict_to_external_visibilities(request)
            try:
                search_response, rerank_response = (
                    await self._retrieval_and_reranking.search(request, top_k=top_k)
                )
            except RetrievalError as error:
                raise GenerationPipelineError(
                    GenerationErrorCode.RETRIEVAL_ERROR,
                    "Authorized retrieval failed.",
                ) from error
            except RerankerInferenceError as error:
                raise GenerationPipelineError(
                    GenerationErrorCode.RERANKING_ERROR,
                    "Passage reranking failed.",
                ) from error
            except ValueError as error:
                # Phase 7/8 validation rejects invalid requests and inconsistent
                # candidates (such as mixed tenants) with ValueError.
                raise GenerationPipelineError(
                    GenerationErrorCode.RERANKING_ERROR,
                    "Retrieval or reranking rejected the request or its candidates.",
                ) from error

            retrieval_ms = search_response.diagnostics.total_seconds * 1000
            reranking_ms = rerank_response.diagnostics.inference_seconds * 1000
            if not rerank_response.results:
                total_ms = (perf_counter() - started) * 1000
                return RagAnswerResponse(
                    answer=NO_EVIDENCE_ANSWER,
                    model=self._settings.openai_model,
                    status=GenerationStatus.NO_EVIDENCE,
                    evidence_chunk_ids=(),
                    evidence=(),
                    request_count=0,
                    context=ContextDiagnostics(
                        available_context_tokens=self._settings.max_context_tokens,
                        estimated_context_tokens=0,
                        estimated_total_input_tokens=0,
                        included_passage_count=0,
                        excluded_passage_count=0,
                        truncated_chunk_ids=(),
                    ),
                    latency=GenerationLatency(
                        retrieval_ms=retrieval_ms,
                        reranking_ms=reranking_ms,
                        generation_ms=0,
                        total_ms=total_ms,
                    ),
                )

            base_user_message = compose_user_message(request.query, "")
            base_tokens = (
                count_generation_tokens(SYSTEM_INSTRUCTION)
                + count_generation_tokens(base_user_message)
                + PROMPT_TOKEN_OVERHEAD
            )
            available_input_tokens = (
                self._settings.max_generation_input_tokens
                - self._settings.max_output_tokens
                - base_tokens
            )
            context_budget = min(
                self._settings.max_context_tokens, available_input_tokens
            )
            if context_budget <= 0:
                raise GenerationPipelineError(
                    GenerationErrorCode.CONTEXT_ERROR,
                    "Generation input budget cannot fit instructions and question.",
                )
            try:
                context_result = build_context(rerank_response.results, context_budget)
            except (ContextBuildError, TypeError, ValueError) as error:
                raise GenerationPipelineError(
                    GenerationErrorCode.CONTEXT_ERROR,
                    "Authorized evidence could not fit the context budget.",
                ) from error

            self._validate_external_data_policy(request, context_result.included)
            user_message = compose_user_message(request.query, context_result.text)
            estimated_total_input_tokens = base_tokens + context_result.estimated_tokens
            generation = await self._generator.generate(
                SYSTEM_INSTRUCTION,
                user_message,
                request_id=request_id,
            )
            total_ms = (perf_counter() - started) * 1000
            return RagAnswerResponse(
                answer=generation.answer,
                model=generation.model,
                status=GenerationStatus.GENERATED,
                evidence_chunk_ids=tuple(
                    item.passage.chunk_id for item in context_result.included
                ),
                evidence=tuple(
                    EvidenceIdentity(
                        chunk_id=item.passage.chunk_id,
                        document_id=item.passage.document_id,
                        document_version=item.passage.document_version,
                        title=item.passage.title,
                        section=item.passage.section,
                        start_page=item.passage.start_page,
                        end_page=item.passage.end_page,
                        source_reference=item.passage.source_reference,
                        rerank_rank=item.rerank_rank,
                    )
                    for item in context_result.included
                ),
                request_count=1,
                provider_response_id=generation.provider_response_id,
                usage=generation.usage,
                cost=generation.cost,
                context=ContextDiagnostics(
                    available_context_tokens=context_result.available_tokens,
                    estimated_context_tokens=context_result.estimated_tokens,
                    estimated_total_input_tokens=estimated_total_input_tokens,
                    included_passage_count=len(context_result.included),
                    excluded_passage_count=len(context_result.excluded_chunk_ids),
                    truncated_chunk_ids=context_result.truncated_chunk_ids,
                ),
                latency=GenerationLatency(
                    retrieval_ms=retrieval_ms,
                    reranking_ms=reranking_ms,
                    generation_ms=generation.latency_ms,
                    total_ms=total_ms,
                ),
            )
        except asyncio.CancelledError:
            raise
        finally:
            logger.info(
                "performance request_id=%s operation=rag_answer total_ms=%.3f",
                request_id,
                (perf_counter() - started) * 1000,
            )

    def _restrict_to_external_visibilities(self, request: SearchRequest) -> SearchRequest:
        """Narrow the trusted scope to visibilities approved for external generation.

        Retrieving only sendable evidence keeps one highly ranked restricted
        passage from failing the whole request after paid-for retrieval work.
        """
        allowed = set(self._settings.externally_allowed_visibilities)
        permitted = request.context.permitted_visibilities
        sendable = tuple(value for value in permitted if value in allowed)
        if not sendable:
            raise GenerationPipelineError(
                GenerationErrorCode.EXTERNAL_DATA_POLICY_ERROR,
                "No authorized visibility is approved for external generation.",
            )
        if sendable == permitted:
            return request
        return replace(
            request, context=replace(request.context, permitted_visibilities=sendable)
        )

    def _validate_external_data_policy(self, request: SearchRequest, passages) -> None:
        """Prevent unauthorized or deployment-disallowed evidence transmission."""
        allowed = set(self._settings.externally_allowed_visibilities)
        for reranked in passages:
            passage = reranked.passage
            # Retrieval remains the authorization boundary. These checks detect
            # contract corruption before evidence leaves the backend.
            if passage.tenant_id != request.context.tenant_id:
                raise GenerationPipelineError(
                    GenerationErrorCode.EXTERNAL_DATA_POLICY_ERROR,
                    "Evidence tenant does not match the trusted request scope.",
                )
            if passage.visibility not in allowed:
                raise GenerationPipelineError(
                    GenerationErrorCode.EXTERNAL_DATA_POLICY_ERROR,
                    "Evidence visibility is not approved for external generation.",
                )
