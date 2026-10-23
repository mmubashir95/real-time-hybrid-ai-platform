"""Orchestrate authorized evidence, structured claims, verification, and rendering."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from time import perf_counter
from typing import Protocol

from pydantic import BaseModel

from app.config import RagGenerationSettings
from app.rag.generation.context_builder import (
    ContextBuildError,
    build_context,
    count_generation_tokens,
)
from app.rag.generation.models import (
    GenerationErrorCode,
    GenerationPipelineError,
    LlmStructuredResult,
)
from app.rag.generation.prompts import compose_user_message
from app.rag.grounding.citations import EvidenceRegistry
from app.rag.grounding.models import (
    ClaimSupport,
    GeneratedClaim,
    GroundedAnswerResponse,
    GroundedDraft,
    GroundingDiagnostics,
    GroundingLatency,
    GroundingStatus,
    ValidatedCitation,
    VerificationBatch,
)
from app.rag.grounding.prompts import (
    GROUNDED_GENERATION_INSTRUCTION,
    VERIFICATION_INSTRUCTION,
    compose_verification_message,
)
from app.rag.reranking.qwen3 import RerankerInferenceError
from app.rag.retrieval.dense import RetrievalError
from app.rag.retrieval.models import SearchRequest


INSUFFICIENT_ANSWER = (
    "The available authorized evidence is insufficient to provide a verified answer."
)
VERIFICATION_LIMITATION = (
    "Semantic support is judged by an automated model and is not a guarantee of factual correctness."
)
PROMPT_TOKEN_OVERHEAD = 32
VERIFICATION_TOKENS_PER_CLAIM = 96


class RetrievalAndReranking(Protocol):
    """Describe the established Phase 7/8 orchestration boundary."""

    async def search(self, request: SearchRequest, *, top_k: int):
        """Return authorized retrieval and reranking responses."""


class StructuredGenerator(Protocol):
    """Describe the existing provider client structured-output boundary."""

    async def generate_structured(
        self,
        system_instruction: str,
        user_message: str,
        output_model: type[BaseModel],
        *,
        request_id: str = "-",
        max_output_tokens: int | None = None,
    ) -> LlmStructuredResult:
        """Return one provider-validated Pydantic result."""


class GroundedRagService:
    """Produce answers containing only claims accepted by bounded verification."""

    def __init__(
        self,
        retrieval_and_reranking: RetrievalAndReranking,
        generator: StructuredGenerator,
        settings: RagGenerationSettings,
    ) -> None:
        """Bind existing retrieval/reranking and OpenAI client components."""
        self._pipeline = retrieval_and_reranking
        self._generator = generator
        self._settings = settings

    async def answer_question(
        self,
        request: SearchRequest,
        *,
        top_k: int = 5,
        request_id: str = "-",
    ) -> GroundedAnswerResponse:
        """Generate, verify once, and deterministically render a grounded answer."""
        started = perf_counter()
        request = self._restrict_to_external_visibilities(request)
        try:
            search_response, rerank_response = await self._pipeline.search(
                request, top_k=top_k
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
            raise GenerationPipelineError(
                GenerationErrorCode.RERANKING_ERROR,
                "Retrieval or reranking rejected the request or its candidates.",
            ) from error
        retrieval_ms = search_response.diagnostics.total_seconds * 1_000
        reranking_ms = rerank_response.diagnostics.inference_seconds * 1_000
        if not rerank_response.results:
            return self._empty_response(started, retrieval_ms, reranking_ms)

        # Reuse the Phase 9 context builder so E-number order and truncation
        # behavior remain identical between comparison modes.
        empty_message = compose_user_message(request.query, "")
        fixed_tokens = (
            count_generation_tokens(GROUNDED_GENERATION_INSTRUCTION)
            + count_generation_tokens(empty_message)
            + PROMPT_TOKEN_OVERHEAD
        )
        context_budget = min(
            self._settings.max_context_tokens,
            self._settings.max_generation_input_tokens
            - self._settings.max_output_tokens
            - fixed_tokens,
        )
        if context_budget <= 0:
            raise GenerationPipelineError(
                GenerationErrorCode.CONTEXT_ERROR,
                "Grounded generation input budget cannot fit instructions and question.",
            )
        try:
            context = build_context(rerank_response.results, context_budget)
        except (ContextBuildError, TypeError, ValueError) as error:
            raise GenerationPipelineError(
                GenerationErrorCode.CONTEXT_ERROR,
                "Authorized evidence could not fit the grounded context budget.",
            ) from error
        try:
            self._validate_evidence_policy(request, context.included)
            registry = EvidenceRegistry(context.included, request.context.tenant_id)
        except ValueError as error:
            raise GenerationPipelineError(
                GenerationErrorCode.EXTERNAL_DATA_POLICY_ERROR,
                "Evidence failed grounding provenance or authorization validation.",
            ) from error
        draft_result = await self._generator.generate_structured(
            GROUNDED_GENERATION_INSTRUCTION,
            compose_user_message(request.query, context.text),
            GroundedDraft,
            request_id=request_id,
        )
        draft = draft_result.output
        if draft.status != GroundingStatus.GROUNDED:
            return self._insufficient_draft_response(
                started, retrieval_ms, reranking_ms, registry, draft_result
            )

        valid_claims, invalid_count = self._resolve_claims(draft.claims, registry)
        if not valid_claims:
            return self._all_rejected_response(
                started,
                retrieval_ms,
                reranking_ms,
                registry,
                draft_result,
                len(draft.claims),
                invalid_count,
            )

        verification_message = compose_verification_message(valid_claims)
        try:
            verification_result = await self._generator.generate_structured(
                VERIFICATION_INSTRUCTION,
                verification_message,
                VerificationBatch,
                request_id=request_id,
                # Each decision carries a label and a short reason; scale the
                # allowance so a full batch is not truncated mid-JSON.
                max_output_tokens=max(
                    self._settings.max_output_tokens,
                    VERIFICATION_TOKENS_PER_CLAIM * len(valid_claims),
                ),
            )
        except GenerationPipelineError:
            return self._verification_failed_response(
                started,
                retrieval_ms,
                reranking_ms,
                registry,
                draft_result,
                len(draft.claims),
                invalid_count,
            )

        decisions = {item.claim_index: item for item in verification_result.output.decisions}
        expected = {item[0] for item in valid_claims}
        # Missing, duplicate, or invented indices make the entire verification
        # result unusable; partial acceptance could otherwise hide a model error.
        if len(decisions) != len(verification_result.output.decisions) or set(decisions) != expected:
            return self._verification_failed_response(
                started,
                retrieval_ms,
                reranking_ms,
                registry,
                draft_result,
                len(draft.claims),
                invalid_count,
                verification_result=verification_result,
            )

        supported = [
            item for item in valid_claims
            if decisions[item[0]].outcome == ClaimSupport.SUPPORTED
        ]
        counts = self._outcome_counts(decisions.values())
        answer, citations = self._render_supported(supported)
        status = GroundingStatus.GROUNDED if supported else GroundingStatus.INSUFFICIENT_EVIDENCE
        if not supported:
            answer = INSUFFICIENT_ANSWER
        return GroundedAnswerResponse(
            status=status,
            answer=answer,
            citations=citations,
            evidence=registry.records,
            grounding=GroundingDiagnostics(
                generated_claim_count=len(draft.claims),
                verified_claim_count=len(decisions),
                supported_claim_count=len(supported),
                unsupported_claim_count=counts[ClaimSupport.UNSUPPORTED],
                contradicted_claim_count=counts[ClaimSupport.CONTRADICTED],
                insufficient_evidence_count=counts[ClaimSupport.INSUFFICIENT_EVIDENCE],
                invalid_citation_count=invalid_count,
                included_evidence_count=len(registry.records),
                verification_limitation=VERIFICATION_LIMITATION,
            ),
            generation_model=draft_result.model,
            verification_model=verification_result.model,
            generation_usage=draft_result.usage,
            verification_usage=verification_result.usage,
            generation_cost=draft_result.cost,
            verification_cost=verification_result.cost,
            request_count=2,
            latency=GroundingLatency(
                retrieval_ms=retrieval_ms,
                reranking_ms=reranking_ms,
                generation_ms=draft_result.latency_ms,
                verification_ms=verification_result.latency_ms,
                total_ms=(perf_counter() - started) * 1_000,
            ),
        )

    def _restrict_to_external_visibilities(self, request: SearchRequest) -> SearchRequest:
        """Narrow retrieval to evidence permitted to leave the deployment."""
        allowed = set(self._settings.externally_allowed_visibilities)
        sendable = tuple(
            item for item in request.context.permitted_visibilities if item in allowed
        )
        if not sendable:
            raise ValueError("No authorized visibility is approved for external generation")
        if sendable == request.context.permitted_visibilities:
            return request
        return replace(
            request,
            context=replace(request.context, permitted_visibilities=sendable),
        )

    def _validate_evidence_policy(self, request: SearchRequest, passages) -> None:
        """Detect corrupted tenant, visibility, or lifecycle data before egress."""
        historical = request.context.historical
        allowed = set(self._settings.externally_allowed_visibilities)
        for item in passages:
            passage = item.passage
            if passage.tenant_id != request.context.tenant_id or passage.visibility not in allowed:
                raise ValueError("Evidence violates trusted tenant or visibility scope")
            if historical is None and passage.status != "active":
                raise ValueError("Archived evidence requires exact historical authorization")
            if historical is not None and (
                passage.document_id != historical.document_id
                or passage.document_version != historical.version
            ):
                raise ValueError("Evidence does not match exact historical authorization")

    @staticmethod
    def _resolve_claims(claims, registry):
        """Separate structurally valid claims without trusting model provenance."""
        valid = []
        invalid_count = 0
        for index, claim in enumerate(claims):
            try:
                evidence = registry.validate_claim(claim)
            except ValueError:
                invalid_count += 1
                continue
            valid.append((index, claim, evidence))
        return tuple(valid), invalid_count

    @staticmethod
    def _outcome_counts(decisions) -> dict[ClaimSupport, int]:
        """Count each verifier label for inspectable final diagnostics."""
        counts = {outcome: 0 for outcome in ClaimSupport}
        for decision in decisions:
            counts[decision.outcome] += 1
        return counts

    @staticmethod
    def _render_supported(supported):
        """Render supported claims and stable citations by first evidence use."""
        number_by_id: dict[str, int] = {}
        citations: list[ValidatedCitation] = []
        lines: list[str] = []
        for _, claim, evidence in supported:
            numbers: list[int] = []
            for record in evidence:
                if record.evidence_id not in number_by_id:
                    number = len(number_by_id) + 1
                    number_by_id[record.evidence_id] = number
                    citations.append(ValidatedCitation(
                        citation_number=number,
                        evidence_id=record.evidence_id,
                        chunk_id=record.chunk_id,
                        document_id=record.document_id,
                        document_version=record.document_version,
                        document_title=record.document_title,
                        section=record.section,
                        start_page=record.start_page,
                        end_page=record.end_page,
                    ))
                numbers.append(number_by_id[record.evidence_id])
            suffix = "".join(f" [{number}]" for number in numbers)
            lines.append(f"{claim.text}{suffix}")
        return "\n\n".join(lines), tuple(citations)

    def _empty_response(self, started, retrieval_ms, reranking_ms):
        """Return explicit no-evidence without making a paid provider call."""
        return self._base_insufficient(
            started, retrieval_ms, reranking_ms, (), 0, 0, 0, None, None, 0
        )

    def _insufficient_draft_response(self, started, retrieval_ms, reranking_ms, registry, draft):
        """Return the model's structured insufficiency without verification."""
        return self._base_insufficient(
            started, retrieval_ms, reranking_ms, registry.records, 0, 0, 0,
            draft, None, 1
        )

    def _all_rejected_response(self, started, retrieval_ms, reranking_ms, registry, draft, count, invalid):
        """Return insufficiency when every generated claim has forged citations."""
        return self._base_insufficient(
            started, retrieval_ms, reranking_ms, registry.records, count, invalid, 0,
            draft, None, 1
        )

    def _base_insufficient(
        self, started, retrieval_ms, reranking_ms, evidence, claim_count,
        invalid_count, verified_count, draft, verification, request_count,
    ):
        """Assemble deterministic insufficient-evidence variants."""
        return GroundedAnswerResponse(
            status=GroundingStatus.INSUFFICIENT_EVIDENCE,
            answer=INSUFFICIENT_ANSWER,
            citations=(), evidence=evidence,
            grounding=GroundingDiagnostics(
                generated_claim_count=claim_count,
                verified_claim_count=verified_count,
                supported_claim_count=0, unsupported_claim_count=0,
                contradicted_claim_count=0, insufficient_evidence_count=0,
                invalid_citation_count=invalid_count,
                included_evidence_count=len(evidence),
                verification_limitation=VERIFICATION_LIMITATION,
            ),
            generation_model=draft.model if draft else self._settings.openai_model,
            generation_usage=draft.usage if draft else None,
            generation_cost=draft.cost if draft else None,
            verification_model=verification.model if verification else None,
            verification_usage=verification.usage if verification else None,
            verification_cost=verification.cost if verification else None,
            request_count=request_count,
            latency=GroundingLatency(
                retrieval_ms=retrieval_ms, reranking_ms=reranking_ms,
                generation_ms=draft.latency_ms if draft else 0,
                verification_ms=verification.latency_ms if verification else None,
                total_ms=(perf_counter() - started) * 1_000,
            ),
        )

    def _verification_failed_response(
        self, started, retrieval_ms, reranking_ms, registry, draft, claim_count,
        invalid_count, *, verification_result=None,
    ):
        """Fail closed when the semantic verifier fails or returns malformed coverage."""
        base = self._base_insufficient(
            started, retrieval_ms, reranking_ms, registry.records, claim_count,
            invalid_count, 0, draft, verification_result,
            2,
        )
        return base.model_copy(update={"status": GroundingStatus.GROUNDING_FAILED})
