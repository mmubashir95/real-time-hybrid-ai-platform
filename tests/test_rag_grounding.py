"""Verify Phase 10 citation authority, semantic filtering, and fail-closed behavior."""

from __future__ import annotations

import unittest

from app.config import RagGenerationSettings
from app.rag.generation.models import LlmStructuredResult
from app.rag.grounding.citations import EvidenceRegistry
from app.rag.grounding.models import (
    ClaimSupport,
    ClaimVerification,
    GeneratedClaim,
    GroundedDraft,
    GroundingStatus,
    VerificationBatch,
)
from app.rag.grounding.service import GroundedRagService
from app.rag.persistence.lifecycle import EmbeddingProfile
from app.rag.reranking.models import RerankedPassage, RerankDiagnostics, RerankResponse
from app.rag.retrieval.models import (
    RetrievedPassage,
    SearchContextOrigin,
    SearchDiagnostics,
    SearchFilters,
    SearchMode,
    SearchRequest,
    SearchResponse,
    TrustedSearchContext,
)


def settings() -> RagGenerationSettings:
    """Build bounded provider and egress settings for service tests."""
    return RagGenerationSettings(
        "test", "gpt-5-nano", 20, 256, 1_000, 2_000, "minimal", 2, ("public",)
    )


def request() -> SearchRequest:
    """Build a trusted active-policy request."""
    profile = EmbeddingProfile("model", "a" * 40, "b" * 64, True, "phase5-v1")
    return SearchRequest(
        "Can a duplicate charge be refunded?", SearchMode.HYBRID, SearchFilters(), 20,
        TrustedSearchContext("asterdesk", ("public",), SearchContextOrigin.INTERNAL),
        profile, 20, 20,
    )


def passage(
    identifier: str = "a", *, tenant: str = "asterdesk", status: str = "active",
) -> RerankedPassage:
    """Build one provenance-complete authorized passage."""
    retrieved = RetrievedPassage(
        "chk_" + identifier * 64, "AST-REF-001", "2.0", tenant,
        "Refund Policy", ("Decision Rules",), None, None,
        "Verified duplicate subscription charges may qualify for a full refund.",
        "documents/asterdesk/refund.md", "md", (1,), status, "public", "2026-01-01",
        SearchMode.HYBRID, 1, fusion_score=0.03,
    )
    return RerankedPassage(
        retrieved, 1, 2.0, 1, "Qwen/Qwen3-Reranker-0.6B", "c" * 40, 100, False
    )


class FakePipeline:
    """Return controlled Phase 7/8 results without database or model work."""

    def __init__(self, results) -> None:
        """Store the authorized results returned to grounding."""
        self.results = results

    async def search(self, search_request, *, top_k):
        """Return paired retrieval and reranking diagnostics."""
        search = SearchResponse(
            tuple(item.passage for item in self.results),
            SearchDiagnostics(1, 1, len(self.results), len(self.results), .01, .01, .01, .01, .05),
        )
        rerank = RerankResponse(
            self.results,
            RerankDiagnostics(len(self.results), len(self.results), 0, 4, .1, .2, "cpu", "float32"),
        )
        return search, rerank


class FakeStructuredGenerator:
    """Return queued typed outputs and record bounded prompts."""

    def __init__(self, *outputs) -> None:
        """Queue generation then verification models."""
        self.outputs = list(outputs)
        self.calls = []

    async def generate_structured(
        self, instruction, message, output_model, *, request_id="-", max_output_tokens=None,
    ):
        """Return the next typed fixture as a measured provider result."""
        self.calls.append((instruction, message, output_model, max_output_tokens))
        output = self.outputs.pop(0)
        return LlmStructuredResult(output, "gpt-5-nano-test", None, None, None, 10)


class EvidenceRegistryTests(unittest.TestCase):
    """Test request-local IDs and provenance rejection independently."""

    def test_registry_assigns_stable_id_and_rejects_unknown_id(self) -> None:
        """Never accept a model citation merely because it resembles an E ID."""
        registry = EvidenceRegistry((passage(),), "asterdesk")
        self.assertEqual(registry.records[0].evidence_id, "E1")
        self.assertEqual(registry.records[0].document_version, "2.0")
        with self.assertRaisesRegex(ValueError, "Unknown evidence"):
            registry.validate_claim(GeneratedClaim(text="Claim", evidence_ids=("E9",)))

    def test_registry_rejects_duplicate_chunks_and_foreign_tenant(self) -> None:
        """Prevent ambiguous identity and cross-tenant evidence registration."""
        with self.assertRaisesRegex(ValueError, "duplicate chunks"):
            EvidenceRegistry((passage(), passage()), "asterdesk")
        with self.assertRaisesRegex(ValueError, "tenant"):
            EvidenceRegistry((passage(tenant="brookhaven"),), "asterdesk")


class GroundedRagServiceTests(unittest.IsolatedAsyncioTestCase):
    """Test accepted, rejected, and insufficient grounded answer behavior."""

    async def test_supported_claim_is_rendered_with_registry_citation(self) -> None:
        """Render only a verifier-supported claim and server provenance."""
        draft = GroundedDraft(
            status=GroundingStatus.GROUNDED,
            claims=(GeneratedClaim(text="Duplicate charges may qualify.", evidence_ids=("E1",)),),
        )
        verification = VerificationBatch(decisions=(
            ClaimVerification(claim_index=0, outcome=ClaimSupport.SUPPORTED, reason="Directly stated."),
        ))
        generator = FakeStructuredGenerator(draft, verification)
        response = await GroundedRagService(
            FakePipeline((passage(),)), generator, settings()
        ).answer_question(request())

        self.assertEqual(response.status, GroundingStatus.GROUNDED)
        self.assertEqual(response.answer, "Duplicate charges may qualify. [1]")
        self.assertEqual(response.citations[0].document_id, "AST-REF-001")
        self.assertEqual(response.grounding.supported_claim_count, 1)
        self.assertNotIn("documents/asterdesk", response.answer)
        self.assertEqual(len(generator.calls), 2)

    async def test_unsupported_and_contradicted_claims_are_not_rendered(self) -> None:
        """Fail safely when no generated material claim survives verification."""
        claims = (
            GeneratedClaim(text="Refunds arrive in 24 hours.", evidence_ids=("E1",)),
            GeneratedClaim(text="Duplicate charges never qualify.", evidence_ids=("E1",)),
        )
        decisions = (
            ClaimVerification(claim_index=0, outcome=ClaimSupport.UNSUPPORTED, reason="No deadline."),
            ClaimVerification(claim_index=1, outcome=ClaimSupport.CONTRADICTED, reason="Conflicts."),
        )
        response = await GroundedRagService(
            FakePipeline((passage(),)),
            FakeStructuredGenerator(
                GroundedDraft(status=GroundingStatus.GROUNDED, claims=claims),
                VerificationBatch(decisions=decisions),
            ),
            settings(),
        ).answer_question(request())

        self.assertEqual(response.status, GroundingStatus.INSUFFICIENT_EVIDENCE)
        self.assertNotIn("24 hours", response.answer)
        self.assertEqual(response.citations, ())
        self.assertEqual(response.grounding.unsupported_claim_count, 1)
        self.assertEqual(response.grounding.contradicted_claim_count, 1)

    async def test_forged_id_skips_verifier_and_cannot_be_grounded(self) -> None:
        """Reject unknown IDs structurally before passage text reaches verification."""
        generator = FakeStructuredGenerator(GroundedDraft(
            status=GroundingStatus.GROUNDED,
            claims=(GeneratedClaim(text="Forged claim.", evidence_ids=("E99",)),),
        ))
        response = await GroundedRagService(
            FakePipeline((passage(),)), generator, settings()
        ).answer_question(request())

        self.assertEqual(response.status, GroundingStatus.INSUFFICIENT_EVIDENCE)
        self.assertEqual(response.grounding.invalid_citation_count, 1)
        self.assertEqual(len(generator.calls), 1)

    async def test_incomplete_verifier_coverage_fails_closed(self) -> None:
        """Never treat a partial verifier response as successful grounding."""
        draft = GroundedDraft(
            status=GroundingStatus.GROUNDED,
            claims=(GeneratedClaim(text="Duplicate charges may qualify.", evidence_ids=("E1",)),),
        )
        response = await GroundedRagService(
            FakePipeline((passage(),)),
            FakeStructuredGenerator(draft, VerificationBatch(decisions=())),
            settings(),
        ).answer_question(request())

        self.assertEqual(response.status, GroundingStatus.GROUNDING_FAILED)
        self.assertEqual(response.citations, ())
        self.assertEqual(response.request_count, 2)

    async def test_empty_retrieval_uses_no_provider_calls(self) -> None:
        """Keep no-evidence distinct and cost-free."""
        generator = FakeStructuredGenerator()
        response = await GroundedRagService(
            FakePipeline(()), generator, settings()
        ).answer_question(request())
        self.assertEqual(response.status, GroundingStatus.INSUFFICIENT_EVIDENCE)
        self.assertEqual(response.request_count, 0)
        self.assertEqual(generator.calls, [])


if __name__ == "__main__":
    unittest.main()
