"""Validate the common Phase 7 search request boundary."""

from __future__ import annotations

import unittest

from app.rag.persistence.lifecycle import EmbeddingProfile
from app.rag.retrieval import (
    SearchContextOrigin,
    SearchFilters,
    SearchMode,
    SearchRequest,
    TrustedSearchContext,
)


PROFILE = EmbeddingProfile(
    "BAAI/bge-small-en-v1.5", "a" * 40, "b" * 64, True, "phase5-v1"
)
CONTEXT = TrustedSearchContext(
    "tenant-a", ("public",), SearchContextOrigin.INTERNAL
)


class SearchContractTests(unittest.TestCase):
    """Reject unsafe or ambiguous requests before either branch executes."""

    def test_all_modes_construct_with_one_consistent_contract(self) -> None:
        """Accept each explicit mode without silently defaulting unknown values."""
        for mode in SearchMode:
            request = SearchRequest(
                "refund policy", mode, SearchFilters(), 5, CONTEXT, PROFILE
            )
            self.assertIs(request.mode, mode)
        with self.assertRaisesRegex(TypeError, "SearchMode"):
            SearchRequest("refund", "unknown", SearchFilters(), 5, CONTEXT, PROFILE)
        keyword = SearchRequest(
            "identifier", SearchMode.KEYWORD, SearchFilters(), 5, CONTEXT
        )
        self.assertIsNone(keyword.profile)
        with self.assertRaisesRegex(TypeError, "require an EmbeddingProfile"):
            SearchRequest("meaning", SearchMode.SEMANTIC, SearchFilters(), 5, CONTEXT)

    def test_query_k_and_candidate_bounds(self) -> None:
        """Bound query work and reject booleans masquerading as integers."""
        with self.assertRaisesRegex(ValueError, "nonempty"):
            SearchRequest(" ", SearchMode.HYBRID, SearchFilters(), 5, CONTEXT, PROFILE)
        with self.assertRaisesRegex(ValueError, "k must"):
            SearchRequest("valid", SearchMode.HYBRID, SearchFilters(), 0, CONTEXT, PROFILE)
        with self.assertRaisesRegex(TypeError, "candidate"):
            SearchRequest(
                "valid", SearchMode.HYBRID, SearchFilters(), 5, CONTEXT, PROFILE,
                keyword_candidate_limit=True,
            )

    def test_filters_validate_dates_and_narrowing_values(self) -> None:
        """Reject malformed metadata filters at the service boundary."""
        with self.assertRaisesRegex(ValueError, "YYYY-MM-DD"):
            SearchFilters(effective_date_from="22 October 2026")
        with self.assertRaisesRegex(ValueError, "must not follow"):
            SearchFilters(
                effective_date_from="2026-10-22", effective_date_to="2026-01-01"
            )
        with self.assertRaisesRegex(ValueError, "nonempty"):
            SearchFilters(document_id="")


if __name__ == "__main__":
    unittest.main()
