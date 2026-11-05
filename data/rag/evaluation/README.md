# Review-required evaluation seed bank

## Phase 11 golden candidates

`dataset_manifest_v1.json` and `golden_questions_v1.jsonl` define the typed
Phase 11 candidate format. The manifest pins the JSONL, corpus catalog, and
section catalog by SHA-256. The loader resolves every answerable gold reference
against both catalogs and verifies quotes directly for readable source formats.
It rejects artifact drift, duplicate IDs, duplicate normalized
questions, cross-split leakage groups, tenant mismatches, lifecycle mismatches,
and contradictory answer/review labels.

All repository candidates are currently `needs_review`. They are proposals
derived from the synthetic corpus, not approved ground truth. Official loading
must use `official=True`, which requires an approved manifest and approved
records with reviewer identity and timestamp.

### Human review workflow

1. Open the exact source document/version/section named by each evidence item.
2. Confirm the quote exists and supports every required fact and the complete
   expected answer. Add evidence items when the answer requires multiple spans.
3. Search the authorized tenant corpus for conflicting active text. For an
   unanswerable record, confirm that no authorized source establishes the
   requested fact. For an authorization probe, confirm that the requested
   material is outside the authenticated tenant scope.
4. Check that question wording is natural and not a near-copy of records in the
   other split. Related questions must share one `leakage_group` and remain in
   the same split.
5. Set `review_status` to `approved`, add `reviewed_by` and an ISO-8601
   `reviewed_at`, and record material reviewer reasoning in `annotation_notes`.
   Use `rejected` instead when a label cannot be made authoritative.
6. After all intended official records are approved, update the manifest review
   status and version. Recalculate catalog hashes only when intentionally
   freezing a new corpus snapshot; never change a hash merely to bypass drift.

At least two reviewers should resolve ambiguous, multi-evidence, historical,
unanswerable, and authorization cases before the held-out split is frozen.

## Measurement artifacts

The Phase 11 aggregator accepts optional JSONL inputs:

- `QuestionResult` rows for retrieval/ranking candidates and metrics;
- `ClaimReview` rows for independent human claim and citation judgments;
- `QuestionReview` rows for answer/abstention outcomes;
- `OperationalSample` rows for stage latency, usage, cost, and reliability.

Run `scripts/evaluate_rag.py --help` for the full aggregation contract. Missing
inputs remain `NOT MEASURED`; they are never converted to zero. Configuration D
can be frozen for Phase 13 only from an approved, complete run with no failed
questions. Retrieval-only configurations A–D must not make OpenAI calls.

## Legacy synthetic CSV

- `heldout_test`: freeze after human review; never train on its query/answer labels or near-duplicate templates.
- `validation`: may guide development, never used as final unbiased test.
- `train_candidate`: queries from eligible document families; manual review before training.
- `expected_no_answer=true`: do not fabricate a source answer.
- `gold_document_id`, `gold_version`, `gold_section_key` are *source anchors*, not implementation chunk IDs.
- Questions generated from uniform templates are NOT an independent industry-quality test. Add manually written difficult questions and external holdout corpus for a credible benchmark.
