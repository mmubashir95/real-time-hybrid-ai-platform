# Month 2 RAG Evaluation Report

## Evaluation objective

Measure retrieval, reranking, grounded-answer quality, citation correctness,
abstention, authorization, reliability, latency, token usage, and cost against
human-reviewed labels and a reproducibly identified corpus.

## Current status

**PENDING HUMAN REVIEW AND LIVE EXECUTION**

The evaluation implementation exists, but the current seven candidate questions
are not approved ground truth. Configurations C and D have exploratory
development-split runs (below); they are not official results. No security,
token, or cost result is reported as measured in this document.

## Dataset

- Dataset: `month2-rag-golden-candidates`
- Version: `1.2.0` (C/D exploratory runs used `1.0.0`; see below)
- Candidate questions: 7
- Approved questions: 0
- Review status: `needs_review`
- Development split: 5
- Held-out test split: 2
- Corpus: fictional synthetic AsterDesk and Brookhaven sources

## Experimental configurations

| ID | Retrieval | Reranker | Grounded answer | Status |
|---|---|---|---|---|
| A | Dense | None | No | NOT MEASURED |
| B | BM25 | None | No | NOT MEASURED |
| C | Hybrid RRF | None | No | EXPLORATORY (`2745324e`) |
| D | Hybrid RRF | Qwen3 pretrained | No | EXPLORATORY (`86b8e42e`) |
| E | Hybrid RRF | Qwen3 pretrained | Yes | NOT MEASURED |

## Exploratory C/D results (not official)

Development split, 5 questions (4 answerable), unapproved labels. Both runs
used dataset `1.0.0`, before equivalent-section credit (`1.1.0`) and the
cut-at-K-then-deduplicate scoring fix, so they must be rerun before any
official claim. Artifacts: `reports/rag_evaluation/<run_id>/`.

| Metric | C hybrid | D hybrid + Qwen3 |
|---|---|---|
| Precision@5 (all 5 questions) | 0.0400 | 0.1600 |
| Precision@5 (answerable only, ceiling 0.2500) | 0.0500 | 0.2000 |
| Recall@5 | 0.2500 | 0.7500 |
| Recall@10 | 0.7500 | 1.0000 |
| Recall@20 | 1.0000 | 1.0000 |
| Complete Evidence Coverage@5 | 0.2500 | 0.7500 |
| MRR@10 | 0.1607 | 0.3857 |
| NDCG@5 | 0.1250 | 0.4234 |
| NDCG@10 | 0.3091 | 0.5067 |
| Latency p50 / p95 ms | 35.1 / 205.4 | 1025.8 / 2053.8 |

- Recall, coverage, MRR, and NDCG average the 4 answerable questions; the
  all-question precision row also includes the no-answer question's zero.
- The precision ceiling is the best P@5 any ranking could reach given the
  gold counts (three questions with one gold section, one with two).
- D's latency includes reranker inference and first-question warm-up.
- Fixed-candidate preservation: verified after the fact — D's candidate set
  equals C's for all 5 questions. Future live D runs enforce this with
  `--candidate-baseline`.
- Helped/hurt questions: NOT MEASURED.

## Answer and grounding quality

- Answer correctness: PENDING HUMAN REVIEW
- Answer completeness: PENDING HUMAN REVIEW
- Answer relevance: PENDING HUMAN REVIEW
- Claim support rate: PENDING HUMAN REVIEW
- Unsupported claim rate: PENDING HUMAN REVIEW
- Citation correctness: PENDING HUMAN REVIEW
- Abstention accuracy: PENDING HUMAN REVIEW
- Grounding failure rate: NOT MEASURED

## Security

- Tenant isolation: NOT EXECUTED FOR PHASE 11
- Historical-version authorization: NOT EXECUTED FOR PHASE 11
- Prompt injection: NOT EXECUTED FOR PHASE 11
- Citation forgery: NOT EXECUTED FOR PHASE 11
- Unauthorized evidence exposure: NOT MEASURED

Existing Phase 10 tests cover these boundaries, but their previous existence is
not reported as a Phase 11 execution result.

## Performance, tokens, and cost

- Stage latency p50/p95: NOT MEASURED
- Generation tokens: NOT MEASURED
- Verification tokens: NOT MEASURED
- Estimated cost: NOT MEASURED
- Reliability/failure rates: NOT MEASURED

## Failure examples

PENDING LIVE EXECUTION.

## Frozen pretrained baseline for Phase 13

NOT FROZEN. The CLI permits freezing only an official, complete configuration D
artifact with no failed questions.

## Limitations

- Human approval is pending.
- The initial dataset is much smaller than the planned 50–80 questions.
- Only exploratory C and D runs exist; A, B, and E (OpenAI) have not been run.
- Generator/verifier judgments cannot replace independent human review.
- The corpus is synthetic and cannot establish performance on real policies.

## Recommendation

Complete annotation review, expand the dataset, execute A–E in an isolated
development environment, collect independent claim/citation reviews, and then
generate the final JSON and Markdown artifacts before accepting Phase 11.
