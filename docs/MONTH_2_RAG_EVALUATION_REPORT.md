# Month 2 RAG Evaluation Report

## Evaluation objective

Measure retrieval, reranking, grounded-answer quality, citation correctness,
abstention, authorization, reliability, latency, token usage, and cost against
human-reviewed labels and a reproducibly identified corpus.

## Current status

**PENDING HUMAN REVIEW AND LIVE EXECUTION**

The evaluation implementation exists, but the current seven candidate questions
are not approved ground truth. No quality, security, latency, token, or cost
result is reported as measured in this document.

## Dataset

- Dataset: `month2-rag-golden-candidates`
- Version: `1.0.0`
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
| C | Hybrid RRF | None | No | NOT MEASURED |
| D | Hybrid RRF | Qwen3 pretrained | No | NOT MEASURED |
| E | Hybrid RRF | Qwen3 pretrained | Yes | NOT MEASURED |

## Retrieval metrics

- Precision@5/10: NOT MEASURED
- Recall@5/10/20: NOT MEASURED
- Complete Evidence Coverage@5/10/20: NOT MEASURED

## Reranker metrics

- MRR@10: NOT MEASURED
- NDCG@5/10: NOT MEASURED
- Helped/hurt questions: NOT MEASURED
- Fixed-candidate preservation: IMPLEMENTED; LIVE VERIFICATION PENDING

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
- Database, Qwen3, and OpenAI evaluations have not been run in this checkpoint.
- Generator/verifier judgments cannot replace independent human review.
- The corpus is synthetic and cannot establish performance on real policies.

## Recommendation

Complete annotation review, expand the dataset, execute A–E in an isolated
development environment, collect independent claim/citation reviews, and then
generate the final JSON and Markdown artifacts before accepting Phase 11.
