# Phase 8 — Pretrained Qwen3 Reranking

## Status

Phase 8 code is implemented. Real model inference, database-backed integration,
and corpus evaluation remain user-run verification steps. No measured result is
recorded until its corresponding command succeeds.

## Architecture

The existing Phase 7 `RetrievalService` remains unchanged and independently
available. `RetrievalRerankingService` requests one authorized hybrid candidate
set and passes only those returned `RetrievedPassage` objects to
`RerankingService`. Reranking never queries the database, changes permissions,
loads other documents, or rewrites source metadata.

Each final result retains the exact original passage object and adds:

- original retrieval rank;
- Qwen3 raw relevance score;
- reranked rank;
- pinned model identity and revision;
- input token count and truncation status.

## Pinned Version 1 configuration

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen3-Reranker-0.6B` |
| Model revision | `fd9fb1d26c07223ced488065909faf522e29cc7d` |
| Tokenizer revision | Same pinned revision |
| License | Apache-2.0 |
| Architecture | Qwen3 causal language model used for text reranking |
| Scoring | Raw `yes` token logit minus `no` token logit; higher is better |
| Instruction | Customer-support policy-passage retrieval instruction in `RerankerConfig` |
| Configured token budget | 8,192 tokens |
| Model-supported context | 32k tokens (the application deliberately uses a smaller bound) |
| Default batch size | 4 |
| Default device | CPU; MPS is an explicit alternative |
| Default precision | `auto` (float32 on CPU; checkpoint dtype on accelerators) |
| Transformers | `4.57.6` |
| Sentence Transformers | `5.1.1` (used by Phase 5 embeddings, not Qwen scoring) |
| PyTorch | `2.14.1` |

The installed Sentence Transformers 5.1.1 `CrossEncoder` implementation loads
`AutoModelForSequenceClassification`; it does not implement the newer causal-LM
`LogitScore` module needed by this Qwen checkpoint. Phase 8 therefore follows
Qwen's documented Transformers implementation directly rather than applying an
incorrect classification-head assumption.

## Input and truncation policy

The adapter builds Qwen's documented system prompt, instruction, query,
document, and assistant suffix. Like Qwen's reference code, it tokenizes the
instruction, query, and passage as one string so the passage's first word keeps
its leading-space token merge; tokenizing the passage separately changes the
token sequence and shifts scores. Over-budget inputs are cut from the tail, and
because the passage comes last, only passage tokens may be truncated.

`auto` precision loads the checkpoint's bfloat16 weights on accelerators but
resolves to float32 on CPU: bfloat16 lacks fast kernels on most CPUs and was
measured at about 8x slower (35.3s vs 4.5s for 20 candidates). If the fixed task and query exceed the entire budget, inference
fails explicitly. Every truncated pair is counted in diagnostics and evaluation.

Phase 4 currently targets 400-token chunks and limits chunks to 500
`o200k_base` tokens. Qwen uses a different tokenizer, so Phase 8 still measures
its actual formatted input instead of assuming Phase 4 counts are identical.

## Security and failure behavior

- Phase 7 establishes tenant, visibility, and active/historical-version scope.
- Phase 8 accepts only the passages returned by that search call.
- Duplicate chunk IDs and empty passage text are rejected.
- The model is not loaded for an empty candidate list.
- Invalid, non-finite, missing, or extra scores fail the request.
- There is no silent fallback to hybrid order after a reranker failure.
- Candidate text is model input data and cannot grant retrieval authorization.

## Commands for user-run verification

The commands below are intentionally documented but were not run by Codex.

Run focused deterministic tests (no model download or database required):

```bash
.venv/bin/python -m unittest tests.test_rag_reranking
```

Run all regression tests:

```bash
.venv/bin/python -m unittest discover -s tests
```

Run the required real-model smoke test on CPU:

```bash
.venv/bin/python scripts/smoke_test_rag_reranker.py --device cpu
```

Optionally compare Apple Silicon MPS after CPU succeeds:

```bash
.venv/bin/python scripts/smoke_test_rag_reranker.py --device mps --precision float16
```

Run a real authorized retrieval-to-reranking inspection (replace the embedding
fingerprint with the imported Phase 5/6 profile):

```bash
.venv/bin/python scripts/rerank_rag_documents.py \
  --query "Can I get a refund after being charged twice?" \
  --tenant asterdesk \
  --visibility public \
  --visibility internal \
  --candidate-k 20 \
  --top-k 5 \
  --config-fingerprint REPLACE_WITH_IMPORTED_PROFILE_FINGERPRINT
```

Run the exploratory benchmark only with explicit acknowledgement that the
current labels are still `synthetic_seed_needs_review`:

```bash
.venv/bin/python scripts/evaluate_rag_reranking.py \
  --config-fingerprint REPLACE_WITH_IMPORTED_PROFILE_FINGERPRINT \
  --candidate-limit 20 \
  --top-k 5 \
  --allow-unreviewed-labels
```

Without `--allow-unreviewed-labels`, the evaluator refuses the current dataset.
Both baselines use the exact same candidate list retrieved once per question.
Both arms keep the full ordering of every candidate (the reranker scores all of
them anyway), so @10 metrics really cover ranks 6-10; `--top-k` is recorded
in the report but does not truncate the evaluated rankings.
The report includes frozen chunk IDs, Recall@5, MRR@10, NDCG@5/10, candidate
Recall@20, p50/p95 latency, truncation count, and authorization violations.
Peak memory is reported as `NOT MEASURED` until a reliable platform-specific
measurement is added.

## What reranking does and does not do

A cross-encoder jointly processes a question and passage, allowing richer
token-level interaction than independent embedding similarity. Qwen3 differs
from a conventional classifier-head cross-encoder: its causal LM produces
relevance through the relative next-token logits for `yes` and `no` under a
specific prompt.

BM25 and semantic retrieval remain necessary because scoring every corpus chunk
with Qwen would be too expensive. They also define the authorized candidate
set. Reranking can move relevant evidence into the top five, but it cannot
recover evidence absent from the top 20 candidates. Retrieval, fusion, and
reranker scores have different meanings and scales, so they are retained as
separate diagnostics rather than added together.

Larger batches may improve throughput but increase peak memory and sometimes
single-query latency. Long inputs also increase compute and may hide evidence
if truncated, which is why token counts and truncation are explicit. Preserving
the original passage object keeps chunk identity, document version, page,
section, and source reference traceable after ranking changes. The benchmark's
quality gains and added warm latency—not model reputation—determine whether the
reranker is worthwhile.

## Known limitations before final acceptance

- Real Qwen3 loading and the required relevance smoke test are not yet measured.
- CPU versus MPS correctness, latency, precision, and memory are not yet measured.
- Live PostgreSQL retrieval-to-reranking authorization is not yet verified.
- Ranking metrics are not defensible until labels receive human review; the
  current opt-in evaluation is exploratory only.
- Phase 13 must reuse the recorded candidate sets and all configuration above
  when comparing a future approved fine-tuned checkpoint.
