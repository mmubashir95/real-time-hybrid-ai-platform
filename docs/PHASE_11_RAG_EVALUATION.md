# Phase 11 — RAG Evaluation

## Checkpoint scope

This checkpoint implements the Phase 11 dataset, metric, live-runner,
human-review, operational-summary, artifact, and reporting infrastructure.
Actual retrieval, reranker, answer-quality, performance, and security results
remain unmeasured until the owner executes the final commands.

## Existing architecture assessment

The evaluation layer can reuse production components without parallel RAG
implementations:

- `RetrievalService` independently executes keyword, semantic, or hybrid/RRF
  retrieval with the same authorization scope.
- `RerankingService` can reorder one frozen candidate set without introducing
  passages; the integrated service keeps blocking inference off the event loop.
- `GroundedRagService` exposes the Phase 10 answer, evidence, grounding counts,
  stage latency, provider usage, and cost required by later evaluation.
- Retrieval SQL applies tenant, visibility, active-version, and exact archived
  version filters before candidates leave the database.

Existing retrieval and reranking scripts remain available, but their legacy
365-row CSV contains synthetic template labels marked for review. It is useful
for exploratory work only and is not an approved golden benchmark.

## Golden dataset contract

The versioned JSONL contract records:

- stable question identity, type, tenant, expected behavior, and split;
- reviewed expected answer and individually testable required facts;
- one or more stable tenant/document/version/section/source anchors;
- a short source quote and binary-or-graded relevance value;
- leakage group, annotation notes, review state, reviewer, and timestamp.

Chunk IDs are deliberately excluded from permanent ground truth. Later runners
must resolve source anchors to the chunks produced by the frozen experiment.

The manifest pins the JSONL, `manifest.csv`, and `sections_index.csv` with
SHA-256 values. Readable source quotes are checked against the source file.
Loading fails on artifact drift, unresolved sources, wrong lifecycle state,
cross-tenant evidence, duplicate IDs or normalized queries, split leakage, or
unapproved official labels.

## Candidate dataset status

The initial JSONL contains seven evidence-backed candidates:

| Category | Count | Review state |
|---|---:|---|
| Direct policy | 2 | Needs review |
| Paraphrase | 1 | Needs review |
| Multi-evidence | 1 | Needs review |
| Historical version | 1 | Needs review |
| Unanswerable | 1 | Needs review |
| Authorization probe | 1 | Needs review |

Five candidates are in `development` and two are in `held_out_test`. None is
approved, so there are no official Phase 11 quality measurements yet.

## Corpus and metadata assessment

The repository catalogs 79 versioned synthetic AsterDesk and Brookhaven source
documents and 655 stable source-section anchors. Available provenance includes
tenant, document ID, version, lifecycle status, visibility, effective date,
source path/format, source SHA-256, and section key. The corpus is explicitly
fictional and must not be represented as real company policy.

## Implemented measurement infrastructure

The repository now includes deterministic implementations for:

- Precision@5/10, Recall@5/10/20, and Complete Evidence Coverage@K;
- MRR@10 and graded NDCG@5/10 with duplicate-source suppression;
- fixed-candidate-set validation for fair Qwen3 comparisons;
- independent human claim support, correctness, completeness, relevance,
  citation correctness, unsupported-claim, abstention, and grounding metrics;
- p50/p95 stage latency, token/cost totals, and reliability failure counts;
- immutable per-question and aggregate JSON artifacts;
- conservative Markdown reports and a guarded pretrained-baseline manifest;
- configurations A–E in the shared run contracts.

`scripts/evaluate_rag.py` validates and aggregates measured JSONL inputs. With
the explicit `--execute-live` flag, it instead invokes the existing Phase 7,
8, and 10 services. Configurations A–D never instantiate the OpenAI client;
configuration E requires OpenAI configuration and preserves complete grounded
responses for independent human review.

## Live execution prerequisites

Before retrieval evaluation:

1. Human reviewers must approve a representative development set and freeze a
   held-out set, expanding toward the planned 50–80 questions without templates
   masquerading as independent judgments.
2. Every unanswerable and authorization record must be confirmed by corpus and
   scope review, not by absence from one retrieval result.
3. Catalog hashes and index/embedding fingerprints must identify the executed
   corpus snapshot.
4. Execute configurations A–E with identical dataset, corpus/index identity,
   authorization scope, and candidate limits. Configuration D validates that
   Qwen3 preserves the frozen hybrid candidate set.

## Execution safety

Without `--execute-live`, the CLI performs offline validation and aggregation
only. Live execution requires an approved dataset unless `--exploratory` is
explicitly supplied. A–D perform no paid generation. E uses the configured
OpenAI client and should be run only after inspecting the approved scope.

Status: **Evaluation infrastructure implemented with limitations; human review
and live A–E execution pending.**
