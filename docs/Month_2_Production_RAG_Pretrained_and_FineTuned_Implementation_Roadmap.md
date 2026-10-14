# Month 2 — Production RAG: Pretrained Baseline + Fine-Tuned Reranker

**Project:** Real-Time Hybrid AI Support & Operations Platform  
**Status:** Implementation roadmap — not a statement of completed features  
**Primary career skill:** Applied AI / production NLP systems engineering  
**Secondary career skill:** Practical supervised model fine-tuning with PyTorch  
**Estimated effort:** 10–12 weeks at ~10 focused hours/week (revise after first two milestones)  
**Deliverable:** One production-style RAG service with two interchangeable reranker configurations, reproducible evaluations, and an honest comparison.

> **Central rule:** Build **Version 1 (pretrained)** first, freeze and measure it, then build **Version 2 (fine-tuned reranker)** by replacing **one model component**. Do **not** maintain two divergent applications. A fine-tuned model is accepted only if held-out evidence justifies it.

---

## 1. Purpose and Scope

### Business problem

Extend the existing customer-support classification and extraction service into a **document-grounded support knowledge system**. The service ingests real support policies, billing and subscription documentation, and help articles, then answers support questions with evidence-backed citations. In Month 3, the same retrieval function will be usable by controlled agents and approved API/database tools.

**Example question:** “I was charged twice for my subscription. What does the refund policy say?”  
**Expected behavior:** Classify/analyze the message using the existing service if appropriate; retrieve authorized and current policies; produce a grounded answer with traceable citations; state insufficient evidence if the documents cannot support an answer. **Do not invent the customer's payment status:** accessing live account data belongs to Month 3 tools.

### Version definitions

| Dimension | Version 1 — Pretrained RAG | Version 2 — Fine-tuned RAG |
|---|---|---|
| Ingestion, documents and metadata | Shared | Exactly the same |
| Document chunks / corpus snapshot | Shared frozen snapshot for comparison | Same snapshot |
| Embedding model | Pinned pretrained model | Same pretrained model |
| Dense + BM25 retrieval | Shared | Same implementation and candidate set |
| Reranker | Pinned pretrained cross-encoder | Fine-tuned checkpoint of the **same base architecture** |
| Context builder, LLM and prompt | Shared | Identical |
| Authentication, tenant filters, citations | Shared | Identical |
| Evaluation questions | Frozen held-out set | The **same** held-out set |
| Model-training step | None | Labeled pairs, supervised fine-tuning, model artifact |
| Output | Measured production-style baseline | Measured model-adaptation experiment |

**Not in scope:** training an LLM from scratch, fine-tuning the answer-generating LLM, learning multiple orchestration frameworks, agents/tool calling (Month 3), Kafka (later), a sophisticated admin dashboard, fully optimized mobile UI, and training both embedding and reranking models simultaneously.

---

## 2. Fit With the Existing Repository

The uploaded project snapshot already contains these **Month 1 components**:

- `app/main.py`: FastAPI routes such as `/health`, `/analyze`, `/analyze/stream`, `/analyses/{id}`.
- `app/services/ai_service.py`: async hosted model call and structured classification/extraction.
- `app/services/streaming_service.py`: SSE progress events for analysis.
- `app/database.py`, `app/models.py`, `app/repositories/analysis_repository.py`: SQLAlchemy/asyncpg persistence.
- `app/config.py`, `app/schemas.py`: configuration and API contracts.
- `schema.sql`, `compose.yaml`, `Dockerfile`: current PostgreSQL and container setup.
- `tests/`, `scripts/benchmark.py`: existing tests and API benchmarking support.
- `docs/`, `ai_workflows/`: the existing roadmap and implementation/review guidance.

**Reuse these foundations.** Review the current repository state before changing code; the uploaded ZIP is a snapshot, not guaranteed to match future branches. Preserve all existing analysis APIs and tests. Add new RAG capabilities incrementally; do not rewrite the Month 1 architecture.

**Infrastructure note:** The current Compose file uses ordinary `postgres:17-bookworm`. That **does not automatically provide pgvector**. Choose and pin a compatible pgvector-enabled PostgreSQL build (or install a compatible extension) and verify `CREATE EXTENSION vector` in an isolated development database. Do not assume existing persistent volumes are re-initialized when `schema.sql` changes; introduce repeatable migrations once RAG schema changes begin.

---

## 3. Target Architecture (One System, Two Configurations)

```mermaid
flowchart TD
    D[Public, permitted source documents] --> J[Idempotent ingestion jobs]
    J --> P[Parse / normalize / chunk / metadata]
    P --> IX[(Document versions + chunks + embeddings + lexical index)]
    Q[Authenticated support question] --> A[Caller identity and authorization scope]
    A --> S[search(query, filters, k, caller_context)]
    IX --> S
    S --> H[Authorized dense + lexical retrieval / fusion]
    H --> C[Same candidate passages]
    C --> R1[Version 1: pretrained cross-encoder]
    C --> R2[Version 2: fine-tuned cross-encoder]
    TD[Labeled train/validation query–passage pairs] --> FT[Fine-tune + export checkpoint]
    FT --> R2
    R1 --> CB[Shared context builder]
    R2 --> CB
    CB --> LLM[Same hosted LLM, prompt and decoding settings]
    LLM --> OUT[Grounded answer / cited evidence or insufficient evidence]
    OUT --> EV[Quality + latency + cost evaluation]
```

The branches represent **experiment configurations**, not two simultaneously answering pipelines. Run the same questions through each configuration separately (or an explicitly controlled A/B harness). Query access rules must be enforced **before returning candidate passages**, independent of what the LLM says.

### Future-compatible retrieval contract

Conceptual interface (not prescribed implementation syntax):

```python
search(
    query: str,
    filters: SearchFilters,
    k: int,
    caller_context: VerifiedCallerContext,
) -> list[RetrievedPassage]
```

Every `RetrievedPassage` should carry a stable `document_id`, `document_version_id`, `section_or_span_reference`, current `chunk_id`, text, retrieval provenance, and citation metadata. **`caller_context` must come from verified server-side authentication**, not a client-provided tenant ID. This retrieval boundary can become a tool in Month 3 without redesigning the index.

---

## 4. Data Plan — Do This Before Writing Retrieval Code

**Choose one coherent business domain:** customer support, billing, subscriptions and refunds. Use real, publicly available, lawfully reusable support documents and PDFs. Record URL, publisher, date accessed, permitted usage, and document type. Avoid treating unrelated companies' policies as if they were one company's authoritative rules; keep sources and organizations separate. Do not ingest real client/customer PII.

### Minimum useful corpus

- Start with approximately **30–100 heterogeneous documents** (HTML help pages, text, and PDFs, including some tables and messy layouts). Document count is a practical starting range, not an accuracy target.
- Include policy documents with version dates and deliberately **create test update/delete/conflict scenarios** without misrepresenting synthetic modifications as real company policy.
- Create **two isolated test tenants** to test the security boundary. Real documents may be duplicated or partitioned for these controlled tests; use clear labels for synthetic fixtures.
- Document known parsing problems such as tables, headers, footers, OCR-only pages, and broken extraction. OCR can be a documented unsupported case for v1 rather than a whole new project.

### Evaluation dataset — create before retrieval experiments

Create **50–80 manually reviewed, representative questions** initially. Include direct matches, paraphrases, exact identifiers, multi-section answers, temporal/version conflicts, no-answer questions, and unauthorized cross-tenant probes.

**Store gold evidence at document + version + section/span level, not only at chunk ID**, because chunk IDs will change when chunking changes. Resolve the expected span into currently indexed chunks during each evaluation run.

Suggested dataset columns:

```text
question_id, query_text, tenant_id, question_type,
expected_answer_or_no_answer, gold_document_id,
gold_document_version_id, gold_section_or_span,
allowed_source_scope, annotation_notes, reviewed_by
```

A question may have **multiple** gold evidence spans. Freeze the test queries and gold labels before tuning. Keep separate **training**, **validation**, and **held-out test** partitions; avoid identical or near-duplicate policies, question templates, and document versions leaking across partitions. For fine-tuning, maintain an independently labeled **query–passage pair dataset** (Section 7).

### Metrics and reporting

| Layer | Required measurements | Why |
|---|---|---|
| Candidate retrieval | Recall@5/10/20 (with span-to-chunk mapping) | Was the evidence found? |
| Ranked retrieval | MRR@10, NDCG@5/10 | Was relevant evidence ranked high? |
| Grounded answer | Supported-answer rate, insufficient-evidence handling, citation correctness | Did the answer stay faithful? |
| Operations | p50/p95 latency for retrieval, reranking and end-to-end; failure rate | Is service behavior usable? |
| Cost | Embedding/LLM usage and estimated cost per request; training cost separately | Is the change worthwhile? |
| Security | Unauthorized passage exposures (must be zero in defined test cases) | Does isolation hold? |

**Important:** Embedding, cross-encoder and LLM scores are **not** automatically calibrated probabilities. Do not label them “confidence of truth.” If using an LLM judge for groundedness, first hand-label ~30–50 judgments and check agreement with the judge; also inspect disagreements. Keep human-reviewed examples in the final report.

---

## 5. Version 1 — Production RAG With Pretrained Models

### Phase 1 — Data, evaluation and architecture (Week 1)

**Build**

1. Confirm current API, configuration, DB schema, tests and dependency versions.
2. Select and document real source corpus and permissions.
3. Create 50–80 gold questions with span-level evidence labels and fixed partitions.
4. Define document/version/chunk/citation data contracts, and choose model versions.
5. Record **initial stack decisions** in `docs/` and explain alternatives.

**Acceptance / exit checks**

- Corpus manifest is version controlled or reproducibly downloadable; no secrets or restricted content are committed.
- Questions include relevant, no-answer, versioned and forbidden-tenant cases.
- Gold answers reference source spans rather than mutable chunk numbers.
- Existing Month 1 tests still pass.

### Phase 2 — Document lifecycle and ingestion (Week 2)

**Build**

1. Parser and normalization for supported formats; preserve page/section references.
2. Deterministic chunking with configurable size/overlap; attach source metadata.
3. Stable `document_id`; distinct `version_id`, content hash and deterministic chunk identifiers within a version.
4. Ingestion job states (`pending`, `processing`, `completed`, `failed`), controlled retries and failure reporting.
5. Idempotent upload/re-index and delete behavior; atomically switch active version when a replacement index is ready.

**Acceptance / exit checks**

- Same document uploaded twice does not create duplicate active chunks.
- Updating a policy changes the active searchable version; deleting it removes it from retrieval (according to documented retention policy).
- Failed parse/index reports a safe error and does not partially expose an incomplete document.
- PDF citations can identify meaningful pages/sections; known unsupported layouts are documented.

### Phase 3 — Dense retrieval baseline (Week 3)

**Build**

1. Pin one pretrained embedding model, dimensions and normalization strategy.
2. Store source/embedding model ID and version with indexed vectors; implement version-compatible re-embedding strategy.
3. Install/verify pgvector and database migrations; create exact vector search baseline first.
4. Implement authorization-aware `search(...)` returning passages and metadata.
5. Capture baseline Recall@K, MRR/NDCG, retrieval latency and failures.

**Acceptance / exit checks**

- Correct evidence is returned for known simple questions on the frozen evaluation set.
- All returned passages are within verified caller scope.
- Results are reproducible for a pinned corpus/index/model snapshot.
- The code does not silently combine incompatible embedding dimensions/models.

### Phase 4 — Lexical retrieval and hybrid fusion (Week 4)

**Build**

1. **Choose BM25 deliberately.** PostgreSQL full-text `ts_rank` is **not BM25**. Options: a lightweight Python BM25 implementation for this bounded corpus, a compatible ParadeDB BM25 extension, or an external search backend if justified. Document trade-offs.
2. Implement and test lexical retrieval; log index and scorer configuration.
3. Combine dense + lexical ranks using Reciprocal Rank Fusion (RRF) or an explicitly evaluated alternative.
4. Apply caller authorization and metadata filters correctly in **both** retrieval channels before content is returned.
5. Benchmark dense-only vs lexical-only vs hybrid.

**Acceptance / exit checks**

- README does not incorrectly call ordinary PostgreSQL FTS “BM25.”
- Hybrid evaluation uses the **same** frozen dataset and relevance labels as dense-only.
- All retrieved passages pass tenant ACL checks, including lexical matches.
- If adding HNSW approximate search, test filtered top-k recall: filtering after ANN scans can yield fewer than `k`; test pgvector iterative scans / index tuning or use exact filtered search for small data. Security checks must never rely on approximate filter behavior.

### Phase 5 — Pretrained reranker and context builder (Week 5)

**Build**

1. Select one small pretrained cross-encoder reranker and pin revision.
2. Re-rank a bounded number of authorized hybrid candidates; keep candidate IDs and scores traceable.
3. Build context with evidence IDs, section/version metadata, token budgeting and overlap reduction.
4. Log which evidence was supplied to the answer model.
5. Compare hybrid without rerank vs hybrid + pretrained rerank (quality **and** added latency).

**Acceptance / exit checks**

- Benchmark shows measurable gains, regressions, or no change; no automatic claim that reranking helps.
- No inaccessible or inactive-version content reaches the context builder.
- Version 1 reranker can be selected through a **trusted configuration** without changing API contract.

### Phase 6 — Answer API, grounding and citations (Week 6)

**Build**

1. Add a RAG answer endpoint using existing FastAPI, Pydantic and async provider patterns.
2. Instruct the LLM to answer **only from supplied authorized evidence**; separate document text from instructions.
3. Return cited `document_id`, version, section/span and source title/URL; reject/flag citations not traceable to supplied passages.
4. Explicitly return `insufficient_evidence` (or an equivalent contract) when evidence is inadequate; do not fabricate policy or live customer status.
5. Preserve source and token-level usage metadata for evaluation; start with a **non-streaming** answer to validate grounding.
6. If streaming is added, define a clear finalization/validation contract: streamed tokens may be provisional; validate final citation references and mark failures rather than implying invalid citations were never shown.

**Acceptance / exit checks**

- Tests cover answerable, unanswerable, conflicting-version and adversarial-document cases.
- Every returned citation maps to an **authorized**, actually supplied source passage (citation existence alone is not proof of claim support).
- Structured API errors safely handle provider outages, timeouts and invalid generated output.

### Phase 7 — Security, deployment and Version 1 release (Weeks 7–8)

**Build**

1. Establish a server-verified identity / tenant context. Do not accept arbitrary `tenant_id` headers or filters as authorization proof.
2. Test cross-tenant and document-ACL isolation before and after retrieval; enforce permissions during ingestion, search and citation access.
3. Add prompt-injection tests with instructions embedded in documents; treat retrieved text as untrusted **data**.
4. Add ingestion limits, basic rate limits, safe logs, request IDs, p95 latency and token/cost accounting.
5. Apply repeatable DB migrations; containerize/deploy to an appropriate environment (not public without real authentication).
6. Run frozen evaluation and save reproducible Version 1 report, versions and failure examples.

**Acceptance / exit checks**

- Defined cross-tenant tests return **zero unauthorized passages**; an injected “ignore previous instructions” policy does not become an instruction.
- Upload/update/delete and job retry behavior is tested; errors do not expose secrets or confidential raw data.
- API runs in Docker and in a deployed test environment, with meaningful logs.
- Baseline artifacts are frozen: corpus version, split IDs, chunking, embeddings, lexical implementation, candidates, reranker revision, prompts, metrics, latency and cost.

### Version 1 definition of done

- [ ] Repeatable ingestion with versioning, deduplication, update/delete and failed-job handling.
- [ ] Dense + **genuine BM25** search, hybrid fusion, metadata filters, and ACL safety.
- [ ] Pretrained reranker plus controllable/traceable context construction.
- [ ] Grounded answer with real source citations and insufficient-evidence responses.
- [ ] Representative frozen dataset; retrieval + answer evaluations; error analysis.
- [ ] Integration/security tests, Docker/deployed test environment, latency/cost logs.
- [ ] Existing `/analyze` behavior remains intact.

---

## 6. Version 2 — Fine-Tune a Small Reranker (Not the Whole RAG)

### Experiment hypothesis

> “On held-out questions about billing, subscription and refund policies, a domain-adapted small cross-encoder reranker may improve passage ranking relative to the same pretrained checkpoint, with an acceptable inference-latency and maintenance cost.”

This is a **hypothesis to test** — not a promised improvement. The whole pipeline stays identical except the reranker weights.

### Phase 8 — Curated supervised data (Week 9)

**Build**

1. Choose the **same architecture/base checkpoint** used for Version 1 (example family: small MS MARCO-tuned MiniLM cross-encoder; confirm license, compatibility and actual revision).
2. Build manually reviewed `query, passage, relevance_label` pairs from the approved corpus.
3. Use positive passages and **hard negatives** (plausible but wrong retrieved passages); include versions, near-duplicates, exact-name distractors and no-answer queries.
4. Split by meaningful grouping (document families, duplicated/versioned content, question patterns) to minimize leakage; do **not** train on the frozen held-out RAG test questions or gold answers.
5. Document labeling guidelines, ambiguous cases and class balance; use a separate validation set for checkpoint selection.

**Acceptance / exit checks**

- An audited training set exists, with reproducible splits and no known eval leakage.
- Every positive/negative pair is traceable to legitimate source evidence; no fabricated “ground truth” treated as real policy.
- Baseline pretrained model was measured on the exact same held-out ranking cases before training.

### Phase 9 — Fine-tuning and model artifact (Week 10)

**Build**

1. Run a small supervised cross-encoder fine-tuning experiment with Sentence Transformers + PyTorch, using a suitable ranking/classification loss for the label format.
2. Establish one controlled configuration: random seed, learning rate, batch size, epochs, max sequence length, optimizer, loss and model/base revision.
3. Record validation metrics, learning curves and training time; use early stopping or the best validation checkpoint.
4. Save model weights, tokenizer, config, dependency versions and training-data manifest; do **not** commit large binaries or private data without a suitable artifact store.
5. Support local CPU and, where compatible, Apple Silicon MPS; do not assume all training libraries run identically on each backend.

**Acceptance / exit checks**

- Training run is reproducible with a script/config and pinned base checkpoint.
- A persisted checkpoint reloads and produces deterministic eval-mode rankings for a fixed test fixture.
- No train/validation leakage or silent preprocessing mismatch between training and serving.

### Phase 10 — Swap, compare and release decision (Week 11; Week 12 buffer)

**Build**

1. Add a minimal model-selector at the existing reranker boundary (`pretrained` vs `fine_tuned`) using **trusted server configuration**.
2. Reuse the **identical retrieval candidate lists** on the same snapshot and evaluation queries to isolate reranker effects.
3. Evaluate original vs fine-tuned reranker on **held-out** NDCG@5/10 and MRR@10; then evaluate full answers for groundedness and citation correctness.
4. Compare p50/p95 reranking and total answer latency, model size, training cost, serving cost and maintenance complexity.
5. Inspect qualitative errors: when each model ranks stale/conflicting/irrelevant passages incorrectly.
6. Deploy the fine-tuned variant in a controlled environment only if justified; retain pretrained rollback.

**Acceptance / exit checks**

- Both variants pass identical authorization, citation and failure-path tests.
- Comparison and configs can be reproduced by another engineer.
- Conclusion explicitly states **improved / unchanged / worse**, with confidence limitations from the evaluation sample.
- Select production default based on measured quality/latency/cost, not on a requirement to use custom weights.

### Version 2 definition of done

- [ ] Clean labeled train/validation datasets and a fully isolated held-out test set.
- [ ] Reproducible fine-tuning script, configuration and versioned checkpoint.
- [ ] Minimal drop-in reranker integration; no duplicate ingestion/search/LLM stacks.
- [ ] Same-candidate, same-question comparison against Version 1.
- [ ] Latency, quality, cost, security and failure comparison with honest limitations.
- [ ] Documented decision and quick rollback to pretrained reranker.

---

## 7. Evaluation Matrix and Final Report

Start these ablations **before** fine-tuning; keep all reference data and system configurations identifiable.

| Configuration | Dense | BM25 | Hybrid/RRF | Reranker | Optional extras |
|---|---:|---:|---:|---|---|
| A — Dense baseline | ✓ | — | — | — | — |
| B — Lexical baseline | — | ✓ | — | — | — |
| C — Hybrid baseline | ✓ | ✓ | ✓ | — | — |
| D — **Version 1** | ✓ | ✓ | ✓ | **Pretrained** | — |
| E — **Version 2** | ✓ | ✓ | ✓ | **Fine-tuned** | — |
| F — Optional ablation | ✓ | ✓ | ✓ | Best measured | Contextual chunk enrichment / other justified experiment |

For **retrieval ablations**, run the frozen query/evidence dataset and log Recall@K, MRR@10, NDCG@5/10, candidate count and latency. For **D vs E**, hold the rerank candidate set fixed; Recall of the initial candidate set will therefore be unchanged, while top-k ranking/answer metrics **may** differ. Do not falsely attribute candidate-stage Recall gains to a reranker.

For **answer ablations**, keep the LLM/prompt/context budget as consistent as possible. Report groundedness, citation correctness, sufficient-evidence behavior, p95 end-to-end latency and token cost. Record model nondeterminism / sample sizes. Human-review a subset of difficult outputs; do not trust uncalibrated LLM-as-judge scores alone.

Final artifact: `docs/MONTH_2_RAG_EXPERIMENT_REPORT.md` (proposed) with actual numbers, architecture, dependency/model versions, plots or tables, retrieval and security failures, limitations, and why the selected configuration won.

**Example result table — populate with measured numbers; do not fabricate metrics:**

| Metric | Dense | Hybrid | Pretrained reranker | Fine-tuned reranker |
|---|---:|---:|---:|---:|
| Recall@10 (candidate retrieval) | TBD | TBD | Same hybrid candidates | Same hybrid candidates |
| MRR@10 | TBD | TBD | TBD | TBD |
| NDCG@5 | TBD | TBD | TBD | TBD |
| Citation correctness | N/A or TBD | N/A or TBD | TBD | TBD |
| Unsupported-answer rate | N/A or TBD | N/A or TBD | TBD | TBD |
| p95 end-to-end latency | TBD | TBD | TBD | TBD |
| Cost/query (excluding one-time training) | TBD | TBD | TBD | TBD |
| One-time fine-tuning cost | — | — | — | TBD |

---

## 8. Proposed API and Repository Layout (Not Yet Implemented)

**Proposed API behavior** — use existing patterns; final endpoints may differ after code review:

| Endpoint / operation | Purpose | Important contract |
|---|---|---|
| `POST /rag/documents` | Submit a document for ingestion | Authenticated, returns document/job ID; no synchronous heavy parsing |
| `GET /rag/jobs/{job_id}` | Inspect ingestion status | Scoped to owner/tenant; no raw internal exception leakage |
| `DELETE /rag/documents/{document_id}` | Retire/delete a document | Authorized; no stale searchable content |
| `POST /rag/search` | Return source passages | Verified caller scope; stable citation/source IDs |
| `POST /rag/answer` | Generate grounded answer | Source citations, insufficient-evidence status and trace ID |
| Evaluation script (not public) | Run offline benchmark matrix | Frozen dataset and config snapshots |

**Possible additions** (illustrative, not a requirement to create every file):

```text
app/
  services/
    rag_ingestion_service.py       # parsing, chunking, lifecycle
    rag_retrieval_service.py       # search contract, dense/BM25 fusion
    rag_reranker.py                # pretrained / fine-tuned strategy
    rag_answer_service.py          # context and grounded LLM answers
  repositories/
    rag_repository.py              # index queries and persistence
  # extend existing models, schemas, routes and config as needed
scripts/
  evaluate_rag.py                  # offline retrieval/answer evaluations
  train_reranker.py                # version 2 supervised fine-tuning
  compare_rag_versions.py         # controlled ablation runner
tests/
  test_rag_ingestion.py
  test_rag_retrieval.py
  test_rag_security.py
  test_rag_answers.py
  test_rag_model_swap.py
  test_rag_evaluation.py
docs/
  Month_2_Production_RAG_Pretrained_and_FineTuned_Implementation_Roadmap.md
  MONTH_2_RAG_EXPERIMENT_REPORT.md  # produced during implementation
```

Add a migrations directory/tool only when selecting the schema-change approach. Do not introduce a generic plugin architecture, a second backend, or a new orchestration framework solely because this list mentions modules.

---

## 9. Key Engineering Decisions to Record

| Decision | Initial default / constraint | What to verify |
|---|---|---|
| Embeddings | One pretrained embedding model, revision pinned | Licensing, dimensions, price, recall, update cost |
| Vectors | PostgreSQL + pgvector | Extension/image compatibility, migrations, tenant-filter recall |
| Lexical retrieval | **Real BM25** (small-corpus Python initially, or supported DB/search extension) | Corpus scale, operational complexity, tokenizer behavior |
| ANN | Start with exact vector search; enable HNSW only when measured need exists | Recall with filters, iterative scans, latency trade-offs |
| Fusion | RRF baseline | Improvement versus best single channel |
| Reranker | One small cross-encoder, pretrained and later fine-tuned | Ranking metrics, hardware performance, serving contract |
| LLM | Existing hosted provider integration and shared prompt | Grounding, citations, cost, safe failures |
| Index lifecycle | Stable source IDs, hash/version, active index switch | Dedup, update/delete, rollback, incompatible embedding revisions |
| Ingestion queue | Simple durable job mechanism appropriate to this project | Restart recovery, backpressure, bounded retries; no Kafka yet |
| Authorization | Verified server-side identity and ACL-aware search | Cross-tenant leakage and metadata/citation exposure |
| Model selection | Internal config for baseline and experimental reranker | Safe rollback and no client ability to bypass security |
| Streaming | Start non-streaming; add SSE only after answer/citation contract works | Provisional tokens and final citation validation |

**Security design requirement:** Tenant filtering is simultaneously a correctness concern and a privacy boundary. Make authorization explicit at storage/search and source-read boundaries. The approximate index's post-filter behavior can affect **recall**, but must **never** be used as a substitute for access control.

---

## 10. Weekly Timebox and Milestones

| Week | Focus | Review checkpoint |
|---|---|---|
| 1 | Corpus, evaluation labels, contracts | **Dataset/eval frozen** |
| 2 | Ingestion, stable IDs, versions, jobs | Deterministic document lifecycle |
| 3 | Embedding + pgvector dense baseline | Dense retrieval measured |
| 4 | Genuine BM25 + hybrid fusion | Dense/lexical/hybrid comparison |
| 5 | Pretrained reranker + context | Reranker ablation completed |
| 6 | Grounded answer API + citations | Valid answers and honest abstentions |
| 7 | Auth/ACL, adversarial/security tests, reliability | Tenant isolation and lifecycle tests pass |
| 8 | Docker deployment, logging, frozen evaluation | **Version 1 shipped and benchmarked** |
| 9 | Human-reviewed training pairs + split audit | Fine-tuning dataset ready |
| 10 | PyTorch cross-encoder fine-tuning | Reloadable custom model checkpoint |
| 11 | Fair A/B evaluation, deployment choice, report | **Version 2 assessed; decision documented** |
| 12 | Contingency: fix failures or improve documentation | No new speculative scope |

At ~10 hours/week, this is a **capacity-based estimate, not a deadline guarantee**. Protect the v1 acceptance criteria before investing extra time in training. If Week 8 is late, extend the schedule rather than silently removing evaluation or security.

---

## 11. Working Method With Codex + Human Review

Apply the repository's existing `ai_workflows/codex_implementation_workflow.md` and `ai_workflows/claude_implementation_review_workflow.md` throughout.

For **each phase**:

1. **Inspect:** Codex identifies what exists and which pieces to reuse, including tests, schema and routes.
2. **Understand:** The engineer explains the input/output contracts, why the method was chosen, and its alternatives before accepting code.
3. **Experiment:** Implement the minimal isolated retrieval/training change.
4. **Integrate:** Preserve Month 1 behavior and avoid duplicate modules.
5. **Test + measure:** Run relevant unit, integration, security and offline evaluation cases; report actual numbers.
6. **Review:** Independent code/architecture review; resolve blockers/high findings; note limitations.
7. **Document:** Record model/data versions, decisions, failure cases, rollback and the next phase.

**Personal learning checkpoint:** Before marking phases 3–5 and 8–10 complete, be able to explain embeddings, hybrid ranking, cross-encoder scoring, the training labels/loss, train-versus-validation split, and why measured metrics changed — without relying on generated code to answer for you.

---

## 12. End-of-Month Portfolio Evidence

The finished repository should show:

- A real-document, reproducible, permission-respecting ingestion and retrieval system.
- A working deployed knowledge-answering API with appropriate citations and honest no-answer behavior.
- A stable **pretrained baseline** and an optional **fine-tuned reranker** behind the same component contract.
- A versioned evaluation dataset with document/span labels and leakage-aware train/validation/test splits.
- A comparison of **dense, lexical, hybrid, pretrained reranked, and fine-tuned reranked** retrieval.
- Measured accuracy, groundedness, citation correctness, latency and request cost; training cost separately.
- Tenant-isolation, prompt-injection, versioning and deletion tests.
- An engineering report with failures, trade-offs, and an explicit model-selection conclusion.

**Success statement:** “I can build, deploy, secure and evaluate a document-grounded RAG system; fine-tune a reranker on domain data; and prove whether that fine-tuning was worth using.”

---

## 13. Mapping to the Original Six-Month Roadmap

**Directly from the original Month 2 outline:** ingestion/parsing/chunking/metadata; stable IDs, deduplication, versioning and re-indexing; embeddings, pgvector, BM25 and hybrid retrieval; query processing; reranking; context construction; grounded citations and insufficient-evidence handling; Recall/Precision/MRR/NDCG and groundedness evaluation; ingestion/retrieval/answer/admin/evaluation APIs; optional mobile streaming/history.

**Added or tightened by this implementation roadmap:** real-source corpus and evidence-span labels; acceptance criteria; explicit genuine-BM25 selection; filtered pgvector ANN tests; tenant/ACL enforcement; ingestion-job reliability; embedding/version migration; calibrated judging; explicit citation validation and streaming semantics; per-request latency/cost tracking; tool-ready search boundary; **Version 2 cross-encoder fine-tuning**; fair controlled ablation and rollback.

**Deliberately deferred:** default multi-query/decomposition, contextual retrieval and additional chunking experiments until evaluation shows value; managed-RAG provider comparison; UI-heavy admin and mobile work; and Month 3 agent/tool integrations.

> **Final principle:** Version 1 is not “incomplete RAG.” It is the deployable baseline. Version 2 is a controlled machine-learning experiment layered on it. Keep whichever model the evidence supports.
