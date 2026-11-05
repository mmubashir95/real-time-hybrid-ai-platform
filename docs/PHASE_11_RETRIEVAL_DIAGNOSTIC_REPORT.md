# Phase 11 Retrieval Diagnostic Report

## A. Executive Summary

Configuration C's reported scores are mathematically consistent with the current evaluator, but they do not cleanly measure whether an answer-bearing passage was retrieved. The strongest confirmed cause of the very low Top-5 precision is an incomplete relevance-label design: a result is relevant only when tenant, document, version, and section key exactly equal a gold anchor. The corpus intentionally repeats the same controlling fact in `SCOPE`, `DECISION`, `MATRIX`, `PROCEDURE`, and `FAQ`. Exact or substantively equivalent passages outside the one labeled section are therefore counted as irrelevant.

Three direct questions demonstrate the impact:

- `P11-CAN-001`: the exact answer is in the rank-1 `FAQ` chunk, but only the rank-3 `DECISION` chunk is gold.
- `P11-ACCESS-001`: the exact answer is in the rank-3 `FAQ` chunk, but only the rank-12 `DECISION` chunk is gold.
- `P11-BKH-001`: the exact answer is in the rank-1 `FAQ` chunk, but only the rank-7 `DECISION` chunk is gold.

There is also a real ranking weakness under the current labels. BM25 alone places three of four answerable questions' labeled evidence in the top five. RRF moves the ownership and receipt labels from BM25 ranks 4 and 5 to hybrid ranks 12 and 7. Conversely, RRF repairs Top-20 coverage for the multi-evidence question by combining evidence found separately by the two channels. Configuration C is therefore not uniformly poor: it improves candidate coverage but worsens early precision for some questions.

The subsequent controlled Configuration D run shows that Qwen3 repairs much of this early-ranking weakness: Recall@5 increases from 0.25 to 0.75 and all labeled evidence is present by rank 10. The trade-off in the local environment is substantial latency: p50 increases from 35.12 ms to 1,025.77 ms.

All seven dataset records remain `needs_review`. No label was changed or approved during this investigation.

## B. Original Run

| Field | Value |
|---|---|
| Run ID | `2745324e-6a82-4a2c-86d1-d57c6cb7a9bd` |
| Dataset | `month2-rag-golden-candidates` v1.0.0 |
| Dataset hash | `405380bdc107fb8acc6ddda6320a94ca695d6e4e63a9f609eb58ad4e116828a5` |
| Split | development |
| Configuration | C, hybrid BM25 + dense + RRF |
| Corpus snapshot | `9c2ce177ccf5705716dcd91c8c5cd7b165702a5ef5ac7ab76e920c86fdc1a2fb` |
| Index version | `000,001,002,003` |
| Embedding | `BAAI/bge-small-en-v1.5@5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` |
| Embedding fingerprint | `2945caacf194aff9aa2d9a26da0a058c98a61a96ee98881e36e3e2301c327d37` |
| Candidate limit | 20 per channel and 20 after fusion |
| Questions / failures | 5 / 0 |

Original metrics:

| P@5 | P@10 | R@5 | R@10 | R@20 | Complete@5 | MRR@10 | NDCG@5 | NDCG@10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0400 | 0.0800 | 0.2500 | 0.7500 | 1.0000 | 0.2500 | 0.1607 | 0.1250 | 0.3091 |

These are exploratory results from five questions and unapproved labels. The original artifact does not persist passage text or separate BM25/dense ranks. Passage previews and channel ranks below were reconstructed read-only from the same pinned local database and controlled A/B runs.

## C. Dataset Integrity

### Question inventory

| ID | Question/type | Scope and expected behavior | Gold mapping | Review |
|---|---|---|---|---|
| `P11-CAN-001` | Cancel monthly workspace; direct | AsterDesk; answer with renewal boundary and paid-through access | AST-CAN-003 v2.0, `DECISION`; one gold unit | `needs_review` |
| `P11-ACCESS-001` | Admin ownership transfer; paraphrase | AsterDesk; answer no without owner approval | AST-ACCESS-016 v2.0, `DECISION`; one gold unit | `needs_review` |
| `P11-MULTI-001` | Identifiers for cancellation and role change; multi-evidence | AsterDesk; answer using both workflows | AST-CAN-003 v2.0 `DECISION` + AST-ACCESS-016 v2.0 `DECISION`; two gold units | `needs_review` |
| `P11-BKH-001` | App-store receipt; direct | Brookhaven; answer that the store issues it | BKH-BILL-103 v2.0, `DECISION`; one gold unit | `needs_review` |
| `P11-NOANSWER-001` | Guaranteed refund time; unanswerable | AsterDesk; abstain | no gold units | `needs_review` |

All questions are distinct. The four answerable questions' source references, versions, sections, and quoted text resolve. Current-policy questions search only active v2.0 documents. The historical and authorization probes are in the held-out split and were not part of this run. Tenant filtering is applied before BM25 materialization and in dense SQL; no cross-tenant result appears in the five original result lists.

### Pending annotation issues

1. **Equivalent answer passages are excluded.** The exact controlling sentence is repeated in `FAQ` for all three direct questions, and repeated elsewhere in `SCOPE`, `MATRIX`, and `PROCEDURE`. The current gold set labels only `DECISION`.
2. **Multi-evidence equivalents are excluded.** The same two evidence checklists appear in each document's `AUDIT` section. The cancellation `AUDIT` chunk is hybrid rank 14 and the role `AUDIT` chunk is rank 8, but both are marked irrelevant.
3. **Gold identity is section-level, not quote/span-level.** `scripts/evaluate_rag.py::_section_key` and the gold dictionary compare `(tenant, document, version, section)`. The immutable `source_quote` is validated when loading the dataset but is not used when matching a retrieved chunk. A later chunk split could grant credit to the wrong chunk in a gold section or deny an equivalent span in another section.
4. **No-answer status is plausible but unapproved.** A corpus search found no universal guaranteed refund processing time. The corpus instead contains explicit unsupported-guarantee handling. Human review is still required.

Smallest proposed annotation correction, pending human approval: represent each required fact as an evidence unit with an approved set of equivalent source spans/chunks, rather than adding whole duplicate sections indiscriminately. This avoids counting a generally related section as relevant while allowing the exact `FAQ` answer to receive credit.

## D. Independent Metric Verification

The verification below was calculated directly from persisted ranks and gold flags, without calling `evaluate_ranking`.

| Question | Gold units | Gold ranks in C | Relevant 1-5 | 6-10 | 11-20 | R@5 | R@10 | R@20 | RR@10 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| CAN | 1 | 3 | 1 | 0 | 0 | 1 | 1 | 1 | 1/3 |
| ACCESS | 1 | 12 | 0 | 0 | 1 | 0 | 0 | 1 | 0 |
| MULTI | 2 | 6, 9 | 0 | 2 | 0 | 0 | 1 | 1 | 1/6 |
| BKH | 1 | 7 | 0 | 1 | 0 | 0 | 1 | 1 | 1/7 |
| NOANSWER | 0 | none | 0 | 0 | 0 | undefined | undefined | undefined | undefined |

Therefore:

- Precision@5 uses fixed denominator 5 and macro-averages all five completed questions: `(1/5 + 0 + 0 + 0 + 0) / 5 = 0.04`.
- Precision@10: `(1/10 + 0 + 2/10 + 1/10 + 0) / 5 = 0.08`.
- Recall excludes the no-answer question: R@5 = `(1+0+0+0)/4 = 0.25`; R@10 = `(1+0+1+1)/4 = 0.75`; R@20 = `4/4 = 1.0`.
- Complete Evidence Coverage uses the same four-question denominator and produces 0.25, 0.75, and 1.0.
- MRR@10 = `(1/3 + 0 + 1/6 + 1/7)/4 = 0.160714...`.
- NDCG uses gain `2^grade-1`, logarithmic discount, and an ideal ranking made from gold grades. Recalculation produces 0.1250 at 5 and 0.309079 at 10.

The implementation handles zero gold, one/multiple gold units, duplicate anchors, fewer than K results, and graded relevance consistently with its documented conventions. Tests cover those cases. Incorrect document versions cannot match because version is part of the evidence key. Missing gold references are rejected by dataset loading. No arithmetic defect was reproduced.

Methodological caution: precision includes the no-answer probe as zero while recall/MRR/NDCG exclude it. That convention is explicit and internally consistent, but reports should publish the denominator and preferably separate answerable retrieval precision from no-answer retrieval behavior.

## E. Per-Question Diagnostics

Ranks marked `FAQ equivalent` contain the direct answer but are not gold under the current annotation.

### `P11-CAN-001`

- First labeled result: 3. Top 5/10/20: yes/yes/yes. Complete gold found.
- Rank 1 preview (`FAQ equivalent`): “If I cancel today ... Self-serve monthly cancellation disables renewal ... leaves paid features accessible through the paid-through timestamp.”
- Gold preview (rank 3, `DECISION`): “Self-serve monthly cancellation disables renewal at the next billing boundary and leaves paid features accessible through the paid-through timestamp.”
- Channel/fusion: FAQ B1/D2/C1; SCOPE B3/D1/C2; gold DECISION B2/D3/C3. Fusion is behaving as designed and keeps all precise variants high.
- Conclusion: reported P@5 undercounts answer-bearing evidence; no material retrieval failure for this question.

### `P11-ACCESS-001`

- First labeled result: 12. Top 5/10/20: no/no/yes. Complete gold only at 20.
- Rank 3 preview (`FAQ equivalent`): “administrators manage most membership settings but cannot transfer ownership without owner approval.”
- Gold preview (rank 12, `DECISION`): the same controlling sentence.
- Irrelevant passages about owner changes outrank the label because the query strongly overlaps AST-OWNER-026. The equivalent AST-ACCESS FAQ is nevertheless rank 3.
- Channel/fusion: gold DECISION B4, absent from dense top 20, C12. RRF demotes a strong BM25-only label because it receives only one contribution. The exact FAQ answer is B5/D4/C3.
- Conclusion: confirmed label/matching undercount plus a real RRF weakness relative to the single-section label.

### `P11-MULTI-001`

- First labeled result: 6; second: 9. Top 5/10/20 complete evidence: no/yes/yes.
- Gold previews: cancellation requires workspace ID, cancellation actor, authenticated session, contract type, paid-through date; role change requires actor subject ID, role, workspace ID, authorization decision, audit event.
- Equivalent `AUDIT` previews occur at C8 (role checklist) and C14 (cancellation checklist) but are not gold.
- Channel/fusion: dense ranks the two labels D2 and D9; BM25 finds only role DECISION at B17 in its top 20; hybrid combines them at C6 and C9.
- Conclusion: hybrid materially improves coverage over BM25, but five broad FAQ/procedure passages precede the first label. This is ranking weakness and incomplete equivalent-span labeling, not a candidate-limit failure at 20.

### `P11-BKH-001`

- First labeled result: 7. Top 5/10/20: no/yes/yes.
- Rank 1 preview (`FAQ equivalent`): “app-store receipts are issued by the store and may not appear as Brookhaven web invoices.”
- Gold preview (rank 7, `DECISION`): the same sentence.
- Channel/fusion: exact FAQ B1/D1/C1; gold DECISION B5/D10/C7. RRF promotes candidates present in both channels and slightly demotes the labeled section.
- Conclusion: the user-answering passage is already rank 1; the low label-aware score is primarily an annotation/matching artifact.

### `P11-NOANSWER-001`

- No gold evidence; retrieval relevance metrics are undefined except precision, which is zero.
- Top results discuss refund procedures, unsupported guarantees, and exception handling, but no retrieved passage establishes a guaranteed processing time for every refund.
- Conclusion: abstention appears appropriate, pending human corpus review. The retrieval report alone does not measure whether the downstream system abstained.

### Original C top-20 provenance

The compact notation is `C rank: document/section [BM25 rank, dense rank]`; `—` means outside that channel's top 20. Asterisks are current gold labels.

- CAN: `1 CAN/FAQ [1,2]`, `2 CAN/SCOPE [3,1]`, `3 CAN/DECISION* [2,3]`, `4 CAN/CASES [4,6]`, `5 COUPON/SCOPE [9,4]`, `6 RENEW/DECISION [13,7]`, `7 CAN/DOC-CONTROL [11,9]`, `8 CAN/PROCEDURE [6,18]`, `9 RENEW/SCOPE [16,8]`, `10 RENEW/FAQ [17,10]`, `11 CAN/MATRIX [5,—]`, `12 TRIAL/SCOPE [—,5]`, `13-15 COUPON/CASES [7/8/10,—]`, `16 USAGE/FAQ [—,11]`, `17 COUPON/MATRIX [12,—]`, `18 USAGE/SCOPE [—,12]`, `19 OWNER/SCOPE [—,13]`, `20 CAN/CASES [14,—]`.
- ACCESS: `1 OWNER/SCOPE [3,3]`, `2 ACCESS/PROCEDURE [7,2]`, `3 ACCESS/FAQ [5,4]`, `4 OWNER/PROCEDURE [10,1]`, `5 OWNER/FAQ [1,13]`, `6 OWNER/DOC-CONTROL [8,7]`, `7 REGION/FAQ [9,10]`, `8-9 OWNER/CASES [13/14,15/17]`, `10 ACCESS/DOC-CONTROL [17,14]`, `11 ACCESS/SCOPE [2,—]`, `12 ACCESS/DECISION* [4,—]`, `13 ONBOARD/PROCEDURE [—,5]`, `14 ACCESS/MATRIX [6,—]`, `15 REGION/PROCEDURE [—,6]`, `16 OWNER/EXCEPTIONS [—,8]`, `17 CAN/PROCEDURE [—,9]`, `18 OWNER/MATRIX [11,—]`, `19 EXPORT/PROCEDURE [—,11]`, `20 REGION/CASES [12,—]`.
- MULTI: `1 ACCESS/FAQ [1,1]`, `2 CAN/FAQ [4,7]`, `3 ACCESS/PROCEDURE [12,3]`, `4 ONBOARD/FAQ [11,4]`, `5 SEAT/FAQ [5,15]`, `6 ACCESS/DECISION* [17,9]`, `7 COUPON/FAQ [15,17]`, `8 ACCESS/AUDIT [2,—]`, `9 CAN/DECISION* [—,2]`, `10 ACCESS/CASES [3,—]`, `11 CAN/SCOPE [—,5]`, `12 SEAT/AUDIT [6,—]`, `13 COUPON/DECISION [—,6]`, `14 CAN/AUDIT [7,—]`, `15 RENEW/FAQ [8,—]`, `16 ACCESS/DOC-CONTROL [—,8]`, `17 PRIV/FAQ [9,—]`, `18 API/FAQ [10,—]`, `19 SEAT/DECISION [—,10]`, `20 CAN/PROCEDURE [—,11]`.
- BKH: `1 BILL/FAQ [1,1]`, `2 REF/FAQ [2,2]`, `3 BILL/SCOPE [6,4]`, `4 REF/DECISION [3,8]`, `5 REF/PROCEDURE [4,7]`, `6 BILL/PROCEDURE [12,3]`, `7 BILL/DECISION* [5,10]`, `8 REF/SCOPE [7,9]`, `9 APP/FAQ [16,5]`, `10 APP/DECISION [11,11]`, `11 APP/PROCEDURE [17,6]`, `12 APP/CASES [13,15]`, `13 APP/SCOPE [19,12]`, `14 REF/EXCEPTIONS [20,20]`, `15 BILL/MATRIX [8,—]`, `16 REF/CASES [9,—]`, `17 REF/MATRIX [10,—]`, `18 BILL/EXCEPTIONS [—,13]`, `19 REF/CASES [14,—]`, `20 PRIV/PROCEDURE [—,14]`.
- NOANSWER: all 20 are non-gold by definition. The top five are REF/PROCEDURE, FAQ/EXCEPTIONS, USAGE/EXCEPTIONS, API/EXCEPTIONS, and OWNER/EXCEPTIONS; their channel ranks are `[5,1]`, `[15,2]`, `[16,6]`, `[20,5]`, and `[17,19]`.

No independent human relevance judgments exist, so the report distinguishes current gold, exact/equivalent answer text, and unreviewed passages rather than inventing “human relevance.”

## F. Configuration Comparison

All measured runs use the same five questions, corpus hash, index versions, authorization rules, model fingerprint, and candidate limit.

| Config | P@5 | P@10 | R@5 | R@10 | R@20 | MRR@10 | NDCG@5 | NDCG@10 | Complete@5 | p50/p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A dense | 0.080 | 0.080 | 0.375 | 0.750 | 0.750 | 0.2333 | 0.2217 | 0.3401 | 0.250 | 12.37 / 49.10 |
| B BM25 | 0.120 | 0.060 | 0.750 | 0.750 | 0.875 | 0.2375 | 0.3621 | 0.3621 | 0.750 | 23.47 / 34.47 |
| C hybrid | 0.040 | 0.080 | 0.250 | 0.750 | 1.000 | 0.1607 | 0.1250 | 0.3091 | 0.250 | 35.12 / 205.44 |
| D Qwen3 | 0.160 | 0.100 | 0.750 | 1.000 | 1.000 | 0.3857 | 0.4234 | 0.5067 | 0.750 | 1025.77 / 2053.82 |

Per-question labeled ranks:

| Question | A dense | B BM25 | C hybrid |
|---|---|---|---|
| CAN | 3 | 2 | 3 |
| ACCESS | outside 20 | 4 | 12 |
| MULTI | 2, 9 | 17, second outside 20 | 6, 9 |
| BKH | 10 | 5 | 7 |

The successful D run is `86b8e42e-efc1-47d3-9473-61cb88efafc7`. It uses the identical dataset hash, corpus snapshot, index version, embedding identity, candidate limit, and development split as C. It completed all five questions without failure and preserved the hybrid candidate set before reranking.

## G. RRF Analysis

The implementation uses standard unweighted RRF, `1/(60 + rank)` per channel, deduplicates by chunk ID, and deterministically sorts by score, then BM25 rank, dense rank, and chunk ID. Recomputed scores match persisted C scores.

Confirmed behavior:

- Candidates present in both top-20 channels dominate single-channel candidates. This is why ACCESS's BM25-rank-4 gold label becomes C12: it is absent from dense top 20 and receives only `1/(60+4)=0.015625`.
- The ACCESS FAQ equivalent is present in both channels (B5/D4) and correctly rises to C3.
- The BKH FAQ equivalent is B1/D1 and correctly ranks C1; the label is B5/D10 and ranks C7.
- MULTI benefits from union coverage: one gold is effectively dense-led and the other weakly present in both channels, producing full C top-10 coverage.
- No duplicate chunk IDs were found in a channel or fused list. Candidate limit 20 truncates one BM25-only multi-evidence label but C recovers it from dense; it does not explain C's reported R@20.

Changing the RRF constant or weights is not justified until equivalent-span gold labels are reviewed. The current behavior is formula-correct.

## H. Chunking Analysis

The active chunker targets 400 tokens, caps at 500, uses 50-token overlap, preserves section heading paths and source block IDs, and does not cross logical section boundaries. The relevant `DECISION`, `FAQ`, and `AUDIT` passages are intact and not truncated.

No confirmed chunk-boundary failure affects these five questions. The important issue is corpus redundancy across sections, not overlap duplication: the same policy statement is intentionally authored multiple times in separate source sections. The evaluator collapses at section identity but does not model cross-section equivalence. Increasing chunk size would not solve that and could reduce precision by combining unrelated policy content.

## I. Qwen3 Analysis

Configuration D materially improves label-aware early ranking: Precision@5 rises from 0.04 to 0.16, Recall@5 and complete evidence coverage at 5 rise from 0.25 to 0.75, and Recall@10 reaches 1.0. MRR@10 rises from 0.1607 to 0.3857. This supports Qwen3 as an effective reranker for the fixed hybrid candidate set under the current labels.

The cost is local latency. D measures 1,025.77 ms p50 and 2,053.82 ms p95, compared with C's 35.12 ms p50 and 205.44 ms p95. These five sequential local samples are insufficient for a production latency decision, but the trade-off is large enough to require explicit performance acceptance and broader measurement.

## J. Root-Cause Summary

| Root cause | Affected questions | Evidence and metric impact | Confidence |
|---|---|---|---|
| `EVALUATION_LABEL_PROBLEM` | CAN, ACCESS, MULTI, BKH | Exact/equivalent FAQ/AUDIT passages are excluded; sharply depresses precision and early recall | Confirmed |
| `EVIDENCE_MATCHING_PROBLEM` | All answerable, future split chunks | Runner matches only section identity and ignores `source_quote`; cannot represent equivalent spans | Confirmed |
| `RRF_RANKING_PROBLEM` | ACCESS, BKH | Single-channel gold evidence falls from B4/B5 to C12/C7 | Confirmed relative to current labels |
| `DENSE_RETRIEVAL_PROBLEM` | ACCESS | Labeled DECISION absent from dense top 20, though equivalent FAQ is D4 | Confirmed label-aware; answer-quality impact low |
| `BM25_RETRIEVAL_PROBLEM` | MULTI | One of two labels outside BM25 top 20 | Confirmed |
| `QUERY_AMBIGUITY` | ACCESS | “ownership” strongly retrieves the separate ownership policy | Strongly suspected |
| `CHUNKING_PROBLEM` | none | Gold and equivalent passages are intact | Not supported |
| `AUTHORIZATION_OR_VERSION_FILTER` | none | All results have correct tenant and active version | Not supported |
| `CANDIDATE_LIMITATION` | MULTI/B only | BM25 misses one label at 20; C still has full Top-20 coverage | Confirmed but not primary for C |
| `RERANKER_LIMITATION` | all | D improves ranking but adds approximately one second median local latency | Confirmed performance trade-off |
| `METRIC_CALCULATION_PROBLEM` | none | Independent arithmetic matches artifact | Not found |

## K. Recommended Corrective Actions

1. Human-review the five development annotations. For each required fact, approve all genuinely equivalent immutable spans, especially the exact FAQ and AUDIT duplicates. Do not approve whole sections solely to improve scores.
2. Define an evidence-unit schema that can map one fact to multiple acceptable source spans/chunks. Preserve document/version/tenant constraints.
3. Add a focused matching regression test: an exact approved quote in an alternate section should receive credit only when explicitly listed as an accepted equivalent; a same-section chunk lacking the quote should not.
4. Recompute A/B/C after label review. Only then assess whether RRF needs a controlled weighting/constant experiment.
5. Measure D on a larger reviewed set and profile its latency/resource use while preserving the exact C candidate set.
6. Publish answerable-only and no-answer behavior separately, and always display sample counts.

## L. Test Results

Commands and results are listed in the final investigation response. No production evaluation or retrieval code was modified. Controlled A/B diagnostic artifacts were written only under `/tmp`; the original report was not overwritten.

## M. Limitations

- Five development questions are too few for stable conclusions.
- All labels are AI-produced candidates marked `needs_review`; no human relevance judgments exist.
- Equivalent-span observations are evidence-backed reviewer proposals, not approved labels.
- A and B are fresh controlled local reproductions; C uses the reported artifact. The pinned corpus/index/model identities match.
- D completed successfully, but its five local latency samples are not a production performance benchmark.
- Original C passage text and per-channel ranks were reconstructed from the unchanged pinned local environment because they were not persisted in `run.json`.
- Latency comparisons are local, sequential, and not a performance benchmark; the original C p95 is one maximum among five samples.

## N. Next Steps

Do not accept the current retrieval-quality numbers as a production baseline. First complete human label review and implement explicit equivalent-span matching. Then rerun A/B/C/D on the identical snapshot and a larger reviewed set. Retrieval optimization should be considered only if the reviewed labels still show the same rank failures; Qwen3's measured ranking gain must also be weighed against its local latency cost.
