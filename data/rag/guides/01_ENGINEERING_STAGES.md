# Practical path: from source documents to two production RAG variants

### Stage 1. Inspect source fidelity

**Build:** Select one PDF and HTML source; inspect headings/tables/pages; preserve anchors.

**Verify:** Compare extracted text and table semantics against source visual; record errors.

### Stage 2. Define evidence dataset

**Build:** Review 15 seed questions; mark doc/version/section or explicit unanswerable.

**Verify:** Every direct answer supported by quoted source text; no training/eval leakage.

### Stage 3. Ingestion and chunking

**Build:** Implement parsers, normalization, chunk metadata and deterministic hashes.

**Verify:** Chunk boundaries sensible; no duplicate current chunks on repeat ingestion.

### Stage 4. Dense retrieval

**Build:** Pin embedding model, create embeddings for active authorized docs.

**Verify:** Recall@K on frozen questions; no cross-tenant passages.

### Stage 5. BM25 and hybrid

**Build:** Implement genuine BM25 and RRF.

**Verify:** Run dense-only, BM25-only, hybrid ablation on same query set.

### Stage 6. Pretrained cross-encoder

**Build:** Rerank top candidate passages; preserve section/citation metadata.

**Verify:** MRR and NDCG measured; latency measured; failure cases recorded.

### Stage 7. Grounded answers

**Build:** Add controlled LLM prompting + citation validation + abstain path.

**Verify:** No made-up live account state; every cited ID maps to authorized source.

### Stage 8. Document lifecycle

**Build:** Handle active/archived versions, delete and re-index jobs.

**Verify:** Current-policy questions cannot quote archived rules.

### Stage 9. Fine-tune reranker

**Build:** Label training pairs, isolate test docs, train modest PyTorch cross-encoder.

**Verify:** Checkpoint reloads; labels reviewed; eval split not reused.

### Stage 10. Compare and deploy

**Build:** Replay identical candidates with base and fine-tuned rerankers.

**Verify:** NDCG/groundedness/citations/p95 latency/cost + failure analysis; keep best.
