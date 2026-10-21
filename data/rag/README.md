# Month 2 — Extended Production RAG Practice Corpus (v2)

> **100% synthetic / fictional.** All policies, persons, companies, transactional values and incidents are invented for education. **This is not a real-world benchmark** and must never be used for actual customer decisions.

## Why this replaces the first ZIP

The original collection contained numerous ~1 KB policies with almost no chunking challenge. This second package contains substantial multi-section documents, multi-page PDFs, semantic HTML tables, text/Markdown, near-duplicate policy revisions, conflicting thresholds, cross-tenant material, long case dossiers and 840 structured supporting records. **The content is more extensive, but still artificial and more regular than real enterprise documentation.**

## Contents (automatically measured)

- **79 long-form documents**, containing **172,785 extracted-source words** (count includes repeated policy references), across AsterDesk and Brookhaven.
- **23 PDF**, **19 HTML**, **20 Markdown**, **17 TXT**.
- **4 archived policy editions**, deliberately different from active revisions.
- **20 long case dossiers** with incident timelines, untrusted embedded notes and multiple policy issues.
- **365 synthetic question/answer/evidence anchors**: doc ID, version, section key (not chunk ID). Some are deliberately unanswerable or cross-tenant probes. **Seeds require human review, not independently validated reference labels.**
- **480 fake payment rows**, **360 fake support-ticket rows**, a policy registry and **104 training-pair suggestions** (not ready to train without review).

## Dataset layout

```text
documents/asterdesk/                  Primary SaaS company policies and casebooks
documents/brookhaven/                 Independent streaming company; isolation tests
structured_data/                     Synthetic ledgers and tickets (NOT policy ground truth)
evaluation/question_bank_*.csv       Gold-candidate questions, doc + section anchors
fine_tuning/seed_training_pairs_*.csv Candidate training data; manual review required
manifest.csv                         IDs, current/archived state, tenant, version, paths and hashes
sections_index.csv                   Human-checkable section anchors
guides/                               Staged exercise guide and business truth rules
scripts/                              Local consistency checker
```

## Very important distinctions

1. **PDF/HTML/Markdown/TXT are ingestible sources.** CSV payment records are synthetic structured fixtures, **not verified live customer state**; a static RAG cannot prove a real payment was reversed.
2. Keep `evaluation/` and `fine_tuning/` **outside** document ingestion. Otherwise you leak expected answers and labels into the retriever.
3. If a PDF/HTML title/section occurs in multiple versions, the **active** version controls a question about present policy. The old version only supports explicitly historical questions.
4. Never combine AsterDesk and Brookhaven policies when answering as a single tenant. Require verified server-side authorization before retrieval.
5. Model versions and parsing/chunking configuration must be pinned for repeatable ablations. Build lexical, dense, hybrid, pretrained reranker, then fine-tuned reranker on the same authorized candidates.
6. Fine-tuning seed pairs include artificial negative passages; manually audit and improve them. **No claim of human-labeling or independent benchmark quality is made.**

## First three practice sessions (90–120 min each)

**Session 1: Document extraction**. Read `manifest.csv`, choose one longer PDF `documents/asterdesk/*REF*pdf` or another document with 4+ pages, extract per-page text and inspect section IDs. Compare parser output with what you can read visually. Note headings, tables, headers, section references and failed extraction.

### Phase 2 parser

From the repository root, validate and parse every manifest document with:

```bash
.venv/bin/python scripts/parse_rag_documents.py
```

Inspect one exact tenant/document/version without printing the full corpus:

```bash
.venv/bin/python scripts/parse_rag_documents.py \
  --inspect-document AST-REF-001 --tenant asterdesk --version 2.0
```

Phase 1 validates checksums and tenant-safe resolved paths before Phase 2 reads
any source. PDF output splits each physical page at `[SECTION-KEY] Title`
lines, so every block carries a 1-based page number and its section; a section
continuing onto the next page keeps its name there. TXT is split at the same
section lines. HTML and Markdown use ordered structural blocks named by their
headings. Empty PDF pages are reported, while wholly empty documents fail
parsing. OCR, layout reconstruction,
normalization, retrieval chunking, embeddings, and persistence are intentionally
outside this phase. PDF reading order follows PyMuPDF and may differ from complex
visual layouts; Markdown recognition is limited to ATX (`#`) headings.

**Session 2: Chunking**. Split 3 documents using at least two strategies (simple character windows vs structure-aware). Display first, middle and boundary chunks, their stable document/version/section IDs, token estimates and overlaps; measure boundary errors.

**Session 3: Seed evaluation**. Review 15 questions against the **source** before search exists. Label exact document/version/section spans, no-answer and historical cases. Freeze these labels before retrieval experiments. Build retrieval only after this checkpoint.

## Phase 5 local embeddings

The production-oriented local embedding pipeline consumes the generated Phase
4 `data/rag/processed/chunks.jsonl` artifact without rerunning or changing
ingestion. See `docs/PHASE_5_PRODUCTION_EMBEDDINGS.md` for the pinned model and
revision, tokenizer compatibility preflight, artifact contract, commands,
tests, and scoped semantic experiment.

## Experiment setup

- Begin with 3–5 documents and expand to both tenants after ingestion and indexing are reliable.
- In your main project, preserve Month 1 FastAPI + DB APIs. Treat these documents as data, not code.
- Use a **real BM25 implementation**, not PostgreSQL `ts_rank` mislabeled BM25.
- Evaluate both retrieval relevance **and** operational correctness: versions, permissions, deletion, missing evidence, prompt injection and approximate-vector filtering behavior.
- Gold evidence should use **document ID + version + section/span**, because chunk IDs change under different chunkers.
- For a credible job portfolio, add permitted real-world documents later. Purely synthetic RAG benchmarks can be unrealistically clean and do not prove external generalization.

## Licensing

This synthetic text was newly authored for this learning exercise and may be used, adapted, and shared with your project. Any real-company names/products mentioned in source examples are illustrative; **AsterDesk and Brookhaven are fictional**.
