# Phase 6.6: Exact dense semantic retrieval

Task 6.6 adds internal, database-backed semantic search. It generates only the
incoming question embedding, reuses stored document vectors, and returns source
chunks. It does not generate an answer, citation text, keyword results, or a
reranked/hybrid result set.

## Data flow and compatibility

`SentenceTransformerEncoder.encode_query()` uses the pinned
`BAAI/bge-small-en-v1.5` revision and prefixes the BGE query instruction. The
384-dimensional float32 vector is checked for shape, finite values, and nonzero
norm. Model ID, immutable revision, artifact configuration fingerprint,
normalization flag, encoding version, query instruction, and vector dimensions
must match persisted rows. Equal dimensions alone do not mean two embedding
spaces are compatible.

`DenseRetriever` sends the vector as a bound asyncpg parameter and explicitly
casts it to `public.vector`. PostgreSQL computes cosine distance with `<=>` and
the service reports similarity as `1 - cosine_distance`. Similarity is a ranking
signal, not calibrated confidence or proof that a passage is factually correct.

The exact query applies these predicates before ordering and limiting:

- trusted tenant identity;
- caller-authorized `public` and/or `internal` visibility;
- `active` status for normal current-policy search;
- one exact document ID and version for explicitly authorized historical search;
- the complete selected embedding profile.

Filtering before Top-K prevents another tenant, inaccessible content, or an old
version from affecting the ranking or entering application memory. Equal
distances use stable chunk-ID ordering. No HNSW or IVFFlat index is created.

## Internal contracts

`TrustedSearchContext` is deliberately named as a server-side authorization
boundary. A future HTTP layer must derive it from authenticated policy; it must
not treat a client-provided tenant string as authorization. The included CLI is
for development/evaluation and its `--tenant` value is only a trusted test
scope. No public retrieval endpoint is added.

`DenseSearchResult` retains stable chunk ID and unmodified text, document and
version identity, title, heading path, page bounds, source block IDs, source
path/format, visibility, and embedding identity. Missing page data remains
`None`; the service does not invent citation fields.

Normal search returns a valid empty response when no eligible compatible row
exists. Historical mode requires both document ID and version and cannot become
a tenant-wide archived search.

## Development CLI

First inspect the compatible fingerprint already imported into the selected
database:

```sql
SELECT embedding_model_id, embedding_model_revision,
       embedding_config_fingerprint, normalize_embeddings,
       encoding_config_version, query_instruction, count(*)
FROM chunk_embeddings
GROUP BY 1, 2, 3, 4, 5, 6;
```

Then run a current-policy query (this is intentionally left for the operator):

```bash
python scripts/search_rag_dense.py \
  --question "Can I get a refund if I was charged twice?" \
  --tenant asterdesk \
  --visibility public \
  --top-k 5 \
  --config-fingerprint <64-character-imported-fingerprint>
```

Use `--question-id AST-REF-001-Q01` instead of `--question` to load a question
and tenant from the existing evaluation bank. Add `--visibility internal` only
when that test context is authorized. Historical retrieval additionally needs
both `--historical-document-id` and `--historical-version`.

The CLI reports model-loading, query-encoding, SQL, and total retrieval time,
plus the eligible candidate count and returned chunk IDs. It loads the model
once per process and never regenerates document embeddings.

## Tests and evaluation

Contract tests run without loading the transformer. Database behavior uses a
disposable schema in a real PostgreSQL/pgvector database:

```bash
python -m unittest tests.test_rag_dense_retrieval.DenseSearchContractTests

TEST_DATABASE_URL=postgresql+asyncpg://localhost:5432/realtime_ai_test \
  python -m unittest tests.test_rag_dense_retrieval.DenseSearchPostgresTests
```

The integration cases cover cosine ordering, NumPy agreement, Top-K count,
tenant isolation, default lifecycle exclusions, visibility, exact historical
scope, embedding-profile exclusion, empty results, stable provenance, and real
SQL parameter binding. They create and remove only unique test schemas.

For corpus evaluation, use the existing review-required question bank and the
same split, tenant/version scope, encoder, vector artifact, and compatibility
profile for both implementations. Phase 5's `evaluate_recall()` defines the
current metric: a hit requires the labeled document, version, and section anchor
within the first K results. Expected-no-answer questions are counted separately
because Recall@K cannot measure abstention. Compare ordered chunk IDs from exact
PostgreSQL search with `run_semantic_experiment()`; differences are valid only
when lifecycle/visibility candidate sets differ or a tie is resolved differently.
Do not compare an active-only database result against an unfiltered artifact
baseline.

Recall@K answers whether at least one labeled evidence section appeared in the
first K candidates. It does not measure exact passage quality, answer accuracy,
groundedness, or calibrated confidence. When scope, vectors, cosine calculation,
and tie-breaking are equivalent, exact PostgreSQL and NumPy rankings should
match.

## Known limitations and Phase 7 boundary

Dense-only search can miss exact identifiers, rare terms, and lexical matches.
The synthetic evaluation labels still require human review. Task 6.6 does not
add BM25, reciprocal-rank fusion, reranking, answer generation, or citation
rendering. Phase 7 should add PostgreSQL keyword retrieval, establish equivalent
authorization filters for both candidate sources, and evaluate hybrid fusion
without weakening this trusted search boundary.
