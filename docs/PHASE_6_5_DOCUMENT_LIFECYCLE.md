# Phase 6.5: Document lifecycle and integrity

## Scope

Task 6.5 adds controlled staging, atomic publication, withdrawal, and a reusable
current-search eligibility query. It reuses the Phase 6.4 artifact validator and
transactional importer. It does not parse, chunk, embed, rank, or implement dense
similarity search.

## Database migration

Revision `003_add_document_lifecycle` is a forward-only migration. It leaves
revision 002 unchanged, preserves every existing row, and keeps
`uq_document_versions_one_active` intact. It extends `status` to:

| Status | Meaning | Current-searchable |
|---|---|---|
| `staged` | Complete persisted candidate awaiting publication | no |
| `active` | Current published version | yes |
| `archived` | Superseded historical version | no |
| `withdrawn` | Explicitly deactivated version | no |

The migration adds nullable `expected_chunk_count` and
`expected_chunk_ids_fingerprint` columns. Nulls preserve compatibility for legacy
active/archived rows. Both values are database-required whenever a row is
`staged`, and the pair must be either both present or both absent.

Apply and inspect it through the existing runner:

```bash
python -m scripts.migrate_database status
python -m scripts.migrate_database upgrade
python -m scripts.migrate_database status
```

Rollback requires a deliberate forward migration. The old two-state constraint
cannot be restored while `staged` or `withdrawn` rows exist, and removing the two
completeness columns would discard lifecycle metadata.

## Staging and authoritative completeness

`stage_document_version` accepts one fully validated `ImportPlan` for exactly one
tenant/document/version. It derives completeness from the plan's complete source
chunk set before any database write. It never derives the expected set from rows
already stored in PostgreSQL.

The fingerprint contract is:

1. reject an empty set or duplicate chunk IDs;
2. sort unique chunk IDs using Python string ordering;
3. serialize the array as UTF-8 JSON with separators `,` and `:` and no spaces;
4. compute lowercase SHA-256 hex over those exact bytes.

JSON array boundaries make the encoding unambiguous. Staging replaces only the
incoming lifecycle status with `staged`, attaches the expected count/fingerprint,
and calls Task 6.4's `import_rag_embeddings`. That importer retains ownership,
content, provenance, model checks, transaction rollback, and exact-row retry
comparison. Baseline corpus imports still preserve their source `active` and
`archived` statuses and store null completeness metadata.

An identical staging retry reuses all rows. A retry with different source fields,
expected count, fingerprint, chunks, or vectors raises `RagImportConflict`; it
never overwrites a prior staged identity.

## Atomic publication

`publish_document_version` runs this sequence in one asyncpg transaction:

```text
set schema search path
  -> resolve tenant/document and SELECT parent FOR UPDATE
  -> read active version after the lock
  -> enforce explicit expected-active precondition
  -> require target status=staged
  -> recompute stored chunk count and fingerprint
  -> compare them with authoritative staging metadata
  -> require one selected-profile embedding for every expected chunk
  -> archive the old active version, if present
  -> activate the target
  -> verify the resulting active identity
  -> commit
```

Completeness and state transition share the same transaction and parent row lock.
Any exception rolls back both status updates. The selected `EmbeddingProfile`
checks model ID, immutable model revision, embedding configuration fingerprint,
normalization, encoding configuration version, non-null input fingerprints, and
384 dimensions.

The parent lock serializes writers for one logical document without locking other
documents. Both concurrent workers evaluate state after acquiring it. The first
valid worker commits; the next sees that its expected active version is stale and
raises `LifecycleConflict`. The partial unique index remains the database backstop.

First publication requires an explicit `expected_active_version=None`. A retry of
the same successful publication returns `already_published` only when the target
is still active and the originally expected predecessor is archived (or the
original operation explicitly expected no active version). Archived or withdrawn
targets cannot be republished.

## Withdrawal

`withdraw_document_version` applies only `active -> withdrawn`. It uses the same
tenant-scoped parent lock and explicit expected-active check as publication. It
preserves chunks, embeddings, hashes, and metadata. A matching retry returns
`already_withdrawn`; staged administrative withdrawal is deliberately unsupported.
There is no hard-delete operation.

## Current and historical visibility

`fetch_current_chunk_ids` is the minimal reusable predicate for Task 6.6. Current
candidates must match all of:

- trusted tenant identity;
- `document_versions.status = 'active'`;
- a caller-approved visibility;
- embedding model ID and immutable revision;
- embedding configuration fingerprint;
- normalization and encoding configuration version.

Staged, archived, and withdrawn rows are excluded by construction. Historical
records remain stored for a future explicitly authorized historical query, but no
general historical lookup API is introduced here. A tenant string supplied by a
script is scoping input, not production authentication.

## Tests

Use a disposable PostgreSQL 17 database whose name contains `test` and has
pgvector available:

```bash
export TEST_DATABASE_URL=postgresql+asyncpg://realtime_ai:password@localhost:5432/realtime_ai_test
python -m unittest \
  tests.test_database_migrations \
  tests.test_rag_schema_integration \
  tests.test_rag_embedding_import \
  tests.test_rag_document_lifecycle -v
python -m unittest discover -s tests -v
```

Lifecycle integration tests create isolated random schemas and cover migration
preservation, state constraints, staging retry/conflict, missing chunks,
same-count wrong-ID sets, missing embeddings, rollback, first publication,
publication retry, withdrawal retry, tenant/model/visibility filtering, and real
concurrent publication using two PostgreSQL connections.

## Known limitations and next task

There is no lifecycle HTTP or CLI surface; the async service is ready for a later
trusted application boundary. Legacy active/archived rows have no staged manifest,
which is intentional because they are not publishable staged candidates. The
current-search helper returns eligible IDs but does not calculate vector distance.
Task 6.6 can reuse this predicate and add PostgreSQL pgvector similarity ordering,
query embedding, limits, and retrieval contracts without redefining lifecycle
eligibility.
