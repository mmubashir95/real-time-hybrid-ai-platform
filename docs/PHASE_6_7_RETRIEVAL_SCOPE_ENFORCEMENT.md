# Phase 6.7: Retrieval scope enforcement

Task 6.7 hardens the existing exact dense retriever. It does not add a second
retrieval engine, a public API, row-level security, or an authentication system.
The enforced guarantee is narrower and precise: PostgreSQL ranks only rows that
match a server-established tenant, visibility set, lifecycle scope, and complete
embedding profile.

## Trusted scope and authorization boundary

`TrustedSearchContext` requires:

- a non-blank tenant ID;
- one or more approved visibilities (`public` and/or `internal`);
- a `SearchContextOrigin` stating whether authenticated application policy or a
  trusted internal/test caller established the scope; and
- for historical retrieval, an exact document ID and version plus explicit
  historical permission.

The type records where authorization must have been established; constructing it
does not authenticate a user. This repository has no production RAG identity
resolver, and the FastAPI application exposes no retrieval route. A future route
must derive tenant, visibility, and historical permission from authenticated
server-side policy. It must never copy a request body's tenant ID into this
context and call that authorization.

`scripts/search_rag_dense.py` remains a development/evaluation interface. Its
`--tenant` and `--visibility` arguments are filter inputs in a trusted operator
context, not proof of end-user access. Historical CLI use additionally requires
`--authorize-historical` together with both exact historical identifiers. This
explicit acknowledgement is not a substitute for production authorization.

## SQL enforcement before Top-K

The existing query joins ownership and content in PostgreSQL:

```sql
FROM documents d
JOIN document_versions v ON v.document_fk = d.id
JOIN document_chunks c ON c.document_version_fk = v.id
JOIN chunk_embeddings e ON e.chunk_id = c.chunk_id
WHERE d.tenant_id = $1
  AND v.visibility = ANY($2::text[])
  AND (
      ($8::text IS NULL AND v.status = 'active')
      OR
      ($8::text IS NOT NULL AND v.status = 'archived'
       AND d.document_id = $8 AND v.version = $9)
  )
```

The same eligible CTE checks model ID, immutable revision, configuration
fingerprint, normalization, encoding version, query instruction, and 384 vector
dimensions. Only then does PostgreSQL compute cosine distance, order candidates,
and apply `LIMIT`. Consequently, a more-similar foreign, invisible, historical,
or incompatible vector cannot consume an authorized Top-K slot or enter Python
application memory. All values are bound parameters; only a strictly validated
schema identifier is interpolated.

An unknown tenant, unknown version, or scope with no eligible rows returns an
empty result and candidate count. There is no global, cross-tenant, or archived
fallback.

## Lifecycle and visibility policy

| Version state | Current search | Exact historical search |
|---|---:|---:|
| `active` | included when otherwise authorized | excluded |
| `archived` | excluded | included when explicitly authorized |
| `staged` | excluded | excluded |
| `withdrawn` | excluded | excluded |

Withdrawal is treated as denial, including in historical mode. Any future need
to inspect withdrawn content requires a separate approved policy; it must not be
folded into ordinary historical access. Staged content is unpublished and is
never retrievable.

Visibility is version metadata and is always checked in SQL for current and
historical requests. Missing visibility permission fails validation, and unknown
labels are rejected rather than mapped to `public`. Visibility labels remain a
policy input, not a complete authorization system.

## Verification

The focused contract tests require no model or database. The PostgreSQL tests
create and later drop a unique disposable schema in the configured test database.
They use active, archived, staged, withdrawn, internal, foreign-tenant, and
incompatible-profile rows, including unauthorized rows with higher similarity
than an authorized match.

Commands are intentionally operator-run:

```bash
python -m unittest tests.test_rag_dense_retrieval.DenseSearchContractTests

TEST_DATABASE_URL=postgresql+asyncpg://localhost:5432/realtime_ai_test \
  python -m unittest tests.test_rag_dense_retrieval.DenseSearchPostgresTests
```

Example trusted internal historical invocation:

```bash
python scripts/search_rag_dense.py \
  --question "What did the earlier refund policy say?" \
  --tenant asterdesk \
  --visibility public \
  --historical-document-id AST-REF-001 \
  --historical-version 1.0 \
  --authorize-historical \
  --config-fingerprint <64-character-imported-fingerprint>
```

These checks validate application-level query isolation. They do not establish
database row-level security or prove that a production caller is authorized.

## Next boundary

Task 6.8 should evaluate retrieval quality and regression behavior without
weakening these predicates. Production API work must first add authenticated
identity-to-tenant and permission resolution, then construct
`TrustedSearchContext` only from that trusted server-side result.
