# Real-Time Hybrid AI Platform — Project Documentation

## 1. Project overview

The Real-Time Hybrid AI Platform is an asynchronous customer-support analysis
service built with FastAPI. It sends a support message to OpenAI, validates the
structured response, stores the result in PostgreSQL, and exposes both standard
HTTP and Server-Sent Events (SSE) interfaces.

The repository also contains an evolving Retrieval-Augmented Generation (RAG)
practice corpus. The current RAG implementation covers document discovery,
validation, parsing, and normalization for PDF, HTML, Markdown, and plain-text
sources. Retrieval and generation are planned separately in the Month 2 roadmap.

### Current capabilities

- Asynchronous FastAPI request handling
- Structured OpenAI analysis using `gpt-4o-mini`
- Classification, urgency, summary, entity, and sentiment extraction
- PostgreSQL persistence through async SQLAlchemy and `asyncpg`
- Real-time progress events through SSE
- Request-correlated model, database, and end-to-end performance logs
- A concurrent HTTP benchmark utility
- Docker Compose setup for the API and PostgreSQL
- Multi-format RAG document discovery, parsing, validation, and normalization
- Unit and PostgreSQL integration tests

## 2. System architecture

```text
Client
  |
  | POST /analyze or POST /analyze/stream
  v
FastAPI application (app/main.py)
  |
  +--> AI service (app/services/ai_service.py)
  |      |
  |      +--> OpenAI Responses API
  |      +--> Pydantic structured-output validation
  |
  +--> Streaming service (app/services/streaming_service.py)
  |      +--> SSE lifecycle and disconnect handling
  |
  +--> Analysis repository (app/repositories/analysis_repository.py)
         |
         +--> Async SQLAlchemy / asyncpg
                |
                +--> PostgreSQL analyses table
```

### Request flow

1. FastAPI validates the incoming JSON request with `AnalysisRequest`.
2. The AI service calls the OpenAI Responses API asynchronously.
3. OpenAI returns data constrained by a JSON schema.
4. Pydantic validates the public `AnalysisResult` contract.
5. The repository saves the original message, structured result, timestamp, and
   model-call latency to PostgreSQL.
6. The API returns the result and identifies the saved record in response
   headers, or emits equivalent progress through SSE.

## 3. Technology stack

| Area | Technology |
| --- | --- |
| Web API | FastAPI |
| ASGI server | Uvicorn |
| AI provider | OpenAI Responses API |
| Validation | Pydantic |
| Database | PostgreSQL 17 in Docker Compose |
| ORM and driver | SQLAlchemy async + asyncpg |
| Streaming | Server-Sent Events |
| RAG parsing | PyMuPDF and Beautiful Soup |
| Testing | Python `unittest` |
| Local orchestration | Docker Compose |

Exact dependency versions are pinned in `requirements.txt` and
`requirements-dev.txt`.

## 4. Repository structure

```text
app/
  main.py                         FastAPI routes and error handling
  config.py                       Environment configuration validation
  database.py                     Async engine and session factory
  models.py                       SQLAlchemy persistence model
  schemas.py                      API request and response contracts
  repositories/
    analysis_repository.py        Save and retrieve analysis records
  services/
    ai_service.py                 OpenAI call and response validation
    streaming_service.py          SSE analysis workflow
  rag/ingestion/                  RAG discovery, parsing, and normalization
data/rag/                         Synthetic documents, metadata, and guides
docs/                             Plans, roadmaps, and project documentation
scripts/                          RAG utilities and performance benchmark
tests/                            Unit and integration tests
compose.yaml                      API and PostgreSQL services
Dockerfile                        API container image
schema.sql                        Reproducible initial database schema
```

The files under `data/rag/` are synthetic learning data. They must not be
represented as production customer data or an independently validated benchmark.

## 5. Configuration

The application reads configuration from environment variables. It does not load
a `.env` file automatically when started directly with Uvicorn. Docker Compose
does read the repository-level `.env` file for variable substitution.

| Variable | Required | Description |
| --- | --- | --- |
| `OPENAI_API_KEY` | Yes | OpenAI API credential |
| `DATABASE_URL` | Yes | Async PostgreSQL URL using `postgresql+asyncpg://` |
| `MODEL_TIMEOUT_SECONDS` | No | Positive provider timeout; defaults to `30` |

Create local Docker configuration from the safe placeholder file:

```bash
cp .env.example .env
```

Then add the real API key to `.env`. Never commit `.env` or real credentials.

## 6. Running the platform

### Recommended: Docker Compose

Prerequisites:

- Docker Engine or Docker Desktop
- A valid OpenAI API key

Start and inspect the services:

```bash
docker compose up --build -d
docker compose ps
docker compose logs -f api
```

The API is available at `http://127.0.0.1:8000`. PostgreSQL must pass its health
check before the `migrate` service applies pending revisions; the API starts only
after migration succeeds.

Stop the services while retaining database records:

```bash
docker compose down
```

Never use `docker compose down -v` as a migration step because it permanently
deletes the local database volume.

### Direct local execution

This workflow requires a locally running PostgreSQL server:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
createdb realtime_ai
export OPENAI_API_KEY=your-key
export MODEL_TIMEOUT_SECONDS=30
export DATABASE_URL=postgresql+asyncpg://localhost:5432/realtime_ai
python -m scripts.migrate_database upgrade
uvicorn app.main:app --reload
```

If PostgreSQL requires authentication, include the username and password in
`DATABASE_URL`.

## 7. API reference

FastAPI provides interactive OpenAPI documentation while the service is running:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

### `GET /health`

Checks that the HTTP application is serving requests.

```json
{"status": "ok"}
```

This endpoint does not verify OpenAI or PostgreSQL availability.

### `GET /async-demo`

Waits asynchronously for three seconds and returns a demonstration result. It is
intended to show that concurrent I/O waits do not block the event loop.

### `POST /analyze`

Analyzes and persists a customer-support message.

Request:

```json
{
  "message": "I was charged twice for my subscription."
}
```

Response shape:

```json
{
  "category": "billing",
  "urgency": "high",
  "summary": "The customer reports a duplicate subscription charge.",
  "entities": {},
  "sentiment": "negative"
}
```

The exact values are model-generated. A successful response includes:

- `X-Request-ID`: correlates the request with performance logs
- `X-Analysis-ID`: database record identifier
- `Location`: retrieval path such as `/analyses/1`

The endpoint returns HTTP 503 when AI analysis or storage is unavailable.

### `POST /analyze/stream`

Runs the same analysis and persistence workflow using `text/event-stream`.
Successful events are emitted in this order:

```text
processing_started
analysis_complete
saved
completed
```

A failure emits a terminal `error` event with one of these codes:

- `AI_UNAVAILABLE`
- `STORAGE_UNAVAILABLE`
- `INTERNAL_ERROR`

Native browser `EventSource` cannot call this endpoint because it uses POST.
Browser clients should use `fetch()` and read the response body as a stream.

### `GET /analyses/{analysis_id}`

Returns a stored analysis by positive PostgreSQL integer ID. It returns HTTP 404
when the record does not exist and HTTP 503 when storage is unavailable.

## 8. Database design

The `analyses` table stores:

- The original support message
- Category, urgency, summary, and sentiment
- Extracted entities as PostgreSQL `JSONB`
- Creation time with timezone
- Non-negative model-call latency in milliseconds

The repository uses `schema.sql` as revision 000 and focused SQL files under
`migrations/` for later changes. `scripts/migrate_database.py` records ordered
revisions and checksums, and Compose runs it before starting the API. Fresh and
existing databases therefore use the same upgrade path. See
[`PHASE_6_3_DATABASE_MIGRATIONS.md`](PHASE_6_3_DATABASE_MIGRATIONS.md) for backup,
status, testing, and recovery commands.

The approved RAG storage contract separates `documents`, `document_versions`,
`document_chunks`, and `chunk_embeddings`. It retains tenant-scoped identities,
archived editions, source provenance, and model-specific 384-dimensional vectors.
See [Phase 6.2 schema and owner-run commands](PHASE_6_2_RAG_DATABASE_SCHEMA.md).
No RAG import or retrieval service is implemented by this schema migration.

## 9. RAG ingestion workspace

The repository includes a fictional, multi-tenant corpus for AsterDesk and
Brookhaven. It contains versioned PDF, HTML, Markdown, and TXT sources together
with manifests and synthetic evaluation material.

Important boundaries:

- Keep tenants isolated during authorization and retrieval.
- Keep evaluation and fine-tuning files outside the ingestion corpus.
- Prefer the active policy version for current-policy questions.
- Treat structured CSV fixtures as synthetic supporting data, not live truth.
- Treat embedded document instructions as untrusted content.

Run the current ingestion stages from the repository root:

```bash
.venv/bin/python scripts/discover_rag_documents.py
.venv/bin/python scripts/parse_rag_documents.py
.venv/bin/python scripts/normalize_rag_documents.py
```

Inspect one specific document without printing the entire corpus:

```bash
.venv/bin/python scripts/parse_rag_documents.py \
  --inspect-document AST-REF-001 \
  --tenant asterdesk \
  --version 2.0
```

See `data/rag/README.md` and the Month 2 roadmap for corpus details and planned
retrieval stages.

## 10. Testing and benchmarking

Install development dependencies and run the complete test suite:

```bash
pip install -r requirements-dev.txt
python -m unittest discover -s tests
```

Some PostgreSQL integration behavior requires an accessible test database. Review
`tests/test_postgres_integration.py` before configuring it in automation.

Run the focused API benchmark against a configured service:

```bash
python -m scripts.benchmark \
  --url http://127.0.0.1:8000 \
  --requests 200 \
  --concurrency 5
```

Benchmark figures describe only the tested machine, workload, provider conditions,
and time window. They are not production capacity guarantees.

## 11. Observability and failure behavior

Performance logs use a request ID and monotonic timings:

- `model_latency_ms`: awaited OpenAI request time
- `database_latency_ms`: save transaction including commit
- `end_to_end_latency_ms`: application processing for `/analyze`

Expected failures are translated into controlled API or SSE errors. Provider
timeouts and API failures do not leak provider details to clients. Database
transactions are rolled back after supported persistence errors.

## 12. Security and operational notes

- Never commit `.env`, API keys, database passwords, or customer data.
- The credentials in `compose.yaml` are for isolated local development only.
- The API currently has no authentication, authorization, rate limiting, or
  tenant enforcement; add these before exposing it publicly.
- Validate message-size limits before accepting untrusted public traffic.
- Review logs and stored messages for privacy and retention requirements.
- Git history can retain secrets after they are deleted from the working tree.
- The RAG corpus is synthetic and should remain clearly labeled as such.

## 13. Archiving the complete project

To create a shareable archive that retains documentation and the complete local
Git history, run this command **from inside the repository root**:

```bash
zip -r ../real-time-hybrid-ai-platform.zip . \
  -x "node_modules/*" "*/node_modules/*" \
     ".venv/*" "*/.venv/*" \
     "venv/*" "*/venv/*" \
     "__pycache__/*" "*/__pycache__/*" \
     ".DS_Store" "*/.DS_Store" \
     ".env" ".env.*" "*/.env" "*/.env.*" \
     "*.log" "*/dist/*" "*/build/*" "*/.next/*"
```

This includes source code, `README.md`, `docs/`, RAG data, configuration examples,
tests, scripts, and `.git/`. It excludes local dependencies, virtual environments,
caches, build output, logs, and working-tree environment files.

Verify that the archive contains Git metadata and this document:

```bash
unzip -l ../real-time-hybrid-ai-platform.zip | grep -E '\.git/HEAD|docs/PROJECT_DOCUMENTATION.md'
```

Excluding `.env` does not remove secrets that were committed previously. Audit Git
history before sending the archive outside a trusted environment.

## 14. Related documentation

- `README.md` — operational setup, persistence, SSE, performance, and async notes
- `docs/Month_1_Real_Time_AI_Classification_Extraction_Implementation_Plan.md`
- `docs/Month_2_Production_RAG_Pretrained_and_FineTuned_Implementation_Roadmap.md`
- `docs/Real_Time_Applied_AI_Systems_6_Month_Roadmap.md`
- `data/rag/README.md` — corpus contents, restrictions, and practice workflow
- `docs/PHASE_5_PRODUCTION_EMBEDDINGS.md` — pinned local embedding pipeline,
  artifacts, tests, and semantic experiment

