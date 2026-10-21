# real-time-hybrid-ai-platform
Real-Time Hybrid AI Support &amp; Operations Platform

## Run locally

Requires a running PostgreSQL server. The commands below connect as your OS user,
which is the default for a local Homebrew or Postgres.app install.

```bash
pip install -r requirements.txt

# One-time database setup: create the database, enable pgvector, and add the table.
createdb realtime_ai
psql -v ON_ERROR_STOP=1 -d realtime_ai -f migrations/001_enable_pgvector.sql
psql -d realtime_ai -f schema.sql

export OPENAI_API_KEY=your-key
export MODEL_TIMEOUT_SECONDS=30  # optional, defaults to 30
export DATABASE_URL=postgresql+asyncpg://localhost:5432/realtime_ai
uvicorn app.main:app --reload
```

The app reads these settings from the environment. A `.env` file is not loaded
automatically. If your PostgreSQL role needs a password, use
`postgresql+asyncpg://user:password@localhost:5432/realtime_ai`. Never commit real
values; `.env.example` holds placeholders only.

## Docker

Prerequisites are Docker Engine or Docker Desktop and a valid OpenAI API key. The
Compose stack contains only the FastAPI service and PostgreSQL; Redis is omitted
because the application does not use it.

Create local runtime configuration and set `OPENAI_API_KEY` in the new `.env` file:

```bash
cp .env.example .env
```

Compose overrides `DATABASE_URL` with an internal async URL whose hostname is
`postgres`. The committed PostgreSQL username/password are development-only values
used solely inside this local Compose network; do not reuse them in production.

Stop any locally running `uvicorn` first. Compose publishes port 8000, and on some
hosts (notably macOS) a local server on `127.0.0.1:8000` keeps answering requests
there instead of the container.

Start the complete stack:

```bash
docker compose up --build -d
docker compose ps
docker compose logs -f api
```

PostgreSQL must pass `pg_isready` before the API starts. On the first creation of
the named volume, the PostgreSQL entrypoint applies `schema.sql` and the pgvector
migration automatically. It does not reapply initialization files to an existing
volume. For an existing volume, follow
[`docs/PHASE_6_1_PGVECTOR_SETUP.md`](docs/PHASE_6_1_PGVECTOR_SETUP.md) before
recreating the PostgreSQL container with the pgvector image.

Verify the API and its existing behavior:

```bash
curl http://127.0.0.1:8000/health

curl -i -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"message":"I was charged twice for my subscription."}'

curl -N -X POST http://127.0.0.1:8000/analyze/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"message":"I was charged twice for my subscription."}'
```

The API container healthcheck uses `/health`; it verifies that FastAPI is
serving HTTP, not that the external model provider is available. Phase 9 performance
logs remain visible through `docker compose logs api`.

Stop containers without deleting stored analyses:

```bash
docker compose down
```

Starting again with `docker compose up -d` reuses the named PostgreSQL volume. Use
`GET /analyses/{id}` to confirm a previously stored record remains available. To
delete all local PostgreSQL data and rerun `schema.sql` on the next start:

```bash
docker compose down -v
```

The `-v` option permanently removes the Compose database volume.

The API image can also be built independently:

```bash
docker build -t realtime-ai-service .
```

Running that image alone still requires `OPENAI_API_KEY`, `MODEL_TIMEOUT_SECONDS`,
and a `DATABASE_URL` reachable from inside the container. Compose is the supported
complete local workflow because it supplies PostgreSQL and initializes its schema.

## PostgreSQL persistence

Phase 7 uses SQLAlchemy's async session support with the `asyncpg` driver. Database
network I/O is awaited, so it does not block the FastAPI event loop. The `analyses`
table stores the submitted message, structured model result, creation time, and
model-call latency. Entities use PostgreSQL JSONB because they are structured JSON
and may need to be queried by key later.

The project uses focused, versioned SQL migrations rather than a migration
framework. [`schema.sql`](schema.sql) remains the reproducible definition of the
initial application table, while `migrations/001_enable_pgvector.sql` enables the
database-scoped extension safely on fresh and existing databases. The app does not
create tables or extensions on startup.

Both `OPENAI_API_KEY` and `DATABASE_URL` are required; startup imports remain safe,
and requests fail with controlled service-unavailable responses when storage is not
configured or available.

**Troubleshooting:** if `/analyze` or `/analyses/{id}` returns 503 "Analysis storage
is temporarily unavailable", check the server log:

- `error_type=UndefinedTableError`: the database exists but `schema.sql` was not
  applied. Run `psql -d realtime_ai -f schema.sql`.
- `error_type=OSError`: PostgreSQL is not running or `DATABASE_URL` points at the
  wrong host or port.
- `error_type=InvalidCatalogNameError`: the database does not exist. Run
  `createdb realtime_ai`.

### Manual verification

Start PostgreSQL, apply `schema.sql`, export the environment variables above, and
start FastAPI. Submit a message with response headers included:

```bash
curl -i -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"message":"I was charged twice for my subscription."}'
```

The response body remains the existing analysis contract. The `X-Analysis-ID`
header contains the stored ID, and `Location` gives its retrieval path. Retrieve
the row using that ID:

```bash
curl http://127.0.0.1:8000/analyses/1
```

The response includes `id`, `message`, all analysis fields, `created_at`, and
`latency_ms`. To verify true persistence, stop FastAPI, start it again with the
same `DATABASE_URL`, and repeat the GET request. The record should still be
returned because it is stored in PostgreSQL rather than application memory.

## Server-Sent Events

`POST /analyze/stream` runs the same AI service and persistence path as
`POST /analyze`, but reports progress over Server-Sent Events. Use `curl -N` to
disable client-side output buffering and see events as they arrive:

```bash
curl -N -X POST http://127.0.0.1:8000/analyze/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"message":"I was charged twice for my subscription."}'
```

A successful stream emits these events in order:

```text
processing_started
analysis_complete
saved
completed
```

Classification and extraction are represented by one `analysis_complete` event
because the provider produces all structured fields in one model call. Each `data:`
value is JSON.

Any failure produces one terminal `error` event, followed by a cleanly closed
connection. Its `code` is `AI_UNAVAILABLE`, `STORAGE_UNAVAILABLE`, or
`INTERNAL_ERROR`, and no `saved` or `completed` event follows it. If the connection
ends without either `completed` or `error`, treat the analysis as failed.

**Disconnects:** when the client disconnects, uvicorn and Starlette cancel the
stream at whatever it is awaiting, including the model call. The stream also checks
for a disconnect just before saving. If the client leaves before `saved`, nothing is
stored and no analysis ID exists, even if `analysis_complete` was already sent. This
differs from `POST /analyze`, which finishes and stores the result even if the
caller disconnects. Cancelling the in-flight model call is cooperative: the async
SDK stops waiting, but the remote provider may already have begun processing (and
billing for) the request.

**Browser clients:** native `EventSource` only supports GET requests, so it cannot
call this POST endpoint. Use `fetch()` and read `response.body` as a stream instead.

## Performance measurement

The service writes correlated performance log lines using a request ID and
monotonic `perf_counter()` timings. All durations are milliseconds:

- `model_latency_ms`: only the awaited OpenAI provider request. Response parsing
  and database work are excluded. Failed and cancelled attempts are also logged.
- `database_latency_ms`: the complete primary save transaction, including add,
  flush, refresh, and commit. A failed transaction includes its rollback attempt.
- `end_to_end_latency_ms`: `/analyze` processing after FastAPI validation and
  dependency setup through the point where the response is ready. It includes AI
  processing and persistence, but not client/network round-trip time.

Each `/analyze` response carries an `X-Request-ID` header matching the
`request_id` in these log lines, so a slow response can be traced to its model and
database timings.

The first requests after startup include one-time costs: building the OpenAI
client (roughly 90 ms locally, counted in end-to-end but not model latency), opening
new database pool connections (counted in database latency), and the first TLS
handshake to the provider (counted in model latency). Don't read a single cold
request's timings as steady-state overhead.

Run the focused benchmark against a configured, running service:

```bash
pip install -r requirements-dev.txt
python -m scripts.benchmark \
  --url http://127.0.0.1:8000 \
  --requests 200 \
  --concurrency 5
```

`--warmup` (default: the concurrency value), `--message`, and `--timeout` are also
configurable. Warm-up requests are sent first and excluded from every result, so
the one-time startup costs above do not land in p95/p99. The benchmark targets only
the normal `/analyze` endpoint so its completion latency is not mixed with SSE timing
semantics.

Latency percentiles are client-observed and cover successful requests only. They
use linear interpolation between closest ranks (the same method as NumPy's default
and Python's `statistics.quantiles(method="inclusive")`):

- p50: the median; half of the measured requests were at or below this value.
- p95: 95% of measured requests were at or below this value.
- p99: 99% of measured requests were at or below this value. It is not the single
  slowest request.

Sample size matters. p95 needs at least about 100 successful requests, and p99
needs several hundred to over a thousand before it is more than a near-maximum.
With 20 requests, p99 is effectively the slowest request. The report prints a
warning when fewer than 100 requests succeed.

RPS (all) is every measured request divided by total benchmark elapsed time; RPS
(successful) counts only 2xx responses, because fast failures such as immediate 503s
would inflate the total. Error rate is transport errors (including timeouts) or any
non-2xx HTTP response divided by measured requests; with the valid default payload,
a 4xx response is therefore a benchmark failure.

The report records the target, request counts, concurrency, model, database, run
time, and local machine description. Results characterize only that tested
environment, workload, and time window; provider latency varies externally, and a
local run is not a production capacity claim.

## Async behavior

Both `/analyze` and `/async-demo` are asynchronous because they wait for I/O-like
operations. `/analyze` awaits the OpenAI SDK's async HTTP call, while `/async-demo`
uses `await asyncio.sleep(...)` to demonstrate the same scheduling behavior without
calling an external service. Start and finish logs for `/async-demo` include a
request ID, so overlapping requests can be observed.

`async def` makes a function a coroutine, but does not make blocking code
asynchronous. At `await`, a non-blocking operation pauses that coroutine and returns
control to the event loop, allowing another request to run. `asyncio.sleep(...)`
registers a wake-up timer with the event loop and yields, whereas `time.sleep(...)`
holds the event-loop thread so no other request can run; it is not used in either
async request path.

Async does not make a single model call faster: one `/analyze` request still takes
as long as the provider takes to respond. The benefit is concurrency for I/O-bound
work. While one request waits on the network, the same worker can serve others.

To see this, run two requests at once. Each waits 3 seconds, and together they
finish in about 3 seconds rather than 6:

```bash
time (
  curl http://127.0.0.1:8000/async-demo &
  curl http://127.0.0.1:8000/async-demo &
  wait
)
```

Cancellation is also delivered at an await point: if the coroutine is cancelled
while waiting, `asyncio.CancelledError` propagates instead of being swallowed.

## Test

```bash
pip install -r requirements-dev.txt
python -m unittest discover -s tests
```
