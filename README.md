# real-time-hybrid-ai-platform
Real-Time Hybrid AI Support &amp; Operations Platform

## Run locally

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your-key
export MODEL_TIMEOUT_SECONDS=30  # optional, defaults to 30
export DATABASE_URL=postgresql+asyncpg://app_user:password@localhost:5432/ai_platform
uvicorn app.main:app --reload
```

The app reads these settings from the environment. A `.env` file is not loaded automatically.

## PostgreSQL persistence

Phase 7 uses SQLAlchemy's async session support with the `asyncpg` driver. Database
network I/O is awaited, so it does not block the FastAPI event loop. The `analyses`
table stores the submitted message, structured model result, creation time, and
model-call latency. Entities use PostgreSQL JSONB because they are structured JSON
and may need to be queried by key later.

This project does not yet have a migration system. For this single initial table,
[`schema.sql`](schema.sql) is the reproducible schema definition:

```bash
createdb ai_platform
psql postgresql://app_user:password@localhost:5432/ai_platform -f schema.sql
```

Set `DATABASE_URL` to the equivalent async URL used by the application. Both
`OPENAI_API_KEY` and `DATABASE_URL` are required; startup imports remain safe, and
requests fail with controlled service-unavailable responses when storage is not
configured or available.

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
