# real-time-hybrid-ai-platform
Real-Time Hybrid AI Support &amp; Operations Platform

## Run locally

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your-key
export MODEL_TIMEOUT_SECONDS=30  # optional, defaults to 30
uvicorn app.main:app --reload
```

The app reads these settings from the environment. A `.env` file is not loaded automatically.

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
