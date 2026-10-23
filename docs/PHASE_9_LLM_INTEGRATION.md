# Phase 9: OpenAI LLM Integration

Phase 9 adds a bounded answer-generation stage after the existing Phase 7 hybrid retrieval and Phase 8 Qwen3 reranking stages. It uses the OpenAI Responses API with `gpt-5-nano`, preserves the exact evidence identities supplied to the model, and reports provider token usage, estimated cost, and stage latency.

## Pipeline

1. Phase 7 retrieves tenant- and visibility-authorized candidates.
2. Phase 8 reranks those same candidates without changing their provenance.
3. The context builder selects complete passages in rerank order within a token budget. It truncates only the lowest-priority tail of the highest-ranked passage when no complete passage fits.
4. Before retrieval, the trusted visibility scope is narrowed to `RAG_EXTERNAL_ALLOWED_VISIBILITIES`, so only sendable evidence is retrieved. Before transmission, the external-data policy checks the tenant and visibility again as a safety net.
5. One OpenAI Responses API request generates the answer. Empty evidence returns a deterministic no-evidence response and makes no paid request.

Retrieved document text is explicitly delimited and treated as untrusted data. It cannot replace system instructions. This phase does not implement user-visible answer citations; citation rendering and verification remain Phase 10 work.

## Configuration

Copy the relevant values from `.env.example` into your local environment. Never commit the API key.

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_MODEL="gpt-5-nano"
export OPENAI_TIMEOUT_SECONDS="30"
export OPENAI_MAX_OUTPUT_TOKENS="512"
export OPENAI_REASONING_EFFORT="minimal"
export OPENAI_MAX_CONCURRENT_REQUESTS="2"
export RAG_MAX_CONTEXT_TOKENS="6000"
export RAG_MAX_GENERATION_INPUT_TOKENS="8000"
export RAG_EXTERNAL_ALLOWED_VISIBILITIES="public"
```

`RAG_EXTERNAL_ALLOWED_VISIBILITIES` is a deployment egress control, not the primary authorization boundary. The default permits only `public` passages to be sent to OpenAI. Add `internal` only after your organization approves that data class for the configured OpenAI account and retention arrangement.

The database must be running and `DATABASE_URL` must include its actual host and port, commonly `5432` for a local PostgreSQL instance:

```bash
export DATABASE_URL="postgresql+asyncpg://USER:PASSWORD@127.0.0.1:5432/DATABASE"
```

The embedding profile fingerprint must be the 64-character SHA-256 value used when the current corpus was imported.

## Manual commands

The standard test suite mocks the OpenAI client and cannot spend API credits. Run these commands manually after activating `.venv`.

Focused offline tests:

```bash
.venv/bin/python -m unittest \
  tests.test_config \
  tests.test_rag_context_builder \
  tests.test_rag_llm_client \
  tests.test_rag_generation_service
```

One small paid OpenAI-only smoke test:

```bash
.venv/bin/python scripts/smoke_test_rag_generation.py
```

End-to-end database, retrieval, reranking, and generation:

```bash
.venv/bin/python scripts/ask_rag.py \
  --query "Can I get a refund after being charged twice?" \
  --tenant asterdesk \
  --visibility public \
  --candidate-k 20 \
  --top-k 5 \
  --config-fingerprint 2945caacf194aff9aa2d9a26da0a058c98a61a96ee98881e36e3e2301c327d37 \
  --reranker-device mps \
  --reranker-precision float16
```

The CLI prints the answer, the exact supplied chunk/document/version/section/page/source identities, estimated and actual token counts, estimated request cost, and separate retrieval/reranking/generation latency.

## Model and pricing note

The implementation keeps `gpt-5-nano` because it is the model selected for this phase. OpenAI currently marks `gpt-5-nano` and its `gpt-5-nano-2025-08-07` snapshot as deprecated, so a later phase should evaluate a supported replacement before production launch. The recorded price constants, effective 2026-10-23, are $0.05 per million uncached input tokens, $0.005 per million cached input tokens, and $0.40 per million output tokens. Cost is an estimate based on provider-reported usage and these versioned constants. Cost is reported only when the responding model is a `gpt-5-nano` snapshot; any other configured model reports no cost rather than a wrong one.

Official references:

- [GPT-5 nano model](https://developers.openai.com/api/docs/models/gpt-5-nano)
- [Create a response](https://developers.openai.com/api/reference/resources/responses/methods/create)
- [Reasoning models](https://developers.openai.com/api/docs/guides/reasoning)

## Known limitations

- There is no streaming response path.
- There is no conversation history or persisted response state (`store=False`).
- The tokenizer budget is a conservative local estimate; provider usage is authoritative after a request.
- Pricing can change and must be reviewed before production cost forecasting.
- Phase 10 must add citation formatting, citation-to-evidence verification, and answer-level support checks.
