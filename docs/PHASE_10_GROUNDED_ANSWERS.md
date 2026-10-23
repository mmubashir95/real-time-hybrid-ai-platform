# Phase 10 — Grounded Answers and Citations

## Objective

Phase 10 turns the Phase 9 answer path into a fail-closed, claim-level grounding
path. A successful answer contains only material policy claims whose citations
exist in the request-local evidence registry and whose complete meaning was
accepted by a bounded semantic verification pass.

Automated verification reduces unsupported output; it does not guarantee
factual correctness. The generator and verifier currently use the same model,
so correlated errors remain possible and require Phase 11 evaluation.

## Data flow

1. Phase 7 retrieves within the trusted tenant, visibility, and active or exact
   historical-version scope.
2. Phase 8 reranks only those authorized passages.
3. The existing Phase 9 context builder selects passages under a token budget
   and assigns `E1`, `E2`, and so on in supplied order.
4. `EvidenceRegistry` independently creates the same request-local mapping from
   the actual included passage objects. The model cannot supply provenance.
5. GPT-5 Nano returns a provider-native `GroundedDraft`: either explicit
   insufficiency or ordered claims containing evidence IDs.
6. Unknown IDs are rejected before verification. One bounded verification call
   receives each remaining claim and only its cited passage text.
7. Only `SUPPORTED` claims are rendered. `UNSUPPORTED`, `CONTRADICTED`, and
   `INSUFFICIENT_EVIDENCE` claims are excluded. If none remain, the response is
   insufficient rather than confidently generated.

The original Phase 9 route remains available through `--mode ungrounded`.

## Response contract

`GroundedAnswerResponse` separates:

- `status`: `grounded`, `insufficient_evidence`, or `grounding_failed`;
- rendered `answer` and validated `citations`;
- the complete server-authoritative evidence registry;
- generation and verification models, usage, cost estimates, and latency;
- counts for generated, verified, supported, rejected, ambiguous, and
  structurally invalid claims;
- the explicit limitation of automated semantic verification.

Unavailable usage or cost remains `null`; it is never reported as zero.

## Evidence registry and provenance

Each registry record preserves the real evidence ID, chunk, tenant, document,
version, title, source reference/format, section, optional page range, and
passage text. IDs are stable only within one answer request. Duplicate chunks,
foreign tenants, unknown IDs, and unauthorized lifecycle states are rejected.

For example, a supported claim citing `E1` may render as:

```text
Verified duplicate subscription charges may qualify for a refund. [1]

[1] AST-REF-001 v2.0 | Decision Rules | page unavailable
```

The displayed document/version/section comes from the retrieved passage, not
the model. Pages are omitted when the source has none.

## Structured generation and support labels

The OpenAI Responses API parses directly into strict Pydantic models. Claim text
is bounded, every grounded claim requires one to five unique evidence IDs, and a
request produces at most twelve claims. Verification output must cover every
structurally valid claim index exactly once.

- `SUPPORTED`: the whole claim, including qualifiers, follows directly.
- `UNSUPPORTED`: it adds information not established by the evidence.
- `CONTRADICTED`: evidence explicitly conflicts with it.
- `INSUFFICIENT_EVIDENCE`: evidence is incomplete or ambiguous.

Missing or malformed verification coverage, provider failure, and truncated or
schema-invalid verifier output yield `grounding_failed`; none becomes a grounded
answer. Invalid or truncated draft output raises a controlled `generation_error`.

Claim text must not carry its own citations. A claim containing a numbered
marker such as `[2]`, an `E` identifier, or a `Source:` label is rejected like a
forged citation, because only the server renders citation markers. There is no correction or regeneration loop.

## Authorization and prompt-injection boundary

The citation layer performs no additional database lookup and cannot broaden
authorization. Default questions accept active passages only. Archived passages
require the existing exact historical document/version authorization. Tenant
and externally permitted visibility are checked again before provider egress.

Both prompts label passage text as untrusted data. Instructions such as “ignore
previous instructions” inside a document remain reference text and have no
authority. Secrets and internal prompts are not placed in the evidence registry
or end-user answer.

## Latency and cost

Grounded mode uses at most two paid requests: one structured draft and one
batched verification. Empty evidence, explicit draft insufficiency, and drafts
whose citations are all invalid skip verification. Generation and verification
usage, costs, and latency are reported separately. The verification output
allowance is the larger of the configured output limit and 96 tokens per
verified claim, so a full batch is not truncated mid-response.

## Developer CLI

Use the existing Phase 10 command with the normal retrieval arguments and the
default grounded mode. Add this flag to compare the original behavior:

```bash
.venv/bin/python scripts/ask_rag.py ... --mode ungrounded
```

The tenant flag is trusted local-development scope, not production
authentication. A grounded CLI run prints `FINAL STATUS: PASSED` only for a
`grounded` response.

## Tests

Focused deterministic tests (no database or paid request):

```bash
.venv/bin/python -m unittest tests.test_rag_grounding tests.test_rag_llm_client
```

Full regression:

```bash
.venv/bin/python -m unittest discover -s tests
```

For a live smoke test, configure the existing database, model artifacts, and
OpenAI credentials, then run the known duplicate-charge CLI query in grounded
mode. Inspect every rendered claim against its cited chunk. Live execution is
intentional and must not be folded into deterministic automated tests.

## Known limitations and Phase 11 prerequisites

- Generator and verifier share GPT-5 Nano and may repeat the same mistake.
- No human-reviewed support/contradiction benchmark or calibrated accuracy is
  claimed yet.
- Multi-passage and long, ambiguous policy reasoning remains difficult.
- Truncated top passages may omit a decisive exception.
- Verification operates at generated-claim granularity, not token-level proof.

Phase 11 should create human-labeled claim/evidence cases covering numbers,
dates, exceptions, negation, conflicting versions, multi-hop claims, ambiguity,
and prompt injection, then measure false-support and false-rejection rates plus
latency and cost distributions.
