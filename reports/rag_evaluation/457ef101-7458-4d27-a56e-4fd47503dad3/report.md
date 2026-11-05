# Month 2 RAG Evaluation Report

## Run identity

- Run ID: `457ef101-7458-4d27-a56e-4fd47503dad3`
- Timestamp: `2026-11-05T14:17:18.454765+00:00`
- Dataset: `month2-rag-golden-candidates` v`1.2.0`
- Split: `development`
- Configuration: `D`
- Corpus snapshot: `9c2ce177ccf5705716dcd91c8c5cd7b165702a5ef5ac7ab76e920c86fdc1a2fb`
- Index version: `000,001,002,003`
- Test environment: local development Qwen3
- Candidate baseline (C) run: `836516ee-b106-46be-b5b7-8dca3587b1c2` (artifact SHA-256 `1b205dd86232ff7440ff2de76158f3829e35168f5c2cac657b8c8f8e6b765a4f`)
- Reranker: `Qwen/Qwen3-Reranker-0.6B@fd9fb1d26c07223ced488065909faf522e29cc7d` on `mps`, precision `auto`, batch size 4, max length 8192
- Reranker model load seconds: 0.70
- Reranker truncated inputs: 0
- Warm-up question (latency includes first-inference cost): `P11-CAN-001`

## Retrieval and ranking

- Questions: 5 (answerable: 4)
- Failed questions: 0
- Precision@5 (all completed questions): 0.2000
- Precision@10 (all completed questions): 0.1000
- Precision@5 (answerable only): 0.2500 of ceiling 0.2500
- Precision@10 (answerable only): 0.1250 of ceiling 0.1250
- Recall, coverage, MRR, and NDCG average answerable questions only.
- Recall@5: 1.0000
- Recall@10: 1.0000
- Recall@20: 1.0000
- Complete Evidence Coverage@5: 1.0000
- MRR@10: 1.0000
- NDCG@5: 0.9799
- NDCG@10: 0.9799
- Latency p50 ms: 1010.6628
- Latency p95 ms: 1141.3655

## Answer quality and grounding

PENDING HUMAN REVIEW

## Security

NOT MEASURED

## Performance, usage, cost, and reliability

- Requests: 5
- Success rate: 1.0000
- Failure rate: 0.0000
- Total tokens: NOT MEASURED
- Total estimated cost USD: NOT MEASURED

## Limitations

- Independent human answer and citation reviews are pending.
- Dataset labels are unapproved; results are exploratory only.
