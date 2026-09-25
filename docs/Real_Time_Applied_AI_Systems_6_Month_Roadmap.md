# 6-Month Real-Time Applied AI Systems Roadmap

**Primary:** Production Applied AI + Real-Time AI Systems  
**Specialization:** Edge AI + Hybrid AI  
**Secondary:** Mobile AI Integration

## 1. Career Target

At the end of the six months, the target positioning is:

> **Applied AI Engineer / AI Systems Engineer** specializing in **Real-Time AI, Edge AI, and Hybrid AI systems**, with strong mobile AI integration capability.

The roadmap is intentionally centered on AI systems rather than mobile application development. Mobile remains important as a client platform and as an edge execution environment, but it is a secondary capability rather than the core career identity.

### Target capability architecture

```text
Data / Events / User Input
          ↓
Real-Time Ingestion
          ↓
AI Service / Orchestrator
          ↓
┌─────────┼─────────────┐
│         │             │
ML      RAG/LLM       Tools
│         │             │
└─────────┼─────────────┘
          ↓
Decision / Response
          ↓
Routing
    ┌─────┴─────┐
    ↓           ↓
 Edge AI      Cloud AI
    ↓           ↓
    └─────┬─────┘
          ↓
Mobile / Web / API
          ↓
Monitoring + Evaluation
```

## 2. Skill Priority

| Area | Priority |
|---|---:|
| Real-Time AI + AI backend/systems | 30% |
| Applied AI: LLM/RAG/Agents | 25% |
| Production/deployment/observability | 20% |
| Edge + Hybrid AI | 15% |
| Mobile AI integration | 10% |

The goal is to use existing mobile engineering experience rather than spend major roadmap time relearning normal mobile application development.

---

# Month 1 - Production AI Service Foundations

## Goal

Learn how an AI model becomes a real, deployed service rather than remaining inside a notebook.

## AI fundamentals

Learn the engineering-level foundations needed to work effectively with modern models:

- Training vs inference
- Tokens and tokenization
- Embeddings
- Context windows
- Sampling basics
- Structured outputs
- Model APIs
- Classification and extraction
- Model inputs and outputs
- Confidence and probabilities at a practical level

## AI backend engineering

Learn and practice:

- Python for AI applications
- FastAPI
- Pydantic
- REST APIs
- `async` / `await`
- Request validation
- Exception handling
- Timeouts
- Configuration and environment variables
- PostgreSQL
- Redis basics
- Docker
- API authentication basics

## Real-time foundations

Understand the difference between standard request-response and streaming systems.

```text
Normal HTTP
Client → Request → Wait → Response

Streaming
Client → Request
       ← token
       ← token
       ← token
       ← token
```

Learn:

- Server-Sent Events (SSE)
- WebSockets conceptually
- Streaming LLM output
- Async model calls
- Cancellation
- Concurrency basics

## Performance from Day 1

Measure:

- Model inference latency
- End-to-end latency
- p50 latency
- p95 latency
- p99 latency
- Requests per second
- Error rate

## Project: Real-Time AI Classification & Extraction Service

```text
Incoming support message
        ↓
FastAPI
        ↓
AI
├── category
├── urgency
├── summary
├── entities
└── sentiment
        ↓
PostgreSQL
        ↓
API / streaming response
```

## Deployment milestone

Do not wait until Month 5 to deploy.

```text
Code
→ Docker image
→ cloud deployment
→ public endpoint
→ logs
```

## Mobile AI integration - secondary

Build only a thin client:

```text
Mobile
→ deployed FastAPI
→ AI result
```

Do not spend major time on visual polish.

## Month 1 outcome

> I can build, containerize, deploy, and measure an AI inference service and consume it from mobile, web, or API clients.

---

# Month 2 - Industry-Grade Production RAG

## Goal

Move beyond tutorial RAG and build a serious retrieval system with ingestion, hybrid retrieval, reranking, grounding, citations, evaluation, and production lifecycle thinking.

## Production retrieval architecture

```text
Documents
   ↓
Ingestion
   ↓
Parsing / Cleaning
   ↓
Chunking
   ↓
Metadata
   ↓
Indexing
   ↓
┌───────────────┐
│ Dense Search  │
│ BM25          │
│ Metadata      │
└───────┬───────┘
        ↓
Hybrid Retrieval
        ↓
Reranking
        ↓
Context Builder
        ↓
LLM
        ↓
Grounded Answer + Citations
```

## Ingestion

Learn:

- Document parsing
- Cleaning and normalization
- Chunking strategies
- Chunk overlap
- Metadata extraction
- Stable document IDs
- Document versioning concepts
- Deduplication
- Incremental re-indexing
- Updating and deleting documents

## Retrieval

Learn:

- Embeddings
- Cosine similarity
- pgvector
- Dense search
- BM25
- Sparse vs dense retrieval
- Hybrid retrieval
- Metadata filtering

## Query processing

Learn:

- Query rewriting
- Intent detection
- Metadata extraction from queries
- Multi-query retrieval
- Query decomposition when useful

## Reranking

Understand candidate generation and reranking:

```text
Initial retrieval
100 candidates
      ↓
filter
      ↓
20 candidates
      ↓
reranker
      ↓
5 best passages
```

Study practical reranking and the role of cross-encoder/reranker models.

## Context construction

Learn:

- Context limits
- Redundancy removal
- Chunk ordering
- Context budgeting
- Source attribution

## Grounded generation

The system must:

- Produce citations
- Prefer evidence-backed answers
- Refuse or explicitly report insufficient evidence
- Avoid inventing information when retrieval is weak

## RAG evaluation

Create a representative evaluation dataset and measure:

- Recall@K
- Precision@K
- MRR / NDCG conceptually where useful
- Answer relevance
- Groundedness
- Citation correctness
- Hallucination rate

## Project: Production Knowledge Intelligence Service

Build more than a chatbot UI:

```text
Ingestion API
Retrieval API
Answer API
Admin / re-index flow
Evaluation runner
Metrics
```

## Mobile AI integration - secondary

The mobile client should support:

- Streaming answers
- Citations
- Conversation history

## Month 2 outcome

> I can design, implement, evaluate, and deploy a production-style RAG pipeline rather than a basic vector-search demo.

---

# Month 3 - Agentic RAG + Tools + Controlled Workflows

## Goal

Extend the Month 2 RAG system into a controlled tool-using AI workflow that can retrieve internal knowledge and fetch live information from approved systems.

## Architecture

```text
                  User Query
                      ↓
                  AI Router
                      ↓
      ┌───────────────┼───────────────┐
      ↓               ↓               ↓
     RAG          Database Tool     API Tool
      ↓               ↓               ↓
Knowledge Base    Live records    External system
      └───────────────┼───────────────┘
                      ↓
              Evidence Aggregation
                      ↓
                     LLM
                      ↓
             Grounded Response
```

## Tool calling

Learn:

- Tool schemas
- Structured arguments
- Argument validation
- Tool selection
- Authentication
- Authorization
- Timeouts
- Retries
- Error responses
- Audit logging

Example tools:

```text
get_customer()
get_order()
search_inventory()
get_account_status()
create_ticket()
search_current_information()
```

## Controlled agents

Do not focus on fully autonomous agents. Learn bounded, production-oriented workflows.

```text
Router
→ approved workflow
→ approved tools
→ validation
→ result
```

Learn:

- Tool/function calling
- Workflow state
- Routing
- Planner/executor concepts
- Retries
- Bounded loops
- Human approval
- Failure recovery
- Idempotency

## RAG fallback logic

```text
Question
   ↓
Retrieve evidence
   ↓
Enough information?
   │
 ┌─┴─┐
Yes  No
 │    │
 ↓    ↓
Answer Determine required tool
          ↓
      API / DB / Search
          ↓
       Evidence
          ↓
         Answer
```

## Agent evaluation

Measure:

- Correct tool selected?
- Correct arguments?
- Unnecessary tool calls?
- Tool-call success?
- Final task success?
- Number of steps?
- Latency?
- Cost?

## Project: Agentic Knowledge & Operations System

Example request:

> "Why is customer #284's payment pending, and what does company policy say about this?"

The system should:

```text
RAG
→ retrieve company payment policy

Tool
→ retrieve customer payment status

LLM
→ combine evidence

Response
→ live status + policy + citations
```

## Mobile AI integration - secondary

The mobile client supports:

- Streamed AI output
- Citations
- Tool progress
- User confirmation for risky actions

## Month 3 outcome

> I can build a controlled tool-using AI workflow combining RAG, databases, APIs, live information, and grounded generation.

---

# Month 4 - Real-Time AI + Edge AI + Multimodal

## Goal

Develop the two main specializations beyond Applied AI: real-time event processing and practical edge/on-device inference.

## Real-time AI systems

Learn:

- Event-driven architecture
- Producers and consumers
- Queues
- Workers
- Message brokers
- Asynchronous pipelines
- Event schemas
- Retries
- Dead-letter queues
- Ordering
- Idempotency
- Backpressure

## Kafka / Redpanda

Do not become a Kafka administrator. Learn enough to build and reason about:

```text
Producer
→ Topic
→ Consumer
→ AI inference
→ Event / result
```

Start with a normal job queue first, then implement one meaningful Kafka/Redpanda pipeline.

## Edge AI foundations

Learn:

- PyTorch inference
- Model export
- ONNX concepts
- Static vs dynamic shapes
- Operators
- CPU vs GPU vs NPU
- Mobile inference runtimes

Focus on:

- ExecuTorch
- Core ML
- LiteRT

You do not need to master all three equally.

## Quantization

Learn:

- FP32
- FP16
- INT8
- INT4 concepts
- Post-training quantization (PTQ)
- Quantization-aware training (QAT) concepts
- Calibration
- Per-channel vs per-tensor quantization
- Mixed precision

Benchmark model variants:

| Model | Accuracy | Size | Latency | RAM |
|---|---:|---:|---:|---:|
| FP32 | | | | |
| FP16 | | | | |
| INT8 | | | | |

## Real-time multimodal AI

Introduce camera, audio, image, and sensor-like input where useful.

```text
Camera frames
      ↓
Local vision model
      ↓
Interesting event?
   │
   ├── No → remain local
   │
   └── Yes
          ↓
       Cloud AI
          ↓
    LLM / RAG / Tools
          ↓
        Action
```

## Mobile AI-specific skills

Learn:

- Local model execution
- Camera-to-model pipeline
- Audio-to-inference pipeline
- Background inference
- Model lifecycle
- Local model loading
- Model storage
- Device resource constraints
- Cloud escalation

## Project: Real-Time Edge-to-Cloud AI System

Example:

```text
Mobile camera
→ local detection
→ confidence decision
→ cloud escalation
→ AI reasoning
→ result / action
```

## Month 4 outcome

> I can run an optimized model on-device and integrate it into a real-time cloud AI workflow.

---

# Month 5 - Production AI Engineering, Deployment & Reliability

## Goal

Turn earlier systems into production-style services with observability, reliability, deployment discipline, and measurable performance.

## Observability

Learn and apply:

- Structured logging
- Request/correlation IDs
- Metrics
- Distributed tracing concepts
- OpenTelemetry concepts
- Model latency tracking
- External API latency
- Database latency
- LLM call tracing
- Tool-call tracing

## Production metrics

### Service

- p50 latency
- p95 latency
- p99 latency
- Throughput
- Requests per second
- Error rate

### LLM

- Time to first token
- Tokens per second
- Input tokens
- Output tokens
- Cost per request

### RAG

- Retrieval quality
- Groundedness
- Citation accuracy

### Agents

- Tool-selection accuracy
- Task success rate
- Retries
- Loop count

### Edge

- Inference latency
- Memory
- Model size
- Cold start
- Thermal degradation
- Battery impact where measurable

## Reliability

Implement:

- Timeouts
- Retries
- Exponential backoff
- Provider fallback
- Model fallback
- Circuit-breaker concepts
- Caching
- Graceful degradation
- Queue recovery
- Dead-letter handling

Example fallback chain:

```text
Primary LLM
   ↓ failure
Secondary LLM
   ↓ failure
Smaller fallback
   ↓
Graceful response
```

## Deployment

Move beyond local Docker execution.

```text
Git
 ↓
Tests
 ↓
Docker build
 ↓
Container registry
 ↓
Deployment
 ↓
Health checks
 ↓
Monitoring
```

Learn:

- Docker
- Docker Compose
- Managed container/cloud deployment
- Environment and secrets
- Database migrations
- Health/readiness endpoints
- CI/CD basics
- Staging vs production
- Release versioning
- Rollback
- Basic autoscaling concepts

## Kubernetes

Postpone deep Kubernetes. Understand what it solves and how it fits into larger systems, but do not spend significant roadmap time on cluster administration.

## Security

Cover practical production basics:

- Authentication
- Authorization
- Secrets
- Prompt injection
- Data leakage
- Tool permission boundaries
- PII handling concepts
- Rate limiting
- Audit logs

## Edge model lifecycle

Add:

- Model versioning
- Model download
- Integrity checks
- Cache management
- Rollback
- OTA model updates
- Failed update recovery

## Month 5 outcome

> I can deploy, observe, troubleshoot, secure, and improve an AI system rather than merely demonstrate one locally.

---

# Month 6 - Flagship Real-Time Hybrid AI Platform

## Goal

Do not start another course. Month 6 is:

> **BUILD → DEPLOY → TEST → BENCHMARK → DOCUMENT**

Combine the entire learning path into one serious system.

## Flagship architecture

```text
                   INPUT
        Text / Voice / Camera / Events
                     │
                     ↓
              Real-Time Gateway
                     │
                     ↓
             AI Orchestrator
                     │
       ┌─────────────┼──────────────┐
       ↓             ↓              ↓
      RAG          Tools          Models
       │             │              │
       └─────────────┼──────────────┘
                     ↓
                AI ROUTER
                     │
       ┌─────────────┴─────────────┐
       ↓                           ↓
    EDGE AI                     CLOUD AI
 Small models                Larger models
 Offline                     RAG
 Private                     Agents
 Fast                        Tool use
 Cheap                       Reasoning
       │                           │
       └─────────────┬─────────────┘
                     ↓
                 RESPONSE
                     │
          ┌──────────┼──────────┐
          ↓          ↓          ↓
        Mobile      API        Web
                     │
                     ↓
              Observability
```

## Hybrid routing

The router should consider:

- Privacy
- Latency SLA
- Internet availability
- Device performance
- RAM
- Battery
- Task complexity
- Confidence
- Cost
- Model capability
- Cloud availability

Example:

```python
if privacy_required:
    route = "local"
elif offline:
    route = "local"
elif local_confidence > threshold:
    route = "local"
elif complex_reasoning_required:
    route = "cloud"
else:
    route = "best_available"
```

## Mobile's role in the flagship

Mobile should demonstrate AI-specific capabilities rather than ordinary app development.

### Input

- Text
- Camera
- Audio where useful

### Local AI

- Local classification
- Computer vision
- Small model / SLM where appropriate

### Cloud AI

- Production RAG
- Tools
- Controlled agents
- Complex reasoning

### Real-time UX

- Streamed output
- Progress events
- Cancel operation
- Retry
- Offline state

### Hybrid

```text
Mobile device
    ↓
local inference
    ↓
router
    ↓
cloud only when required
```

## Month 6 outcome

> I can architect and ship a complete real-time hybrid AI system spanning ingestion, retrieval, tools, cloud AI, edge AI, routing, mobile integration, evaluation, deployment, and observability.

---

# 9. One Main Project Should Evolve Across All Six Months

Avoid six disconnected tutorial repositories.

```text
MONTH 1
AI Service
    ↓
MONTH 2
AI Service + Production RAG
    ↓
MONTH 3
RAG + Agent + Tools
    ↓
MONTH 4
Agentic System + Real-Time Pipeline + Edge Model
    ↓
MONTH 5
Deployment + Monitoring + Reliability
    ↓
MONTH 6
Complete Hybrid AI Platform
```

This teaches how a real AI product evolves from prototype to production.

---

# 10. Recommended Flagship Project

## Real-Time AI Support & Operations Platform

Core workflow:

```text
Incoming customer message / event
            ↓
Intent / classification
            ↓
Production RAG
            ↓
Enough information?
       ┌────┴────┐
       ↓         ↓
      Yes        No
       │          │
       │     Agent / Tools
       │     ├─ Customer API
       │     ├─ Orders
       │     ├─ Payment system
       │     └─ Search
       │          │
       └────┬─────┘
            ↓
      AI reasoning
            ↓
      response / action
            ↓
       streaming UI
```

Add mobile/edge capabilities:

```text
Mobile
   ↓
local classification
   ↓
privacy / latency-sensitive tasks local
   ↓
cloud AI for complex work
```

Add an event-driven path:

```text
Support events
      ↓
Queue / Kafka
      ↓
AI processor
      ↓
risk / priority
      ↓
notification / action
```

This project can demonstrate almost the entire target skill profile.

---

# 11. Weekly Learning Workflow

Use the same learning cycle throughout the six months.

## Phase 1 - Understand

Before implementation, answer:

- What problem are we solving?
- Why is this component needed?
- How does it work?
- What alternatives exist?
- What trade-offs are involved?

## Phase 2 - Small experiment

Implement the concept in isolation before putting it into the main architecture.

Example:

```text
Learn reranking
→ tiny retrieval/reranking experiment
```

## Phase 3 - Integrate

Add the concept to the main system.

## Phase 4 - Measure

Ask:

- Did quality improve?
- Did latency improve or regress?
- Did cost change?
- What failed?
- What should be changed?

## Phase 5 - Document

For each major feature document:

- Architecture
- Why it was chosen
- Implementation
- Metrics
- Trade-offs
- Failures
- Lessons learned

This keeps AI coding tools from replacing understanding.

---

# 12. Depth Targets

## Learn strongly

- Python AI engineering
- FastAPI
- Async systems
- Production RAG
- Retrieval evaluation
- LLM/tool calling
- Controlled agents
- AI workflow design
- Docker
- Redis/PostgreSQL
- Deployment
- Logging and monitoring
- Latency/performance
- Failure handling
- PyTorch inference
- Quantization
- Local/cloud architecture

## Intermediate

- Kafka/Redpanda
- Core ML
- LiteRT
- ExecuTorch
- Multimodal pipelines
- CI/CD
- OpenTelemetry
- Cloud architecture

## Understand conceptually

- Kubernetes
- QAT in depth
- Sophisticated distributed inference
- GPU serving clusters
- Advanced model training

---

# 13. Explicitly Postpone

Postpone topics that dilute the six-month transition:

- LLM pretraining
- Distributed model training
- Advanced CUDA
- Custom kernels
- Kubernetes administration
- Deep Terraform
- Advanced SRE
- Deep computer-vision research
- Robotics
- TinyML
- GANs
- Reinforcement learning
- Learning many agent frameworks
- Learning many vector databases
- Implementing transformers from scratch

These are valuable topics, but they are not the highest-return priorities for this transition.

---

# 14. Portfolio at the End

## Portfolio 1 - Production Agentic RAG

Proves:

- RAG
- Hybrid retrieval
- Reranking
- Citations
- Evaluation
- Tools
- Controlled workflows
- APIs

## Portfolio 2 - Real-Time Edge AI System

Proves:

- PyTorch
- Quantization
- Device inference
- Real-time processing
- Profiling
- Mobile integration

## Portfolio 3 - Flagship Hybrid AI Platform

Proves:

- Real-time systems
- Applied AI
- Production RAG
- Agents
- Tools
- Edge
- Cloud
- Mobile
- Deployment
- Monitoring
- Evaluation
- Reliability

---

# 15. End-of-Six-Month Skill Matrix

| Skill | Target Level |
|---|---|
| Python AI engineering | Job-ready |
| FastAPI AI services | Job-ready |
| LLM application engineering | Job-ready |
| Production RAG | Job-ready |
| Tool calling | Job-ready |
| Controlled AI agents/workflows | Job-ready |
| RAG/agent evaluation | Job-ready |
| Docker | Job-ready |
| Production deployment | Strong practical |
| PostgreSQL/Redis | Strong practical |
| Async/streaming AI | Strong practical |
| Event-driven AI | Intermediate |
| Kafka/Redpanda | Intermediate |
| Observability | Strong practical |
| CI/CD | Intermediate |
| PyTorch inference | Strong practical |
| Quantization | Intermediate / strong practical |
| Edge AI deployment | Intermediate |
| Hybrid AI architecture | Strong practical |
| Mobile AI integration | Strong practical |
| Kubernetes | Foundational only |
| Large-scale MLOps | Foundational only |
| Training large models from scratch | Not a target |

---

# 16. Final Capability Test

At the end of the roadmap, you should be able to receive a requirement such as:

> "We receive live customer and application events. Use AI to classify them immediately. Search our internal knowledge base using hybrid retrieval and reranking. If the required information is not available, call approved internal APIs/tools. Stream the answer to the user. Run privacy-sensitive or latency-sensitive inference locally on the phone where appropriate and fall back to cloud models for complex reasoning. Deploy the whole system, monitor latency and quality, and handle provider/model failures."

You should be able to reason through and implement:

```text
Architecture
↓
Models
↓
RAG
↓
Tools
↓
Streaming
↓
Edge
↓
Hybrid routing
↓
Backend
↓
Deployment
↓
Observability
↓
Evaluation
↓
Mobile integration
```

## North Star

**Build production, real-time AI systems.**

Then specialize in:

**Edge AI + Hybrid AI**, with **mobile AI integration as a strong secondary capability and deployment surface.**
