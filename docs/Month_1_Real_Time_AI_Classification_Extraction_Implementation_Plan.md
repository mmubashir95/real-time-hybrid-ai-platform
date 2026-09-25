# Month 1 Implementation Plan — Real-Time AI Classification & Extraction Service

## Month 1 Goal

Build the first working version of a real AI backend system that can:

```text
Incoming support message
        ↓
      FastAPI
        ↓
        AI
        ↓
 category
 urgency
 summary
 entities
 sentiment
        ↓
   PostgreSQL
        ↓
API / streaming response
```

The objective is to understand how an AI capability becomes a real deployed service rather than remaining only inside a notebook or isolated script.

---

## Phase 0 — Understand the System Before Coding

Before implementation, understand what each part of the system does.

```text
Client
  ↓
FastAPI
  ↓
Validation
  ↓
AI
  ↓
Structured result
  ↓
Database
  ↓
Response
```

### Learn / Review

- Request-response flow
- Classification vs extraction
- Structured outputs
- Model inputs and outputs
- Where FastAPI fits in an AI system

### Exit Condition

You can explain the complete Month 1 architecture in your own words before writing code.

---

## Phase 1 — Create the Basic FastAPI Service

Build the smallest working backend.

```text
Client
  ↓
POST /analyze
  ↓
FastAPI
  ↓
JSON response
```

Example request:

```json
{
  "message": "I was charged twice for my subscription."
}
```

Initial response:

```json
{
  "status": "received"
}
```

### Learn

- FastAPI basics
- REST APIs
- Route handlers
- Request / response flow

### Exit Condition

You can run FastAPI locally and successfully call `POST /analyze`.

---

## Phase 2 — Add Pydantic and Structured Contracts

Define exactly what the service accepts and returns.

```text
HTTP Request
    ↓
Pydantic validation
    ↓
FastAPI
    ↓
Pydantic response
```

Suggested contracts:

```text
AnalysisRequest
- message

AnalysisResult
- category
- urgency
- summary
- entities
- sentiment
```

### Learn

- Pydantic models
- Required and optional fields
- Request validation
- Response validation
- Structured outputs

### Exit Condition

Invalid requests are rejected and valid requests return a predictable response schema.

---

## Phase 3 — Create a Mock AI Layer

Before connecting a real model, create an AI service layer that returns fake but realistic results.

```text
FastAPI
   ↓
AI Service
   ↓
Mock classification / extraction
   ↓
Structured result
```

Example:

```json
{
  "category": "billing",
  "urgency": "high",
  "summary": "Customer reports duplicate billing.",
  "entities": {},
  "sentiment": "negative"
}
```

### Why This Phase Exists

This allows you to understand the system architecture separately from model behavior.

### Learn

- Separation of concerns
- AI input / output contracts
- Classification vs extraction
- Service-layer design

### Exit Condition

`POST /analyze` behaves like a real AI endpoint even though the result is still mocked.

---

## Phase 4 — Connect a Real Hosted Model API

Replace the mock AI implementation with a real hosted model API.

```text
Support message
      ↓
FastAPI
      ↓
Model API
      ↓
AI analysis
      ↓
Structured result
```

The model should return:

```text
category
urgency
summary
entities
sentiment
```

### Learn

- Model APIs
- Model inputs and outputs
- Prompt / instruction design
- Structured model responses
- Practical confidence / probability concepts where applicable

### Exit Condition

Real support messages produce valid classification and extraction results.

---

## Phase 5 — Make the AI Integration Reliable

Once the happy path works, handle failure cases.

```text
Model request
   ↓
success?
 ┌─┴─┐
yes  no
 ↓    ↓
result timeout / error
```

### Implement

- Exception handling
- Model API timeouts
- Malformed response handling
- Environment variables
- Configuration
- API key / secret handling

### Exit Condition

The API fails gracefully instead of crashing when the model provider fails or returns invalid output.

---

## Phase 6 — Add Async/Await and Concurrency Basics

Use asynchronous I/O correctly for model calls and other I/O operations.

```text
Request A → model call ───────→ response
                 ↑
Request B → model call ───────→ response
```

### Learn

- `async` / `await`
- Blocking vs non-blocking operations
- Async model calls
- Basic concurrency
- Cancellation concepts

### Exit Condition

You can explain why the endpoint is asynchronous, not just write `async def`.

---

## Phase 7 — Add PostgreSQL Persistence

Store incoming requests and AI results.

```text
Incoming message
      ↓
AI processing
      ↓
PostgreSQL
      ↓
stored request + result
```

Suggested data:

```text
request
- message

result
- category
- urgency
- summary
- entities
- sentiment

metadata
- created_at
- latency
```

### Learn

- PostgreSQL basics
- Persistence
- Database models / records
- Saving AI outputs
- Retrieving stored results

### Exit Condition

You can submit a message, analyze it, and retrieve the stored result from the database.

---

## Phase 8 — Add Real-Time / Streaming Behavior

First keep the normal request-response endpoint.

```text
Normal HTTP

Client → Request → Wait → Complete Response
```

Then add streaming.

```text
Streaming

Client → Request
       ← event
       ← event
       ← event
       ← completion
```

Possible SSE events:

```text
processing_started
classification_complete
extraction_complete
saved
completed
```

### Learn

- Server-Sent Events (SSE)
- Streaming responses
- Conceptual WebSockets
- Async model calls
- Cancellation
- Concurrency

### Exit Condition

You can demonstrate and explain both normal request-response and streaming behavior.

---

## Phase 9 — Add Performance Measurement

Measure the system instead of only checking whether it works.

### Measure

- Model inference latency
- End-to-end latency
- p50 latency
- p95 latency
- p99 latency
- Requests per second
- Error rate

Example:

```text
Model call:      720 ms
Database:         30 ms
Other backend:    20 ms
------------------------
Total:           770 ms
```

### Exit Condition

You can report real performance numbers for the service.

---

## Phase 10 — Dockerize the Application

Package the service and its dependencies.

```text
FastAPI
+
dependencies
+
configuration
        ↓
Docker image
```

Possible local setup:

```text
Docker Compose
├── FastAPI
├── PostgreSQL
└── Redis if / when useful
```

### Learn

- Docker
- Dockerfile
- Docker Compose basics
- Environment configuration

### Exit Condition

The project can be started in a reproducible environment without manually rebuilding the Python setup.

---

## Phase 11 — Deploy to the Cloud

Deploy the service instead of leaving it local.

```text
Code
→ Docker image
→ cloud deployment
→ public endpoint
→ logs
```

### Implement

- Build Docker image
- Deploy to a managed cloud/container platform
- Expose a public endpoint
- Verify environment variables / secrets
- Inspect logs

### Exit Condition

The API is accessible outside your local machine and you can inspect logs when something fails.

---

## Phase 12 — Add the Thin Mobile Client

Use mobile only as a thin client in Month 1.

```text
Mobile App
    ↓
Deployed FastAPI
    ↓
AI analysis
    ↓
Mobile App
```

Suggested UI:

```text
┌───────────────────────────────┐
│ Support message               │
│                               │
│ [...........................]  │
│                               │
│         Analyze               │
│                               │
│ Category: Billing             │
│ Urgency: High                 │
│ Sentiment: Negative           │
│ Summary: ...                  │
│ Entities: ...                 │
└───────────────────────────────┘
```

### Important

Do not spend major time on visual polish. The mobile app is only there to prove that a real client can consume the deployed AI service.

### Exit Condition

A mobile app can send a real request to the deployed FastAPI service and display the AI result.

---

# Final Month 1 Architecture

```text
                 Mobile / API Client
                         ↓
                    FastAPI
                         ↓
                Pydantic Validation
                         ↓
                  AI Service Layer
                         ↓
                  Hosted AI Model
                         ↓
        Classification + Extraction
                         ↓
                Structured Result
                         ↓
                   PostgreSQL
                         ↓
                API / SSE Response

                         +

              Errors / Timeouts
              Async / Concurrency
              Authentication basics
              Performance Metrics
              Docker
              Cloud Deployment
              Logs
```

---

# Month 1 Completion Outcome

By the end of Month 1, you should be able to say:

> I can build, containerize, deploy, and measure an AI inference service and consume it from mobile, web, or API clients.

---

# Recommended Learning Workflow for Every Phase

For each phase, use the same cycle:

```text
Understand
   ↓
Small Experiment
   ↓
Integrate
   ↓
Measure
   ↓
Document
```

## 1. Understand

Before implementation, answer:

- What problem are we solving?
- Why is this component needed?
- How does it work?
- What alternatives exist?
- What trade-offs are involved?

## 2. Small Experiment

Implement the concept in isolation first.

## 3. Integrate

Add the concept into the main Month 1 project.

## 4. Measure

Check:

- Did it work correctly?
- Did latency improve or regress?
- Did cost change?
- What failed?
- What should be changed?

## 5. Document

Record:

- Architecture
- Why the approach was chosen
- Implementation notes
- Metrics
- Trade-offs
- Failures
- Lessons learned

---

# Month 1 Phase Summary

```text
Phase 0  → Understand the architecture
Phase 1  → Basic FastAPI service
Phase 2  → Pydantic + structured contracts
Phase 3  → Mock AI layer
Phase 4  → Real hosted model API
Phase 5  → Error handling + reliability
Phase 6  → Async / await + concurrency
Phase 7  → PostgreSQL persistence
Phase 8  → SSE / streaming
Phase 9  → Performance measurement
Phase 10 → Docker
Phase 11 → Cloud deployment
Phase 12 → Thin mobile client
```

The key rule is: **do not jump ahead. Complete and understand each phase before moving to the next one.**
