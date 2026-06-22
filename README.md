# Semantic AI Context Gateway

> A production-grade AI gateway that reduces LLM token usage by up to **96.5%** using
> Redis Stack HNSW vector similarity search — with real-time SSE streaming, per-user
> session isolation, Redis rate limiting, and token cost metrics API.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)
![Redis Stack](https://img.shields.io/badge/Redis-Stack-red)
![Groq](https://img.shields.io/badge/LLM-Groq%20LLaMA%203.1-orange)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## What It Does

The gateway sits between your application and the LLM API. Every prompt passes
through a semantic compaction pipeline — sentences are split, encoded into 384-dimension
vectors using a local AI model, and checked against a Redis HNSW vector index.
Near-duplicate sentences (cosine distance ≤ 0.05) are dropped before forwarding to Groq.
The LLM response streams back word-by-word via Server-Sent Events (SSE).

**Result:** Less tokens sent → lower cost → faster responses → same quality answers.

---

## Benchmark Results (Real Load Test)

Tested across **1,750 requests** with up to **200 concurrent connections**.

| Tier | Concurrency | Requests | Success | Compacted | Avg Savings | Throughput |
|---|---|---|---|---|---|---|
| Warm-up | 10 | 50 | 100% | 82% | 71.8% | 9.23 req/s |
| Medium | 50 | 200 | 100% | 82% | 71.8% | 7.57 req/s |
| Heavy | 100 | 500 | 98% | 95.3% | 92.8% | 7.89 req/s |
| Stress | 200 | 1,000 | 100% | 97.7% | 96.5% | 6.62 req/s |

### Latency (Semantic Compaction Pipeline Only)

| Metric | Tier 1 (10c) | Tier 4 (200c) |
|---|---|---|
| Min | 399 ms | 2,476 ms |
| Avg | 1,042 ms | 5,251 ms |
| P50 | 610 ms | 5,280 ms |
| P95 | 1,910 ms | 8,209 ms |
| P99 | 1,929 ms | 10,079 ms |

### Token & Cost Savings (Across Full Test Run)

| Metric | Value |
|---|---|
| Total requests processed | 1,740 |
| Total chars saved | 186,219 |
| Estimated tokens saved | ~46,553 |
| Compaction rate | **94.8%** |
| Average savings % | **91.88%** |
| Estimated cost saved | **$0.0466 USD** |
| Rate limiter blocked | 10 excess requests  |
| Failed requests | 0  |

> Cost reference: GPT-3.5-turbo pricing ($0.001/1K tokens).
> At GPT-4 pricing ($0.03/1K tokens), the same run saves **$1.39 USD**.

---

## Architecture

```
User Request (multi-sentence prompt)
         │
         ▼
┌─────────────────────────────────┐
│     Rate Limiter                │  Redis INCR + EXPIRE
│     50 req / 60s / session      │  → HTTP 429 if exceeded
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│     Sentence Splitter           │  look-behind regex
│     "A. B. C." → [A, B, C]     │  punctuation preserved
└─────────────────────────────────┘
         │
         ▼ (per sentence)
┌─────────────────────────────────┐
│     Embedding Model             │  all-MiniLM-L6-v2
│     sentence → 384-dim vector   │  runs in thread executor
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│     Redis HNSW Vector Search    │  cosine distance KNN
│     find nearest neighbour      │  dialect 2, FLOAT32
└─────────────────────────────────┘
         │
    ┌────┴────┐
    │         │
distance    distance
≤ 0.05      > 0.05
    │         │
  DROP       KEEP + CACHE
    │         │
    └────┬────┘
         │
         ▼
┌─────────────────────────────────┐
│     Groq LLaMA 3.1 8B           │  httpx AsyncClient
│     compacted prompt →          │  streaming=True
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│     SSE Token Stream            │  word by word
│     → User                     │  text/event-stream
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│     Metrics Logger              │  tiktoken token count
│     Redis HASH  (24h TTL)       │  cost saved in USD
└─────────────────────────────────┘
```

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| API Framework | FastAPI 0.110 | Async endpoints, SSE streaming |
| Vector Database | Redis Stack HNSW | Sentence similarity search |
| Embedding Model | all-MiniLM-L6-v2 (384-dim) | Sentence vector encoding |
| LLM Provider | Groq LLaMA 3.1 8B Instant | Fast inference, free tier |
| Token Counter | tiktoken cl100k_base | Accurate cost calculation |
| HTTP Client | httpx AsyncClient | Non-blocking LLM streaming proxy |
| Containerisation | Docker + Compose | One-command deployment |
| Settings | pydantic-settings | .env config management |

---

## Quick Start

### Option A — Docker (recommended, one command)

```bash
git clone https://github.com/abhijitkk358/semantic_api_gateway
cd semantic-api-gateway

# Add your free Groq API key (console.groq.com)
cp .env.example .env
nano .env

# Start Redis + Gateway together
docker compose up
```

Open `http://localhost:8000/docs` for interactive Swagger UI.

### Option B — Local Development

```bash
# Terminal 1 — Redis only
docker compose up semantic-redis-cache

# Terminal 2 — FastAPI gateway
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/v1/chat/completions` | Main — compact prompt + stream Groq response |
| GET | `/v1/metrics` | Token savings, cost stats, session leaderboard |
| DELETE | `/v1/metrics/reset` | Wipe metrics (dev/testing only) |
| POST | `/v1/stream/completions` | Direct streaming endpoint (internal) |
| GET | `/v1/stream/health` | Groq API key + model status |
| GET | `/health` | Full system health — Redis, model, version |

---

## curl Examples

### 1. Streaming request — duplicate sentence gets dropped

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Machine learning is a subset of AI. Explain neural networks. Machine learning is a subset of AI.",
    "stream": true,
    "session_id": "user123"
  }' --no-buffer
```

**Expected SSE output:**
```
data: {"type": "metadata", "gateway_status": "COMPACTED", "savings_percentage": 52.3, "original_chars": 98, "transmitted_chars": 47}

data: {"type": "token", "content": "Neural"}
data: {"type": "token", "content": " networks"}
data: {"type": "token", "content": " are"}
...
data: {"type": "done", "tokens_used": 12, "tokens_saved": 13, "llm_latency_ms": 834.2}
data: [DONE]
```

### 2. Non-streaming — compaction report only

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is deep learning?", "stream": false, "session_id": "user123"}' \
  | python3 -m json.tool
```

**Expected output:**
```json
{
    "request_id": "a1b2c3d4",
    "session_id": "user123",
    "gateway_status": "NO_REDUNDANCY",
    "original_chars": 22,
    "transmitted_chars": 22,
    "savings_percentage": 0.0,
    "compaction_latency_ms": 18.4,
    "final_prompt_sent": "What is deep learning?"
}
```

### 3. Token savings metrics

```bash
curl -s "http://localhost:8000/v1/metrics?sessions=true" | python3 -m json.tool
```

**Expected output:**
```json
{
    "total_requests": 1740,
    "total_tokens_used": 2645,
    "total_tokens_saved": 33660,
    "total_chars_saved": 186219,
    "average_savings_pct": 91.88,
    "estimated_cost_saved_display": "$0.0337",
    "total_compacted_requests": 1649,
    "top_sessions": [
        {
            "session_id": "loadtest_session_1",
            "requests": 90,
            "tokens_saved": 1834,
            "cost_saved": "$0.0018"
        }
    ]
}
```

### 4. Rate limiter test

```bash
# Fire 55 requests — after 50 you get HTTP 429
for i in {1..55}; do
  curl -s -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Test.", "stream": false, "session_id": "testuser"}' \
    | python3 -m json.tool
done
```

---

## How It Works

1. **Split** — Prompt split into sentences using look-behind regex. Punctuation stays attached to preserve vector meaning.

2. **Embed** — Each sentence encoded into 384-dim FLOAT32 vector by `all-MiniLM-L6-v2` model in a thread executor (keeps async event loop free).

3. **Search** — Redis HNSW index finds nearest cached sentence using cosine distance KNN (Dialect 2). Session-scoped keys prevent cross-user cache collisions.

4. **Filter** — Sentences with cosine distance ≤ 0.05 (≥ 95% similar) are dropped. Sentences under 15 characters always pass through.

5. **Stream** — Compacted prompt forwarded to Groq via httpx async streaming. Each token yielded as SSE event the moment Groq generates it.

6. **Metrics** — tiktoken counts tokens used vs saved. Results stored in Redis HASH with 24h TTL. Accessible via `/v1/metrics`.

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | Required. Free at console.groq.com |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model name |
| `GROQ_BASE_URL` | `https://api.groq.com/openai/v1/` | OpenAI-compatible endpoint |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection string |

---

## Rate Limiting

Every session gets **50 requests per 60 seconds** via Redis sliding window counter.
Exceeding the limit returns HTTP 429 with `Retry-After` header.
Each `session_id` has an independent counter — users never block each other.

```json
{
    "error": "rate_limit_exceeded",
    "message": "Too many requests. You have sent 51 requests in 60s. Limit is 50.",
    "retry_after": 43,
    "session_id": "user123"
}
```

---

## Load Testing

```bash
# Install httpx (already in requirements.txt)
python load_test.py
```

Runs 4 tiers: 10 → 50 → 100 → 200 concurrent connections across 1,750 total requests.
Outputs real CV-ready stats: throughput, P95/P99 latency, tokens saved, cost saved.

---

## Project Structure

```
semantic_api_gateway/
├── .env                    # Your secrets (never commit)
├── .env.example            # Template for new developers
├── .gitignore              # Excludes .env, venv/, __pycache__/
├── .dockerignore           # Excludes venv/, .env from Docker image
├── Dockerfile              # Container build — python:3.12-slim
├── docker-compose.yml      # Redis Stack + Gateway services
├── requirements.txt        # Python dependencies
├── load_test.py            # Async load tester (4 tiers, 1750 requests)
└── app/
    ├── __init__.py
    ├── config.py           # Pydantic settings (reads .env)
    ├── main.py             # FastAPI app, lifespan, main endpoint
    └── routers/
        ├── __init__.py
        ├── stream.py       # Groq SSE streaming router
        ├── ratelimit.py    # Redis sliding window rate limiter
        └── metrics.py      # Token savings metrics router
```

---

## License

MIT License — free to use, modify, and distribute.
