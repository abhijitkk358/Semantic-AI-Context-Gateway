"""
load_test.py — Semantic AI Context Gateway Heavy Load Tester
=============================================================
Run from project root:
    python load_test.py

Generates real benchmark stats:
- Concurrent requests (10, 50, 100, 200)
- Response times (min, max, avg, p50, p95, p99)
- Token savings across sessions
- Cost saved in USD
- Compaction rate %
- Throughput (req/sec)
- Rate limiter validation

Results saved to: load_test_results.json + printed to terminal
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import httpx

# ─── Config ──────────────────────────────────────────────────────────────────
GATEWAY_URL    = "http://localhost:8000"
TIMEOUT        = 30.0   # seconds per request
COST_PER_1K    = 0.001  # USD per 1000 tokens (gpt-3.5 reference pricing)

# ─── Test prompts — mix of long/short, duplicates, unique sentences ───────────
PROMPTS = [
    # Long with embedded duplicates — high compaction expected
    "Machine learning is a subset of artificial intelligence. "
    "It enables systems to learn from data. "
    "Machine learning is a subset of artificial intelligence. "
    "Deep learning uses neural networks with many layers.",

    "Python is a high-level programming language. "
    "It is widely used in data science and web development. "
    "Python is a high-level programming language. "
    "FastAPI is a modern web framework for building APIs.",

    "Redis is an in-memory data structure store. "
    "It supports various data structures like strings and hashes. "
    "Redis is an in-memory data structure store. "
    "Redis Stack adds vector search capabilities to Redis.",

    "Neural networks are inspired by the human brain. "
    "They consist of layers of interconnected nodes. "
    "Neural networks are inspired by the human brain. "
    "Convolutional neural networks are used for image recognition.",

    "Docker is a platform for containerising applications. "
    "It allows developers to package apps with their dependencies. "
    "Docker is a platform for containerising applications. "
    "Docker Compose orchestrates multi-container applications.",

    # Medium prompts with partial duplicates
    "FastAPI uses Python type hints for request validation. "
    "It automatically generates OpenAPI documentation. "
    "FastAPI uses Python type hints for request validation.",

    "Vector embeddings represent text as numerical arrays. "
    "Similar sentences produce vectors that are close in space. "
    "Cosine similarity measures the angle between two vectors.",

    "Rate limiting protects APIs from abuse and DDoS attacks. "
    "A sliding window algorithm tracks requests over time. "
    "Rate limiting protects APIs from abuse and DDoS attacks.",

    # Short unique prompts — should never be dropped
    "What is artificial intelligence?",
    "Explain the transformer architecture briefly.",
    "How does gradient descent work?",
    "What are the benefits of microservices?",
    "Define cosine similarity in simple terms.",
    "What is the difference between SQL and NoSQL?",
    "How does Redis handle persistence?",
    "What is an API gateway used for?",
]


# ─── Result dataclass ─────────────────────────────────────────────────────────
@dataclass
class RequestResult:
    request_id:        str
    session_id:        str
    success:           bool
    status_code:       int
    latency_ms:        float
    gateway_status:    str   = "UNKNOWN"
    original_chars:    int   = 0
    transmitted_chars: int   = 0
    savings_pct:       float = 0.0
    error:             str   = ""


@dataclass
class BenchmarkResult:
    label:              str
    concurrency:        int
    total_requests:     int
    successful:         int
    failed:             int
    rate_limited:       int
    compacted:          int
    no_redundancy:      int
    total_chars_saved:  int
    avg_savings_pct:    float
    throughput_rps:     float
    latency_min_ms:     float
    latency_max_ms:     float
    latency_avg_ms:     float
    latency_p50_ms:     float
    latency_p95_ms:     float
    latency_p99_ms:     float
    total_duration_sec: float
    estimated_tokens_saved: int   = 0
    estimated_cost_saved_usd: float = 0.0


# ─── Single request ───────────────────────────────────────────────────────────
async def send_request(
    client: httpx.AsyncClient,
    prompt: str,
    session_id: str,
) -> RequestResult:
    req_id = str(uuid.uuid4())[:8]
    t0 = time.perf_counter()

    try:
        resp = await client.post(
            f"{GATEWAY_URL}/v1/chat/completions",
            json={
                "prompt":     prompt,
                "stream":     False,   # non-streaming for clean latency measurement
                "session_id": session_id,
                "temperature": 0.7,
            },
            timeout=TIMEOUT,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if resp.status_code == 429:
            return RequestResult(
                request_id=req_id, session_id=session_id,
                success=False, status_code=429,
                latency_ms=latency_ms, gateway_status="RATE_LIMITED",
            )

        if resp.status_code != 200:
            return RequestResult(
                request_id=req_id, session_id=session_id,
                success=False, status_code=resp.status_code,
                latency_ms=latency_ms, error=resp.text[:200],
            )

        data = resp.json()
        return RequestResult(
            request_id=req_id,
            session_id=session_id,
            success=True,
            status_code=200,
            latency_ms=latency_ms,
            gateway_status=data.get("gateway_status", "UNKNOWN"),
            original_chars=data.get("original_chars", 0),
            transmitted_chars=data.get("transmitted_chars", 0),
            savings_pct=data.get("savings_percentage", 0.0),
        )

    except httpx.TimeoutException:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return RequestResult(
            request_id=req_id, session_id=session_id,
            success=False, status_code=0,
            latency_ms=latency_ms, error="TIMEOUT",
        )
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return RequestResult(
            request_id=req_id, session_id=session_id,
            success=False, status_code=0,
            latency_ms=latency_ms, error=str(e)[:200],
        )


# ─── Run one benchmark tier ───────────────────────────────────────────────────
async def run_benchmark(
    label: str,
    concurrency: int,
    total_requests: int,
) -> BenchmarkResult:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Concurrency: {concurrency}  |  Total requests: {total_requests}")
    print(f"{'='*60}")

    results: List[RequestResult] = []
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_request(prompt: str, session_id: str) -> RequestResult:
        async with semaphore:
            return await send_request(client, prompt, session_id)

    t_start = time.perf_counter()

    async with httpx.AsyncClient() as client:
        tasks = []
        for i in range(total_requests):
            prompt     = PROMPTS[i % len(PROMPTS)]
            # Spread across sessions to test session isolation
            # Use fewer sessions than requests to trigger duplicates
            session_id = f"loadtest_session_{i % max(1, concurrency // 5)}"
            tasks.append(bounded_request(prompt, session_id))

        # Run all tasks with progress reporting every 50 requests
        completed = 0
        chunk_size = min(50, total_requests)
        for i in range(0, len(tasks), chunk_size):
            chunk = tasks[i:i + chunk_size]
            chunk_results = await asyncio.gather(*chunk)
            results.extend(chunk_results)
            completed += len(chunk_results)
            success_count = sum(1 for r in results if r.success)
            print(f"  Progress: {completed}/{total_requests} "
                  f"({success_count} ok, "
                  f"{sum(1 for r in results if r.status_code == 429)} rate-limited)")

    total_duration = time.perf_counter() - t_start

    # ── Compute stats ─────────────────────────────────────────────────────────
    successful   = [r for r in results if r.success]
    failed       = [r for r in results if not r.success and r.status_code != 429]
    rate_limited = [r for r in results if r.status_code == 429]
    compacted    = [r for r in successful if r.gateway_status == "COMPACTED"]
    no_redundancy= [r for r in successful if r.gateway_status == "NO_REDUNDANCY"]

    latencies = [r.latency_ms for r in successful] or [0]
    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)

    def percentile(data, pct):
        if not data:
            return 0.0
        idx = int(len(data) * pct / 100)
        return data[min(idx, len(data) - 1)]

    total_chars_saved = sum(
        r.original_chars - r.transmitted_chars
        for r in successful
        if r.original_chars > r.transmitted_chars
    )
    avg_savings = (
        statistics.mean(r.savings_pct for r in successful)
        if successful else 0.0
    )
    # Estimate tokens: ~4 chars per token average
    tokens_saved = total_chars_saved // 4
    cost_saved   = (tokens_saved / 1000.0) * COST_PER_1K

    bench = BenchmarkResult(
        label=label,
        concurrency=concurrency,
        total_requests=total_requests,
        successful=len(successful),
        failed=len(failed),
        rate_limited=len(rate_limited),
        compacted=len(compacted),
        no_redundancy=len(no_redundancy),
        total_chars_saved=total_chars_saved,
        avg_savings_pct=round(avg_savings, 2),
        throughput_rps=round(len(successful) / total_duration, 2),
        latency_min_ms=round(min(latencies), 2),
        latency_max_ms=round(max(latencies), 2),
        latency_avg_ms=round(statistics.mean(latencies), 2),
        latency_p50_ms=round(percentile(latencies_sorted, 50), 2),
        latency_p95_ms=round(percentile(latencies_sorted, 95), 2),
        latency_p99_ms=round(percentile(latencies_sorted, 99), 2),
        total_duration_sec=round(total_duration, 2),
        estimated_tokens_saved=tokens_saved,
        estimated_cost_saved_usd=round(cost_saved, 6),
    )

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n  ✅ Successful      : {bench.successful} / {bench.total_requests}")
    print(f"  ❌ Failed          : {bench.failed}")
    print(f"  🚫 Rate Limited   : {bench.rate_limited}")
    print(f"  📦 Compacted      : {bench.compacted} ({round(bench.compacted/max(1,bench.successful)*100,1)}%)")
    print(f"  🔄 No Redundancy  : {bench.no_redundancy}")
    print(f"\n  ⏱  Latency (ms)")
    print(f"     Min    : {bench.latency_min_ms}")
    print(f"     Avg    : {bench.latency_avg_ms}")
    print(f"     P50    : {bench.latency_p50_ms}")
    print(f"     P95    : {bench.latency_p95_ms}")
    print(f"     P99    : {bench.latency_max_ms}")
    print(f"     Max    : {bench.latency_max_ms}")
    print(f"\n  🚀 Throughput     : {bench.throughput_rps} req/sec")
    print(f"  ⏳ Total Duration : {bench.total_duration_sec}s")
    print(f"\n  💾 Chars Saved    : {bench.total_chars_saved:,}")
    print(f"  🪙 Tokens Saved   : ~{bench.estimated_tokens_saved:,}")
    print(f"  💰 Cost Saved     : ${bench.estimated_cost_saved_usd:.6f}")
    print(f"  📉 Avg Savings    : {bench.avg_savings_pct}%")

    return bench


# ─── Health check ─────────────────────────────────────────────────────────────
async def check_health() -> bool:
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{GATEWAY_URL}/health", timeout=5.0)
            data = resp.json()
            print(f"  Gateway version : {data.get('version', 'unknown')}")
            print(f"  Redis           : {data.get('redis', False)}")
            print(f"  Model loaded    : {data.get('model_loaded', False)}")
            print(f"  API key set     : {data.get('api_key_set', False)}")
            print(f"  Threshold       : {data.get('threshold', 'unknown')}")
            return (
                data.get("redis", False) and
                data.get("model_loaded", False) and
                data.get("api_key_set", False)
            )
    except Exception as e:
        print(f"  Health check failed: {e}")
        return False


# ─── Main ─────────────────────────────────────────────────────────────────────
async def main():
    print("\n" + "="*60)
    print("  SEMANTIC AI CONTEXT GATEWAY — LOAD TEST")
    print("="*60)

    print("\n[1] Health Check ...")
    healthy = await check_health()
    if not healthy:
        print("\n❌ Gateway not healthy. Start it first:")
        print("   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")
        return

    print("\n✅ Gateway healthy. Starting load tests...\n")
    print("NOTE: Using stream=false for accurate latency measurement.")
    print("      Rate limiter is 50 req/60s — tests above that will")
    print("      see 429s which is EXPECTED and proves it works.\n")

    all_results = []

    # ── Tier 1 — Warm up (10 concurrent) ─────────────────────────────────────
    r1 = await run_benchmark(
        label="TIER 1 — Warm-up (10 concurrent, 50 requests)",
        concurrency=10,
        total_requests=50,
    )
    all_results.append(r1)
    await asyncio.sleep(2)

    # ── Tier 2 — Medium load (50 concurrent) ─────────────────────────────────
    r2 = await run_benchmark(
        label="TIER 2 — Medium Load (50 concurrent, 200 requests)",
        concurrency=50,
        total_requests=200,
    )
    all_results.append(r2)
    await asyncio.sleep(2)

    # ── Tier 3 — Heavy load (100 concurrent) ─────────────────────────────────
    r3 = await run_benchmark(
        label="TIER 3 — Heavy Load (100 concurrent, 500 requests)",
        concurrency=100,
        total_requests=500,
    )
    all_results.append(r3)
    await asyncio.sleep(2)

    # ── Tier 4 — Stress test (200 concurrent) ────────────────────────────────
    r4 = await run_benchmark(
        label="TIER 4 — Stress Test (200 concurrent, 1000 requests)",
        concurrency=200,
        total_requests=1000,
    )
    all_results.append(r4)

    # ── Pull final metrics from gateway ──────────────────────────────────────
    print("\n\n" + "="*60)
    print("  FINAL GATEWAY METRICS")
    print("="*60)
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{GATEWAY_URL}/v1/metrics?sessions=true&logs=false",
                timeout=10.0,
            )
            metrics = resp.json()
            print(f"\n  Total requests processed : {metrics.get('total_requests', 0):,}")
            print(f"  Total tokens used        : {metrics.get('total_tokens_used', 0):,}")
            print(f"  Total tokens saved       : {metrics.get('total_tokens_saved', 0):,}")
            print(f"  Total chars saved        : {metrics.get('total_chars_saved', 0):,}")
            print(f"  Avg savings %            : {metrics.get('average_savings_pct', 0):.2f}%")
            print(f"  Cost saved (USD)         : {metrics.get('estimated_cost_saved_display', '$0')}")
            print(f"  Compacted requests       : {metrics.get('total_compacted_requests', 0):,}")
            sessions = metrics.get("top_sessions", [])
            if sessions:
                print(f"\n  Top Sessions:")
                for s in sessions[:5]:
                    print(f"    {s['session_id'][:20]:20s}  "
                          f"reqs={s['requests']:4d}  "
                          f"tokens_saved={s['tokens_saved']:5d}  "
                          f"cost_saved={s['cost_saved']}")
    except Exception as e:
        print(f"  Could not fetch final metrics: {e}")
        metrics = {}

    # ── Aggregate summary ─────────────────────────────────────────────────────
    print("\n\n" + "="*60)
    print("  AGGREGATE SUMMARY — ALL TIERS")
    print("="*60)

    total_reqs    = sum(r.total_requests for r in all_results)
    total_success = sum(r.successful for r in all_results)
    total_fail    = sum(r.failed for r in all_results)
    total_rl      = sum(r.rate_limited for r in all_results)
    total_compact = sum(r.compacted for r in all_results)
    total_chars   = sum(r.total_chars_saved for r in all_results)
    total_tokens  = sum(r.estimated_tokens_saved for r in all_results)
    total_cost    = sum(r.estimated_cost_saved_usd for r in all_results)
    avg_p95       = statistics.mean(r.latency_p95_ms for r in all_results)
    avg_p99       = statistics.mean(r.latency_p99_ms for r in all_results)
    max_rps       = max(r.throughput_rps for r in all_results)

    print(f"""
  Total Requests Fired   : {total_reqs:,}
  Successful             : {total_success:,}
  Failed (errors)        : {total_fail:,}
  Rate Limited (429)     : {total_rl:,}  ← proves rate limiter works
  Compacted              : {total_compact:,} ({round(total_compact/max(1,total_success)*100,1)}%)

  Peak Throughput        : {max_rps} req/sec
  Avg P95 Latency        : {round(avg_p95,2)} ms
  Avg P99 Latency        : {round(avg_p99,2)} ms

  Total Chars Saved      : {total_chars:,}
  Estimated Tokens Saved : ~{total_tokens:,}
  Estimated Cost Saved   : ${total_cost:.4f} USD

  ─────────────────────────────────────────────
  CV STATS TO USE:
  ─────────────────────────────────────────────
  • Handles {max_rps:.0f}+ req/sec peak throughput
  • {round(total_compact/max(1,total_success)*100,1)}% of repeated context successfully compacted
  • P95 latency under {round(avg_p95,0):.0f}ms for semantic deduplication
  • ~{total_tokens:,} tokens saved across {total_reqs:,} test requests
  • ${total_cost:.4f} USD saved in single test run
  • Rate limiter correctly blocked {total_rl:,} excess requests
    """)

    # ── Save JSON results ─────────────────────────────────────────────────────
    output = {
        "test_timestamp": int(time.time()),
        "gateway_url":    GATEWAY_URL,
        "tiers":          [asdict(r) for r in all_results],
        "aggregate": {
            "total_requests":        total_reqs,
            "successful":            total_success,
            "failed":                total_fail,
            "rate_limited":          total_rl,
            "compacted":             total_compact,
            "compaction_rate_pct":   round(total_compact/max(1,total_success)*100, 1),
            "peak_throughput_rps":   max_rps,
            "avg_p95_latency_ms":    round(avg_p95, 2),
            "avg_p99_latency_ms":    round(avg_p99, 2),
            "total_chars_saved":     total_chars,
            "estimated_tokens_saved": total_tokens,
            "estimated_cost_saved_usd": total_cost,
        },
        "gateway_metrics": metrics,
    }

    with open("load_test_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("  Full results saved to: load_test_results.json")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
