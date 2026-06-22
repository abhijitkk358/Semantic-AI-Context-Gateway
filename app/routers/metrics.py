"""
metrics.py — Token Savings Metrics Router
==========================================
Logs every request result to Redis and exposes
a GET /v1/metrics endpoint for reporting.

Redis keys used:
  metrics:global           — HASH  global counters
  metrics:session:<id>     — HASH  per-session counters
  metrics:log:<timestamp>  — HASH  individual request log (TTL 24h)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

from fastapi import APIRouter
from redis import Redis

log = logging.getLogger("semantic_gateway.metrics")

router = APIRouter(prefix="/v1", tags=["metrics"])

# Cost per 1000 tokens for llama-3.1-8b-instant on Groq (free but show value)
# Using OpenAI gpt-3.5-turbo pricing as reference: $0.001 per 1k tokens
COST_PER_1K_TOKENS_USD = 0.001


# ---------------------------------------------------------------------------
# Write metrics after every request
# ---------------------------------------------------------------------------
def record_request_metrics(
    r: Redis,
    request_id: str,
    session_id: str,
    gateway_status: str,
    original_chars: int,
    transmitted_chars: int,
    savings_pct: float,
    tokens_used: int,
    tokens_saved: int,
    compaction_latency_ms: float,
    llm_latency_ms: float = 0.0,
) -> None:
    """
    Write metrics for one completed request into Redis.
    Uses atomic pipeline for all writes.
    """
    try:
        chars_saved = original_chars - transmitted_chars
        cost_saved = (tokens_saved / 1000.0) * COST_PER_1K_TOKENS_USD
        timestamp = int(time.time())
        is_compacted = 1 if gateway_status == "COMPACTED" else 0

        pipe = r.pipeline(transaction=True)

        # ── Global counters (HASH increments) ─────────────────────────────
        pipe.hincrbyfloat("metrics:global", "total_requests",          1)
        pipe.hincrbyfloat("metrics:global", "total_tokens_used",       tokens_used)
        pipe.hincrbyfloat("metrics:global", "total_tokens_saved",      tokens_saved)
        pipe.hincrbyfloat("metrics:global", "total_chars_saved",       chars_saved)
        pipe.hincrbyfloat("metrics:global", "total_cost_saved_usd",    cost_saved)
        pipe.hincrbyfloat("metrics:global", "total_compacted",         is_compacted)
        pipe.hincrbyfloat("metrics:global", "total_savings_pct_sum",   savings_pct)

        # ── Per-session counters ───────────────────────────────────────────
        session_key = f"metrics:session:{session_id[:32]}"
        pipe.hincrbyfloat(session_key, "requests",      1)
        pipe.hincrbyfloat(session_key, "tokens_used",   tokens_used)
        pipe.hincrbyfloat(session_key, "tokens_saved",  tokens_saved)
        pipe.hincrbyfloat(session_key, "chars_saved",   chars_saved)
        pipe.hincrbyfloat(session_key, "cost_saved_usd", cost_saved)
        pipe.expire(session_key, 86400)  # 24 hour TTL per session

        # ── Individual request log ─────────────────────────────────────────
        log_key = f"metrics:log:{timestamp}:{request_id}"
        pipe.hset(log_key, mapping={
            "request_id":           request_id,
            "session_id":           session_id,
            "gateway_status":       gateway_status,
            "original_chars":       original_chars,
            "transmitted_chars":    transmitted_chars,
            "chars_saved":          chars_saved,
            "savings_pct":          round(savings_pct, 2),
            "tokens_used":          tokens_used,
            "tokens_saved":         tokens_saved,
            "cost_saved_usd":       round(cost_saved, 6),
            "compaction_latency_ms": round(compaction_latency_ms, 2),
            "llm_latency_ms":       round(llm_latency_ms, 2),
            "timestamp":            timestamp,
        })
        pipe.expire(log_key, 86400)  # logs expire after 24 hours

        # ── Track unique sessions for leaderboard ─────────────────────────
        pipe.zadd("metrics:sessions", {session_id[:32]: timestamp})

        pipe.execute()

        log.info(
            "[%s] Metrics saved  tokens_saved=%d  cost_saved=$%.6f  chars_saved=%d",
            request_id, tokens_saved, cost_saved, chars_saved,
        )

    except Exception as exc:
        # Never crash the main request over metrics failure
        log.error("[%s] Failed to save metrics: %s", request_id, exc)


# ---------------------------------------------------------------------------
# Read global metrics
# ---------------------------------------------------------------------------
def _safe_float(val) -> float:
    try:
        if val is None:
            return 0.0
        return float(val.decode() if isinstance(val, bytes) else val)
    except Exception:
        return 0.0


def get_global_metrics(r: Redis) -> dict:
    try:
        raw = r.hgetall("metrics:global")
        if not raw:
            return _empty_metrics()

        total_requests   = _safe_float(raw.get(b"total_requests", 0))
        tokens_used      = _safe_float(raw.get(b"total_tokens_used", 0))
        tokens_saved     = _safe_float(raw.get(b"total_tokens_saved", 0))
        chars_saved      = _safe_float(raw.get(b"total_chars_saved", 0))
        cost_saved       = _safe_float(raw.get(b"total_cost_saved_usd", 0))
        total_compacted  = _safe_float(raw.get(b"total_compacted", 0))
        savings_pct_sum  = _safe_float(raw.get(b"total_savings_pct_sum", 0))

        avg_savings = (
            round(savings_pct_sum / total_requests, 2)
            if total_requests > 0 else 0.0
        )
        no_redundancy = total_requests - total_compacted

        return {
            "total_requests":          int(total_requests),
            "total_tokens_used":       int(tokens_used),
            "total_tokens_saved":      int(tokens_saved),
            "total_chars_saved":       int(chars_saved),
            "total_compacted_requests": int(total_compacted),
            "total_no_redundancy_requests": int(no_redundancy),
            "average_savings_pct":     avg_savings,
            "estimated_cost_saved_usd": round(cost_saved, 6),
            "estimated_cost_saved_display": f"${cost_saved:.4f}",
        }
    except Exception as exc:
        log.error("Failed to read global metrics: %s", exc)
        return _empty_metrics()


def _empty_metrics() -> dict:
    return {
        "total_requests":               0,
        "total_tokens_used":            0,
        "total_tokens_saved":           0,
        "total_chars_saved":            0,
        "total_compacted_requests":     0,
        "total_no_redundancy_requests": 0,
        "average_savings_pct":          0.0,
        "estimated_cost_saved_usd":     0.0,
        "estimated_cost_saved_display": "$0.0000",
    }


# ---------------------------------------------------------------------------
# Read per-session leaderboard
# ---------------------------------------------------------------------------
def get_top_sessions(r: Redis, top_n: int = 5) -> list:
    try:
        # Get all tracked session IDs
        session_ids = r.zrange("metrics:sessions", 0, -1)
        sessions = []

        for sid_raw in session_ids:
            sid = sid_raw.decode() if isinstance(sid_raw, bytes) else sid_raw
            key = f"metrics:session:{sid}"
            raw = r.hgetall(key)
            if not raw:
                continue
            sessions.append({
                "session_id":   sid,
                "requests":     int(_safe_float(raw.get(b"requests", 0))),
                "tokens_used":  int(_safe_float(raw.get(b"tokens_used", 0))),
                "tokens_saved": int(_safe_float(raw.get(b"tokens_saved", 0))),
                "chars_saved":  int(_safe_float(raw.get(b"chars_saved", 0))),
                "cost_saved":   f"${_safe_float(raw.get(b'cost_saved_usd', 0)):.4f}",
            })

        # Sort by tokens_saved descending
        sessions.sort(key=lambda x: x["tokens_saved"], reverse=True)
        return sessions[:top_n]

    except Exception as exc:
        log.error("Failed to read session leaderboard: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Recent request logs
# ---------------------------------------------------------------------------
def get_recent_logs(r: Redis, limit: int = 10) -> list:
    try:
        pattern = "metrics:log:*"
        keys = r.keys(pattern)
        if not keys:
            return []

        # Sort keys by timestamp (embedded in key name)
        keys_sorted = sorted(
            keys,
            key=lambda k: int(
                (k.decode() if isinstance(k, bytes) else k).split(":")[2]
            ),
            reverse=True,
        )[:limit]

        logs = []
        for key in keys_sorted:
            raw = r.hgetall(key)
            if not raw:
                continue
            entry = {}
            for k, v in raw.items():
                field = k.decode() if isinstance(k, bytes) else k
                value = v.decode() if isinstance(v, bytes) else v
                try:
                    value = float(value) if "." in value else int(value)
                except (ValueError, TypeError):
                    pass
                entry[field] = value
            logs.append(entry)

        return logs

    except Exception as exc:
        log.error("Failed to read recent logs: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
@router.get("/metrics")
async def get_metrics(
    sessions: bool = True,
    logs: bool = False,
    top_n: int = 5,
) -> dict:
    """
    GET /v1/metrics — Returns token savings and cost metrics.

    Query params:
      sessions=true  — include per-session leaderboard (default: true)
      logs=true      — include recent request logs (default: false)
      top_n=5        — how many top sessions to return
    """
    from app.main import app_state

    r = app_state.redis
    if not r:
        return {"error": "Redis not connected"}

    result = get_global_metrics(r)

    if sessions:
        result["top_sessions"] = get_top_sessions(r, top_n)

    if logs:
        result["recent_logs"] = get_recent_logs(r, limit=10)

    return result


# ---------------------------------------------------------------------------
# Reset endpoint (useful for testing)
# ---------------------------------------------------------------------------
@router.delete("/metrics/reset")
async def reset_metrics() -> dict:
    """
    DELETE /v1/metrics/reset — Wipes all metrics data.
    Use only in development/testing.
    """
    from app.main import app_state

    r = app_state.redis
    if not r:
        return {"error": "Redis not connected"}

    try:
        keys = r.keys("metrics:*")
        if keys:
            r.delete(*keys)
        log.info("Metrics reset — deleted %d keys", len(keys))
        return {"status": "ok", "deleted_keys": len(keys)}
    except Exception as exc:
        log.error("Failed to reset metrics: %s", exc)
        return {"error": str(exc)}
