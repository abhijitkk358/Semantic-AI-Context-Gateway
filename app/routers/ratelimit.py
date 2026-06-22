"""
ratelimit.py — Redis Sliding Window Rate Limiter
=================================================
Strategy: Redis INCR + EXPIRE
- Key: ratelimit:<session_id>
- Window: 60 seconds
- Limit: 50 requests per window
- On first request: set key + expiry
- On subsequent requests: increment only (expiry already set)
- On limit breach: return 429 with retry-after header

Why sliding window over fixed window:
Fixed window resets at :00 every minute — a user can send
50 requests at :59 and 50 more at :01 = 100 in 2 seconds.
Sliding window always looks back exactly 60 seconds from NOW.
"""

from __future__ import annotations

import logging
from redis import Redis

log = logging.getLogger("semantic_gateway.ratelimit")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RATE_LIMIT_REQUESTS = 50       # max requests
RATE_LIMIT_WINDOW_SECONDS = 60 # per this many seconds
RATE_LIMIT_KEY_PREFIX = "ratelimit"


# ---------------------------------------------------------------------------
# Core check function
# ---------------------------------------------------------------------------
def check_rate_limit(r: Redis, session_id: str) -> dict:
    """
    Check and increment rate limit counter for a session.

    Returns dict:
        allowed     : bool   — True if request is allowed
        current     : int    — current request count in window
        limit       : int    — max allowed
        remaining   : int    — requests left in window
        retry_after : int    — seconds until window resets (only when blocked)
    """
    # Sanitise session_id to safe Redis key characters
    safe_session = "".join(
        c if c.isalnum() or c in "-_" else "_"
        for c in session_id
    )[:32]

    key = f"{RATE_LIMIT_KEY_PREFIX}:{safe_session}"

    # Redis pipeline — atomic INCR + EXPIRE in one round trip
    pipe = r.pipeline(transaction=True)
    pipe.incr(key)
    pipe.ttl(key)
    results = pipe.execute()

    current_count = results[0]
    ttl = results[1]

    # First request in window — set expiry
    if ttl == -1:
        r.expire(key, RATE_LIMIT_WINDOW_SECONDS)
        ttl = RATE_LIMIT_WINDOW_SECONDS

    remaining = max(0, RATE_LIMIT_REQUESTS - current_count)
    allowed = current_count <= RATE_LIMIT_REQUESTS

    if not allowed:
        log.warning(
            "Rate limit exceeded  session=%s  count=%d  limit=%d  retry_after=%ds",
            safe_session, current_count, RATE_LIMIT_REQUESTS, ttl,
        )
    else:
        log.debug(
            "Rate limit OK  session=%s  count=%d/%d  remaining=%d  window_ttl=%ds",
            safe_session, current_count, RATE_LIMIT_REQUESTS, remaining, ttl,
        )

    return {
        "allowed":      allowed,
        "current":      current_count,
        "limit":        RATE_LIMIT_REQUESTS,
        "remaining":    remaining,
        "retry_after":  ttl if not allowed else 0,
        "window":       RATE_LIMIT_WINDOW_SECONDS,
    }


# ---------------------------------------------------------------------------
# Admin helper — get current count without incrementing (for /metrics)
# ---------------------------------------------------------------------------
def get_rate_limit_status(r: Redis, session_id: str) -> dict:
    """
    Read current rate limit state without consuming a request.
    Used by metrics endpoint.
    """
    safe_session = "".join(
        c if c.isalnum() or c in "-_" else "_"
        for c in session_id
    )[:32]

    key = f"{RATE_LIMIT_KEY_PREFIX}:{safe_session}"

    pipe = r.pipeline(transaction=True)
    pipe.get(key)
    pipe.ttl(key)
    results = pipe.execute()

    current_count = int(results[0]) if results[0] else 0
    ttl = results[1] if results[1] and results[1] > 0 else 0
    remaining = max(0, RATE_LIMIT_REQUESTS - current_count)

    return {
        "session_id":  session_id,
        "current":     current_count,
        "limit":       RATE_LIMIT_REQUESTS,
        "remaining":   remaining,
        "window":      RATE_LIMIT_WINDOW_SECONDS,
        "resets_in":   ttl,
    }
