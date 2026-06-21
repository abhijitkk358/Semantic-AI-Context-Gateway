"""
Semantic AI Context Gateway — main.py  v2.1.0
===============================================
Production fixes in this version:
1. Per-session Redis namespacing  — session_id isolates each user's cache
2. Stricter similarity threshold  — 0.05 distance, only near-exact duplicates
3. Short sentence bypass          — sentences under min_sentence_length always kept
4. Sliding TTL                    — cache expiry resets on every cache hit
5. Empty compaction fallback      — always sends something to LLM
6. Auto retry on 429              — handled in stream.py
7. Request ID tracing             — UUID on every request for log correlation
8. Better sentence splitter       — handles ? ! . and newlines correctly
9. Dropped sentences tracked      — passed to LLM metadata for token savings calc
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from functools import partial
from typing import AsyncGenerator, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from redis import Redis
from redis.commands.search.field import TagField, TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.routers.stream import StreamRequest, groq_stream, router as stream_router

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("semantic_gateway")


# ---------------------------------------------------------------------------
# App state singletons
# ---------------------------------------------------------------------------
class AppState:
    model: Optional[SentenceTransformer] = None
    redis: Optional[Redis] = None


app_state = AppState()


# ---------------------------------------------------------------------------
# Redis index
# ---------------------------------------------------------------------------
def ensure_redis_index(r: Redis) -> None:
    idx = settings.index_name
    expected_dim = settings.vector_dim
    expected_metric = settings.distance_metric.upper()
    needs_create = False

    try:
        info = r.ft(idx).info()
        attrs = info.get("attributes", [])
        current_dim: Optional[int] = None
        current_metric: Optional[str] = None

        for attr in attrs:
            if isinstance(attr, (list, tuple)):
                attr_dict = {}
                it = iter(attr)
                for k in it:
                    try:
                        v = next(it)
                        attr_dict[k.lower() if isinstance(k, str) else k] = v
                    except StopIteration:
                        break
                if attr_dict.get(b"type", attr_dict.get("type", b"")).upper() in (
                    b"VECTOR", "VECTOR",
                ):
                    params = attr_dict.get(b"attributes", attr_dict.get("attributes", []))
                    if isinstance(params, (list, tuple)):
                        params_dict: dict = {}
                        it2 = iter(params)
                        for k2 in it2:
                            try:
                                v2 = next(it2)
                                params_dict[
                                    k2.decode() if isinstance(k2, bytes) else k2
                                ] = v2.decode() if isinstance(v2, bytes) else v2
                            except StopIteration:
                                break
                        current_dim = int(params_dict.get("DIM", 0))
                        current_metric = params_dict.get("DISTANCE_METRIC", "")

        if current_dim != expected_dim or (
            current_metric and current_metric.upper() != expected_metric
        ):
            log.warning("Index schema mismatch — dropping and recreating.")
            r.ft(idx).dropindex(delete_documents=True)
            needs_create = True
        else:
            log.info("Existing index '%s' matches schema — reusing.", idx)

    except Exception as exc:
        err_msg = str(exc).lower()
        if "unknown" in err_msg or "no such" in err_msg or "not found" in err_msg:
            log.info("Index '%s' does not exist — will create.", idx)
            needs_create = True
        else:
            raise

    if not needs_create:
        return

    schema = (
        TagField("sentence_id"),
        TagField("session_id"),
        TextField("text"),
        VectorField(
            "embedding",
            "HNSW",
            {
                "TYPE": "FLOAT32",
                "DIM": expected_dim,
                "DISTANCE_METRIC": expected_metric,
                "M": settings.hnsw_m,
                "EF_CONSTRUCTION": settings.hnsw_ef_construction,
            },
        ),
    )

    definition = IndexDefinition(
        prefix=["sentence:"],
        index_type=IndexType.HASH,
    )

    r.ft(idx).create_index(schema, definition=definition)
    log.info(
        "Created index '%s'  DIM=%d  METRIC=%s  M=%d  EF_CONSTRUCTION=%d",
        idx, expected_dim, expected_metric,
        settings.hnsw_m, settings.hnsw_ef_construction,
    )


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("=== Semantic Gateway Startup v2.1.0 ===")

    loop = asyncio.get_running_loop()
    log.info("Loading SentenceTransformer model '%s' ...", settings.model_name)
    app_state.model = await loop.run_in_executor(
        None, partial(SentenceTransformer, settings.model_name)
    )
    log.info("Model loaded  dim=%d", settings.vector_dim)

    log.info("Connecting to Redis at %s ...", settings.redis_url)
    r = Redis.from_url(settings.redis_url, decode_responses=False)
    r.ping()
    app_state.redis = r
    log.info("Redis connection OK.")

    ensure_redis_index(r)

    log.info("LLM model     : %s", settings.groq_model)
    log.info("LLM base URL  : %s", settings.groq_base_url)
    log.info("API key set   : %s", bool(settings.groq_api_key))
    log.info("Threshold     : %.3f (cosine distance)", settings.similarity_threshold)
    log.info("Min sent len  : %d chars", settings.min_sentence_length)
    log.info("Cache TTL     : %ds (sliding)", settings.context_ttl_seconds)
    log.info("=== Startup complete. Gateway is ready. ===")
    yield

    log.info("=== Semantic Gateway Shutdown ===")
    if app_state.redis:
        app_state.redis.close()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Semantic AI Context Gateway",
    version="2.1.0",
    lifespan=lifespan,
)

app.include_router(stream_router)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ChatCompletionRequest(BaseModel):
    prompt: str            = Field(..., description="Prompt to compact and send.")
    temperature: float     = Field(0.7, ge=0.0, le=2.0)
    stream: bool           = Field(True, description="Stream via SSE if True.")
    # FIX 1: per-user session isolation
    session_id: str        = Field("default", description="Unique user/session ID.")


class ChatCompletionResponse(BaseModel):
    request_id: str
    session_id: str
    gateway_status: str
    original_chars: int
    transmitted_chars: int
    savings_percentage: float
    compaction_latency_ms: float
    final_prompt_sent: str


# ---------------------------------------------------------------------------
# FIX 8: Better sentence splitter
# Handles . ! ? and newlines, keeps punctuation attached
# ---------------------------------------------------------------------------
_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(text: str) -> List[str]:
    raw = _SPLIT_RE.split(text.strip())
    cleaned = []
    for s in raw:
        s = s.strip()
        if not s:
            continue
        cleaned.append(s)
    return cleaned


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------
def _encode_sync(model: SentenceTransformer, text: str) -> np.ndarray:
    vec = model.encode(text, normalize_embeddings=True, convert_to_numpy=True)
    return vec.astype(np.float32)


async def encode_sentence(text: str) -> np.ndarray:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, partial(_encode_sync, app_state.model, text)
    )


def vec_to_bytes(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()


# ---------------------------------------------------------------------------
# FIX 1: Session-scoped Redis key
# Each session gets its own namespace — no cross-user cache collisions
# ---------------------------------------------------------------------------
def redis_key(sentence: str, session_id: str) -> str:
    digest = hashlib.sha256(sentence.encode()).hexdigest()[:16]
    # Sanitise session_id — remove spaces and special chars
    safe_session = re.sub(r"[^a-zA-Z0-9_-]", "_", session_id)[:32]
    return f"sentence:{safe_session}:{digest}"


# ---------------------------------------------------------------------------
# FIX 4: Sliding TTL — store with expiry, refresh on every cache hit
# ---------------------------------------------------------------------------
def store_sentence(
    r: Redis, sentence: str, vec: np.ndarray,
    ttl: int, session_id: str
) -> None:
    key = redis_key(sentence, session_id)
    mapping = {
        "sentence_id": str(uuid.uuid4()),
        "session_id":  session_id,
        "text":        sentence,
        "embedding":   vec_to_bytes(vec),
    }
    r.hset(key, mapping=mapping)
    r.expire(key, ttl)
    log.debug("  [STORE] key=%s", key)


def refresh_ttl(r: Redis, key: str, ttl: int) -> None:
    """Sliding TTL — reset expiry on every cache hit."""
    r.expire(key, ttl)
    log.debug("  [TTL REFRESH] key=%s  new_ttl=%ds", key, ttl)


# ---------------------------------------------------------------------------
# FIX 1: Session-scoped vector search
# ---------------------------------------------------------------------------
def search_similar(
    r: Redis, vec: np.ndarray, session_id: str, k: int = 1
) -> List[dict]:
    """
    Search only within the current session's cached sentences.
    Uses key prefix filter: sentence:<session_id>:*
    """
    query_bytes = vec_to_bytes(vec)
    safe_session = re.sub(r"[^a-zA-Z0-9_-]", "_", session_id)[:32]

    try:
        raw = r.execute_command(
            "FT.SEARCH", settings.index_name,
            f"(@session_id:{{{safe_session}}})=>[KNN 1 @embedding $query_vector AS score]",
            "PARAMS", "2",
            "query_vector", query_bytes,
            "SORTBY", "score", "ASC",
            "RETURN", "3", "text", "score", "session_id",
            "DIALECT", "2",
            "LIMIT", "0", str(k),
        )
    except Exception as exc:
        log.error("Vector search error: %s", exc, exc_info=True)
        return []

    docs = []

    if isinstance(raw, dict):
        total = raw.get(b"total_results", raw.get("total_results", 0))
        if total == 0:
            return []

        results_list = raw.get(b"results", raw.get("results", []))
        for result in results_list:
            key_raw = result.get(b"id", result.get("id", b""))
            key = key_raw.decode() if isinstance(key_raw, bytes) else key_raw

            attrs = result.get(b"extra_attributes", result.get("extra_attributes", {}))

            score_raw = attrs.get(b"score", attrs.get("score", b"1.0"))
            score_str = score_raw.decode() if isinstance(score_raw, bytes) else str(score_raw)
            raw_score = max(0.0, float(score_str))
            similarity = 1.0 - raw_score

            text_raw = attrs.get(b"text", attrs.get("text", b""))
            text = text_raw.decode() if isinstance(text_raw, bytes) else text_raw

            log.info(
                "        MATCH → distance=%.6f  similarity=%.6f  text='%.60s'",
                raw_score, similarity, text,
            )

            # FIX 4: refresh TTL on cache hit
            refresh_ttl(r, key, settings.context_ttl_seconds)

            docs.append({
                "key": key,
                "text": text,
                "distance": raw_score,
                "similarity": similarity,
            })

    elif isinstance(raw, list):
        if not raw or raw[0] == 0:
            return []
        i = 1
        while i < len(raw):
            key = raw[i].decode() if isinstance(raw[i], bytes) else raw[i]
            fields_list = raw[i + 1] if i + 1 < len(raw) else []
            i += 2
            field_dict = {}
            it = iter(fields_list)
            for f in it:
                try:
                    v = next(it)
                    fname = f.decode() if isinstance(f, bytes) else f
                    fval  = v.decode() if isinstance(v, bytes) else v
                    field_dict[fname] = fval
                except StopIteration:
                    break
            raw_score = max(0.0, float(field_dict.get("score", 1.0)))
            similarity = 1.0 - raw_score
            text = field_dict.get("text", "")
            log.info(
                "        MATCH → distance=%.6f  similarity=%.6f  text='%.60s'",
                raw_score, similarity, text,
            )
            refresh_ttl(r, key, settings.context_ttl_seconds)
            docs.append({
                "key": key,
                "text": text,
                "distance": raw_score,
                "similarity": similarity,
            })

    return docs


# ---------------------------------------------------------------------------
# Core compaction
# ---------------------------------------------------------------------------
async def compact_prompt(
    prompt: str,
    session_id: str,
    request_id: str,
) -> Tuple[str, int, int, List[str], List[str]]:
    """
    Returns (final_prompt, original_chars, transmitted_chars, kept, dropped)
    """
    r = app_state.redis
    sentences = split_sentences(prompt)
    original_chars = len(prompt)

    log.info("[%s] ─── Compaction start ───────────────────────────", request_id)
    log.info("[%s] Session        : %s", request_id, session_id)
    log.info("[%s] Input sentences: %d", request_id, len(sentences))
    log.info("[%s] Original chars : %d", request_id, original_chars)

    kept: List[str] = []
    dropped: List[str] = []

    for i, sentence in enumerate(sentences):
        log.info("[%s]   [%d/%d] '%s'", request_id, i + 1, len(sentences), sentence[:80])

        # FIX 3: short sentences bypass deduplication — always kept
        if len(sentence) < settings.min_sentence_length:
            log.info("[%s]         → KEPT  (too short to deduplicate — %d chars)",
                     request_id, len(sentence))
            kept.append(sentence)
            continue

        vec = await encode_sentence(sentence)

        loop = asyncio.get_running_loop()
        matches = await loop.run_in_executor(
            None, partial(search_similar, r, vec, session_id)
        )

        log.info("[%s]         Search returned %d result(s).", request_id, len(matches))

        is_duplicate = False
        if matches:
            best = matches[0]
            log.info(
                "[%s]         Best → distance=%.6f  threshold=%.4f  text='%.50s'",
                request_id, best["distance"],
                settings.similarity_threshold, best["text"],
            )
            if best["distance"] <= settings.similarity_threshold:
                is_duplicate = True
                log.info("[%s]         → DROPPED (near-exact duplicate)", request_id)
            else:
                log.info("[%s]         → KEPT  (different enough)", request_id)
        else:
            log.info("[%s]         → KEPT  (nothing in session cache yet)", request_id)

        if not is_duplicate:
            kept.append(sentence)
            await loop.run_in_executor(
                None,
                partial(
                    store_sentence, r, sentence, vec,
                    settings.context_ttl_seconds, session_id
                ),
            )
        else:
            dropped.append(sentence)

    final_prompt = " ".join(kept)
    transmitted_chars = len(final_prompt)

    log.info("[%s] Kept     : %d / %d", request_id, len(kept), len(sentences))
    log.info("[%s] Dropped  : %d / %d", request_id, len(dropped), len(sentences))
    log.info("[%s] ─── Compaction end ──────────────────────────────", request_id)

    return final_prompt, original_chars, transmitted_chars, kept, dropped


# ---------------------------------------------------------------------------
# Main endpoint
# ---------------------------------------------------------------------------
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt must not be empty.")

    # FIX 7: unique request ID for tracing across logs
    request_id = str(uuid.uuid4())[:8]
    session_id = request.session_id or "default"

    t_start = time.perf_counter()
    final_prompt, original_chars, transmitted_chars, kept, dropped = \
        await compact_prompt(request.prompt, session_id, request_id)
    compaction_latency_ms = (time.perf_counter() - t_start) * 1000.0

    saved = original_chars - transmitted_chars
    savings_pct = (saved / original_chars * 100.0) if original_chars > 0 else 0.0
    status = "COMPACTED" if saved > 0 else "NO_REDUNDANCY"

    log.info(
        "[%s] status=%s  savings=%.2f%%  latency=%.1fms  session=%s",
        request_id, status, savings_pct, compaction_latency_ms, session_id,
    )

    # Non-streaming mode — return JSON only
    if not request.stream:
        return ChatCompletionResponse(
            request_id=request_id,
            session_id=session_id,
            gateway_status=status,
            original_chars=original_chars,
            transmitted_chars=transmitted_chars,
            savings_percentage=round(savings_pct, 2),
            compaction_latency_ms=round(compaction_latency_ms, 2),
            final_prompt_sent=final_prompt,
        )

    # FIX 5: always send something to LLM — use original if compacted is empty
    prompt_to_send = final_prompt.strip() if final_prompt.strip() else request.prompt
    if not final_prompt.strip():
        log.info("[%s] Compacted empty — using original prompt for LLM answer.", request_id)

    metadata = {
        "request_id":          request_id,
        "session_id":          session_id,
        "gateway_status":      status,
        "savings_percentage":  round(savings_pct, 2),
        "original_chars":      original_chars,
        "transmitted_chars":   transmitted_chars,
        "dropped_content":     " ".join(dropped),
        "compaction_latency_ms": round(compaction_latency_ms, 2),
    }

    return StreamingResponse(
        groq_stream(prompt_to_send, request.temperature, metadata),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> dict:
    redis_ok = False
    try:
        if app_state.redis:
            app_state.redis.ping()
            redis_ok = True
    except Exception:
        pass
    return {
        "status":            "ok" if redis_ok else "degraded",
        "version":           "2.1.0",
        "redis":             redis_ok,
        "model_loaded":      app_state.model is not None,
        "llm_model":         settings.groq_model,
        "llm_base_url":      settings.groq_base_url,
        "api_key_set":       bool(settings.groq_api_key),
        "threshold":         settings.similarity_threshold,
        "min_sentence_len":  settings.min_sentence_length,
        "cache_ttl_seconds": settings.context_ttl_seconds,
    }
