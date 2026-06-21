"""
Semantic AI Context Gateway — main.py
======================================
Phase 2: Added Gemini SSE streaming proxy.
- POST /v1/chat/completions  → compacts prompt + streams Gemini response
- POST /v1/stream/completions → direct streaming endpoint
- GET  /v1/stream/health      → Gemini API key status
- GET  /health                → Redis + model status
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
from typing import AsyncGenerator, List, Optional

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
from app.routers.stream import StreamRequest, gemini_stream, router as stream_router

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("semantic_gateway")


# ---------------------------------------------------------------------------
# Application-level singletons
# ---------------------------------------------------------------------------
class AppState:
    model: Optional[SentenceTransformer] = None
    redis: Optional[Redis] = None


app_state = AppState()


# ---------------------------------------------------------------------------
# Redis index management
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
    log.info("=== Semantic Gateway Startup ===")

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

    log.info("Gemini model: %s", settings.gemini_model)
    log.info("Gemini API key set: %s", bool(settings.gemini_api_key))
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
    version="2.0.0",
    lifespan=lifespan,
)

# Register streaming router
app.include_router(stream_router)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class ChatCompletionRequest(BaseModel):
    prompt: str = Field(..., description="Multi-sentence prompt to compact then stream.")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    stream: bool = Field(True, description="Stream response via SSE (default: True)")


class ChatCompletionResponse(BaseModel):
    gateway_status: str
    original_chars: int
    transmitted_chars: int
    savings_percentage: float
    latency_ms: float
    final_prompt_sent: str


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------
_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|(?<=\n)")


def split_sentences(text: str) -> List[str]:
    raw = _SPLIT_RE.split(text.strip())
    return [s.strip() for s in raw if s.strip()]


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
# Redis helpers
# ---------------------------------------------------------------------------
def redis_key(sentence: str) -> str:
    digest = hashlib.sha256(sentence.encode()).hexdigest()[:16]
    return f"sentence:{digest}"


def store_sentence(r: Redis, sentence: str, vec: np.ndarray, ttl: int) -> None:
    key = redis_key(sentence)
    mapping = {
        "sentence_id": str(uuid.uuid4()),
        "text": sentence,
        "embedding": vec_to_bytes(vec),
    }
    r.hset(key, mapping=mapping)
    r.expire(key, ttl)
    log.debug("  [STORE] key=%s  sentence='%.60s'", key, sentence)


def search_similar(r: Redis, vec: np.ndarray, k: int = 1) -> List[dict]:
    query_bytes = vec_to_bytes(vec)

    try:
        raw = r.execute_command(
            "FT.SEARCH", settings.index_name,
            "(*)=>[KNN 1 @embedding $query_vector AS score]",
            "PARAMS", "2",
            "query_vector", query_bytes,
            "SORTBY", "score", "ASC",
            "RETURN", "2", "text", "score",
            "DIALECT", "2",
            "LIMIT", "0", str(k),
        )
    except Exception as exc:
        log.error("Vector search error: %s", exc, exc_info=True)
        return []

    docs = []

    if isinstance(raw, dict):
        total = raw.get(b"total_results", raw.get("total_results", 0))
        log.info("        Dict response — total_results: %d", total)
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
                "        MATCH → key=%s  distance=%.6f  similarity=%.6f  text='%.60s'",
                key, raw_score, similarity, text,
            )
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
                "        MATCH → key=%s  distance=%.6f  similarity=%.6f  text='%.60s'",
                key, raw_score, similarity, text,
            )
            docs.append({
                "key": key,
                "text": text,
                "distance": raw_score,
                "similarity": similarity,
            })

    return docs


# ---------------------------------------------------------------------------
# Core compaction logic
# ---------------------------------------------------------------------------
async def compact_prompt(prompt: str) -> tuple[str, int, int, List[str], List[str]]:
    """
    Returns (final_prompt, original_chars, transmitted_chars, kept, dropped)
    """
    r = app_state.redis
    sentences = split_sentences(prompt)
    original_chars = len(prompt)

    log.info("─── Compaction start ───────────────────────────────")
    log.info("Input sentences : %d", len(sentences))
    log.info("Original chars  : %d", original_chars)

    kept: List[str] = []
    dropped: List[str] = []

    for i, sentence in enumerate(sentences):
        log.info("  [%d/%d] Processing: '%.80s'", i + 1, len(sentences), sentence)

        vec = await encode_sentence(sentence)

        loop = asyncio.get_running_loop()
        matches = await loop.run_in_executor(
            None, partial(search_similar, r, vec)
        )

        log.info("        Search returned %d result(s).", len(matches))

        is_duplicate = False
        if matches:
            best = matches[0]
            log.info(
                "        Best match → distance=%.6f  similarity=%.6f  "
                "threshold_distance=%.4f  text='%.60s'",
                best["distance"], best["similarity"],
                settings.similarity_threshold, best["text"],
            )
            if best["distance"] <= settings.similarity_threshold:
                is_duplicate = True
                log.info("        → DROPPED  (duplicate / near-duplicate detected)")
            else:
                log.info("        → KEPT  (sufficiently different)")
        else:
            log.info("        → KEPT  (no existing embeddings to compare against)")

        if not is_duplicate:
            kept.append(sentence)
            await loop.run_in_executor(
                None,
                partial(store_sentence, r, sentence, vec, settings.context_ttl_seconds),
            )
        else:
            dropped.append(sentence)

    final_prompt = " ".join(kept)
    transmitted_chars = len(final_prompt)

    log.info("Kept sentences   : %d / %d", len(kept), len(sentences))
    log.info("Dropped sentences: %d / %d", len(dropped), len(sentences))
    log.info("Transmitted chars: %d", transmitted_chars)
    log.info("─── Compaction end ─────────────────────────────────")

    return final_prompt, original_chars, transmitted_chars, kept, dropped


# ---------------------------------------------------------------------------
# Main endpoint — compact + stream
# ---------------------------------------------------------------------------
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt must not be empty.")

    t_start = time.perf_counter()
    final_prompt, original_chars, transmitted_chars, kept, dropped = \
        await compact_prompt(request.prompt)
    latency_ms = (time.perf_counter() - t_start) * 1000.0

    saved = original_chars - transmitted_chars
    savings_pct = (saved / original_chars * 100.0) if original_chars > 0 else 0.0
    status = "COMPACTED" if saved > 0 else "NO_REDUNDANCY"

    log.info(
        "Compaction done → status=%s  savings=%.2f%%  latency=%.1fms",
        status, savings_pct, latency_ms,
    )

    # If stream=False return JSON only (no LLM call)
    if not request.stream:
        return ChatCompletionResponse(
            gateway_status=status,
            original_chars=original_chars,
            transmitted_chars=transmitted_chars,
            savings_percentage=round(savings_pct, 2),
            latency_ms=round(latency_ms, 2),
            final_prompt_sent=final_prompt,
        )

    # If compacted prompt is empty nothing to send to LLM
    if not final_prompt.strip():
        return ChatCompletionResponse(
            gateway_status=status,
            original_chars=original_chars,
            transmitted_chars=0,
            savings_percentage=round(savings_pct, 2),
            latency_ms=round(latency_ms, 2),
            final_prompt_sent="",
        )

    # Stream compacted prompt to Gemini and pipe SSE back to user
    metadata = {
        "gateway_status":    status,
        "savings_percentage": round(savings_pct, 2),
        "original_chars":    original_chars,
        "transmitted_chars": transmitted_chars,
        "dropped_content":   " ".join(dropped),
    }

    return StreamingResponse(
        gemini_stream(final_prompt, request.temperature, metadata),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


# ---------------------------------------------------------------------------
# Health check
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
        "status":       "ok" if redis_ok else "degraded",
        "redis":        redis_ok,
        "model_loaded": app_state.model is not None,
        "gemini_key":   bool(settings.gemini_api_key),
        "version":      "2.0.0",
    }
