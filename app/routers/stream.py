"""
stream.py — Groq SSE Streaming Router
=======================================
Forwards compacted prompt to Groq API and streams
tokens back to user via Server-Sent Events (SSE).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import AsyncGenerator

import httpx
import tiktoken
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import settings

log = logging.getLogger("semantic_gateway.stream")

router = APIRouter(prefix="/v1", tags=["stream"])

try:
    _enc = tiktoken.get_encoding("cl100k_base")
except Exception:
    _enc = None


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class StreamRequest(BaseModel):
    compacted_prompt: str     = Field(..., description="Compacted prompt from gateway")
    original_prompt: str      = Field(..., description="Original prompt before compaction")
    temperature: float        = Field(0.7, ge=0.0, le=2.0)
    gateway_status: str       = Field("NO_REDUNDANCY")
    savings_percentage: float = Field(0.0)
    original_chars: int       = Field(0)
    transmitted_chars: int    = Field(0)
    request_id: str           = Field("", description="Unique request trace ID")
    session_id: str           = Field("default", description="User session ID")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def count_tokens(text: str) -> int:
    if _enc is None or not text:
        return 0
    try:
        return len(_enc.encode(text))
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# SSE generator with retry
# ---------------------------------------------------------------------------
async def groq_stream(
    prompt: str,
    temperature: float,
    metadata: dict,
) -> AsyncGenerator[str, None]:
    """
    Streams Groq LLM response as SSE events.
    Auto retries on 429 rate limit.
    """
    request_id = metadata.get("request_id", "unknown")

    # First event — gateway metadata
    yield f"data: {json.dumps({'type': 'metadata', **metadata})}\n\n"

    headers = {
        "Authorization": f"Bearer {settings.groq_api_key}",
        "Content-Type":  "application/json",
    }

    payload = {
        "model":       settings.groq_model,
        "messages":    [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "stream":      True,
    }

    log.info("[%s] Forwarding to Groq  model=%s  prompt_len=%d",
             request_id, settings.groq_model, len(prompt))

    full_response = []
    t_start = time.perf_counter()
    attempt = 0

    while attempt < settings.llm_max_retries:
        attempt += 1
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                async with client.stream(
                    "POST",
                    f"{settings.groq_base_url}chat/completions",
                    headers=headers,
                    json=payload,
                ) as response:

                    if response.status_code == 429:
                        if attempt < settings.llm_max_retries:
                            log.warning(
                                "[%s] Groq rate limit (429) — retry %d/%d in %.1fs",
                                request_id, attempt,
                                settings.llm_max_retries,
                                settings.llm_retry_delay_seconds,
                            )
                            yield f"data: {json.dumps({'type': 'warning', 'message': f'Rate limited. Retrying in {settings.llm_retry_delay_seconds}s...'})}\n\n"
                            await asyncio.sleep(settings.llm_retry_delay_seconds)
                            continue
                        else:
                            log.error("[%s] Groq rate limit exhausted after %d retries",
                                      request_id, attempt)
                            yield f"data: {json.dumps({'type': 'error', 'code': 429, 'message': 'Groq rate limit exceeded. Please try again in a moment.'})}\n\n"
                            yield "data: [DONE]\n\n"
                            return

                    if response.status_code != 200:
                        error_body = await response.aread()
                        log.error("[%s] Groq API error %d: %s",
                                  request_id, response.status_code, error_body)
                        yield f"data: {json.dumps({'type': 'error', 'code': response.status_code, 'message': f'Groq API returned {response.status_code}'})}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    # Stream tokens
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        if not line.startswith("data:"):
                            continue

                        raw = line[5:].strip()

                        if raw == "[DONE]":
                            break

                        try:
                            chunk = json.loads(raw)
                        except json.JSONDecodeError:
                            continue

                        delta = (
                            chunk.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content", "")
                        )

                        if delta:
                            full_response.append(delta)
                            log.debug("[%s] Token: %r", request_id, delta)
                            yield f"data: {json.dumps({'type': 'token', 'content': delta})}\n\n"

                    break  # success — exit retry loop

        except httpx.TimeoutException:
            log.error("[%s] Groq request timed out on attempt %d", request_id, attempt)
            if attempt >= settings.llm_max_retries:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Groq request timed out. Please try again.'})}\n\n"
                yield "data: [DONE]\n\n"
                return
            await asyncio.sleep(settings.llm_retry_delay_seconds)
            continue

        except Exception as exc:
            log.error("[%s] Streaming error: %s", request_id, exc, exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': 'Internal streaming error.'})}\n\n"
            yield "data: [DONE]\n\n"
            return

    # Final done event
    latency_ms = (time.perf_counter() - t_start) * 1000.0
    full_text = "".join(full_response)
    tokens_used = count_tokens(prompt)
    tokens_saved = count_tokens(metadata.get("dropped_content", ""))

    log.info(
        "[%s] Groq stream complete  tokens_used=%d  tokens_saved=%d  latency=%.1fms",
        request_id, tokens_used, tokens_saved, latency_ms,
    )

    yield f"data: {json.dumps({'type': 'done', 'full_response': full_text, 'tokens_used': tokens_used, 'tokens_saved': tokens_saved, 'llm_latency_ms': round(latency_ms, 2)})}\n\n"
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Direct streaming endpoint
# ---------------------------------------------------------------------------
@router.post("/stream/completions")
async def stream_completions(request: StreamRequest) -> StreamingResponse:
    if not request.compacted_prompt.strip():
        raise HTTPException(status_code=400, detail="Compacted prompt is empty.")

    metadata = {
        "gateway_status":      request.gateway_status,
        "savings_percentage":  request.savings_percentage,
        "original_chars":      request.original_chars,
        "transmitted_chars":   request.transmitted_chars,
        "dropped_content":     request.original_prompt,
        "request_id":          request.request_id,
        "session_id":          request.session_id,
    }

    return StreamingResponse(
        groq_stream(request.compacted_prompt, request.temperature, metadata),
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
@router.get("/stream/health")
async def stream_health() -> dict:
    key_loaded = bool(settings.groq_api_key)
    return {
        "status":      "ok" if key_loaded else "missing_api_key",
        "model":       settings.groq_model,
        "base_url":    settings.groq_base_url,
        "api_key_set": key_loaded,
    }
