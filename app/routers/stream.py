"""
stream.py — Gemini SSE Streaming Router
========================================
Receives the compacted prompt from main.py, forwards it to Gemini
via OpenAI-compatible API, and streams tokens back to the user via SSE.
"""

from __future__ import annotations

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

# tiktoken encoding for token counting (cl100k_base works for all modern models)
try:
    _enc = tiktoken.get_encoding("cl100k_base")
except Exception:
    _enc = None


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class StreamRequest(BaseModel):
    compacted_prompt: str = Field(..., description="Already compacted prompt from gateway")
    original_prompt: str  = Field(..., description="Original prompt before compaction")
    temperature: float    = Field(0.7, ge=0.0, le=2.0)
    gateway_status: str   = Field("NO_REDUNDANCY")
    savings_percentage: float = Field(0.0)
    original_chars: int   = Field(0)
    transmitted_chars: int = Field(0)


# ---------------------------------------------------------------------------
# Token counting helper
# ---------------------------------------------------------------------------
def count_tokens(text: str) -> int:
    if _enc is None or not text:
        return 0
    try:
        return len(_enc.encode(text))
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# SSE token generator
# ---------------------------------------------------------------------------
async def gemini_stream(
    prompt: str,
    temperature: float,
    metadata: dict,
) -> AsyncGenerator[str, None]:
    """
    Opens a streaming connection to Gemini OpenAI-compatible endpoint.
    Yields SSE-formatted strings one token at a time.
    """

    # First event — send gateway metadata so client knows savings upfront
    yield f"data: {json.dumps({'type': 'metadata', **metadata})}\n\n"

    headers = {
        "Authorization": f"Bearer {settings.gemini_api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": settings.gemini_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "stream": True,
    }

    log.info("Forwarding compacted prompt to Gemini  model=%s", settings.gemini_model)
    log.info("Prompt preview: '%.100s'", prompt)

    full_response = []
    token_count = 0
    t_start = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                f"{settings.gemini_base_url}chat/completions",
                headers=headers,
                json=payload,
            ) as response:

                if response.status_code != 200:
                    error_body = await response.aread()
                    log.error("Gemini API error %d: %s", response.status_code, error_body)
                    yield f"data: {json.dumps({'type': 'error', 'code': response.status_code, 'message': error_body.decode()})}\n\n"
                    return

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
                        token_count += 1
                        log.debug("Token: %r", delta)
                        yield f"data: {json.dumps({'type': 'token', 'content': delta})}\n\n"

    except httpx.TimeoutException:
        log.error("Gemini request timed out")
        yield f"data: {json.dumps({'type': 'error', 'message': 'Gemini request timed out'})}\n\n"
        return

    except Exception as exc:
        log.error("Streaming error: %s", exc, exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        return

    # Final event — send completion stats
    latency_ms = (time.perf_counter() - t_start) * 1000.0
    full_text = "".join(full_response)
    tokens_used = count_tokens(prompt)
    tokens_saved = count_tokens(metadata.get("dropped_content", ""))

    log.info(
        "Stream complete  tokens=%d  latency=%.1fms  tokens_saved=%d",
        token_count, latency_ms, tokens_saved,
    )

    yield f"data: {json.dumps({'type': 'done', 'full_response': full_text, 'tokens_used': tokens_used, 'tokens_saved': tokens_saved, 'llm_latency_ms': round(latency_ms, 2)})}\n\n"
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
@router.post("/stream/completions")
async def stream_completions(request: StreamRequest) -> StreamingResponse:
    """
    Accepts a pre-compacted prompt and streams Gemini response via SSE.
    Called internally by main.py after the semantic compaction step.
    """
    if not request.compacted_prompt.strip():
        raise HTTPException(
            status_code=400,
            detail="Compacted prompt is empty — nothing to forward to LLM."
        )

    metadata = {
        "gateway_status":      request.gateway_status,
        "savings_percentage":  request.savings_percentage,
        "original_chars":      request.original_chars,
        "transmitted_chars":   request.transmitted_chars,
        "dropped_content":     request.original_prompt,
    }

    return StreamingResponse(
        gemini_stream(request.compacted_prompt, request.temperature, metadata),
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
@router.get("/stream/health")
async def stream_health() -> dict:
    key_loaded = bool(settings.gemini_api_key)
    return {
        "status":       "ok" if key_loaded else "missing_api_key",
        "model":        settings.gemini_model,
        "api_key_set":  key_loaded,
    }
