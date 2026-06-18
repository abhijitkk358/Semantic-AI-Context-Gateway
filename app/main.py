from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager
import httpx
import time
from app.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the gateway application lifecycle. Provisioning a globally shared,
    asynchronous HTTPX client connection pool at boot avoids resource exhaustion.
    """
    app.state.http_client = httpx.AsyncClient()
    yield
    await app.state.http_client.aclose()

app = FastAPI(
    title="Semantic AI API Gateway",
    description="An enterprise-grade, async caching proxy layer for LLM transactions.",
    lifespan=lifespan
)

# --- PYDANTIC VALIDATION SCHEMAS ---

class ChatCompletionRequest(BaseModel):
    """Strict schema enforcement for incoming user LLM payloads."""
    prompt: str = Field(..., min_length=1, description="The user query or instruction for the LLM.")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature.")
    max_tokens: Optional[int] = Field(default=256, ge=1, description="Maximum tokens to generate.")

class GatewayResponse(BaseModel):
    """Standardized operational contract returned to the client application."""
    gateway_status: str = Field(..., description="Status flag indicating CACHE_HIT or CACHE_MISS.")
    latency_ms: float = Field(..., description="Total processing latency in milliseconds.")
    user_prompt: str
    llm_response: Dict[str, Any]


# --- CORE ROUTING PLUMBING ---

@app.get("/health")
async def health_check():
    """Internal diagnostic endpoint to verify operational availability."""
    return {"status": "healthy", "gateway": "active"}

@app.post("/v1/chat/completions", response_model=GatewayResponse)
async def proxy_llm_request(request_data: ChatCompletionRequest):
    """
    Validates, timestamps, and intercepts incoming client queries, 
    forwarding them asynchronously to the downstream backend target.
    """
    client: httpx.AsyncClient = app.state.http_client
    start_time = time.perf_counter() # Precise tracking of processing time
    
    try:
        # Perform non-blocking socket fetch from the external downstream target
        response = await client.get(settings.MOCK_LLM_URL, timeout=5.0)
        response.raise_for_status()
        
        # Calculate exactly how many milliseconds the network hop took
        execution_latency = (time.perf_counter() - start_time) * 1000
        
        # Stream the structured contract back up to the client application
        return GatewayResponse(
            gateway_status="SUCCESS_CACHE_MISS",
            latency_ms=round(execution_latency, 2),
            user_prompt=request_data.prompt,
            llm_response=response.json()
        )
        
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code, 
            detail="Downstream backend target returned an error."
        )
    except httpx.RequestError:
        raise HTTPException(
            status_code=503, 
            detail="Downstream backend cluster is completely unreachable."
        )
