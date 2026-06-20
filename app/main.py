import asyncio
import logging
import re
import time
from contextlib import asynccontextmanager
from typing import Any, Literal, Optional

import numpy as np
import redis.asyncio as redis
from fastapi import FastAPI
from pydantic import BaseModel, Field
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
REDIS_URL = "redis://localhost:6379"
INDEX_NAME = "semantic_context_idx"
KEY_PREFIX = "ctx:"
MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_DIMENSION = 384
SIMILARITY_THRESHOLD = 0.65
CONTEXT_TTL_SECONDS = 600
MIN_SENTENCE_CHARS = 5

# 🎯 CRITICAL FIX: Added mandatory (*) parentheses for Dialect 2 Vector Search
KNN_QUERY = "(*)=>[KNN 1 @embedding $query_vector AS similarity_score]"

logger = logging.getLogger("semantic_ai_gateway")
logging.basicConfig(level=logging.INFO)

# --- SCHEMAS ---
class ChatCompletionRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

class ChatCompletionResponse(BaseModel):
    gateway_status: Literal["COMPACTED", "NO_REDUNDANCY"]
    original_chars: int
    transmitted_chars: int
    savings_percentage: float
    latency_ms: float
    final_prompt_sent: str

# --- GLOBAL STATE ---
class AppState:
    redis_client: Optional[redis.Redis] = None
    model: Optional[SentenceTransformer] = None

state = AppState()

# --- HELPER UTILITIES ---
def split_prompt(raw_prompt: str) -> list[str]:
    return [
        chunk.strip()
        for chunk in re.split(r"[.!?\n]", raw_prompt)
        if len(chunk.strip()) >= MIN_SENTENCE_CHARS
    ]

async def encode_sentences(sentences: list[str]) -> np.ndarray:
    if state.model is None:
        raise RuntimeError("SentenceTransformer model is not loaded")

    loop = asyncio.get_running_loop()
    embeddings = await loop.run_in_executor(
        None,
        lambda: state.model.encode(sentences, convert_to_numpy=True, normalize_embeddings=False)
    )
    return np.asarray(embeddings, dtype=np.float32)

async def ensure_redis_index(redis_client: redis.Redis) -> None:
    try:
        await redis_client.ft(INDEX_NAME).info()
        logger.info("✅ Redis vector index '%s' verified and responsive.", INDEX_NAME)
        return
    except Exception as exc:
        message = str(exc).lower()
        if "unknown index name" not in message and "no such index" not in message:
            raise
        
        logger.info("🏗️ Building fresh HNSW Schema vector mappings...")
        schema = (
            TextField("$.text", as_name="text"),
            VectorField("$.embedding", "HNSW", {
                "TYPE": "FLOAT32",
                "DIM": VECTOR_DIMENSION,
                "DISTANCE_METRIC": "COSINE"
            }, as_name="embedding")
        )
        await redis_client.ft(INDEX_NAME).create_index(
            fields=schema,
            definition=IndexDefinition(prefix=[KEY_PREFIX], index_type=IndexType.JSON)
        )
        logger.info("🚀 Vector search fields built successfully.")

# --- CORE COMPACTOR LOGIC ---
async def dynamic_semantic_compactor(raw_prompt: str) -> tuple[str, float]:
    sentences = split_prompt(raw_prompt)
    if not sentences:
        return raw_prompt, 0.0

    optimized_sentences = []
    saved_chars = 0
    total_chars = len(raw_prompt)

    embeddings = await encode_sentences(sentences)
    r_client = state.redis_client

    for idx, sentence in enumerate(sentences):
        flat_vector = np.asarray(embeddings[idx], dtype=np.float32).flatten()
        embedding_bytes = flat_vector.tobytes()

        # 🎯 FIX: Removed .sort_by() to let Redis default to its fast automatic KNN sort
        vector_query = (
            Query(KNN_QUERY)
            .paging(0, 1)
            .return_fields("similarity_score", "text")
            .dialect(2)
        )

        try:
            results = await r_client.ft(INDEX_NAME).search(
                vector_query, 
                query_params={"query_vector": embedding_bytes}
            )

            if results.docs:
                # Read similarity score directly from Redis response attributes
                raw_distance = float(results.docs[0].similarity_score)
                true_similarity = 1.0 - raw_distance

                logger.info(f"🔮 MATH MATCH: Comparing against vector cache. Score: {round(true_similarity, 4)}")

                if true_similarity >= SIMILARITY_THRESHOLD:
                    logger.info(f"🎯 Redundancy Intercepted! Dropping sentence: '{sentence}'")
                    saved_chars += len(sentence) + 1
                    continue
            else:
                logger.info("ℹ️ No matching vectors found in database cache.")
                
        except Exception as e:
            # 🚨 This was catching the sort_by error silently!
            logger.error(f"❌ DATABASE ERROR DURING VECTOR LOOKUP: {e}", exc_info=True)

        optimized_sentences.append(sentence)
        cache_key = f"{KEY_PREFIX}{time.time_ns()}"
        await r_client.json().set(cache_key, "$", {
            "text": sentence,
            "embedding": flat_vector.tolist()
        })
        await r_client.expire(cache_key, CONTEXT_TTL_SECONDS)

    compacted_prompt = ". ".join(optimized_sentences) + "." if optimized_sentences else "None"
    savings = (saved_chars / total_chars) * 100.0 if total_chars > 0 else 0.0
    return compacted_prompt, round(savings, 2)
# --- FASTAPI LIFECYCLE & ROUTES ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🧠 Loading AI Model weights into Memory...")
    state.model = SentenceTransformer(MODEL_NAME)
    logger.info("🔌 Setting up active connection to Redis Stack...")
    state.redis_client = redis.from_url(REDIS_URL, decode_responses=False)
    await ensure_redis_index(state.redis_client)
    yield
    logger.info("🛑 Closing down system pipeline connections...")
    await state.redis_client.close()

app = FastAPI(title="Semantic AI Gateway", lifespan=lifespan)

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def process_semantic_proxy(payload: ChatCompletionRequest):
    start_time = time.perf_counter()
    compacted_prompt, savings_percent = await dynamic_semantic_compactor(payload.prompt)
    execution_latency = (time.perf_counter() - start_time) * 1000
    status_flag = "COMPACTED" if savings_percent > 0 else "NO_REDUNDANCY"

    return ChatCompletionResponse(
        gateway_status=status_flag,
        original_chars=len(payload.prompt),
        transmitted_chars=len(compacted_prompt),
        savings_percentage=savings_percent,
        latency_ms=round(execution_latency, 2),
        final_prompt_sent=compacted_prompt
    )
