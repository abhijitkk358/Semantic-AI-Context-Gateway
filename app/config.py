from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Redis connection
    redis_url: str = "redis://localhost:6379"

    # Embedding model
    model_name: str = "all-MiniLM-L6-v2"

    # Vector index configuration
    index_name: str = "semantic_context_idx"
    vector_dim: int = 384
    distance_metric: str = "COSINE"
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200

    # Semantic deduplication threshold.
    # Stored embeddings use COSINE *distance* (0.0 = identical, 2.0 = opposite).
    # We keep a sentence only when its nearest neighbour distance > this value.
    # 0.15 ≈ cosine *similarity* of 0.85, a conservative duplicate boundary.
    similarity_threshold: float = 0.15

    # TTL for cached embeddings (seconds)
    context_ttl_seconds: int = 600

    class Config:
        env_file = ".env"


settings = Settings()
