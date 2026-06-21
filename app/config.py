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

    # Semantic deduplication threshold (cosine distance)
    similarity_threshold: float = 0.15

    # TTL for cached embeddings (seconds)
    context_ttl_seconds: int = 600

    # Gemini API
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    gemini_base_url: str = "https://api.groq.com/openai/v1/"
    class Config:
        env_file = ".env"


settings = Settings()
