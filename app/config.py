from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Redis
    redis_url: str = "redis://localhost:6379"

    # Embedding model
    model_name: str = "all-MiniLM-L6-v2"

    # Vector index
    index_name: str = "semantic_context_idx"
    vector_dim: int = 384
    distance_metric: str = "COSINE"
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200

    # Similarity threshold — only near-exact duplicates dropped
    similarity_threshold: float = 0.05

    # Minimum sentence length to attempt deduplication
    min_sentence_length: int = 15

    # Sliding TTL — resets on every cache hit
    context_ttl_seconds: int = 1800

    # Groq API
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    groq_base_url: str = "https://api.groq.com/openai/v1/"

    # Retry config for 429 rate limit errors
    llm_max_retries: int = 2
    llm_retry_delay_seconds: float = 5.0

    class Config:
        env_file = ".env"


settings = Settings()
