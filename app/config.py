from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Works with OpenAI or any OpenAI-compatible server (e.g. Ollama).
    openai_api_key: str = "ollama"
    openai_base_url: str | None = None  # e.g. http://localhost:11434/v1 for Ollama
    openai_chat_model: str = "gpt-4.1-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "palm_mind_documents"
    redis_url: str = "redis://localhost:6379/0"
    database_url: str = "sqlite:///./palm_mind.db"
    embedding_dimensions: int = 1536
    chunk_size: int = 800
    chunk_overlap: int = 120
    retrieval_top_k: int = 5
    redis_memory_ttl_seconds: int = 86400
    max_upload_size_mb: int = 15

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
