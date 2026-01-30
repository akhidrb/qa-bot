import os
from functools import lru_cache
from pydantic import BaseModel, Field


class Settings(BaseModel):
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o-mini")
    embedding_model: str = "text-embedding-3-small"
    openai_timeout: int = Field(default=30)

    # --- Limits & safety ---
    max_questions: int = Field(default=50)
    max_doc_bytes: int = Field(default=10 * 1024 * 1024)  # 10 MB
    retrieval_k: int = Field(default=8)  # Increased for better recall

    # --- Chunking ---
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    
    # --- Performance optimizations ---
    embedding_batch_size: int = Field(default=100, description="Batch size for embedding generation")
    use_mmr: bool = Field(default=False, description="Use MMR for diverse retrieval (may reduce recall)")


@lru_cache
def get_settings() -> Settings:
    """
    Load and cache application settings from environment variables.

    Using lru_cache ensures:
    - env vars are read once at startup
    - fast access across requests
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")

    return Settings(
        openai_api_key=api_key,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )
