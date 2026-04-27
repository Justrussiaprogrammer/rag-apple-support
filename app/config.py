from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    raw_docs_path: Path = PROJECT_ROOT / "data" / "raw" / "apple_docs.jsonl"
    chunks_path: Path = PROJECT_ROOT / "data" / "index" / "chunks.jsonl"
    faiss_index_path: Path = PROJECT_ROOT / "data" / "index" / "faiss.index"

    embedding_model_name: str = "intfloat/multilingual-e5-small"
    embedding_batch_size: int = 32

    retrieve_top_k: int = 3
    retrieve_score_threshold: float = 0.25

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:1.5b"
    ollama_timeout_seconds: float = 180.0

    temperature: float = 0.1
    max_context_chunks: int = 3


settings = Settings()
