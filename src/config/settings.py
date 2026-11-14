from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    llm_provider: str = "openai"
    openai_api_key: str = ""
    gemini_api_key: str = ""

    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    temperature: float = 0.0

    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k_results: int = 4

    raw_pdf_dir: Path = Path("data/raw_pdfs")
    chroma_persist_dir: Path = Path("data/chroma_db")
    audit_log_path: Path = Path("data/logs/audit_trail.jsonl")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
