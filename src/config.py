"""Application configuration and paths."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Paths
    patient_csv_path: Path = PROJECT_ROOT / "data" / "raw" / "patient_data.csv"
    documents_dir: Path = PROJECT_ROOT / "data" / "documents"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    chroma_dir: Path = PROJECT_ROOT / "data" / "chroma"
    checkpoint_db_path: Path = PROJECT_ROOT / "data" / "checkpoints.db"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_base_url: str = "http://localhost:8000"

    # LLM
    llm_provider: Literal["openai", "anthropic"] = Field(
        default="openai",
        validation_alias="LLM_PROVIDER",
    )
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", validation_alias="ANTHROPIC_API_KEY")
    llm_model_routing: str = "gpt-4o-mini"
    llm_model_synthesis: str = "gpt-4o"

    # LangSmith
    langchain_tracing_v2: bool = Field(default=False, validation_alias="LANGCHAIN_TRACING_V2")
    langchain_api_key: str = Field(default="", validation_alias="LANGCHAIN_API_KEY")
    langchain_project: str = Field(default="data-doctor", validation_alias="LANGCHAIN_PROJECT")

    # Orchestrator / guardrails
    chat_max_message_chars: int = 4000
    routing_confidence_threshold: float = 0.6

    # SQL / data agent
    sql_max_rows: int = 1000

    # RAG / vectorstore
    rag_collection_name: str = "clinical_documents"
    rag_top_k: int = 5
    rag_retrieve_k: int = 15
    rag_embedding_model: str = "text-embedding-3-small"
    rag_index_batch_size: int = 128
    rag_min_relevant_chunks: int = 1
    rag_max_grounding_retries: int = 1


settings = Settings()
