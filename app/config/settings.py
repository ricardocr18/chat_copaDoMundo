"""
Configurações centralizadas — Arquitetura Dual de LLMs.

Llama 3.3 (Bedrock) → perguntas históricas + RAG
Gemini 1.5 Flash (Google) → dados em tempo real Copa 2026
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Aplicação ──────────────────────────────────────────────────────
    app_name: str = Field(default="world-cup-agent")
    app_env: str = Field(default="development")
    log_level: str = Field(default="DEBUG")

    # ── AWS / Bedrock — Llama 3.3 para histórico ───────────────────────
    aws_region: str = Field(default="us-east-1")
    aws_access_key_id: str | None = Field(default=None)
    aws_secret_access_key: str | None = Field(default=None)
    bedrock_model_id: str = Field(
        default="us.meta.llama3-3-70b-instruct-v1:0",
        description="LLM para perguntas históricas",
    )
    bedrock_embeddings_model_id: str = Field(
        default="amazon.titan-embed-text-v2:0",
    )

    # ── Google — Gemini 1.5 Flash para tempo real ──────────────────────
    google_api_key: str | None = Field(
        default=None,
        description="Chave Google AI Studio para Gemini + web search",
    )

    # ── OpenAI — mantido como fallback ────────────────────────────────
    openai_api_key: str | None = Field(
        default=None,
        description="Chave OpenAI (fallback se Gemini não disponível)",
    )

    # ── LangFuse ───────────────────────────────────────────────────────
    langfuse_public_key: str | None = Field(default=None)
    langfuse_secret_key: str | None = Field(default=None)
    langfuse_host: str = Field(default="https://cloud.langfuse.com")

    # ── Vector Store ───────────────────────────────────────────────────
    chroma_persist_dir: str = Field(
        default="./app/data/processed/chroma_db",
    )

    # ── APIs Externas ──────────────────────────────────────────────────
    rapidapi_key: str | None = Field(default=None)
    news_api_key: str | None = Field(default=None)

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @property
    def has_realtime_search(self) -> bool:
        """Verifica se algum web search está disponível."""
        return bool(self.google_api_key or self.openai_api_key)


settings = Settings()