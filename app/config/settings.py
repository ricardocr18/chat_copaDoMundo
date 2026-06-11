"""
Configurações centralizadas da aplicação — Fase 4.

Novidades:
- Adicionadas chaves para APIs externas (RapidAPI, NewsAPI)
- Campos opcionais: se não configurados, o sistema usa mocks
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

    # ── AWS / Bedrock ──────────────────────────────────────────────────
    aws_region: str = Field(default="us-east-1")
    aws_access_key_id: str | None = Field(default=None)
    aws_secret_access_key: str | None = Field(default=None)
    bedrock_model_id: str = Field(default="us.meta.llama3-3-70b-instruct-v1:0")
    bedrock_embeddings_model_id: str = Field(default="amazon.titan-embed-text-v2:0")

    # ── LangFuse ───────────────────────────────────────────────────────
    langfuse_public_key: str | None = Field(default=None)
    langfuse_secret_key: str | None = Field(default=None)
    langfuse_host: str = Field(default="https://cloud.langfuse.com")

    # ── Vector Store ───────────────────────────────────────────────────
    chroma_persist_dir: str = Field(default="./app/data/processed/chroma_db")

    # ── APIs Externas (Fase 4) ─────────────────────────────────────────
    # RapidAPI — acesso ao api-football.com
    # Cadastro gratuito em: https://rapidapi.com/api-sports/api/api-football
    rapidapi_key: str | None = Field(
        default=None,
        description="Chave da RapidAPI para dados de futebol em tempo real",
    )

    # NewsAPI — notícias em tempo real
    # Cadastro gratuito em: https://newsapi.org
    news_api_key: str | None = Field(
        default=None,
        description="Chave da NewsAPI para notícias recentes",
    )

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @property
    def has_external_api(self) -> bool:
        """Verifica se alguma API externa está configurada."""
        return bool(self.rapidapi_key or self.news_api_key)


settings = Settings()
