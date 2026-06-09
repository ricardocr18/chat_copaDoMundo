"""
Configurações centralizadas da aplicação.

Por que Pydantic Settings?
- Lê variáveis de ambiente automaticamente
- Valida os tipos (se AWS_REGION não for string, já falha na inicialização)
- Documenta quais configs existem em um só lugar
- Suporta arquivos .env nativamente
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Todas as configurações da aplicação.
    
    Pydantic lê automaticamente do ambiente ou do arquivo .env.
    Campos com default=None são opcionais (usados em fases futuras).
    """

    model_config = SettingsConfigDict(
        env_file=".env",          # Lê o arquivo .env na raiz do projeto
        env_file_encoding="utf-8",
        case_sensitive=False,     # AWS_REGION == aws_region
        extra="ignore",           # Ignora variáveis de ambiente desconhecidas
    )

    # ── Aplicação ─────────────────────────────────────────────────────────
    app_name: str = Field(default="world-cup-agent", description="Nome da aplicação")
    app_env: str = Field(default="development", description="Ambiente: development | production")
    log_level: str = Field(default="DEBUG", description="Nível de log")

    # ── AWS / Bedrock ──────────────────────────────────────────────────────
    aws_region: str = Field(default="us-east-1", description="Região AWS")
    aws_access_key_id: str | None = Field(default=None, description="AWS Access Key")
    aws_secret_access_key: str | None = Field(default=None, description="AWS Secret Key")

    bedrock_model_id: str = Field(
        default="anthropic.claude-3-sonnet-20240229-v1:0",
        description="ID do modelo no Bedrock",
    )
    bedrock_embeddings_model_id: str = Field(
        default="amazon.titan-embed-text-v2:0",
        description="ID do modelo de embeddings no Bedrock",
    )

    # ── LangFuse ───────────────────────────────────────────────────────────
    langfuse_public_key: str | None = Field(default=None)
    langfuse_secret_key: str | None = Field(default=None)
    langfuse_host: str = Field(default="https://cloud.langfuse.com")

    # ── Vector Store ───────────────────────────────────────────────────────
    chroma_persist_dir: str = Field(
        default="./app/data/processed/chroma_db",
        description="Diretório onde o ChromaDB persiste os dados",
    )

    @property
    def is_production(self) -> bool:
        """Atalho para verificar se estamos em produção."""
        return self.app_env == "production"

    @property
    def langfuse_enabled(self) -> bool:
        """LangFuse só é ativado se as chaves estiverem configuradas."""
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


# Instância única (Singleton) — importada por todos os módulos
# Uso: from app.config.settings import settings
settings = Settings()
