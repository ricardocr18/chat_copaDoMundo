"""
Schemas da API — Fase 6.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONCEITO: O QUE SÃO SCHEMAS?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Schemas definem o "contrato" da API:
- O que o cliente ENVIA (Request)
- O que o servidor RETORNA (Response)

Usamos Pydantic para validação automática:
  Cliente envia: {"message": "Quem ganhou 1970?"}
  Pydantic valida: é string? não está vazio?
  Se inválido: retorna erro 422 automaticamente
  Se válido: passa para o endpoint
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from pydantic import BaseModel, Field
from typing import Any


class ChatRequest(BaseModel):
    """
    Corpo da requisição POST /chat.

    O cliente envia este JSON:
    {
        "message": "Quem ganhou a Copa de 1970?",
        "session_id": "usuario-123"  (opcional)
    }
    """
    message: str = Field(
        ...,  # ... significa obrigatório
        min_length=1,
        max_length=1000,
        description="Pergunta do usuário",
        examples=["Quem ganhou a Copa do Mundo de 1970?"],
    )
    session_id: str | None = Field(
        default=None,
        description="ID da sessão para manter histórico. Se não enviado, gera automaticamente.",
        examples=["usuario-abc123"],
    )


class ChatResponse(BaseModel):
    """
    Corpo da resposta POST /chat (modo não-streaming).

    O servidor retorna:
    {
        "response": "O Brasil ganhou a Copa de 1970...",
        "session_id": "usuario-123",
        "intent": "rag",
        "latency_ms": 6500.0,
        "source": "📚 RAG (3 trechos)"
    }
    """
    response: str = Field(description="Resposta gerada pelo agente")
    session_id: str = Field(description="ID da sessão")
    intent: str | None = Field(default=None, description="Rota tomada pelo router")
    latency_ms: float | None = Field(default=None, description="Latência total em ms")
    source: str | None = Field(default=None, description="Fonte da resposta")
    node_path: list[str] | None = Field(default=None, description="Caminho percorrido no grafo")


class HealthResponse(BaseModel):
    """
    Resposta do GET /health.
    Usada por load balancers para verificar se a API está viva.
    """
    status: str = Field(description="'ok' se saudável")
    app: str = Field(description="Nome da aplicação")
    version: str = Field(description="Versão da API")
    services: dict[str, str] = Field(description="Status de cada serviço")


class HistoryResponse(BaseModel):
    """Resposta do GET /history/{session_id}."""
    session_id: str
    message_count: int
    messages: list[dict[str, Any]]


class ErrorResponse(BaseModel):
    """Resposta de erro padronizada."""
    error: str
    detail: str | None = None
    session_id: str | None = None
