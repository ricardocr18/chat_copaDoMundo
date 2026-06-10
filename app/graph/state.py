"""
Estado do Grafo — Fase 4.

O que mudou:
- Adicionado campo 'intent' que agora é preenchido pelo router
- Valores possíveis: "rag" | "api" | "off_topic" | "direct"
- O builder usa esse campo para decidir a rota via edges condicionais
"""

from typing import Annotated, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class GraphState(TypedDict):
    """
    Estado compartilhado — a "mochila" que viaja entre os nós.

    Novidade na Fase 4:
        intent: preenchido pelo router_node com a decisão de rota.
                O builder usa esse valor para escolher o próximo nó.

        api_data: dados retornados pela API externa (Fase 4).
                  Preenchido pelo api_node quando a rota é "api".
    """
    messages:          Annotated[list[BaseMessage], add_messages]
    user_input:        str
    intent:            str | None   # "rag" | "api" | "off_topic" | "direct"
    retrieved_context: list[str] | None
    api_data:          dict[str, Any] | None  # NOVO: dados da API externa
    final_response:    str | None
    error:             str | None
    metadata:          dict[str, Any] | None
