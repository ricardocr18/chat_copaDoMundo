"""
Definição do Estado do Grafo (GraphState).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONCEITO FUNDAMENTAL: O QUE É "ESTADO" NO LANGGRAPH?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Imagine o estado como uma "mochila" que viaja entre todos os agentes.
Cada nó do grafo pode:
  - LER qualquer campo da mochila
  - ESCREVER novos valores nos campos

Quando o nó A termina e passa para o nó B, o nó B recebe
a mochila com tudo que o nó A colocou lá.

No LangGraph, o estado é definido como um TypedDict — um
dicionário com tipagem estática, garantindo que todos os
agentes "falem a mesma língua".

Por que TypedDict e não uma classe normal?
  - LangGraph foi projetado para funcionar com TypedDict
  - Permite usar Annotated para controlar como os campos
    são atualizados (ex: listas que se acumulam vs. sobrescrevem)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from typing import Annotated, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class GraphState(TypedDict):
    """
    Estado compartilhado entre todos os nós do grafo.

    Cada campo representa uma "gaveta" da mochila.
    Todos os agentes leem e escrevem aqui.

    Campos:
        messages: Histórico completo da conversa.
                  O `add_messages` é um "reducer" especial do LangGraph:
                  em vez de SUBSTITUIR a lista, ele ACUMULA novas mensagens.
                  Isso é o que mantém o histórico da conversa intacto.

        user_input: A pergunta mais recente do usuário (string pura,
                    sem formatação de mensagem). Facilita o acesso
                    nos nós sem precisar parsear o histórico.

        intent: A intenção detectada pelo router (ex: "rag", "api", "direct").
                Preenchida pelo Router Agent na Fase 4.

        retrieved_context: Trechos de documentos encontrados pelo RAG.
                           Lista de strings com os chunks mais relevantes.

        final_response: A resposta final que será enviada ao usuário.
                        O último nó do grafo preenche este campo.

        error: Se algo deu errado, o erro é registrado aqui.
               Permite que nós subsequentes saibam que houve falha
               e tomem decisões de fallback.

        metadata: Informações auxiliares (latência, tokens usados, etc.).
                  Usado principalmente pela observabilidade (Fase 5).
    """

    # Histórico de mensagens — add_messages acumula em vez de substituir
    messages: Annotated[list[BaseMessage], add_messages]

    # Pergunta atual do usuário
    user_input: str

    # Intenção detectada pelo roteador (preenchida na Fase 4)
    intent: str | None

    # Contexto recuperado pelo RAG (preenchido na Fase 3)
    retrieved_context: list[str] | None

    # Resposta final gerada
    final_response: str | None

    # Erro, se houver
    error: str | None

    # Metadados para observabilidade
    metadata: dict[str, Any] | None
