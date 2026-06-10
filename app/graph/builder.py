"""
Construtor do Grafo LangGraph — Fase 3.

O que mudou:
- Adicionado rag_node entre input_node e process_node
- O fluxo agora é: START → input → rag → process → output → END

Por que o rag_node fica ANTES do process_node?
  O RAG precisa buscar o contexto ANTES do LLM responder.
  O process_node lê o retrieved_context do estado
  e o injeta no prompt — por isso a ordem importa.
"""

from langgraph.graph import END, START, StateGraph
from app.graph.nodes import (
    error_node,
    input_node,
    output_node,
    process_node,
    rag_node,
)
from app.graph.state import GraphState


def build_graph() -> StateGraph:
    """
    Constrói e compila o grafo com o nó RAG.

    Fluxo da Fase 3:
      START → input_node → rag_node → process_node → output_node → END
                               ↑
                    Busca documentos no ChromaDB
                    antes do LLM responder

    Returns:
        Grafo compilado.
    """
    graph = StateGraph(GraphState)

    # ── Registra os nós ───────────────────────────────────────────────
    graph.add_node("input_node",   input_node)
    graph.add_node("rag_node",     rag_node)      # NOVO na Fase 3
    graph.add_node("process_node", process_node)
    graph.add_node("output_node",  output_node)
    graph.add_node("error_node",   error_node)

    # ── Define o fluxo ────────────────────────────────────────────────
    graph.add_edge(START,          "input_node")
    graph.add_edge("input_node",   "rag_node")    # NOVO: vai para o RAG primeiro
    graph.add_edge("rag_node",     "process_node")
    graph.add_edge("process_node", "output_node")
    graph.add_edge("output_node",  END)
    graph.add_edge("error_node",   END)

    compiled = graph.compile()

    print("✅ Grafo compilado com sucesso!")
    print("   Fluxo: START → input_node → rag_node → process_node → output_node → END")

    return compiled


_graph_instance = None


def get_graph() -> StateGraph:
    """Retorna o singleton do grafo."""
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = build_graph()
    return _graph_instance
