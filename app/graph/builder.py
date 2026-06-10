"""
Construtor do Grafo — Fase 4.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONCEITO: EDGES CONDICIONAIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fase 3 usava edges SIMPLES:
  A → B → C → D  (sempre o mesmo caminho)

Fase 4 usa edges CONDICIONAIS:
  A → router → ? (depende do intent)
               ├→ B (se intent == "rag")
               ├→ C (se intent == "api")
               ├→ D (se intent == "direct")
               └→ E (se intent == "off_topic")

add_conditional_edges(origem, função, mapa):
  - origem: nó de onde sai a aresta
  - função: retorna uma string baseada no estado
  - mapa: {string: próximo_nó}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from langgraph.graph import END, START, StateGraph
from app.graph.nodes import (
    input_node,
    router_node,
    routing_function,
    rag_node,
    api_node,
    direct_node,
    off_topic_node,
    process_node,
    output_node,
    error_node,
)
from app.graph.state import GraphState


def build_graph() -> StateGraph:
    """
    Constrói o grafo com roteamento condicional.

    Fluxo completo:
      START
        ↓
      input_node
        ↓
      router_node (LLM decide a rota)
        ↓
        ├──→ rag_node      → process_node → output_node → END
        ├──→ api_node      → process_node → output_node → END
        ├──→ direct_node   → process_node → output_node → END
        └──→ off_topic_node → process_node → output_node → END
    """
    graph = StateGraph(GraphState)

    # ── Registra todos os nós ─────────────────────────────────────────
    graph.add_node("input_node",    input_node)
    graph.add_node("router_node",   router_node)
    graph.add_node("rag_node",      rag_node)
    graph.add_node("api_node",      api_node)
    graph.add_node("direct_node",   direct_node)
    graph.add_node("off_topic_node",off_topic_node)
    graph.add_node("process_node",  process_node)
    graph.add_node("output_node",   output_node)
    graph.add_node("error_node",    error_node)

    # ── Edges simples ─────────────────────────────────────────────────
    graph.add_edge(START,           "input_node")
    graph.add_edge("input_node",    "router_node")

    # ── Edge CONDICIONAL — o coração da Fase 4 ────────────────────────
    # Após o router_node, chama routing_function(state)
    # O retorno decide qual nó executar a seguir
    graph.add_conditional_edges(
        "router_node",      # nó de origem
        routing_function,   # função que retorna a chave
        {                   # mapa: chave → próximo nó
            "rag":       "rag_node",
            "api":       "api_node",
            "direct":    "direct_node",
            "off_topic": "off_topic_node",
        }
    )

    # ── Todos os caminhos convergem para process_node ─────────────────
    graph.add_edge("rag_node",       "process_node")
    graph.add_edge("api_node",       "process_node")
    graph.add_edge("direct_node",    "process_node")
    graph.add_edge("off_topic_node", "process_node")

    # ── Final do fluxo ────────────────────────────────────────────────
    graph.add_edge("process_node",  "output_node")
    graph.add_edge("output_node",   END)
    graph.add_edge("error_node",    END)

    compiled = graph.compile()

    print("✅ Grafo compilado com sucesso!")
    print("   Fluxo: START → input → router → [rag|api|direct|off_topic] → process → output → END")

    return compiled


_graph_instance = None


def get_graph() -> StateGraph:
    """Retorna o singleton do grafo."""
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = build_graph()
    return _graph_instance
