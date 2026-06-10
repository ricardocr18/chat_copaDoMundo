"""
Construtor do Grafo LangGraph.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONCEITO: COMO O LANGGRAPH MONTA UM GRAFO?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Um grafo no LangGraph tem 3 componentes:

1. NÓS (Nodes): funções que processam o estado
   graph.add_node("nome", funcao)

2. ARESTAS (Edges): conexões entre nós
   graph.add_edge("no_origem", "no_destino")
   - Aresta simples: sempre vai de A para B
   - Aresta condicional: decide o destino baseado no estado

3. PONTOS ESPECIAIS:
   - START: entrada do grafo (nó virtual de início)
   - END: saída do grafo (nó virtual de fim)

Fluxo da Fase 1:
  START → input_node → process_node → output_node → END

Nas fases futuras, o fluxo terá ramificações:
  START → input_node → router_node → [rag_node | api_node] → validator_node → END
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from langgraph.graph import END, START, StateGraph

from app.graph.nodes import error_node, input_node, output_node, process_node
from app.graph.state import GraphState


def build_graph() -> StateGraph:
    """
    Constrói e compila o grafo LangGraph.

    A "compilação" é o passo que valida o grafo:
    - Verifica se não há nós isolados (sem conexão)
    - Verifica se START e END estão conectados
    - Prepara o grafo para execução

    Returns:
        Grafo compilado, pronto para invocar com .invoke()
    """

    # 1. Inicializa o builder com nosso estado
    #    StateGraph é o "papel em branco" onde desenhamos o grafo
    graph = StateGraph(GraphState)

    # ── 2. Adiciona os Nós ────────────────────────────────────────────────
    # Cada nó tem um NOME (string) e uma FUNÇÃO
    # O nome é como referenciamos o nó nas arestas
    graph.add_node("input_node", input_node)
    graph.add_node("process_node", process_node)
    graph.add_node("output_node", output_node)
    graph.add_node("error_node", error_node)

    # ── 3. Adiciona as Arestas (fluxo linear por enquanto) ────────────────
    # START é uma constante do LangGraph que representa a entrada
    graph.add_edge(START, "input_node")
    graph.add_edge("input_node", "process_node")
    graph.add_edge("process_node", "output_node")
    graph.add_edge("output_node", END)

    # O error_node é um "nó de escape" — não tem aresta de entrada aqui,
    # pois nas fases futuras será conectado via arestas condicionais.
    # Por ora, está no grafo mas não é chamado automaticamente.
    graph.add_edge("error_node", END)

    # ── 4. Compila o grafo ────────────────────────────────────────────────
    # compile() valida a estrutura e retorna o grafo executável
    compiled = graph.compile()

    print("✅ Grafo compilado com sucesso!")
    print("   Fluxo: START → input_node → process_node → output_node → END")

    return compiled


# Instância do grafo — exportada para uso em outros módulos
# Criada uma vez na inicialização do módulo (lazy initialization)
_graph_instance = None


def get_graph() -> StateGraph:
    """
    Retorna a instância do grafo (Singleton).

    Padrão Singleton: o grafo é construído uma vez e reutilizado.
    Isso evita o overhead de recompilar o grafo a cada requisição.

    Returns:
        Instância compilada do grafo.
    """
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = build_graph()
    return _graph_instance
