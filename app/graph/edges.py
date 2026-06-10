"""
Arestas (Edges) do Grafo LangGraph.

Em LangGraph existem dois tipos de arestas:

1. ARESTA DIRETA (add_edge):
   - Sempre vai do nó A para o nó B
   - Exemplo: input_processor → query_classifier (sempre)
   - Uso: quando não há decisão a tomar

2. ARESTA CONDICIONAL (add_conditional_edges):
   - Decide para qual nó ir baseado no estado atual
   - Uma função examina o estado e retorna o NOME do próximo nó
   - Exemplo: após classificar, pode ir para "resposta" ou "erro"
   - Uso: quando há múltiplos caminhos possíveis

Este arquivo contém as FUNÇÕES DE ROTEAMENTO — o cérebro das decisões.
"""

import structlog

from app.graph.state import AgentStatus, GraphState

logger = structlog.get_logger(__name__)


# ==============================================================================
# Funções de Roteamento (Conditional Edge Functions)
# ==============================================================================
# Cada função recebe o estado e retorna uma STRING com o nome do próximo nó.
# Essa string deve corresponder EXATAMENTE ao nome usado em add_node().


def route_after_classification(state: GraphState) -> str:
    """
    Determina o próximo nó após a classificação da query.

    Esta função é a "chave" da aresta condicional que sai do
    query_classifier_node. Ela examina o estado e decide:

    - Se houve erro de iteração → vai para error_handler
    - Qualquer outro caso → vai para response_generator

    Na Fase 4, esta função ficará mais rica:
    - "off_topic" → guardrail_node
    - "precisa_api" → search_agent_node
    - "precisa_rag" → rag_agent_node
    - "pode_responder_direto" → response_generator_node

    Args:
        state: Estado atual do grafo.

    Returns:
        str: Nome do próximo nó a ser executado.
    """
    status = state.get("agent_status", AgentStatus.PENDING)
    iteration = state.get("iteration_count", 0)
    max_iterations = 10  # Será lido de settings na Fase 2

    # --- Proteção contra loop infinito ---
    if iteration >= max_iterations:
        logger.warning(
            "route_after_classification.max_iterations_reached",
            iteration=iteration,
            max_iterations=max_iterations,
        )
        return "error_handler"

    # --- Roteamento por status ---
    if status == AgentStatus.ERROR:
        logger.info("route_after_classification.routing_to_error")
        return "error_handler"

    # Fallback (off-topic) e Processing vão para o gerador de resposta
    # O gerador sabe como tratar cada caso pelo campo "agent_status"
    logger.info(
        "route_after_classification.routing_to_response_generator",
        status=status,
    )
    return "response_generator"


def route_after_response(state: GraphState) -> str:
    """
    Determina o próximo nó após a geração da resposta.

    Na Fase 1: Sempre vai para END (termina o grafo).

    Na Fase 5: Passará pelo validator_node antes de finalizar:
    - resposta válida → END
    - resposta inválida → response_generator (retry)
    - resposta perigosa → safety_filter_node

    Args:
        state: Estado atual do grafo.

    Returns:
        str: "__end__" para terminar, ou nome de outro nó para continuar.
    """
    status = state.get("agent_status", AgentStatus.PENDING)

    if status == AgentStatus.ERROR:
        # Se o gerador de resposta falhou, vai para o error handler
        return "error_handler"

    # Sucesso ou fallback → finaliza
    logger.info("route_after_response.routing_to_end", status=status)
    return "__end__"


# ==============================================================================
# Constantes para os nomes dos nós
# (Evita strings mágicas espalhadas pelo código)
# ==============================================================================

class NodeNames:
    """
    Constantes com os nomes dos nós do grafo.

    Por que usar constantes?
    - Evita erros de digitação: "resposne_generator" vs "response_generator"
    - Facilita refatoração: muda em um lugar, atualiza em todos
    - IDE consegue detectar referências e dar autocomplete

    Uso:
        graph.add_node(NodeNames.INPUT_PROCESSOR, input_processor_node)
        graph.add_edge(NodeNames.INPUT_PROCESSOR, NodeNames.QUERY_CLASSIFIER)
    """
    INPUT_PROCESSOR = "input_processor"
    QUERY_CLASSIFIER = "query_classifier"
    RESPONSE_GENERATOR = "response_generator"
    ERROR_HANDLER = "error_handler"
