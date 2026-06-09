"""
Nós do Grafo LangGraph.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONCEITO: O QUE É UM "NÓ"?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Um nó é simplesmente uma FUNÇÃO PYTHON que:
  1. Recebe o estado atual (a "mochila")
  2. Faz algum processamento
  3. Retorna um DICIONÁRIO com os campos que quer atualizar

O LangGraph pega esse dicionário e mescla com o estado atual.
Você não precisa retornar o estado completo — só o que mudou.

Exemplo:
  def meu_no(state: GraphState) -> dict:
      return {"final_response": "Olá!"}  # Só atualiza este campo

Nesta Fase 1, os nós são MOCKS — simulam comportamento sem LLM.
Nas fases seguintes, vamos substituir os mocks por implementações reais.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import time
from langchain_core.messages import AIMessage, HumanMessage

from app.graph.state import GraphState


# ─────────────────────────────────────────────────────────────────────────────
# NÓ 1: Recepção da entrada do usuário
# ─────────────────────────────────────────────────────────────────────────────

def input_node(state: GraphState) -> dict:
    """
    Primeiro nó do grafo — recebe e prepara a entrada do usuário.

    Responsabilidades:
    - Registrar o timestamp de início (para medir latência)
    - Formatar a mensagem do usuário para o histórico
    - Inicializar campos que serão preenchidos adiante

    Args:
        state: Estado atual do grafo.

    Returns:
        Dicionário com os campos atualizados.
    """
    user_input = state["user_input"]

    print(f"\n📥 [input_node] Recebendo pergunta: '{user_input}'")

    return {
        # Adiciona a mensagem humana ao histórico
        # add_messages vai ACUMULAR (não substituir) graças ao Annotated
        "messages": [HumanMessage(content=user_input)],

        # Inicializa metadados com timestamp de início
        "metadata": {
            "start_time": time.time(),
            "node_path": ["input_node"],  # Rastreia por quais nós passou
        },

        # Limpa campos de execuções anteriores
        "error": None,
        "final_response": None,
        "intent": None,
        "retrieved_context": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NÓ 2: Processamento (MOCK — será substituído pelo LLM na Fase 2)
# ─────────────────────────────────────────────────────────────────────────────

def process_node(state: GraphState) -> dict:
    """
    Nó de processamento — gera uma resposta para a pergunta.

    ⚠️  FASE 1: Este nó é um MOCK que simula respostas sem LLM.
        Na Fase 2, será substituído por uma chamada real ao Bedrock.

    Args:
        state: Estado atual com user_input preenchido.

    Returns:
        Dicionário atualizando final_response e metadata.
    """
    user_input = state["user_input"].lower()

    print(f"⚙️  [process_node] Processando: '{user_input}'")

    # ── Lógica de mock: respostas simples baseadas em palavras-chave ──────
    mock_responses = {
        "campeão": "🏆 O Brasil é o maior campeão mundial com 5 títulos (1958, 1962, 1970, 1994, 2002).",
        "brasil": "🇧🇷 O Brasil é o único país a disputar todas as Copas do Mundo, com 5 títulos.",
        "artilheiro": "⚽ Miroslav Klose (Alemanha) é o maior artilheiro da história com 16 gols.",
        "gol": "⚽ O maior artilheiro histórico é Miroslav Klose (Alemanha) com 16 gols.",
        "alemanha": "🇩🇪 A Alemanha conquistou 4 títulos mundiais: 1954, 1974, 1990 e 2014.",
        "copa": "🌍 A Copa do Mundo FIFA é o maior torneio de futebol do mundo, realizado a cada 4 anos.",
        "1970": "🏆 A Copa de 1970 foi disputada no México. O Brasil sagrou-se tricampeão, com Pelé sendo o grande destaque.",
        "pelé": "⭐ Pelé é considerado o maior jogador de todos os tempos, sendo tricampeão mundial (1958, 1962, 1970).",
        "próxima": "📅 A próxima Copa do Mundo será em 2026, sediada nos Estados Unidos, México e Canadá.",
        "2026": "🌎 A Copa do Mundo de 2026 será realizada em conjunto pelos EUA, México e Canadá.",
    }

    # Busca pela primeira palavra-chave encontrada na pergunta
    response = None
    for keyword, answer in mock_responses.items():
        if keyword in user_input:
            response = answer
            break

    # Fallback se nenhuma palavra-chave for encontrada
    if response is None:
        response = (
            "🤔 Essa é uma boa pergunta sobre a Copa do Mundo! "
            "Por enquanto estou em modo de demonstração (Fase 1). "
            "Em breve terei acesso ao banco de conhecimento completo."
        )

    print(f"💬 [process_node] Resposta gerada: '{response[:60]}...'")

    # Atualiza o caminho percorrido nos metadados
    metadata = state.get("metadata", {}) or {}
    metadata["node_path"] = metadata.get("node_path", []) + ["process_node"]

    return {
        "final_response": response,
        "metadata": metadata,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NÓ 3: Saída — formata e finaliza a resposta
# ─────────────────────────────────────────────────────────────────────────────

def output_node(state: GraphState) -> dict:
    """
    Último nó do grafo — finaliza e formata a resposta.

    Responsabilidades:
    - Adicionar a resposta ao histórico de mensagens
    - Calcular a latência total
    - Preparar o estado final para retorno ao usuário

    Args:
        state: Estado com final_response preenchida.

    Returns:
        Dicionário com a mensagem AI adicionada ao histórico.
    """
    final_response = state.get("final_response") or "Não consegui gerar uma resposta."

    print(f"📤 [output_node] Finalizando resposta")

    # Calcula latência
    metadata = state.get("metadata", {}) or {}
    start_time = metadata.get("start_time", time.time())
    latency_ms = round((time.time() - start_time) * 1000, 2)

    metadata["node_path"] = metadata.get("node_path", []) + ["output_node"]
    metadata["latency_ms"] = latency_ms

    print(f"⏱️  [output_node] Latência total: {latency_ms}ms")

    return {
        # Adiciona a resposta da IA ao histórico
        "messages": [AIMessage(content=final_response)],
        "metadata": metadata,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NÓ DE ERRO — tratamento centralizado de falhas
# ─────────────────────────────────────────────────────────────────────────────

def error_node(state: GraphState) -> dict:
    """
    Nó de tratamento de erros — chamado quando algo falha.

    No LangGraph, podemos redirecionar o fluxo para este nó
    usando edges condicionais (implementadas na Fase 4).
    Por ora, ele é chamado manualmente quando um nó lança exceção.

    Args:
        state: Estado com o campo `error` preenchido.

    Returns:
        Dicionário com mensagem de fallback.
    """
    error = state.get("error", "Erro desconhecido")

    print(f"❌ [error_node] Tratando erro: {error}")

    fallback_response = (
        "Desculpe, encontrei um problema ao processar sua pergunta. "
        "Por favor, tente novamente ou reformule sua pergunta."
    )

    return {
        "final_response": fallback_response,
        "messages": [AIMessage(content=fallback_response)],
    }
