"""
Router Agent — Fase 4.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONCEITO: O QUE É UM ROUTER AGENT?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
O Router Agent é o "gerente" do sistema. Ele não responde
perguntas — ele DECIDE quem vai responder.

Funciona assim:
1. Recebe a pergunta do usuário
2. Envia para o LLM com um prompt especial pedindo
   uma decisão de roteamento em JSON
3. O LLM retorna: {"intent": "rag", "reason": "..."}
4. O Router salva o intent no estado
5. O builder usa o intent para escolher o próximo nó

Por que usar LLM para rotear?
  Keywords: "Copa" → rag, "clima" → off_topic
  Problema: "Qual o CLIMA de expectativa para a Copa?" → errado!

  LLM entende CONTEXTO e INTENÇÃO, não só palavras.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import re
from langchain_core.messages import HumanMessage, SystemMessage

from app.graph.state import GraphState


# ── Prompt do Router ───────────────────────────────────────────────────────────
# Este prompt é o mais importante da Fase 4.
# Ele instrui o LLM a agir como um classificador, não como um respondedor.
ROUTER_SYSTEM_PROMPT = """Você é um sistema de roteamento para um chatbot especializado em Copa do Mundo FIFA.

Sua ÚNICA função é analisar a pergunta do usuário e decidir qual agente deve responder.
Você NÃO responde perguntas — apenas classifica a intenção.

ROTAS DISPONÍVEIS:

1. "rag" — Use quando a pergunta for sobre:
   - História da Copa do Mundo (campeões, datas, sedes)
   - Artilheiros e estatísticas históricas
   - Curiosidades e recordes históricos
   - Jogadores históricos (Pelé, Maradona, Ronaldo, etc.)
   - Comparações entre edições passadas

2. "api" — Use quando a pergunta for sobre:
   - Dados em tempo real ou recentes
   - Rankings e classificações atuais de seleções
   - Notícias recentes sobre Copa do Mundo
   - Próximas partidas ou calendário
   - Estatísticas atualizadas de jogadores ativos

3. "direct" — Use quando a pergunta for:
   - Saudações (olá, bom dia, tudo bem?)
   - Agradecimentos (obrigado, valeu)
   - Perguntas sobre o que o bot pode fazer
   - Respostas curtas que não precisam de dados externos

4. "off_topic" — Use quando a pergunta for:
   - Completamente fora do tema Copa do Mundo
   - Sobre outros esportes sem relação com Copa
   - Assuntos pessoais, política, tecnologia, etc.
   - Pedidos inadequados ou inapropriados

RESPONDA APENAS com um JSON válido neste formato exato:
{
  "intent": "rag" | "api" | "direct" | "off_topic",
  "reason": "explicação curta em português de por que escolheu esta rota",
  "confidence": 0.0 a 1.0
}

Não adicione nada além do JSON. Sem texto antes ou depois."""


def classify_intent(user_input: str) -> tuple[str, str, float]:
    """
    Usa o LLM para classificar a intenção da pergunta.

    Esta função envia a pergunta ao Llama com um prompt especial
    que instrui o modelo a retornar apenas um JSON com a rota.

    Args:
        user_input: Pergunta do usuário.

    Returns:
        Tupla (intent, reason, confidence).
        intent é sempre um dos 4 valores válidos.
    """
    from app.config.aws_config import get_llm

    llm = get_llm()

    messages = [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=f"Classifique esta pergunta: {user_input}"),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content.strip()

        # ── Extrai o JSON da resposta ─────────────────────────────────
        # O modelo às vezes adiciona texto antes/depois do JSON
        # Usamos regex para extrair apenas o bloco JSON
        json_match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = json.loads(raw)

        intent = data.get("intent", "rag")
        reason = data.get("reason", "")
        confidence = float(data.get("confidence", 0.8))

        # Valida que o intent é um dos valores permitidos
        valid_intents = {"rag", "api", "direct", "off_topic"}
        if intent not in valid_intents:
            intent = "rag"  # fallback seguro

        return intent, reason, confidence

    except (json.JSONDecodeError, Exception) as e:
        # Se o LLM não retornar JSON válido, usa RAG como fallback
        print(f"   ⚠️  Erro ao parsear JSON do router: {e}")
        return "rag", "Fallback por erro de parsing", 0.5


def router_node(state: GraphState) -> dict:
    """
    Nó de roteamento — decide qual caminho o grafo vai seguir.

    Este nó chama o LLM, obtém o intent e salva no estado.
    O builder.py usa o intent para escolher o próximo nó
    via add_conditional_edges.

    Args:
        state: Estado com user_input preenchido.

    Returns:
        Dicionário com intent e metadata atualizados.
    """
    user_input = state["user_input"]
    print(f"🧭 [router_node] Analisando intenção...")

    intent, reason, confidence = classify_intent(user_input)

    # Ícones para deixar o terminal mais legível
    intent_icons = {
        "rag":       "📚",
        "api":       "🌐",
        "direct":    "💬",
        "off_topic": "🚫",
    }
    icon = intent_icons.get(intent, "❓")

    print(f"   {icon} Intent: '{intent}' (confiança: {confidence:.0%})")
    print(f"   💭 Razão: {reason}")

    metadata = state.get("metadata", {}) or {}
    metadata["node_path"] = metadata.get("node_path", []) + ["router_node"]
    metadata["intent"] = intent
    metadata["router_confidence"] = confidence
    metadata["router_reason"] = reason

    return {
        "intent": intent,
        "metadata": metadata,
    }


def routing_function(state: GraphState) -> str:
    """
    Função de roteamento — lida pelo LangGraph para decidir a aresta.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    CONCEITO: COMO O LANGGRAPH USA ESTA FUNÇÃO
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    No builder.py, registramos assim:

    graph.add_conditional_edges(
        "router_node",    ← após este nó...
        routing_function, ← ...chama esta função...
        {                 ← ...e usa o retorno como chave
            "rag":       "rag_node",
            "api":       "api_node",
            "direct":    "direct_node",
            "off_topic": "off_topic_node",
        }
    )

    O LangGraph chama routing_function(state) e usa o
    valor retornado para escolher qual nó executar.
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Args:
        state: Estado com intent preenchido pelo router_node.

    Returns:
        String com o nome da rota ("rag", "api", etc.)
    """
    intent = state.get("intent") or "rag"
    return intent
