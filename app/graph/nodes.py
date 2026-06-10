"""
Nós do Grafo LangGraph — Fase 4.

Novidades em relação à Fase 3:
- router_node importado do router_agent
- api_node: consulta APIs externas de esportes
- direct_node: responde perguntas simples sem RAG
- off_topic_node: trata perguntas fora do escopo
- process_node: agora recebe também api_data

Fluxo:
  input → router → [rag | api | direct | off_topic] → process → output
"""

import json
import time
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from app.graph.state import GraphState


WORLD_CUP_SYSTEM_PROMPT = """Você é um especialista em Copa do Mundo FIFA com conhecimento \
profundo sobre toda a história do torneio desde 1930 até os dias atuais.

Suas responsabilidades:
- Responder perguntas sobre história, estatísticas e curiosidades da Copa do Mundo
- Fornecer informações precisas sobre campeões, artilheiros, recordes e jogadores históricos
- Manter um tom entusiasmado mas preciso, como um comentarista esportivo experiente

Diretrizes:
- Responda SEMPRE em português brasileiro
- Se não souber algo com certeza, diga claramente
- Foque exclusivamente em Copa do Mundo
- Quando houver contexto de documentos ou API, PRIORIZE essas informações
- Máximo de 3 parágrafos"""


# ─────────────────────────────────────────────────────────────────────────────
# NÓ 1: Input — sem alterações
# ─────────────────────────────────────────────────────────────────────────────

def input_node(state: GraphState) -> dict:
    """Recebe e prepara a entrada do usuário."""
    user_input = state["user_input"]
    print(f"\n📥 [input_node] Recebendo: '{user_input}'")

    return {
        "messages": [HumanMessage(content=user_input)],
        "metadata": {
            "start_time": time.time(),
            "node_path": ["input_node"],
        },
        "error": None,
        "final_response": None,
        "intent": None,
        "retrieved_context": None,
        "api_data": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NÓ 2: Router — importado do router_agent
# ─────────────────────────────────────────────────────────────────────────────

def router_node(state: GraphState) -> dict:
    """Delega para o router_agent."""
    from app.agents.router_agent import router_node as _router
    return _router(state)


def routing_function(state: GraphState) -> str:
    """Função de roteamento para edges condicionais."""
    from app.agents.router_agent import routing_function as _route
    return _route(state)


# ─────────────────────────────────────────────────────────────────────────────
# NÓ 3: RAG — sem alterações da Fase 3
# ─────────────────────────────────────────────────────────────────────────────

def rag_node(state: GraphState) -> dict:
    """Busca documentos relevantes no ChromaDB."""
    from app.tools.vector_store import search_similar_documents

    user_input = state["user_input"]
    print(f"🔍 [rag_node] Buscando documentos relevantes...")

    try:
        contexts = search_similar_documents(query=user_input, k=3)

        if contexts:
            print(f"   ✅ {len(contexts)} trechos encontrados")
        else:
            print(f"   ⚠️  Nenhum trecho relevante encontrado")

        metadata = state.get("metadata", {}) or {}
        metadata["node_path"] = metadata.get("node_path", []) + ["rag_node"]
        metadata["rag_chunks_found"] = len(contexts)

        return {"retrieved_context": contexts or [], "metadata": metadata}

    except Exception as e:
        print(f"   ⚠️  Erro no RAG: {str(e)[:80]}")
        metadata = state.get("metadata", {}) or {}
        metadata["node_path"] = metadata.get("node_path", []) + ["rag_node"]
        return {"retrieved_context": [], "metadata": metadata}


# ─────────────────────────────────────────────────────────────────────────────
# NÓ 4: API — NOVO na Fase 4
# ─────────────────────────────────────────────────────────────────────────────

def api_node(state: GraphState) -> dict:
    """
    Nó de API — consulta fontes externas de dados esportivos.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Decisão interna: qual endpoint chamar?
    Analisa palavras-chave no user_input para decidir:
    - "classificação", "ranking" → standings
    - "notícia", "recente"      → news
    - nome de seleção           → team info
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    from app.tools.http_client import (
        fetch_world_cup_standings,
        fetch_world_cup_news,
    )

    user_input = state["user_input"].lower()
    print(f"🌐 [api_node] Consultando API externa...")

    try:
        # Decide qual endpoint chamar baseado na pergunta
        if any(w in user_input for w in ["notícia", "noticia", "recente", "novo", "atualidade"]):
            print(f"   📰 Buscando notícias recentes...")
            api_data = fetch_world_cup_news()

        else:
            # Default: classificação/standings
            print(f"   📊 Buscando classificação/dados gerais...")
            api_data = fetch_world_cup_standings()

        source = api_data.get("source", "unknown")
        print(f"   ✅ Dados recebidos (fonte: {source})")

        metadata = state.get("metadata", {}) or {}
        metadata["node_path"] = metadata.get("node_path", []) + ["api_node"]
        metadata["api_source"] = source

        return {"api_data": api_data, "metadata": metadata}

    except Exception as e:
        print(f"   ❌ Erro na API: {str(e)[:80]}")
        metadata = state.get("metadata", {}) or {}
        metadata["node_path"] = metadata.get("node_path", []) + ["api_node"]
        return {
            "api_data": {"source": "error", "error": str(e)},
            "metadata": metadata,
        }


# ─────────────────────────────────────────────────────────────────────────────
# NÓ 5: Direct — NOVO na Fase 4
# ─────────────────────────────────────────────────────────────────────────────

def direct_node(state: GraphState) -> dict:
    """
    Nó de resposta direta — para saudações e perguntas simples.

    Não chama RAG nem API — responde diretamente com o LLM.
    Mais rápido que o process_node completo pois não
    precisa montar contexto de documentos.
    """
    print(f"💬 [direct_node] Resposta direta (sem RAG/API)...")

    metadata = state.get("metadata", {}) or {}
    metadata["node_path"] = metadata.get("node_path", []) + ["direct_node"]

    return {"metadata": metadata}


# ─────────────────────────────────────────────────────────────────────────────
# NÓ 6: Off-Topic — NOVO na Fase 4
# ─────────────────────────────────────────────────────────────────────────────

def off_topic_node(state: GraphState) -> dict:
    """
    Nó off-topic — trata perguntas fora do escopo.

    Não chama o LLM — retorna resposta padrão diretamente.
    Economiza tokens e tempo para perguntas irrelevantes.
    """
    print(f"🚫 [off_topic_node] Pergunta fora do escopo")

    off_topic_response = (
        "Desculpe, sou especializado exclusivamente em Copa do Mundo FIFA! 🏆\n\n"
        "Posso te ajudar com:\n"
        "• História e campeões das Copas\n"
        "• Artilheiros e estatísticas\n"
        "• Curiosidades e recordes\n"
        "• Informações sobre jogadores históricos\n\n"
        "Tem alguma pergunta sobre a Copa do Mundo?"
    )

    metadata = state.get("metadata", {}) or {}
    metadata["node_path"] = metadata.get("node_path", []) + ["off_topic_node"]

    return {
        "final_response": off_topic_response,
        "metadata": metadata,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NÓ 7: Process — MODIFICADO para tratar RAG + API + Direct
# ─────────────────────────────────────────────────────────────────────────────

def process_node(state: GraphState) -> dict:
    """
    Nó de processamento — chama o LLM com o contexto apropriado.

    Detecta qual rota foi tomada e monta o prompt adequado:
    - Rota RAG:    injeta chunks dos documentos
    - Rota API:    injeta dados da API externa
    - Rota Direct: usa só o system prompt base
    """
    from app.config.aws_config import get_llm

    # Se já tem resposta (off_topic_node preencheu), pula o LLM
    if state.get("final_response"):
        print(f"⚙️  [process_node] Resposta já gerada, pulando LLM")
        metadata = state.get("metadata", {}) or {}
        metadata["node_path"] = metadata.get("node_path", []) + ["process_node"]
        return {"metadata": metadata}

    print(f"⚙️  [process_node] Chamando LLM...")
    intent = state.get("intent", "rag")

    try:
        # ── Monta contexto baseado na rota ────────────────────────────
        retrieved_context = state.get("retrieved_context") or []
        api_data = state.get("api_data")

        if retrieved_context and intent == "rag":
            # Rota RAG: injeta documentos
            context_text = "\n\n---\n\n".join(retrieved_context)
            system_content = f"""{WORLD_CUP_SYSTEM_PROMPT}

CONTEXTO DOS DOCUMENTOS (use como fonte principal):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{context_text}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Responda baseado principalmente neste contexto."""
            print(f"   📚 Modo RAG ({len(retrieved_context)} trechos)")

        elif api_data and intent == "api":
            # Rota API: injeta dados externos
            api_text = json.dumps(api_data, ensure_ascii=False, indent=2)
            system_content = f"""{WORLD_CUP_SYSTEM_PROMPT}

DADOS DA API EXTERNA (informações atualizadas):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{api_text}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use estes dados para responder. Se forem dados de exemplo,
mencione isso na resposta."""
            print(f"   🌐 Modo API (fonte: {api_data.get('source', '?')})")

        else:
            # Rota Direct: sem contexto externo
            system_content = WORLD_CUP_SYSTEM_PROMPT
            print(f"   💬 Modo Direct (só conhecimento do modelo)")

        # ── Envia ao LLM ──────────────────────────────────────────────
        messages_to_send = [
            SystemMessage(content=system_content),
            *state["messages"],
        ]

        llm = get_llm()
        print(f"   🌐 Enviando {len(messages_to_send)} mensagens...")
        response = llm.invoke(messages_to_send)
        answer = response.content
        print(f"   ✅ Resposta recebida ({len(answer)} chars)")

        metadata = state.get("metadata", {}) or {}
        metadata["node_path"] = metadata.get("node_path", []) + ["process_node"]
        metadata["used_rag"] = bool(retrieved_context and intent == "rag")
        metadata["used_api"] = bool(api_data and intent == "api")

        return {"final_response": answer, "metadata": metadata}

    except Exception as e:
        error_msg = str(e)
        print(f"   ❌ Erro: {error_msg[:80]}")
        metadata = state.get("metadata", {}) or {}
        metadata["node_path"] = metadata.get("node_path", []) + ["process_node"]
        return {"error": error_msg, "final_response": None, "metadata": metadata}


# ─────────────────────────────────────────────────────────────────────────────
# NÓ 8: Output — atualizado para mostrar a rota usada
# ─────────────────────────────────────────────────────────────────────────────

def output_node(state: GraphState) -> dict:
    """Finaliza e formata a resposta."""
    error = state.get("error")

    if error:
        print(f"⚠️  [output_node] Erro detectado, usando fallback")
        final_response = "Desculpe, tive um problema. Por favor, tente novamente."
    else:
        final_response = state.get("final_response") or "Não consegui gerar uma resposta."

    print(f"📤 [output_node] Finalizando")

    metadata = state.get("metadata", {}) or {}
    start_time = metadata.get("start_time", time.time())
    latency_ms = round((time.time() - start_time) * 1000, 2)
    metadata["node_path"] = metadata.get("node_path", []) + ["output_node"]
    metadata["latency_ms"] = latency_ms

    # Indicador de fonte baseado na rota
    intent = metadata.get("intent", state.get("intent", "?"))
    source_map = {
        "rag":       f"📚 RAG ({metadata.get('rag_chunks_found', 0)} trechos)",
        "api":       f"🌐 API ({metadata.get('api_source', 'externa')})",
        "direct":    "💬 Direto",
        "off_topic": "🚫 Off-topic",
    }
    source_indicator = source_map.get(intent, "🧠 Modelo")
    metadata["source_indicator"] = source_indicator

    print(f"⏱️  Latência: {latency_ms}ms | Rota: {source_indicator}")

    return {
        "messages": [AIMessage(content=final_response)],
        "final_response": final_response,
        "metadata": metadata,
    }


def error_node(state: GraphState) -> dict:
    """Tratamento centralizado de erros."""
    fallback = "Desculpe, encontrei um problema. Por favor, tente novamente."
    return {"final_response": fallback, "messages": [AIMessage(content=fallback)]}
