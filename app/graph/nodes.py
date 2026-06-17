"""
Nós do Grafo LangGraph — Arquitetura Dual de LLMs.

ARQUITETURA:
  Llama 3.3 (Bedrock) → perguntas históricas + RAG
  GPT-4o-mini (OpenAI) → dados em tempo real Copa 2026 via web search
"""

import json
import time
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from app.graph.state import GraphState


WORLD_CUP_SYSTEM_PROMPT = """Você é um especialista em Copa do Mundo FIFA com conhecimento \
profundo sobre toda a história do torneio desde 1930 até 2026.

Diretrizes:
- Responda SEMPRE em português brasileiro
- Para dados históricos: use os documentos fornecidos
- Para Copa 2026: priorize dados em tempo real quando disponíveis
- Seja conciso — máximo 2 parágrafos"""


# ─────────────────────────────────────────────────────────────────────────────
# NÓ 1: Input + Guardrail
# ─────────────────────────────────────────────────────────────────────────────

def input_node(state: GraphState) -> dict:
    from app.guardrails.response_validator import run_input_guardrail
    from app.observability.langfuse_client import get_tracker

    user_input = state["user_input"]
    print(f"\n📥 [input_node] Recebendo: '{user_input}'")

    tracker = get_tracker()
    session_id = (state.get("metadata") or {}).get("session_id")
    trace_id = tracker.start_trace(user_input=user_input, session_id=session_id)
    tracker.start_span("input_node", input_data=user_input)

    passed, block_reason = run_input_guardrail(user_input)

    if not passed:
        print(f"🛡️  [input_node] Bloqueado: {block_reason}")
        tracker.log_score("guardrail_input", 0.0, comment=block_reason)
        tracker.end_span("input_node", output="BLOQUEADO")
        block_response = f"⚠️ Não consigo processar esta solicitação. {block_reason}"
        tracker.end_trace(output=block_response, intent="blocked")
        return {
            "messages": [HumanMessage(content=user_input)],
            "final_response": block_response,
            "error": "input_blocked",
            "metadata": {
                "start_time": time.time(),
                "node_path": ["input_node"],
                "trace_id": trace_id,
            },
        }

    tracker.log_score("guardrail_input", 1.0)
    tracker.end_span("input_node", output="valid")

    return {
        "messages": [HumanMessage(content=user_input)],
        "metadata": {
            "start_time": time.time(),
            "node_path": ["input_node"],
            "trace_id": trace_id,
        },
        "error": None,
        "final_response": None,
        "intent": None,
        "retrieved_context": None,
        "api_data": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NÓ 2: Router
# ─────────────────────────────────────────────────────────────────────────────

def router_node(state: GraphState) -> dict:
    from app.agents.router_agent import router_node as _router
    from app.observability.langfuse_client import get_tracker

    if state.get("error") == "input_blocked":
        return {}

    tracker = get_tracker()
    tracker.start_span("router_node", input_data=state["user_input"])
    result = _router(state)
    tracker.end_span("router_node", output=result.get("intent"))
    return result


def routing_function(state: GraphState) -> str:
    if state.get("error") == "input_blocked":
        return "off_topic"
    from app.agents.router_agent import routing_function as _route
    return _route(state)


# ─────────────────────────────────────────────────────────────────────────────
# NÓ 3: RAG — Llama 3.3 + ChromaDB para histórico
# ─────────────────────────────────────────────────────────────────────────────

def rag_node(state: GraphState) -> dict:
    from app.tools.vector_store import search_similar_documents
    from app.observability.langfuse_client import get_tracker

    user_input = state["user_input"]
    print(f"🔍 [rag_node] Buscando documentos relevantes...")

    tracker = get_tracker()
    tracker.start_span("rag_node", input_data=user_input)

    try:
        contexts = search_similar_documents(query=user_input, k=3)
        print(f"   ✅ {len(contexts)} trechos encontrados")

        metadata = state.get("metadata", {}) or {}
        metadata["node_path"] = metadata.get("node_path", []) + ["rag_node"]
        metadata["rag_chunks_found"] = len(contexts)

        tracker.end_span("rag_node", output=f"{len(contexts)} chunks")
        return {"retrieved_context": contexts or [], "metadata": metadata}

    except Exception as e:
        print(f"   ⚠️ Erro no RAG: {str(e)[:80]}")
        metadata = state.get("metadata", {}) or {}
        metadata["node_path"] = metadata.get("node_path", []) + ["rag_node"]
        tracker.end_span("rag_node", output="error")
        return {"retrieved_context": [], "metadata": metadata}


# ─────────────────────────────────────────────────────────────────────────────
# NÓ 4: API — GPT-4o-mini + Web Search para Copa 2026 em tempo real
# ─────────────────────────────────────────────────────────────────────────────

def api_node(state: GraphState) -> dict:
    """
    Nó de dados em tempo real.

    USA GPT-4o-mini com web_search para buscar dados atuais da Copa 2026.
    Não chama o Llama — retorna a resposta diretamente no api_data.
    O process_node detecta isso e usa a resposta sem chamar outro LLM.
    """
    from app.tools.web_search_tool import search_copa_2026_realtime
    from app.observability.langfuse_client import get_tracker

    user_input = state["user_input"]
    print(f"🌐 [api_node] Buscando Copa 2026 em tempo real...")    

    tracker = get_tracker()
    tracker.start_span("api_node", input_data=user_input)

    try:
        # GPT-4o-mini busca na web e retorna resposta já formatada
        resposta_openai = search_copa_2026_realtime(user_input)
        print(f"   ✅ Resposta recebida do GPT-4o-mini ({len(resposta_openai)} chars)")

        metadata = state.get("metadata", {}) or {}
        metadata["node_path"] = metadata.get("node_path", []) + ["api_node"]
        metadata["api_source"] = "openai_web_search"

        tracker.end_span("api_node", output="openai_web_search")

        # Salva a resposta pronta no api_data com flag especial
        # O process_node vai detectar "resposta_direta" e não chamar o Llama
        return {
            "api_data": {
                "source": "openai_web_search",
                "resposta_direta": resposta_openai,
            },
            "metadata": metadata,
        }

    except Exception as e:
        print(f"   ❌ Erro no web search: {str(e)[:80]}")
        metadata = state.get("metadata", {}) or {}
        metadata["node_path"] = metadata.get("node_path", []) + ["api_node"]
        tracker.end_span("api_node", output="error")
        return {
            "api_data": {"source": "error", "error": str(e)},
            "metadata": metadata,
        }


# ─────────────────────────────────────────────────────────────────────────────
# NÓ 5: Direct
# ─────────────────────────────────────────────────────────────────────────────

def direct_node(state: GraphState) -> dict:
    from app.observability.langfuse_client import get_tracker
    print(f"💬 [direct_node] Resposta direta...")
    tracker = get_tracker()
    tracker.start_span("direct_node")
    tracker.end_span("direct_node")
    metadata = state.get("metadata", {}) or {}
    metadata["node_path"] = metadata.get("node_path", []) + ["direct_node"]
    return {"metadata": metadata}


# ─────────────────────────────────────────────────────────────────────────────
# NÓ 6: Off-topic
# ─────────────────────────────────────────────────────────────────────────────

def off_topic_node(state: GraphState) -> dict:
    from app.observability.langfuse_client import get_tracker

    if state.get("error") == "input_blocked":
        metadata = state.get("metadata", {}) or {}
        metadata["node_path"] = metadata.get("node_path", []) + ["off_topic_node"]
        return {"metadata": metadata}

    print(f"🚫 [off_topic_node] Fora do escopo")
    tracker = get_tracker()
    tracker.start_span("off_topic_node")
    tracker.end_span("off_topic_node")

    response = (
        "Desculpe, sou especializado exclusivamente em Copa do Mundo FIFA! 🏆\n\n"
        "Posso te ajudar com:\n"
        "• História e campeões das Copas (1930-2022)\n"
        "• Resultados e dados em tempo real da Copa 2026\n"
        "• Artilheiros e estatísticas\n"
        "• Curiosidades e recordes\n\n"
        "Tem alguma pergunta sobre a Copa do Mundo?"
    )

    metadata = state.get("metadata", {}) or {}
    metadata["node_path"] = metadata.get("node_path", []) + ["off_topic_node"]
    return {"final_response": response, "metadata": metadata}


# ─────────────────────────────────────────────────────────────────────────────
# NÓ 7: Process — detecta qual LLM usar
# ─────────────────────────────────────────────────────────────────────────────

def process_node(state: GraphState) -> dict:
    """
    Nó de processamento com lógica dual de LLMs:

    ROTA API (tempo real):
      → api_node já chamou o GPT-4o-mini
      → api_data["resposta_direta"] já tem a resposta
      → process_node só passa adiante SEM chamar Llama

    ROTA RAG/DIRECT (histórico):
      → Chama Llama 3.3 (Bedrock) com contexto do ChromaDB
    """
    from app.config.aws_config import get_llm
    from app.observability.langfuse_client import get_tracker

    if state.get("final_response"):
        print(f"⚙️  [process_node] Resposta já gerada, pulando LLM")
        metadata = state.get("metadata", {}) or {}
        metadata["node_path"] = metadata.get("node_path", []) + ["process_node"]
        return {"metadata": metadata}

    intent = state.get("intent", "rag")
    api_data = state.get("api_data") or {}
    tracker = get_tracker()
    tracker.start_span("process_node", input_data={"intent": intent})

    # ── ROTA API: resposta já veio do GPT-4o-mini ─────────────────────
    if intent == "api" and api_data.get("resposta_direta"):
        print(f"⚙️  [process_node] Usando resposta do GPT-4o-mini (web search)")
        answer = api_data["resposta_direta"]
        print(f"   ✅ Resposta direta: {len(answer)} chars")

        tracker.end_span("process_node", output=answer[:200])
        metadata = state.get("metadata", {}) or {}
        metadata["node_path"] = metadata.get("node_path", []) + ["process_node"]
        metadata["llm_usado"] = "gpt-4o-mini (openai web search)"

        return {"final_response": answer, "metadata": metadata}

    # ── ROTA RAG/DIRECT: chama Llama 3.3 (Bedrock) ────────────────────
    print(f"⚙️  [process_node] Chamando Llama 3.3 (Bedrock)...")

    try:
        retrieved_context = state.get("retrieved_context") or []

        if retrieved_context and intent == "rag":
            context_text = "\n\n---\n\n".join(retrieved_context)
            system_content = f"""{WORLD_CUP_SYSTEM_PROMPT}

CONTEXTO DOS DOCUMENTOS:
━━━━━━━━━━━━━━━━━━━━━━━━
{context_text}
━━━━━━━━━━━━━━━━━━━━━━━━
Responda baseado principalmente neste contexto."""
            print(f"   📚 Modo RAG ({len(retrieved_context)} trechos) — Llama 3.3")
        else:
            system_content = WORLD_CUP_SYSTEM_PROMPT
            print(f"   💬 Modo Direct — Llama 3.3")

        messages_to_send = [SystemMessage(content=system_content), *state["messages"]]
        llm = get_llm()
        start_llm = time.time()
        print(f"   🌐 Enviando {len(messages_to_send)} mensagens ao Llama 3.3...")
        response = llm.invoke(messages_to_send)
        llm_latency = round((time.time() - start_llm) * 1000, 2)
        answer = response.content
        print(f"   ✅ Resposta Llama 3.3: {len(answer)} chars em {llm_latency}ms")

        tracker.log_llm_call(
            "llama_generation",
            system_content[:300],
            answer[:300],
            latency_ms=llm_latency,
        )
        tracker.end_span("process_node", output=answer[:200])

        metadata = state.get("metadata", {}) or {}
        metadata["node_path"] = metadata.get("node_path", []) + ["process_node"]
        metadata["llm_usado"] = "llama-3.3-70b (bedrock)"
        metadata["used_rag"] = bool(retrieved_context and intent == "rag")

        return {"final_response": answer, "metadata": metadata}

    except Exception as e:
        error_msg = str(e)
        print(f"   ❌ Erro: {error_msg[:80]}")
        tracker.end_span("process_node", output="error")
        metadata = state.get("metadata", {}) or {}
        metadata["node_path"] = metadata.get("node_path", []) + ["process_node"]
        return {"error": error_msg, "final_response": None, "metadata": metadata}


# ─────────────────────────────────────────────────────────────────────────────
# NÓ 8: Output
# ─────────────────────────────────────────────────────────────────────────────

def output_node(state: GraphState) -> dict:
    from app.guardrails.response_validator import run_output_guardrail
    from app.observability.langfuse_client import get_tracker

    error = state.get("error")
    intent = state.get("intent", "rag")
    tracker = get_tracker()
    tracker.start_span("output_node")

    if error and error != "input_blocked":
        final_response = "Desculpe, tive um problema. Por favor, tente novamente."
    else:
        final_response = state.get("final_response") or "Não consegui gerar uma resposta."

    if error != "input_blocked":
        passed, final_response, guard_reason = run_output_guardrail(final_response, intent)
        tracker.log_score("guardrail_output", 1.0 if passed else 0.0, comment=guard_reason)

    print(f"📤 [output_node] Finalizando")

    metadata = state.get("metadata", {}) or {}
    start_time = metadata.get("start_time", time.time())
    latency_ms = round((time.time() - start_time) * 1000, 2)
    metadata["node_path"] = metadata.get("node_path", []) + ["output_node"]
    metadata["latency_ms"] = latency_ms

    llm_usado = metadata.get("llm_usado", "")
    source_map = {
        "rag":       f"📚 RAG + Llama 3.3",
        "api":       f"🌐 GPT-4o-mini + Web Search",
        "direct":    "💬 Llama 3.3 Direto",
        "off_topic": "🚫 Off-topic",
    }
    source_indicator = source_map.get(intent, "🧠 Modelo")
    metadata["source_indicator"] = source_indicator

    print(f"⏱️  Latência: {latency_ms}ms | Fonte: {source_indicator}")

    tracker.end_span("output_node", output=final_response[:200])
    tracker.log_score("total_latency_ok", 1.0 if latency_ms < 20000 else 0.5)
    tracker.end_trace(output=final_response, intent=intent, latency_ms=latency_ms)

    return {
        "messages": [AIMessage(content=final_response)],
        "final_response": final_response,
        "metadata": metadata,
    }


def error_node(state: GraphState) -> dict:
    fallback = "Desculpe, encontrei um problema. Por favor, tente novamente."
    return {"final_response": fallback, "messages": [AIMessage(content=fallback)]}
