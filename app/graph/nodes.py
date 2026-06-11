"""
Nós do Grafo LangGraph — Fase 5.

Novidades em relação à Fase 4:
- Guardrails de entrada no input_node
- Guardrails de saída no output_node
- LangFuse tracking em todos os nós
- Scores de qualidade registrados automaticamente
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
# NÓ 1: Input — MODIFICADO com Guardrail de Entrada + LangFuse
# ─────────────────────────────────────────────────────────────────────────────

def input_node(state: GraphState) -> dict:
    """
    Recebe a entrada e aplica guardrail de entrada.

    Novidades Fase 5:
    - Valida a pergunta antes de processar
    - Inicia o trace no LangFuse
    - Bloqueia conteúdo inadequado imediatamente
    """
    from app.guardrails.response_validator import run_input_guardrail
    from app.observability.langfuse_client import get_tracker

    user_input = state["user_input"]
    print(f"\n📥 [input_node] Recebendo: '{user_input}'")

    tracker = get_tracker()

    # ── Inicia o trace no LangFuse ────────────────────────────────────
    session_id = state.get("metadata", {}) and state["metadata"].get("session_id")
    trace_id = tracker.start_trace(
        user_input=user_input,
        session_id=session_id,
    )
    tracker.start_span("input_node", input_data=user_input)

    # ── Guardrail de Entrada ──────────────────────────────────────────
    passed, block_reason = run_input_guardrail(user_input)

    if not passed:
        print(f"🛡️  [input_node] Guardrail bloqueou a entrada")

        # Registra o bloqueio no LangFuse
        tracker.log_score("guardrail_input", 0.0, comment=block_reason)
        tracker.end_span("input_node", output=f"BLOQUEADO: {block_reason}")

        # Resposta de bloqueio — não processa mais nada
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
                "guardrail_blocked": True,
            },
        }

    # ── Entrada válida — continua o fluxo normal ──────────────────────
    tracker.log_score("guardrail_input", 1.0, comment="Input válido")
    tracker.end_span("input_node", output="valid")

    return {
        "messages": [HumanMessage(content=user_input)],
        "metadata": {
            "start_time": time.time(),
            "node_path": ["input_node"],
            "trace_id": trace_id,
            "guardrail_blocked": False,
        },
        "error": None,
        "final_response": None,
        "intent": None,
        "retrieved_context": None,
        "api_data": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NÓ 2: Router — com LangFuse span
# ─────────────────────────────────────────────────────────────────────────────

def router_node(state: GraphState) -> dict:
    """Router com rastreamento LangFuse."""
    from app.agents.router_agent import router_node as _router
    from app.observability.langfuse_client import get_tracker

    # Se entrada foi bloqueada, pula o router
    if state.get("error") == "input_blocked":
        return {}

    tracker = get_tracker()
    tracker.start_span("router_node", input_data=state["user_input"])

    result = _router(state)

    tracker.end_span(
        "router_node",
        output=result.get("intent"),
        metadata={
            "confidence": result.get("metadata", {}).get("router_confidence"),
            "reason": result.get("metadata", {}).get("router_reason"),
        },
    )
    return result


def routing_function(state: GraphState) -> str:
    """Função de roteamento — pula se entrada bloqueada."""
    if state.get("error") == "input_blocked":
        return "off_topic"  # redireciona para nó que não chama LLM
    from app.agents.router_agent import routing_function as _route
    return _route(state)


# ─────────────────────────────────────────────────────────────────────────────
# NÓ 3: RAG — com LangFuse span
# ─────────────────────────────────────────────────────────────────────────────

def rag_node(state: GraphState) -> dict:
    """Busca documentos com rastreamento LangFuse."""
    from app.tools.vector_store import search_similar_documents
    from app.observability.langfuse_client import get_tracker

    user_input = state["user_input"]
    print(f"🔍 [rag_node] Buscando documentos relevantes...")

    tracker = get_tracker()
    tracker.start_span("rag_node", input_data=user_input)

    try:
        contexts = search_similar_documents(query=user_input, k=3)

        if contexts:
            print(f"   ✅ {len(contexts)} trechos encontrados")
        else:
            print(f"   ⚠️  Nenhum trecho encontrado")

        metadata = state.get("metadata", {}) or {}
        metadata["node_path"] = metadata.get("node_path", []) + ["rag_node"]
        metadata["rag_chunks_found"] = len(contexts)

        tracker.end_span("rag_node", output=f"{len(contexts)} chunks", metadata={"chunks": len(contexts)})
        return {"retrieved_context": contexts or [], "metadata": metadata}

    except Exception as e:
        print(f"   ⚠️  Erro no RAG: {str(e)[:80]}")
        metadata = state.get("metadata", {}) or {}
        metadata["node_path"] = metadata.get("node_path", []) + ["rag_node"]
        tracker.end_span("rag_node", output=f"error: {str(e)[:50]}")
        return {"retrieved_context": [], "metadata": metadata}


# ─────────────────────────────────────────────────────────────────────────────
# NÓ 4: API — com LangFuse span
# ─────────────────────────────────────────────────────────────────────────────

def api_node(state: GraphState) -> dict:
    """Consulta API externa com rastreamento LangFuse."""
    from app.tools.http_client import fetch_world_cup_standings, fetch_world_cup_news
    from app.observability.langfuse_client import get_tracker

    user_input = state["user_input"].lower()
    print(f"🌐 [api_node] Consultando API externa...")

    tracker = get_tracker()
    tracker.start_span("api_node", input_data=user_input)

    try:
        if any(w in user_input for w in ["notícia", "noticia", "recente", "novo"]):
            print(f"   📰 Buscando notícias...")
            api_data = fetch_world_cup_news()
        else:
            print(f"   📊 Buscando classificação...")
            api_data = fetch_world_cup_standings()

        source = api_data.get("source", "unknown")
        print(f"   ✅ Dados recebidos (fonte: {source})")

        metadata = state.get("metadata", {}) or {}
        metadata["node_path"] = metadata.get("node_path", []) + ["api_node"]
        metadata["api_source"] = source

        tracker.end_span("api_node", output=source, metadata={"source": source})
        return {"api_data": api_data, "metadata": metadata}

    except Exception as e:
        print(f"   ❌ Erro na API: {str(e)[:80]}")
        metadata = state.get("metadata", {}) or {}
        metadata["node_path"] = metadata.get("node_path", []) + ["api_node"]
        tracker.end_span("api_node", output=f"error: {str(e)[:50]}")
        return {"api_data": {"source": "error", "error": str(e)}, "metadata": metadata}


# ─────────────────────────────────────────────────────────────────────────────
# NÓ 5: Direct — com LangFuse span
# ─────────────────────────────────────────────────────────────────────────────

def direct_node(state: GraphState) -> dict:
    """Resposta direta com rastreamento."""
    from app.observability.langfuse_client import get_tracker

    print(f"💬 [direct_node] Resposta direta...")
    tracker = get_tracker()
    tracker.start_span("direct_node")
    tracker.end_span("direct_node", output="direct_response")

    metadata = state.get("metadata", {}) or {}
    metadata["node_path"] = metadata.get("node_path", []) + ["direct_node"]
    return {"metadata": metadata}


# ─────────────────────────────────────────────────────────────────────────────
# NÓ 6: Off-Topic — com LangFuse span
# ─────────────────────────────────────────────────────────────────────────────

def off_topic_node(state: GraphState) -> dict:
    """Off-topic com rastreamento."""
    from app.observability.langfuse_client import get_tracker

    # Se foi bloqueado pelo guardrail, não exibe a mensagem de off-topic
    if state.get("error") == "input_blocked":
        metadata = state.get("metadata", {}) or {}
        metadata["node_path"] = metadata.get("node_path", []) + ["off_topic_node"]
        return {"metadata": metadata}

    print(f"🚫 [off_topic_node] Pergunta fora do escopo")

    tracker = get_tracker()
    tracker.start_span("off_topic_node")
    tracker.end_span("off_topic_node", output="off_topic_response")

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

    return {"final_response": off_topic_response, "metadata": metadata}


# ─────────────────────────────────────────────────────────────────────────────
# NÓ 7: Process — com LangFuse generation tracking
# ─────────────────────────────────────────────────────────────────────────────

def process_node(state: GraphState) -> dict:
    """Processamento com rastreamento completo de geração LLM."""
    from app.config.aws_config import get_llm
    from app.observability.langfuse_client import get_tracker

    if state.get("final_response"):
        print(f"⚙️  [process_node] Resposta já gerada, pulando LLM")
        metadata = state.get("metadata", {}) or {}
        metadata["node_path"] = metadata.get("node_path", []) + ["process_node"]
        return {"metadata": metadata}

    print(f"⚙️  [process_node] Chamando LLM...")
    intent = state.get("intent", "rag")
    tracker = get_tracker()
    tracker.start_span("process_node", input_data={"intent": intent})

    try:
        retrieved_context = state.get("retrieved_context") or []
        api_data = state.get("api_data")

        if retrieved_context and intent == "rag":
            context_text = "\n\n---\n\n".join(retrieved_context)
            system_content = f"""{WORLD_CUP_SYSTEM_PROMPT}

CONTEXTO DOS DOCUMENTOS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{context_text}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Responda baseado principalmente neste contexto."""
            print(f"   📚 Modo RAG ({len(retrieved_context)} trechos)")

        elif api_data and intent == "api":
            api_text = json.dumps(api_data, ensure_ascii=False, indent=2)
            system_content = f"""{WORLD_CUP_SYSTEM_PROMPT}

DADOS DA API EXTERNA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{api_text}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use estes dados para responder."""
            print(f"   🌐 Modo API")
        else:
            system_content = WORLD_CUP_SYSTEM_PROMPT
            print(f"   💬 Modo Direct")

        messages_to_send = [
            SystemMessage(content=system_content),
            *state["messages"],
        ]

        llm = get_llm()
        start_llm = time.time()
        print(f"   🌐 Enviando {len(messages_to_send)} mensagens...")
        response = llm.invoke(messages_to_send)
        llm_latency = round((time.time() - start_llm) * 1000, 2)

        answer = response.content
        print(f"   ✅ Resposta recebida ({len(answer)} chars)")

        # Registra a chamada LLM no LangFuse
        tracker.log_llm_call(
            name="main_generation",
            prompt=system_content[:500],
            response=answer[:500],
            latency_ms=llm_latency,
        )
        tracker.end_span("process_node", output=answer[:200])

        metadata = state.get("metadata", {}) or {}
        metadata["node_path"] = metadata.get("node_path", []) + ["process_node"]
        metadata["used_rag"] = bool(retrieved_context and intent == "rag")
        metadata["used_api"] = bool(api_data and intent == "api")
        metadata["llm_latency_ms"] = llm_latency

        return {"final_response": answer, "metadata": metadata}

    except Exception as e:
        error_msg = str(e)
        print(f"   ❌ Erro: {error_msg[:80]}")
        tracker.end_span("process_node", output=f"error: {error_msg[:50]}")
        metadata = state.get("metadata", {}) or {}
        metadata["node_path"] = metadata.get("node_path", []) + ["process_node"]
        return {"error": error_msg, "final_response": None, "metadata": metadata}


# ─────────────────────────────────────────────────────────────────────────────
# NÓ 8: Output — MODIFICADO com Guardrail de Saída + finaliza LangFuse
# ─────────────────────────────────────────────────────────────────────────────

def output_node(state: GraphState) -> dict:
    """
    Finaliza com guardrail de saída e fecha o trace LangFuse.

    Novidades Fase 5:
    - Valida a resposta antes de entregar
    - Registra score de qualidade no LangFuse
    - Fecha o trace com todas as métricas
    """
    from app.guardrails.response_validator import run_output_guardrail
    from app.observability.langfuse_client import get_tracker

    error = state.get("error")
    intent = state.get("intent", "rag")
    tracker = get_tracker()
    tracker.start_span("output_node")

    if error and error != "input_blocked":
        print(f"⚠️  [output_node] Erro detectado, usando fallback")
        final_response = "Desculpe, tive um problema. Por favor, tente novamente."
    else:
        final_response = state.get("final_response") or "Não consegui gerar uma resposta."

    # ── Guardrail de Saída ────────────────────────────────────────────
    if error != "input_blocked":
        passed, final_response, guard_reason = run_output_guardrail(
            final_response, intent
        )

        if not passed:
            print(f"🛡️  [output_node] Guardrail de saída bloqueou resposta")
            final_response = (
                "Desculpe, não consegui gerar uma resposta adequada. "
                "Por favor, reformule sua pergunta."
            )
            tracker.log_score("guardrail_output", 0.0, comment=guard_reason)
        else:
            tracker.log_score("guardrail_output", 1.0, comment="Output válido")

    print(f"📤 [output_node] Finalizando")

    # ── Métricas finais ───────────────────────────────────────────────
    metadata = state.get("metadata", {}) or {}
    start_time = metadata.get("start_time", time.time())
    latency_ms = round((time.time() - start_time) * 1000, 2)
    metadata["node_path"] = metadata.get("node_path", []) + ["output_node"]
    metadata["latency_ms"] = latency_ms

    intent_display = metadata.get("intent", intent)
    source_map = {
        "rag":       f"📚 RAG ({metadata.get('rag_chunks_found', 0)} trechos)",
        "api":       f"🌐 API ({metadata.get('api_source', 'externa')})",
        "direct":    "💬 Direto",
        "off_topic": "🚫 Off-topic",
        "blocked":   "🛡️ Bloqueado",
    }
    source_indicator = source_map.get(intent_display, "🧠 Modelo")
    metadata["source_indicator"] = source_indicator

    print(f"⏱️  Latência: {latency_ms}ms | Rota: {source_indicator}")

    # ── Fecha o trace no LangFuse ─────────────────────────────────────
    tracker.end_span("output_node", output=final_response[:200])
    tracker.log_score(
        "total_latency_ok",
        1.0 if latency_ms < 15000 else 0.5,
        comment=f"Latência: {latency_ms}ms",
    )
    tracker.end_trace(
        output=final_response,
        intent=intent_display,
        latency_ms=latency_ms,
    )

    return {
        "messages": [AIMessage(content=final_response)],
        "final_response": final_response,
        "metadata": metadata,
    }


def error_node(state: GraphState) -> dict:
    """Tratamento centralizado de erros."""
    fallback = "Desculpe, encontrei um problema. Por favor, tente novamente."
    return {"final_response": fallback, "messages": [AIMessage(content=fallback)]}
