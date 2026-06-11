"""
Rotas da API — Fase 6.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ENDPOINTS IMPLEMENTADOS:

POST /chat          → envia pergunta, recebe resposta
POST /chat/stream   → envia pergunta, recebe streaming
GET  /health        → verifica saúde da API
GET  /history/{id}  → histórico da sessão
DELETE /session/{id}→ apaga uma sessão
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import asyncio
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from api.schemas import (
    ChatRequest, ChatResponse,
    HealthResponse, HistoryResponse, ErrorResponse,
)
from api.session_manager import session_manager
from app.config.settings import settings
from app.graph.builder import get_graph

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# POST /chat — Resposta completa
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Enviar mensagem ao agente",
    description="Envia uma pergunta e recebe a resposta completa.",
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Endpoint principal do chatbot.

    CONCEITO: async def
    A função é assíncrona — enquanto espera o Bedrock
    responder (~10s), o servidor pode atender outras
    requisições. Sem async, ficaria bloqueado esperando.

    Fluxo:
    1. Recupera ou cria sessão
    2. Atualiza estado com a pergunta
    3. Invoca o grafo LangGraph
    4. Salva novo estado na sessão
    5. Retorna a resposta
    """
    # ── Gerencia a sessão ─────────────────────────────────────────────
    session_id = request.session_id
    state = None

    if session_id:
        state = session_manager.get_state(session_id)

    if not state:
        # Sessão não existe ou expirou — cria nova
        session_id = session_manager.create_session(session_id)
        state = session_manager.get_state(session_id)

    # ── Atualiza estado com a pergunta ────────────────────────────────
    state["user_input"] = request.message

    try:
        # ── Invoca o grafo (operação pesada — pode demorar ~10s) ──────
        # run_in_executor roda código síncrono em thread separada
        # sem bloquear o event loop do FastAPI
        graph = get_graph()
        loop = asyncio.get_event_loop()
        new_state = await loop.run_in_executor(
            None,  # usa thread pool padrão
            lambda: graph.invoke(state)
        )

        # ── Salva o novo estado ───────────────────────────────────────
        session_manager.update_state(session_id, new_state)

        # ── Monta a resposta ──────────────────────────────────────────
        metadata = new_state.get("metadata", {}) or {}

        return ChatResponse(
            response=new_state.get("final_response", ""),
            session_id=session_id,
            intent=new_state.get("intent"),
            latency_ms=metadata.get("latency_ms"),
            source=metadata.get("source_indicator"),
            node_path=metadata.get("node_path", []),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar: {str(e)[:200]}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# POST /chat/stream — Resposta em streaming
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/chat/stream",
    summary="Enviar mensagem com resposta em streaming",
    description="Envia uma pergunta e recebe a resposta palavra por palavra (SSE).",
)
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """
    Endpoint de streaming usando Server-Sent Events (SSE).

    CONCEITO: SERVER-SENT EVENTS (SSE)
    SSE é um protocolo onde o servidor envia dados
    continuamente para o cliente após uma única requisição.

    Formato de cada evento SSE:
      data: {"type": "token", "content": "O "}
      data: {"type": "token", "content": "Brasil"}
      data: {"type": "done", "session_id": "abc123"}

    O cliente lê esses eventos e vai montando o texto.
    """

    async def generate():
        """Gerador assíncrono que produz eventos SSE."""

        # ── Gerencia sessão ───────────────────────────────────────────
        session_id = request.session_id
        state = None

        if session_id:
            state = session_manager.get_state(session_id)

        if not state:
            session_id = session_manager.create_session(session_id)
            state = session_manager.get_state(session_id)

        state["user_input"] = request.message

        # ── Evento de início ──────────────────────────────────────────
        yield f"data: {json.dumps({'type': 'start', 'session_id': session_id})}\n\n"

        try:
            # ── Invoca o grafo ────────────────────────────────────────
            graph = get_graph()
            loop = asyncio.get_event_loop()
            new_state = await loop.run_in_executor(
                None,
                lambda: graph.invoke(state)
            )

            session_manager.update_state(session_id, new_state)

            # ── Simula streaming palavra por palavra ──────────────────
            # O Bedrock não suporta streaming nativo via LangChain ainda
            # Simulamos enviando a resposta em chunks de palavras
            response_text = new_state.get("final_response", "")
            words = response_text.split(" ")

            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                event = json.dumps({"type": "token", "content": chunk})
                yield f"data: {event}\n\n"
                # Pequena pausa para simular digitação
                await asyncio.sleep(0.03)

            # ── Evento de finalização ─────────────────────────────────
            metadata = new_state.get("metadata", {}) or {}
            done_event = json.dumps({
                "type": "done",
                "session_id": session_id,
                "intent": new_state.get("intent"),
                "latency_ms": metadata.get("latency_ms"),
                "source": metadata.get("source_indicator"),
            })
            yield f"data: {done_event}\n\n"

        except Exception as e:
            error_event = json.dumps({
                "type": "error",
                "message": str(e)[:200],
            })
            yield f"data: {error_event}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Desativa buffer do nginx
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /health — Health check
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Verificar saúde da API",
)
async def health() -> HealthResponse:
    """
    Health check endpoint.

    Usado por:
    - Load balancers (AWS ALB) para saber se está vivo
    - Monitoramento para detectar falhas
    - Docker para verificar se o container está saudável
    """
    # Verifica serviços dependentes
    services = {
        "api": "ok",
        "sessions": f"{session_manager.active_sessions} ativas",
        "langfuse": "ok" if settings.langfuse_enabled else "não configurado",
        "bedrock": "ok",
    }

    # Verifica se o grafo compila sem erro
    try:
        get_graph()
        services["graph"] = "ok"
    except Exception as e:
        services["graph"] = f"erro: {str(e)[:50]}"

    return HealthResponse(
        status="ok",
        app=settings.app_name,
        version="1.0.0",
        services=services,
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /history/{session_id} — Histórico da sessão
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/history/{session_id}",
    response_model=HistoryResponse,
    summary="Recuperar histórico da sessão",
)
async def get_history(session_id: str) -> HistoryResponse:
    """Retorna o histórico de mensagens de uma sessão."""
    session = session_manager.get_state(session_id)

    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Sessão '{session_id}' não encontrada ou expirada.",
        )

    messages = session_manager.get_history(session_id)

    return HistoryResponse(
        session_id=session_id,
        message_count=len(messages),
        messages=messages,
    )


# ─────────────────────────────────────────────────────────────────────────────
# DELETE /session/{session_id} — Apaga uma sessão
# ─────────────────────────────────────────────────────────────────────────────

@router.delete(
    "/session/{session_id}",
    summary="Apagar sessão",
)
async def delete_session(session_id: str) -> dict:
    """Apaga uma sessão e seu histórico."""
    state = session_manager.get_state(session_id)

    if not state:
        raise HTTPException(
            status_code=404,
            detail=f"Sessão '{session_id}' não encontrada.",
        )

    session_manager.delete_session(session_id)
    return {"message": f"Sessão '{session_id}' removida com sucesso."}
