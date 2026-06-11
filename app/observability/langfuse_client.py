"""
Cliente LangFuse — Observabilidade e Rastreamento.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONCEITO: O QUE É OBSERVABILIDADE?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Trace = uma conversa completa (do input ao output)
Span  = cada etapa dentro da conversa

  Trace: "Quem ganhou 1970?"
    ├── Span: input_node     (2ms)
    ├── Span: router_node    (1823ms)
    ├── Span: rag_node       (312ms)
    ├── Span: process_node   (7891ms)
    └── Span: output_node    (1ms)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CORREÇÕES DESTA VERSÃO:
- flush_at=1 e flush_interval=0: envia imediatamente
- auth_check() na inicialização para validar conexão
- Reseta _current_trace a cada novo trace (evita vazamento)
- Tratamento robusto de erros em todos os métodos
"""

import time
from typing import Any
from app.config.settings import settings


class LangFuseTracker:
    """
    Wrapper do LangFuse para rastreamento de conversas.
    Se o LangFuse falhar, o chatbot continua normalmente.
    """

    def __init__(self):
        self._client = None
        self._current_trace = None
        self._spans: dict[str, Any] = {}
        self._enabled = False
        self._initialize()

    def _initialize(self) -> None:
        """Inicializa e VALIDA a conexão com o LangFuse."""
        if not settings.langfuse_enabled:
            print("ℹ️  LangFuse não configurado — observabilidade desativada")
            print("   Configure LANGFUSE_PUBLIC_KEY e LANGFUSE_SECRET_KEY no .env")
            return

        try:
            from langfuse import Langfuse

            self._client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
                # ── Correções críticas para envio imediato ────────────
                # flush_at=1: envia cada evento individualmente
                # sem esperar acumular um lote
                flush_at=1,
                # flush_interval=0: sem delay entre envios
                flush_interval=0,
            )

            # Valida a conexão ANTES de marcar como enabled
            # auth_check() faz uma chamada real à API do LangFuse
            is_connected = self._client.auth_check()

            if is_connected:
                self._enabled = True
                print("✅ LangFuse conectado — observabilidade ativa")
                print(f"   Host: {settings.langfuse_host}")
            else:
                print("⚠️  LangFuse: auth_check() falhou — verifique as chaves no .env")

        except ImportError:
            print("⚠️  Biblioteca langfuse não instalada")
            print("   Execute: poetry add langfuse")
        except Exception as e:
            print(f"⚠️  Erro ao conectar LangFuse: {e}")
            print(f"   Verifique LANGFUSE_HOST no .env: {settings.langfuse_host}")

    @property
    def enabled(self) -> bool:
        return self._enabled and self._client is not None

    # ── Gerenciamento de Traces ────────────────────────────────────────

    def start_trace(
        self,
        user_input: str,
        session_id: str | None = None,
    ) -> str | None:
        """
        Inicia um novo trace para uma conversa.

        CORREÇÃO: reseta _current_trace antes de criar novo
        para evitar que traces se misturem entre conversas.
        """
        if not self.enabled:
            return None

        # Reseta o trace anterior
        self._current_trace = None
        self._spans = {}

        try:
            self._current_trace = self._client.trace(
                name="world-cup-chat",
                input=user_input,
                session_id=session_id,
                metadata={
                    "app": settings.app_name,
                    "env": settings.app_env,
                    "model": settings.bedrock_model_id,
                },
                tags=["world-cup", "chatbot", settings.app_env],
            )

            trace_id = self._current_trace.id
            print(f"   📡 LangFuse trace iniciado: {trace_id[:8]}...")
            return trace_id

        except Exception as e:
            print(f"⚠️  LangFuse start_trace error: {e}")
            return None

    def end_trace(
        self,
        output: str,
        intent: str | None = None,
        latency_ms: float | None = None,
    ) -> None:
        """
        Finaliza o trace e envia IMEDIATAMENTE para o LangFuse.

        CORREÇÃO: flush() explícito garante envio mesmo
        em ambientes com buffer de rede.
        """
        if not self.enabled or not self._current_trace:
            return

        try:
            self._current_trace.update(
                output=output,
                metadata={
                    "intent": intent,
                    "latency_ms": latency_ms,
                },
            )

            # Flush forçado — envia agora, não espera o buffer
            self._client.flush()
            print(f"   📡 LangFuse trace finalizado e enviado")

        except Exception as e:
            print(f"⚠️  LangFuse end_trace error: {e}")

    # ── Gerenciamento de Spans ─────────────────────────────────────────

    def start_span(self, name: str, input_data: Any = None) -> None:
        """Inicia um span para rastrear uma etapa específica."""
        if not self.enabled or not self._current_trace:
            return

        try:
            span = self._current_trace.span(
                name=name,
                input=str(input_data)[:500] if input_data else None,
            )
            self._spans[name] = span

        except Exception as e:
            print(f"⚠️  LangFuse start_span({name}) error: {e}")

    def end_span(
        self,
        name: str,
        output: Any = None,
        metadata: dict | None = None,
    ) -> None:
        """Finaliza um span com o resultado da etapa."""
        if not self.enabled or name not in self._spans:
            return

        try:
            span = self._spans.pop(name)
            span.end(
                output=str(output)[:500] if output else None,
                metadata=metadata or {},
            )

        except Exception as e:
            print(f"⚠️  LangFuse end_span({name}) error: {e}")

    def log_llm_call(
        self,
        name: str,
        prompt: str,
        response: str,
        model: str | None = None,
        latency_ms: float | None = None,
    ) -> None:
        """Registra uma chamada ao LLM com detalhes completos."""
        if not self.enabled or not self._current_trace:
            return

        try:
            self._current_trace.generation(
                name=name,
                model=model or settings.bedrock_model_id,
                input=prompt[:1000],
                output=response[:1000],
                metadata={"latency_ms": latency_ms},
            )

        except Exception as e:
            print(f"⚠️  LangFuse log_llm_call error: {e}")

    def log_score(
        self,
        name: str,
        value: float,
        comment: str | None = None,
    ) -> None:
        """Registra uma pontuação de qualidade para o trace."""
        if not self.enabled or not self._current_trace:
            return

        try:
            self._current_trace.score(
                name=name,
                value=value,
                comment=comment,
            )

        except Exception as e:
            print(f"⚠️  LangFuse log_score error: {e}")


# ── Singleton do tracker ───────────────────────────────────────────────────────
_tracker_instance: LangFuseTracker | None = None


def get_tracker() -> LangFuseTracker:
    """Retorna o singleton do LangFuseTracker."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = LangFuseTracker()
    return _tracker_instance