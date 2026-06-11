"""
Gerenciador de Sessões — Fase 6.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONCEITO: POR QUE PRECISAMOS DE SESSÕES?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
No terminal (Fases 1-5), havia um loop contínuo:
  while True: pergunta → resposta → pergunta...
  O estado (histórico) ficava vivo na variável `current_state`

Na API HTTP, cada requisição é INDEPENDENTE:
  POST /chat → processa → retorna → ESQUECE
  POST /chat → processa → retorna → ESQUECE

Para ter histórico entre requisições, precisamos
guardar o estado de cada usuário em memória.

SESSÃO = estado de um usuário específico
  session_id: "usuario-abc123"
  state: {messages: [...], intent: "rag", ...}

Analogia: é como um garçom que lembra o pedido
de cada mesa pelo número dela.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import uuid
from datetime import datetime, timedelta
from app.graph.state import GraphState


class SessionManager:
    """
    Gerencia o estado de cada sessão de usuário em memória.

    IMPORTANTE: Esta implementação usa memória RAM.
    Em produção (Fase 7), usaríamos Redis para persistência.
    Se o servidor reiniciar, as sessões são perdidas.
    """

    def __init__(self, session_timeout_minutes: int = 30):
        # Dicionário: session_id → {state, last_activity}
        self._sessions: dict[str, dict] = {}
        self._timeout = timedelta(minutes=session_timeout_minutes)

    def create_session(self, session_id: str | None = None) -> str:
        """
        Cria uma nova sessão e retorna o ID.

        Se session_id for fornecido, usa ele.
        Se não, gera um UUID único.
        """
        if not session_id:
            session_id = str(uuid.uuid4())

        # Estado inicial — igual ao main.py da Fase 5
        initial_state: GraphState = {
            "messages": [],
            "user_input": "",
            "intent": None,
            "retrieved_context": None,
            "api_data": None,
            "final_response": None,
            "error": None,
            "metadata": {"session_id": session_id},
        }

        self._sessions[session_id] = {
            "state": initial_state,
            "created_at": datetime.now(),
            "last_activity": datetime.now(),
            "message_count": 0,
        }

        return session_id

    def get_state(self, session_id: str) -> GraphState | None:
        """
        Retorna o estado da sessão.
        Retorna None se sessão não existe ou expirou.
        """
        session = self._sessions.get(session_id)
        if not session:
            return None

        # Verifica timeout
        if datetime.now() - session["last_activity"] > self._timeout:
            self.delete_session(session_id)
            return None

        return session["state"]

    def update_state(self, session_id: str, new_state: GraphState) -> None:
        """Atualiza o estado após uma interação."""
        if session_id not in self._sessions:
            return

        self._sessions[session_id]["state"] = new_state
        self._sessions[session_id]["last_activity"] = datetime.now()
        self._sessions[session_id]["message_count"] += 1

    def get_history(self, session_id: str) -> list[dict]:
        """
        Retorna o histórico de mensagens formatado para a API.
        """
        session = self._sessions.get(session_id)
        if not session:
            return []

        messages = session["state"].get("messages", [])
        return [
            {
                "role": "user" if msg.type == "human" else "assistant",
                "content": msg.content,
            }
            for msg in messages
        ]

    def delete_session(self, session_id: str) -> None:
        """Remove uma sessão."""
        self._sessions.pop(session_id, None)

    def cleanup_expired(self) -> int:
        """Remove sessões expiradas. Retorna quantas foram removidas."""
        now = datetime.now()
        expired = [
            sid for sid, data in self._sessions.items()
            if now - data["last_activity"] > self._timeout
        ]
        for sid in expired:
            del self._sessions[sid]
        return len(expired)

    @property
    def active_sessions(self) -> int:
        """Número de sessões ativas."""
        return len(self._sessions)


# Singleton — uma instância compartilhada por toda a aplicação
session_manager = SessionManager()
