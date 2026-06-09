"""
Ponto de entrada da aplicação — Interface de linha de comando (CLI).

Por que começar com CLI e não com API?
  - Mais simples para testar e debugar
  - Sem camada HTTP para adicionar complexidade
  - Foco total na lógica do agente
  - A API REST será adicionada na Fase 6

Como executar:
  python -m app.main
  ou
  python app/main.py
"""

import sys
from app.graph.builder import get_graph
from app.graph.state import GraphState


def run_chat() -> None:
    """
    Loop principal do chatbot em modo CLI.

    Mantém um histórico de mensagens entre as perguntas
    para simular uma conversa real com contexto.
    """
    print("=" * 60)
    print("🏆  BEM-VINDO AO CHATBOT DA COPA DO MUNDO  🏆")
    print("=" * 60)
    print("Fase 1 — Modo demonstração (sem LLM)")
    print("Digite 'sair' ou 'exit' para encerrar")
    print("Digite 'estado' para ver o estado atual do grafo")
    print("-" * 60)

    # Obtém o grafo compilado
    graph = get_graph()

    # Estado inicial — começa vazio, será preenchido ao longo da conversa
    current_state: GraphState = {
        "messages": [],
        "user_input": "",
        "intent": None,
        "retrieved_context": None,
        "final_response": None,
        "error": None,
        "metadata": None,
    }

    while True:
        try:
            # Lê a entrada do usuário
            user_input = input("\n👤 Você: ").strip()

            # Comandos especiais
            if not user_input:
                continue

            if user_input.lower() in ("sair", "exit", "quit"):
                print("\n👋 Até logo! Que venha a próxima Copa! 🏆")
                break

            if user_input.lower() == "estado":
                _print_state(current_state)
                continue

            # ── Atualiza o estado com a nova pergunta ─────────────────────
            current_state["user_input"] = user_input

            # ── Invoca o grafo ─────────────────────────────────────────────
            # O grafo recebe o estado atual e retorna o estado atualizado
            # Todos os nós são executados em sequência
            print("\n🔄 Processando...\n")
            current_state = graph.invoke(current_state)

            # ── Exibe a resposta ───────────────────────────────────────────
            response = current_state.get("final_response", "Sem resposta")
            print(f"\n🤖 Agente: {response}")

            # Exibe metadados em modo desenvolvimento
            metadata = current_state.get("metadata", {})
            if metadata:
                latency = metadata.get("latency_ms", "?")
                path = " → ".join(metadata.get("node_path", []))
                print(f"\n   📊 Latência: {latency}ms | Caminho: {path}")

        except KeyboardInterrupt:
            print("\n\n👋 Interrompido pelo usuário. Até logo!")
            break
        except Exception as e:
            print(f"\n❌ Erro inesperado: {e}")
            print("   Tente novamente ou verifique os logs.")


def _print_state(state: GraphState) -> None:
    """Exibe o estado atual do grafo de forma legível."""
    print("\n" + "=" * 40)
    print("📋 ESTADO ATUAL DO GRAFO")
    print("=" * 40)
    print(f"• Mensagens no histórico: {len(state.get('messages', []))}")
    print(f"• Última entrada: {state.get('user_input', 'N/A')}")
    print(f"• Intenção detectada: {state.get('intent', 'N/A')}")
    print(f"• Erro: {state.get('error', 'Nenhum')}")

    messages = state.get("messages", [])
    if messages:
        print("\n📜 Histórico de mensagens:")
        for i, msg in enumerate(messages[-6:], 1):  # Últimas 6 mensagens
            role = "👤" if msg.type == "human" else "🤖"
            content = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
            print(f"   {i}. {role} {content}")
    print("=" * 40)


if __name__ == "__main__":
    run_chat()
