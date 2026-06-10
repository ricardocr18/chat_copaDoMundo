"""
Ponto de entrada da aplicação — Interface CLI.
Fase 2: Atualizado para refletir o uso do LLM real.
"""

import sys
from app.graph.builder import get_graph
from app.graph.state import GraphState


def run_chat() -> None:
    """Loop principal do chatbot em modo CLI."""
    print("=" * 60)
    print("🏆  BEM-VINDO AO CHATBOT DA COPA DO MUNDO  🏆")
    print("=" * 60)
    print("Fase 2 — Claude via Amazon Bedrock (LLM real)")
    print("Digite 'sair' ou 'exit' para encerrar")
    print("Digite 'estado' para ver o estado atual do grafo")
    print("Digite 'limpar' para reiniciar o histórico")
    print("-" * 60)

    graph = get_graph()

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
            user_input = input("\n👤 Você: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("sair", "exit", "quit"):
                print("\n👋 Até logo! Que venha a próxima Copa! 🏆")
                break

            if user_input.lower() == "estado":
                _print_state(current_state)
                continue

            if user_input.lower() == "limpar":
                current_state["messages"] = []
                print("🗑️  Histórico limpo! Nova conversa iniciada.")
                continue

            current_state["user_input"] = user_input

            print("\n🔄 Consultando Claude...\n")
            current_state = graph.invoke(current_state)

            response = current_state.get("final_response", "Sem resposta")
            print(f"\n🤖 Agente: {response}")

            metadata = current_state.get("metadata", {})
            if metadata:
                latency = metadata.get("latency_ms", "?")
                model = metadata.get("model_id", "?")
                path = " → ".join(metadata.get("node_path", []))
                print(f"\n   📊 Latência: {latency}ms | Modelo: {model}")
                print(f"   🗺️  Caminho: {path}")

        except KeyboardInterrupt:
            print("\n\n👋 Interrompido. Até logo!")
            break
        except Exception as e:
            print(f"\n❌ Erro inesperado: {e}")


def _print_state(state: GraphState) -> None:
    """Exibe o estado atual do grafo."""
    print("\n" + "=" * 40)
    print("📋 ESTADO ATUAL DO GRAFO")
    print("=" * 40)
    print(f"• Mensagens no histórico: {len(state.get('messages', []))}")
    print(f"• Última entrada: {state.get('user_input', 'N/A')}")
    print(f"• Erro: {state.get('error', 'Nenhum')}")

    messages = state.get("messages", [])
    if messages:
        print("\n📜 Histórico:")
        for i, msg in enumerate(messages[-6:], 1):
            role = "👤" if msg.type == "human" else "🤖"
            content = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
            print(f"   {i}. {role} {content}")
    print("=" * 40)


if __name__ == "__main__":
    run_chat()
