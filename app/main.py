"""
Ponto de entrada — Fase 6.
Suporta modo CLI (terminal) e modo API (FastAPI).

Uso:
  CLI:  poetry run python -m app.main
  API:  poetry run uvicorn api.app:app --reload --port 8000
"""

from app.graph.builder import get_graph
from app.graph.state import GraphState


def run_chat() -> None:
    """Loop do chatbot em modo CLI."""
    print("=" * 60)
    print("🏆  BEM-VINDO AO CHATBOT DA COPA DO MUNDO  🏆")
    print("=" * 60)
    print("Fase 6 — API REST disponível em http://localhost:8000")
    print("CLI ativo | Digite 'sair' para encerrar")
    print("-" * 60)

    graph = get_graph()
    current_state: GraphState = {
        "messages": [],
        "user_input": "",
        "intent": None,
        "retrieved_context": None,
        "api_data": None,
        "final_response": None,
        "error": None,
        "metadata": None,
    }

    while True:
        try:
            user_input = input("\n👤 Você: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("sair", "exit"):
                print("\n👋 Até logo! 🏆")
                break
            if user_input.lower() == "estado":
                _print_state(current_state)
                continue
            if user_input.lower() == "limpar":
                current_state["messages"] = []
                print("🗑️  Histórico limpo!")
                continue

            current_state["user_input"] = user_input
            print("\n🔄 Processando...\n")
            current_state = graph.invoke(current_state)

            response = current_state.get("final_response", "Sem resposta")
            print(f"\n🤖 Agente: {response}")

            metadata = current_state.get("metadata", {}) or {}
            if metadata:
                latency = metadata.get("latency_ms", "?")
                source = metadata.get("source_indicator", "?")
                intent = metadata.get("intent", "?")
                confidence = metadata.get("router_confidence", 0)
                path = " → ".join(metadata.get("node_path", []))
                print(f"\n   📊 Latência: {latency}ms | Fonte: {source}")
                print(f"   🧭 Rota: {intent} ({confidence:.0%} confiança)")
                print(f"   🗺️  Caminho: {path}")

        except KeyboardInterrupt:
            print("\n\n👋 Até logo!")
            break
        except Exception as e:
            print(f"\n❌ Erro: {e}")


def _print_state(state: GraphState) -> None:
    print("\n" + "=" * 45)
    print("📋 ESTADO ATUAL")
    print("=" * 45)
    print(f"• Mensagens: {len(state.get('messages', []))}")
    print(f"• Intent: {state.get('intent', 'N/A')}")
    messages = state.get("messages", [])
    if messages:
        print("\n📜 Histórico:")
        for i, msg in enumerate(messages[-4:], 1):
            role = "👤" if msg.type == "human" else "🤖"
            content = msg.content[:70] + "..." if len(msg.content) > 70 else msg.content
            print(f"   {i}. {role} {content}")
    print("=" * 45)


if __name__ == "__main__":
    run_chat()
