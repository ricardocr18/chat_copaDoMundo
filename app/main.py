"""Ponto de entrada — Fase 4: Multi-agentes com roteamento."""

from app.graph.builder import get_graph
from app.graph.state import GraphState


def run_chat() -> None:
    print("=" * 60)
    print("🏆  BEM-VINDO AO CHATBOT DA COPA DO MUNDO  🏆")
    print("=" * 60)
    print("Fase 4 — Multi-agentes com Router inteligente")
    print("Rotas disponíveis: 📚 RAG | 🌐 API | 💬 Direto | 🚫 Off-topic")
    print("Digite 'sair' para encerrar | 'estado' para histórico")
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
                print("\n👋 Até logo! Que venha a próxima Copa! 🏆")
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
    print(f"• Intent atual: {state.get('intent', 'N/A')}")
    ctx = state.get("retrieved_context")
    print(f"• Chunks RAG: {len(ctx) if ctx else 0}")
    print(f"• API data: {'Sim' if state.get('api_data') else 'Não'}")
    messages = state.get("messages", [])
    if messages:
        print("\n📜 Últimas mensagens:")
        for i, msg in enumerate(messages[-4:], 1):
            role = "👤" if msg.type == "human" else "🤖"
            content = msg.content[:70] + "..." if len(msg.content) > 70 else msg.content
            print(f"   {i}. {role} {content}")
    print("=" * 45)


if __name__ == "__main__":
    run_chat()
