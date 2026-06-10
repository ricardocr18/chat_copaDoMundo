"""
Nós do Grafo LangGraph — Fase 2.

O que mudou em relação à Fase 1:
- process_node agora chama o Claude via Bedrock
- Adicionado system prompt temático sobre Copa do Mundo
- Histórico de conversa enviado ao modelo (memória real)
- Tratamento de erros de API (timeout, throttling)

O que NÃO mudou:
- input_node: idêntico à Fase 1
- output_node: idêntico à Fase 1
- error_node: idêntico à Fase 1
- Assinatura das funções: (state) -> dict
- Contrato com o GraphState: mesmo de sempre

Isso demonstra na prática a separação de responsabilidades:
mudamos o comportamento interno sem afetar a interface.
"""

import time
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.graph.state import GraphState


# ── System Prompt ──────────────────────────────────────────────────────────────
# O system prompt é a "personalidade" e as "regras" do assistente.
# É enviado ao modelo em TODA conversa, antes das mensagens do usuário.
# Define escopo, tom, e comportamento esperado.
WORLD_CUP_SYSTEM_PROMPT = """Você é um especialista em Copa do Mundo FIFA com conhecimento \
profundo sobre toda a história do torneio desde 1930 até os dias atuais.

Suas responsabilidades:
- Responder perguntas sobre história, estatísticas e curiosidades da Copa do Mundo
- Fornecer informações precisas sobre campeões, artilheiros, recordes e jogadores históricos
- Contextualizar eventos históricos de forma didática e envolvente
- Manter um tom entusiasmado mas preciso, como um comentarista esportivo experiente

Diretrizes importantes:
- Responda SEMPRE em português brasileiro
- Se não souber algo com certeza, diga claramente
- Foque exclusivamente em Copa do Mundo — para outros assuntos de futebol,
  redirecione gentilmente ao tema principal
- Para perguntas completamente fora do futebol, explique educadamente
  que seu escopo é a Copa do Mundo FIFA

Formato das respostas:
- Respostas diretas e objetivas para perguntas simples
- Respostas mais detalhadas para perguntas históricas ou comparativas
- Use números e estatísticas quando relevante
- Máximo de 3 parágrafos para não sobrecarregar o usuário"""


# ─────────────────────────────────────────────────────────────────────────────
# NÓ 1: Recepção — IDÊNTICO à Fase 1
# ─────────────────────────────────────────────────────────────────────────────

def input_node(state: GraphState) -> dict:
    """
    Primeiro nó — recebe e prepara a entrada do usuário.
    Sem alterações em relação à Fase 1.
    """
    user_input = state["user_input"]
    print(f"\n📥 [input_node] Recebendo pergunta: '{user_input}'")

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
    }


# ─────────────────────────────────────────────────────────────────────────────
# NÓ 2: Processamento — SUBSTITUÍDO pelo Claude via Bedrock
# ─────────────────────────────────────────────────────────────────────────────

def process_node(state: GraphState) -> dict:
    """
    Nó de processamento — chama o Claude via Amazon Bedrock.

    FASE 2: Substituímos o dicionário de keywords por uma
    chamada real ao LLM. O modelo recebe:
    1. System prompt com as regras e personalidade
    2. Histórico completo da conversa (memória)
    3. A pergunta atual do usuário

    Tratamento de erros:
    - Se a chamada falhar, registra o erro no estado
    - O output_node detecta o erro e usa mensagem de fallback
    - Nunca deixa o usuário sem resposta

    Args:
        state: Estado com messages (histórico) e user_input.

    Returns:
        Dicionário com final_response ou error preenchido.
    """
    # Importação local para evitar erro se as credenciais
    # não estiverem configuradas durante os testes
    from app.config.aws_config import get_llm

    print(f"⚙️  [process_node] Chamando Claude via Bedrock...")

    try:
        # ── Monta as mensagens para enviar ao modelo ──────────────────────
        #
        # CONCEITO: Como o Claude "enxerga" a conversa
        #
        # O modelo recebe uma lista de mensagens nesta ordem:
        # [SystemMessage, HumanMessage, AIMessage, HumanMessage, ...]
        #
        # SystemMessage  → as regras/personalidade (nosso WORLD_CUP_SYSTEM_PROMPT)
        # HumanMessage   → mensagens do usuário
        # AIMessage      → respostas anteriores do modelo
        #
        # Enviar o histórico completo é o que dá "memória" ao chatbot.
        # Sem isso, cada pergunta seria tratada de forma isolada.

        messages_to_send = [
            SystemMessage(content=WORLD_CUP_SYSTEM_PROMPT),
            # Histórico da conversa (acumulado pelo add_messages no state)
            *state["messages"],
        ]

        # ── Chama o modelo ────────────────────────────────────────────────
        llm = get_llm()
        print(f"   🌐 Enviando {len(messages_to_send)} mensagens ao modelo...")

        response = llm.invoke(messages_to_send)

        # response é um AIMessage com o campo .content
        answer = response.content
        print(f"   ✅ Resposta recebida ({len(answer)} caracteres)")

        # Atualiza o caminho nos metadados
        metadata = state.get("metadata", {}) or {}
        metadata["node_path"] = metadata.get("node_path", []) + ["process_node"]
        metadata["model_id"] = settings_info()

        return {
            "final_response": answer,
            "metadata": metadata,
        }

    except Exception as e:
        # ── Tratamento de erros ───────────────────────────────────────────
        #
        # Erros comuns do Bedrock:
        # - ThrottlingException: muitas chamadas por segundo
        # - ModelNotReadyException: modelo ainda inicializando
        # - ValidationException: parâmetros inválidos
        # - ConnectionError: problema de rede/credenciais
        #
        # Em vez de deixar o programa crashar, registramos o erro
        # no estado para que o output_node possa tratá-lo.

        error_msg = str(e)
        print(f"   ❌ Erro ao chamar Bedrock: {error_msg[:100]}")

        metadata = state.get("metadata", {}) or {}
        metadata["node_path"] = metadata.get("node_path", []) + ["process_node"]

        return {
            "error": error_msg,
            "final_response": None,
            "metadata": metadata,
        }


def settings_info() -> str:
    """Retorna o model_id configurado para logging."""
    try:
        from app.config.settings import settings
        return settings.bedrock_model_id
    except Exception:
        return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# NÓ 3: Saída — LEVEMENTE MODIFICADO para tratar erros
# ─────────────────────────────────────────────────────────────────────────────

def output_node(state: GraphState) -> dict:
    """
    Último nó — finaliza a resposta.

    Modificação em relação à Fase 1:
    Verifica se houve erro no process_node e usa
    mensagem de fallback adequada.
    """
    # Verifica se houve erro
    error = state.get("error")
    if error:
        print(f"⚠️  [output_node] Erro detectado, usando fallback")
        # Identifica o tipo de erro para mensagem mais útil
        if "ThrottlingException" in error:
            final_response = (
                "Estou recebendo muitas perguntas no momento. "
                "Aguarde alguns segundos e tente novamente."
            )
        elif "credentials" in error.lower() or "auth" in error.lower():
            final_response = (
                "Problema de autenticação com a AWS. "
                "Verifique suas credenciais no arquivo .env."
            )
        else:
            final_response = (
                "Desculpe, tive um problema ao processar sua pergunta. "
                "Por favor, tente novamente."
            )
    else:
        final_response = state.get("final_response") or "Não consegui gerar uma resposta."

    print(f"📤 [output_node] Finalizando resposta")

    # Calcula latência
    metadata = state.get("metadata", {}) or {}
    start_time = metadata.get("start_time", time.time())
    latency_ms = round((time.time() - start_time) * 1000, 2)
    metadata["node_path"] = metadata.get("node_path", []) + ["output_node"]
    metadata["latency_ms"] = latency_ms

    print(f"⏱️  [output_node] Latência total: {latency_ms}ms")

    return {
        "messages": [AIMessage(content=final_response)],
        "final_response": final_response,
        "metadata": metadata,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NÓ DE ERRO — IDÊNTICO à Fase 1
# ─────────────────────────────────────────────────────────────────────────────

def error_node(state: GraphState) -> dict:
    """Tratamento centralizado de erros — sem alterações."""
    error = state.get("error", "Erro desconhecido")
    print(f"❌ [error_node] Tratando erro: {error}")

    fallback_response = (
        "Desculpe, encontrei um problema ao processar sua pergunta. "
        "Por favor, tente novamente."
    )

    return {
        "final_response": fallback_response,
        "messages": [AIMessage(content=fallback_response)],
    }
