"""
Nós do Grafo LangGraph — Fase 3.

O que mudou em relação à Fase 2:
- Adicionado rag_node: busca documentos relevantes antes de responder
- process_node agora recebe contexto do RAG e o injeta no prompt
- O prompt foi enriquecido para usar o contexto encontrado

Fluxo da Fase 3:
  input_node → rag_node → process_node → output_node
                  ↑
          NOVO: busca no ChromaDB
"""

import time
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from app.graph.state import GraphState


# ── System Prompt Base ─────────────────────────────────────────────────────────
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
- Quando houver contexto de documentos fornecido, PRIORIZE essas informações

Formato das respostas:
- Respostas diretas e objetivas para perguntas simples
- Respostas mais detalhadas para perguntas históricas ou comparativas
- Use números e estatísticas quando relevante
- Máximo de 3 parágrafos para não sobrecarregar o usuário"""


# ─────────────────────────────────────────────────────────────────────────────
# NÓ 1: Input — IDÊNTICO à Fase 2
# ─────────────────────────────────────────────────────────────────────────────

def input_node(state: GraphState) -> dict:
    """Recebe e prepara a entrada do usuário."""
    user_input = state["user_input"]
    print(f"\n📥 [input_node] Recebendo: '{user_input}'")

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
# NÓ 2: RAG — NOVO na Fase 3
# ─────────────────────────────────────────────────────────────────────────────

def rag_node(state: GraphState) -> dict:
    """
    Nó de RAG — busca documentos relevantes no ChromaDB.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    COMO FUNCIONA:
    1. Pega a pergunta do usuário (user_input)
    2. Converte em embedding via Titan (chamada AWS)
    3. ChromaDB compara com todos os chunks armazenados
    4. Retorna os 3 chunks mais semanticamente similares
    5. Salva esses chunks em state["retrieved_context"]
    6. O process_node vai usar esse contexto no prompt
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Args:
        state: Estado com user_input preenchido.

    Returns:
        Dicionário com retrieved_context preenchido.
    """
    from app.tools.vector_store import search_similar_documents

    user_input = state["user_input"]
    print(f"🔍 [rag_node] Buscando documentos relevantes...")

    try:
        # Busca os 3 chunks mais relevantes para a pergunta
        contexts = search_similar_documents(query=user_input, k=3)

        if contexts:
            print(f"   ✅ {len(contexts)} trechos encontrados no vector store")
            for i, ctx in enumerate(contexts, 1):
                preview = ctx[:80].replace('\n', ' ')
                print(f"   {i}. {preview}...")
        else:
            print(f"   ⚠️  Nenhum trecho relevante encontrado")

        # Atualiza metadados
        metadata = state.get("metadata", {}) or {}
        metadata["node_path"] = metadata.get("node_path", []) + ["rag_node"]
        metadata["rag_chunks_found"] = len(contexts)

        return {
            "retrieved_context": contexts if contexts else [],
            "metadata": metadata,
        }

    except FileNotFoundError as e:
        # Vector store não foi criado ainda
        print(f"   ⚠️  Vector store não encontrado: {e}")
        print(f"   ℹ️  Respondendo sem RAG (só com conhecimento do modelo)")

        metadata = state.get("metadata", {}) or {}
        metadata["node_path"] = metadata.get("node_path", []) + ["rag_node"]
        metadata["rag_chunks_found"] = 0

        return {
            "retrieved_context": [],
            "metadata": metadata,
        }

    except Exception as e:
        print(f"   ❌ Erro no RAG: {str(e)[:100]}")

        metadata = state.get("metadata", {}) or {}
        metadata["node_path"] = metadata.get("node_path", []) + ["rag_node"]

        return {
            "retrieved_context": [],
            "metadata": metadata,
        }


# ─────────────────────────────────────────────────────────────────────────────
# NÓ 3: Process — MODIFICADO para usar contexto do RAG
# ─────────────────────────────────────────────────────────────────────────────

def process_node(state: GraphState) -> dict:
    """
    Nó de processamento — chama o LLM com contexto do RAG.

    Diferença da Fase 2:
    Se o RAG encontrou documentos relevantes, injeta
    esses trechos no prompt antes da pergunta.
    O modelo passa a responder COM BASE nos documentos.

    Args:
        state: Estado com messages, user_input e retrieved_context.

    Returns:
        Dicionário com final_response preenchido.
    """
    from app.config.aws_config import get_llm

    print(f"⚙️  [process_node] Chamando LLM...")

    try:
        # ── Verifica se temos contexto do RAG ────────────────────────
        retrieved_context = state.get("retrieved_context") or []

        if retrieved_context:
            # Monta o contexto formatado para injetar no prompt
            context_text = "\n\n---\n\n".join(retrieved_context)

            # Prompt enriquecido com o contexto dos documentos
            # Este é o coração do RAG — o modelo "lê" os documentos
            # antes de responder
            rag_prompt = f"""{WORLD_CUP_SYSTEM_PROMPT}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTEXTO DOS DOCUMENTOS (use como fonte principal):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{context_text}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Com base PRINCIPALMENTE nas informações acima, responda
a pergunta do usuário. Se as informações não forem
suficientes, complemente com seu conhecimento."""

            system_content = rag_prompt
            print(f"   📚 Usando RAG ({len(retrieved_context)} trechos injetados no prompt)")
        else:
            # Sem contexto — usa apenas o conhecimento do modelo
            system_content = WORLD_CUP_SYSTEM_PROMPT
            print(f"   🧠 Usando apenas conhecimento do modelo (sem RAG)")

        # ── Monta e envia as mensagens ────────────────────────────────
        messages_to_send = [
            SystemMessage(content=system_content),
            *state["messages"],
        ]

        llm = get_llm()
        print(f"   🌐 Enviando {len(messages_to_send)} mensagens ao modelo...")

        response = llm.invoke(messages_to_send)
        answer = response.content
        print(f"   ✅ Resposta recebida ({len(answer)} caracteres)")

        metadata = state.get("metadata", {}) or {}
        metadata["node_path"] = metadata.get("node_path", []) + ["process_node"]
        metadata["used_rag"] = len(retrieved_context) > 0

        return {
            "final_response": answer,
            "metadata": metadata,
        }

    except Exception as e:
        error_msg = str(e)
        print(f"   ❌ Erro: {error_msg[:100]}")

        metadata = state.get("metadata", {}) or {}
        metadata["node_path"] = metadata.get("node_path", []) + ["process_node"]

        return {
            "error": error_msg,
            "final_response": None,
            "metadata": metadata,
        }


# ─────────────────────────────────────────────────────────────────────────────
# NÓ 4: Output — MODIFICADO para mostrar se usou RAG
# ─────────────────────────────────────────────────────────────────────────────

def output_node(state: GraphState) -> dict:
    """Finaliza a resposta e calcula métricas."""
    error = state.get("error")

    if error:
        print(f"⚠️  [output_node] Erro detectado, usando fallback")
        if "ThrottlingException" in error:
            final_response = "Estou recebendo muitas perguntas. Aguarde e tente novamente."
        elif "credentials" in error.lower() or "auth" in error.lower():
            final_response = "Problema de autenticação AWS. Verifique o .env."
        else:
            final_response = "Desculpe, tive um problema. Por favor, tente novamente."
    else:
        final_response = state.get("final_response") or "Não consegui gerar uma resposta."

    print(f"📤 [output_node] Finalizando resposta")

    metadata = state.get("metadata", {}) or {}
    start_time = metadata.get("start_time", time.time())
    latency_ms = round((time.time() - start_time) * 1000, 2)
    metadata["node_path"] = metadata.get("node_path", []) + ["output_node"]
    metadata["latency_ms"] = latency_ms

    # Indica se a resposta foi fundamentada em documentos
    used_rag = metadata.get("used_rag", False)
    rag_chunks = metadata.get("rag_chunks_found", 0)
    rag_indicator = f"📚 RAG ({rag_chunks} trechos)" if used_rag else "🧠 Modelo"
    metadata["source_indicator"] = rag_indicator

    print(f"⏱️  [output_node] Latência: {latency_ms}ms | Fonte: {rag_indicator}")

    return {
        "messages": [AIMessage(content=final_response)],
        "final_response": final_response,
        "metadata": metadata,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NÓ DE ERRO — IDÊNTICO às fases anteriores
# ─────────────────────────────────────────────────────────────────────────────

def error_node(state: GraphState) -> dict:
    """Tratamento centralizado de erros."""
    error = state.get("error", "Erro desconhecido")
    print(f"❌ [error_node] Tratando: {error}")
    fallback = "Desculpe, encontrei um problema. Por favor, tente novamente."
    return {
        "final_response": fallback,
        "messages": [AIMessage(content=fallback)],
    }
