"""
Web Search Tool — Gemini + GPT-4o-mini com fallback automático.

Hierarquia de LLMs para dados em tempo real da Copa 2026:
  1. Gemini 1.5 Flash (Google) — primário, mais barato
  2. GPT-4o-mini (OpenAI)      — fallback automático
  3. Mensagem de erro          — último recurso

Arquitetura dual do projeto:
  Llama 3.3 (Bedrock)  → perguntas históricas + RAG
  Gemini / GPT-4o-mini → dados em tempo real Copa 2026
"""

import re
from app.config.settings import settings


# ── Prompt padrão usado por todos os modelos ──────────────────────────────────
def _build_prompt(user_input: str) -> str:
    return f"""Você é especialista em Copa do Mundo FIFA 2026.

REGRAS OBRIGATÓRIAS:
- Responda SEMPRE em português brasileiro
- NUNCA inclua links ou URLs
- NUNCA cite fontes (ESPN, FIFA.com, El País, etc.)
- Seja direto — máximo 5 linhas
- Jogos: ⚽ Time A X x X Time B (Cidade)
- Artilheiros: 🥇 Jogador (Seleção) — X gols
- Classificação: 1º Time — X pts (X jogos)

Pergunta: {user_input}"""


# ── Limpeza de links residuais ────────────────────────────────────────────────
def _clean_links(texto: str) -> str:
    texto = re.sub(r'\(https?://[^\)]+\)', '', texto)
    texto = re.sub(r'https?://\S+', '', texto)
    texto = re.sub(r'\[([^\]]+)\]\([^\)]*\)', r'\1', texto)
    texto = re.sub(r'\([a-zA-Z0-9]+\.[a-z]{2,3}\)', '', texto)
    return texto.strip()


# ─────────────────────────────────────────────────────────────────────────────
# FUNÇÃO PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def search_copa_2026_realtime(user_input: str) -> str:
    """
    Busca dados em tempo real da Copa 2026.

    Tenta o Gemini primeiro. Se falhar por quota ou erro,
    usa o GPT-4o-mini automaticamente como fallback.

    Args:
        user_input: Pergunta do usuário.

    Returns:
        Resposta em português, sem links, concisa.
    """
    # Tenta Gemini primeiro
    if settings.google_api_key:
        resultado = _buscar_gemini(user_input)
        if resultado:
            return resultado

    # Fallback para OpenAI
    if settings.openai_api_key:
        print("   🔄 Usando GPT-4o-mini como fallback...")
        return _buscar_openai(user_input)

    # Nenhuma chave configurada
    return _fallback_sem_chave()


# ─────────────────────────────────────────────────────────────────────────────
# GEMINI 1.5 FLASH — Primário
# ─────────────────────────────────────────────────────────────────────────────

def _buscar_gemini(user_input: str) -> str | None:
    """
    Busca via Gemini 1.5 Flash + Google Search.
    Retorna None se falhar (permite fallback para OpenAI).
    """
    try:
        from google import genai
        from google.genai import types

        print("   🤖 Tentando Gemini 1.5 Flash + Google Search...")

        client = genai.Client(api_key=settings.google_api_key)

        response = client.models.generate_content(
            model="models/gemini-2.0-flash-lite",
            contents=_build_prompt(user_input),
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                response_modalities=["TEXT"],
            ),
        )

        texto = _clean_links(response.text)
        print(f"   ✅ Gemini respondeu ({len(texto)} chars)")
        return texto

    except ImportError:
        print("   ⚠️ google-genai não instalado")
        return None

    except Exception as e:
        error = str(e)
        print(f"   ⚠️ Gemini falhou: {error[:100]}")

        # Erros que permitem fallback para OpenAI
        if any(x in error for x in ["429", "quota", "EXHAUSTED", "limit", "unavailable"]):
            print("   🔄 Quota/limite Gemini — ativando fallback OpenAI")
            return None

        # Erro de chave inválida — não tenta fallback
        if any(x in error.lower() for x in ["api_key", "invalid", "unauthorized"]):
            return "⚠️ GOOGLE_API_KEY inválida. Verifique o .env."

        # Outros erros — tenta fallback
        return None


# ─────────────────────────────────────────────────────────────────────────────
# GPT-4o-mini — Fallback automático
# ─────────────────────────────────────────────────────────────────────────────

def _buscar_openai(user_input: str) -> str:
    """
    Busca via GPT-4o-mini + web_search_preview.
    Usado como fallback quando o Gemini falha.
    """
    try:
        from openai import OpenAI

        print("   🤖 GPT-4o-mini + Web Search...")

        client = OpenAI(api_key=settings.openai_api_key)

        response = client.responses.create(
            model="gpt-4o-mini",
            tools=[{"type": "web_search_preview"}],
            input=_build_prompt(user_input),
        )

        texto = _clean_links(response.output_text)
        print(f"   ✅ GPT-4o-mini respondeu ({len(texto)} chars)")
        return texto

    except ImportError:
        return "⚠️ Biblioteca openai não instalada. Execute: poetry add openai"

    except Exception as e:
        error = str(e)
        print(f"   ❌ GPT-4o-mini falhou: {error[:100]}")

        if "quota" in error.lower() or "limit" in error.lower():
            return "⚠️ Limite de uso da OpenAI atingido. Tente novamente mais tarde."
        elif "api_key" in error.lower() or "invalid" in error.lower():
            return "⚠️ OPENAI_API_KEY inválida. Verifique o .env."
        else:
            return f"Não consegui buscar dados em tempo real. ({error[:60]})"


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK FINAL — nenhuma chave configurada
# ─────────────────────────────────────────────────────────────────────────────

def _fallback_sem_chave() -> str:
    return (
        "⚠️ Para dados em tempo real da Copa 2026, configure no .env:\n"
        "  GOOGLE_API_KEY=... (Gemini — recomendado)\n"
        "  OPENAI_API_KEY=... (GPT-4o-mini — fallback)\n\n"
        "Posso responder sobre história da Copa do Mundo até 2022!"
    )