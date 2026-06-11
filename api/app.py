"""
Aplicação FastAPI — Fase 6.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONCEITO: FASTAPI vs FLASK vs DJANGO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Django:  framework completo (ORM, admin, templates)
         pesado, para apps web complexas

Flask:   micro-framework, flexível
         síncrono por padrão, menos performático

FastAPI: moderno, assíncrono, tipado
  ✅ async/await nativo (perfeito para LLMs)
  ✅ Validação automática com Pydantic
  ✅ Documentação automática (Swagger/OpenAPI)
  ✅ Alta performance (comparável ao Node.js)

Para APIs que chamam LLMs (operações lentas de I/O),
FastAPI com async é a escolha ideal.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from api.session_manager import session_manager
from app.config.settings import settings
from app.graph.builder import get_graph


# ── Lifespan — executado na inicialização e encerramento ──────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Código executado quando a API inicia e quando encerra.

    CONCEITO: LIFESPAN
    É como um "construtor e destrutor" da aplicação.
    Usamos para:
    - Pré-carregar o grafo (evita cold start na 1ª requisição)
    - Iniciar tarefas de background (limpeza de sessões)
    - Liberar recursos ao encerrar
    """
    # ── STARTUP ───────────────────────────────────────────────────────
    print("🚀 Iniciando World Cup Agent API...")
    print(f"   Ambiente: {settings.app_env}")
    print(f"   Modelo: {settings.bedrock_model_id}")

    # Pré-compila o grafo para evitar delay na primeira requisição
    print("   Compilando grafo LangGraph...")
    get_graph()

    # Inicia limpeza periódica de sessões expiradas (a cada 5 minutos)
    async def cleanup_task():
        while True:
            await asyncio.sleep(300)  # 5 minutos
            removed = session_manager.cleanup_expired()
            if removed > 0:
                print(f"🗑️  {removed} sessões expiradas removidas")

    task = asyncio.create_task(cleanup_task())
    print("✅ API pronta para receber requisições!")

    yield  # API rodando aqui

    # ── SHUTDOWN ──────────────────────────────────────────────────────
    task.cancel()
    print("👋 API encerrada.")


# ── Criação da aplicação ───────────────────────────────────────────────────────

app = FastAPI(
    title="World Cup Agent API",
    description="Chatbot especializado em Copa do Mundo FIFA, powered by LangGraph + Amazon Bedrock",
    version="1.0.0",
    lifespan=lifespan,
    # Documentação automática disponível em /docs (Swagger) e /redoc
)

# ── CORS — permite que frontends acessem a API ────────────────────────────────
# CONCEITO: CORS (Cross-Origin Resource Sharing)
# Sem CORS, um frontend em localhost:3000 não consegue
# chamar a API em localhost:8000 — o browser bloqueia.
# Este middleware libera o acesso.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if not settings.is_production else ["https://seudominio.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

# ── Registra as rotas ─────────────────────────────────────────────────────────
# prefix="/api/v1" — todas as rotas ficam em /api/v1/chat, /api/v1/health, etc.
app.include_router(router, prefix="/api/v1", tags=["chatbot"])


# ── Rota raiz ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["info"])
async def root():
    """Informações básicas da API."""
    return {
        "name": "World Cup Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
        "chat": "POST /api/v1/chat",
    }
