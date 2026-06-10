# 🏆 World Cup Agent

Chatbot sobre Copa do Mundo construído com **LangGraph** e **Amazon Bedrock**.

## Fases do Projeto
- ✅ **Fase 1** — Fundação e Estado (você está aqui)
- ⏳ **Fase 2** — LLM com Amazon Bedrock
- ⏳ **Fase 3** — RAG (Retrieval-Augmented Generation)
- ⏳ **Fase 4** — Múltiplos Agentes e Roteamento
- ⏳ **Fase 5** — Guardrails e Observabilidade (LangFuse)
- ⏳ **Fase 6** — API REST com FastAPI
- ⏳ **Fase 7** — Containerização e Deploy AWS

## Setup

```bash
# 1. Instalar dependências
pip install poetry
poetry install

# 2. Configurar variáveis de ambiente
cp .env.example .env
# Edite o .env com suas credenciais

# 3. Executar o chatbot
poetry run python -m app.main

# 4. Executar testes
poetry run pytest tests/test_phase1.py -v
```
