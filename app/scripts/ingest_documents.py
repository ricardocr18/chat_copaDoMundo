"""
Script de Ingestão de Documentos — Fase 3.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
O QUE ESTE SCRIPT FAZ?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Este script é executado UMA VEZ para preparar a base
de conhecimento. Depois disso, o chatbot usa o ChromaDB
já criado sem precisar reprocessar os documentos.

Fluxo:
  1. Lê os .txt de app/data/raw/
  2. Divide em chunks de ~500 caracteres
  3. Envia cada chunk ao Titan Embeddings (AWS)
  4. Salva os vetores no ChromaDB local

Como executar:
  poetry run python -m app.scripts.ingest_documents
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys
import shutil
from pathlib import Path

# Garante que o Python encontra os módulos do projeto
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.tools.vector_store import (
    load_documents,
    split_documents,
    create_vector_store,
    PERSIST_DIR,
)


def main() -> None:
    """Executa o pipeline completo de ingestão."""

    print("=" * 55)
    print("📚 INGESTÃO DE DOCUMENTOS — COPA DO MUNDO RAG")
    print("=" * 55)

    # ── Passo 1: Limpa vector store antigo se existir ─────────────────
    if PERSIST_DIR.exists():
        print(f"\n🗑️  Limpando vector store antigo em {PERSIST_DIR}...")
        shutil.rmtree(PERSIST_DIR)
        print("   ✅ Limpo!")

    # ── Passo 2: Carrega documentos ───────────────────────────────────
    print("\n📄 Passo 1/3 — Carregando documentos...")
    try:
        documents = load_documents()
    except FileNotFoundError as e:
        print(f"\n❌ Erro: {e}")
        sys.exit(1)

    # ── Passo 3: Divide em chunks ─────────────────────────────────────
    print("\n✂️  Passo 2/3 — Dividindo em chunks...")
    chunks = split_documents(documents)

    # Mostra um exemplo de chunk para entender o processo
    if chunks:
        print(f"\n   Exemplo de chunk gerado:")
        print(f"   {'─' * 40}")
        example = chunks[0].page_content[:200]
        print(f"   {example}...")
        print(f"   {'─' * 40}")
        print(f"   Fonte: {chunks[0].metadata.get('source', 'N/A')}")

    # ── Passo 4: Cria vector store com embeddings ─────────────────────
    print("\n🔢 Passo 3/3 — Gerando embeddings e criando vector store...")
    print("   (Fazendo chamadas ao Amazon Bedrock Titan Embeddings)")
    print("   (Aguarde — pode levar 1-3 minutos)")

    try:
        vector_store = create_vector_store(chunks)
    except Exception as e:
        print(f"\n❌ Erro ao criar vector store: {e}")
        print("\nVerifique:")
        print("  1. Suas credenciais AWS no .env")
        print("  2. Se o Titan Embeddings está disponível na us-east-1")
        print("  3. Conectividade com a internet")
        sys.exit(1)

    # ── Resumo Final ──────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("✅ INGESTÃO CONCLUÍDA COM SUCESSO!")
    print("=" * 55)
    print(f"\n📊 Resumo:")
    print(f"   • Documentos processados: {len(documents)}")
    print(f"   • Chunks criados:         {len(chunks)}")
    print(f"   • Vector store salvo em:  {PERSIST_DIR}")
    print(f"\n🚀 Próximo passo:")
    print(f"   poetry run python -m app.main")
    print(f"   (O chatbot agora usará o RAG para responder!)")


if __name__ == "__main__":
    main()
