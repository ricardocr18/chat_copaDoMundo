"""
Interface com o ChromaDB — Vector Store do projeto.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONCEITO: O QUE É UM VECTOR STORE?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Um Vector Store é um banco de dados especializado em
armazenar e buscar VETORES (listas de números/embeddings).

Diferença de um banco relacional:
  SQL:    SELECT * WHERE nome = 'Brasil'  (busca exata)
  Vector: busca os documentos cujo SIGNIFICADO
          é mais próximo da pergunta (busca semântica)

Isso permite encontrar "Copa de 70" quando o usuário
pergunta "Mundial do México" — mesmo sem palavras iguais.

ChromaDB é nossa escolha porque:
  ✅ Roda localmente (sem servidor externo)
  ✅ Persiste dados em disco
  ✅ Integração nativa com LangChain
  ✅ Gratuito e open source
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
from pathlib import Path
from langchain_aws import BedrockEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.config.settings import settings


# ── Configurações do Vector Store ─────────────────────────────────────────────
COLLECTION_NAME = "world_cup_knowledge"
RAW_DOCS_DIR = Path("app/data/raw")
PERSIST_DIR = Path(settings.chroma_persist_dir)


def get_embeddings_model() -> BedrockEmbeddings:
    """
    Cria o modelo de embeddings usando Amazon Titan via Bedrock.

    BedrockEmbeddings é a classe do LangChain para gerar
    embeddings com modelos da AWS. Usamos o Titan V2 que:
    - É nativo da AWS (sem formulário extra)
    - Gera vetores de 1024 dimensões
    - Tem ótimo custo-benefício

    Returns:
        Modelo de embeddings pronto para uso.
    """
    import boto3

    # Cria sessão AWS com as credenciais do .env
    if settings.aws_access_key_id and settings.aws_secret_access_key:
        session = boto3.Session(
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region,
        )
    else:
        session = boto3.Session(region_name=settings.aws_region)

    bedrock_client = session.client("bedrock-runtime")

    return BedrockEmbeddings(
        client=bedrock_client,
        model_id=settings.bedrock_embeddings_model_id,
    )


def load_documents() -> list[Document]:
    """
    Carrega todos os documentos .txt da pasta raw/.

    TextLoader lê arquivos de texto e os converte em
    objetos Document do LangChain, que carregam:
    - page_content: o texto em si
    - metadata: informações sobre a origem (nome do arquivo, etc.)

    Returns:
        Lista de documentos carregados.
    """
    documents = []
    txt_files = list(RAW_DOCS_DIR.glob("*.txt"))

    if not txt_files:
        raise FileNotFoundError(
            f"Nenhum arquivo .txt encontrado em {RAW_DOCS_DIR}. "
            "Verifique se os documentos foram criados corretamente."
        )

    for file_path in txt_files:
        print(f"   📄 Carregando: {file_path.name}")
        loader = TextLoader(str(file_path), encoding="utf-8")
        docs = loader.load()
        documents.extend(docs)

    print(f"   ✅ {len(documents)} documentos carregados de {len(txt_files)} arquivos")
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """
    Divide documentos em chunks (pedaços menores).

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    CONCEITO: POR QUE DIVIDIR EM CHUNKS?
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Imagine enviar um livro inteiro para o modelo a cada
    pergunta. Seria lento, caro e confuso.

    Com chunks, enviamos apenas os TRECHOS relevantes:
    - Pergunta sobre 1994? → Envia só o chunk de 1994
    - Pergunta sobre artilheiros? → Envia o chunk de artilheiros

    RecursiveCharacterTextSplitter divide de forma
    inteligente, respeitando parágrafos e frases.

    chunk_size=500    → cada pedaço tem ~500 caracteres
    chunk_overlap=100 → 100 chars de sobreposição entre
                        chunks para não perder contexto
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "],  # tenta dividir por parágrafo primeiro
    )

    chunks = splitter.split_documents(documents)
    print(f"   ✅ {len(documents)} documentos → {len(chunks)} chunks")
    return chunks


def create_vector_store(chunks: list[Document]) -> Chroma:
    """
    Cria o Vector Store com os chunks e salva no disco.

    Este processo:
    1. Pega cada chunk de texto
    2. Envia para o Titan Embeddings → recebe vetor de números
    3. Salva o par (texto, vetor) no ChromaDB

    É o passo mais demorado — cada chunk faz uma chamada
    à API do Bedrock para gerar o embedding.

    Args:
        chunks: Lista de chunks de texto.

    Returns:
        Vector store criado e persistido.
    """
    print(f"   🔢 Gerando embeddings com Titan...")
    print(f"   ⚠️  Isso pode levar alguns minutos ({len(chunks)} chunks)...")

    embeddings = get_embeddings_model()

    # Cria o diretório de persistência se não existir
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIR),
    )

    print(f"   ✅ Vector store criado com {len(chunks)} chunks em {PERSIST_DIR}")
    return vector_store


def load_vector_store() -> Chroma:
    """
    Carrega um Vector Store já existente do disco.

    Após a ingestão inicial, não precisamos reprocessar
    os documentos — basta carregar o ChromaDB do disco.

    Returns:
        Vector store carregado.

    Raises:
        FileNotFoundError: Se o vector store não foi criado ainda.
    """
    if not PERSIST_DIR.exists():
        raise FileNotFoundError(
            f"Vector store não encontrado em {PERSIST_DIR}. "
            "Execute o script de ingestão primeiro: "
            "poetry run python -m app.scripts.ingest_documents"
        )

    embeddings = get_embeddings_model()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIR),
    )

    return vector_store


def search_similar_documents(
    query: str,
    k: int = 3,
) -> list[str]:
    """
    Busca os k documentos mais relevantes para a query.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    CONCEITO: SIMILARITY SEARCH
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    1. query → Titan Embeddings → vetor da pergunta
    2. ChromaDB compara esse vetor com todos os chunks
    3. Retorna os k chunks com menor "distância"
       (maior similaridade semântica)

    k=3 significa: retorna os 3 trechos mais relevantes.
    Mais chunks = mais contexto, mas também mais tokens.
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Args:
        query: Pergunta do usuário.
        k: Número de documentos a retornar.

    Returns:
        Lista de strings com o conteúdo dos chunks relevantes.
    """
    vector_store = load_vector_store()
    results = vector_store.similarity_search(query, k=k)

    # Extrai só o texto (page_content) de cada resultado
    contexts = [doc.page_content for doc in results]

    return contexts


# ── Singleton do Vector Store ──────────────────────────────────────────────────
_vector_store_instance: Chroma | None = None


def get_vector_store() -> Chroma:
    """
    Retorna singleton do vector store.
    Carrega do disco na primeira chamada e reutiliza depois.
    """
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = load_vector_store()
    return _vector_store_instance
