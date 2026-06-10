"""
Configuração da conexão com Amazon Bedrock.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONCEITO: O QUE É O BOTO3?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Boto3 é a biblioteca oficial da AWS para Python.
É o "controle remoto" programático da AWS — a mesma
coisa que o AWS CLI faz no terminal, o boto3 faz
dentro do código Python.

Aqui usamos boto3 para criar uma "sessão autenticada"
com a AWS, que o LangChain usa por baixo dos panos
para chamar o Bedrock.

CONCEITO: O QUE É O ChatBedrock?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ChatBedrock é a classe do LangChain que "embrulha"
o Bedrock com a interface padrão do LangChain.

Isso significa que podemos trocar de modelo
(Claude → Llama → Titan) sem mudar o código
dos agentes — só mudamos a configuração aqui.

É o padrão de projeto "Adapter":
  Código do agente → ChatBedrock → Bedrock API → Claude
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import boto3
from langchain_aws import ChatBedrock
from app.config.settings import settings


def create_bedrock_session() -> boto3.Session:
    """
    Cria uma sessão autenticada com a AWS.

    O boto3 busca as credenciais nesta ordem:
    1. Variáveis de ambiente (AWS_ACCESS_KEY_ID, etc.)
    2. Arquivo ~/.aws/credentials (gerado pelo aws configure)
    3. IAM Role (quando rodando em EC2/ECS na AWS)

    Como configuramos o .env e rodamos 'aws configure',
    as credenciais serão encontradas automaticamente.

    Returns:
        Sessão boto3 autenticada.
    """
    # Se as credenciais estão no .env, passa explicitamente
    # Caso contrário, boto3 usa o ~/.aws/credentials
    if settings.aws_access_key_id and settings.aws_secret_access_key:
        session = boto3.Session(
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region,
        )
    else:
        # Usa as credenciais do 'aws configure' automaticamente
        session = boto3.Session(region_name=settings.aws_region)

    return session


def create_chat_model() -> ChatBedrock:
    """
    Cria o modelo de chat conectado ao Bedrock.

    Parâmetros importantes:
    - model_id: qual modelo usar (definido no .env)
    - temperature: criatividade das respostas
      0.0 = respostas determinísticas e conservadoras
      1.0 = respostas criativas e variadas
      0.7 = equilíbrio entre criatividade e precisão
    - max_tokens: limite de tokens na resposta

    Returns:
        Instância do ChatBedrock pronta para uso.
    """
    session = create_bedrock_session()

    # Cria o cliente boto3 para o Bedrock Runtime
    # "bedrock-runtime" é o endpoint que processa as chamadas ao modelo
    # "bedrock" (sem runtime) é o endpoint de gerenciamento (listar modelos, etc.)
    bedrock_client = session.client("bedrock-runtime")

    llm = ChatBedrock(
        client=bedrock_client,
        model_id=settings.bedrock_model_id,
        model_kwargs={
            "temperature": 0.7,    # Equilíbrio criatividade/precisão
            "max_gen_len": 150,    # Máximo de tokens na resposta
        },
    )

    return llm


# ── Instância singleton do modelo ─────────────────────────────────────────────
# Criada uma vez e reutilizada em todas as chamadas
# Evita overhead de reconectar a cada requisição
_llm_instance: ChatBedrock | None = None


def get_llm() -> ChatBedrock:
    """
    Retorna a instância singleton do modelo.

    Singleton aqui é importante: cada criação do ChatBedrock
    abre uma conexão com a AWS. Reutilizar a mesma instância
    é mais eficiente e evita problemas de rate limiting.

    Returns:
        Instância do ChatBedrock.
    """
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = create_chat_model()
    return _llm_instance
