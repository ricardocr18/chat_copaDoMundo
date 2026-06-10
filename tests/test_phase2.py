"""
Testes da Fase 2 — LLM com Amazon Bedrock.

IMPORTANTE: Os testes desta fase são divididos em dois grupos:

1. Testes UNITÁRIOS (sem AWS): validam a lógica de configuração
   e estrutura do código sem fazer chamadas reais ao Bedrock.
   Rodam sempre, mesmo sem credenciais configuradas.

2. Testes de INTEGRAÇÃO (com AWS): fazem chamadas reais ao Bedrock.
   Só rodam se as credenciais estiverem no .env.
   Marcados com @pytest.mark.integration

Como rodar:
  Só unitários:    pytest tests/test_phase2.py -v -m "not integration"
  Com integração:  pytest tests/test_phase2.py -v
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.graph.state import GraphState
from app.config.settings import settings


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def base_state() -> GraphState:
    """Estado base para testes."""
    return {
        "messages": [HumanMessage(content="Quem ganhou a Copa de 1970?")],
        "user_input": "Quem ganhou a Copa de 1970?",
        "intent": None,
        "retrieved_context": None,
        "final_response": None,
        "error": None,
        "metadata": {"start_time": 0, "node_path": ["input_node"]},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Testes Unitários — Configuração
# ─────────────────────────────────────────────────────────────────────────────

class TestSettings:
    def test_bedrock_model_id_configured(self):
        """Model ID deve estar configurado."""
        assert settings.bedrock_model_id is not None
        assert "claude" in settings.bedrock_model_id.lower() or \
               "anthropic" in settings.bedrock_model_id.lower()

    def test_aws_region_configured(self):
        """Região AWS deve estar configurada."""
        assert settings.aws_region is not None
        assert len(settings.aws_region) > 0

    def test_embeddings_model_configured(self):
        """Modelo de embeddings deve estar configurado."""
        assert settings.bedrock_embeddings_model_id is not None


# ─────────────────────────────────────────────────────────────────────────────
# Testes Unitários — System Prompt
# ─────────────────────────────────────────────────────────────────────────────

class TestSystemPrompt:
    def test_system_prompt_exists(self):
        """System prompt deve existir e ter conteúdo."""
        from app.graph.nodes import WORLD_CUP_SYSTEM_PROMPT
        assert WORLD_CUP_SYSTEM_PROMPT is not None
        assert len(WORLD_CUP_SYSTEM_PROMPT) > 100

    def test_system_prompt_in_portuguese(self):
        """System prompt deve instruir respostas em português."""
        from app.graph.nodes import WORLD_CUP_SYSTEM_PROMPT
        assert "português" in WORLD_CUP_SYSTEM_PROMPT.lower()

    def test_system_prompt_mentions_world_cup(self):
        """System prompt deve ser específico para Copa do Mundo."""
        from app.graph.nodes import WORLD_CUP_SYSTEM_PROMPT
        assert "copa do mundo" in WORLD_CUP_SYSTEM_PROMPT.lower() or \
               "copa" in WORLD_CUP_SYSTEM_PROMPT.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Testes Unitários — process_node com Mock
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessNodeMocked:
    """
    Testa o process_node sem chamar o Bedrock de verdade.

    Usamos 'mock' — um objeto falso que simula o comportamento
    do LLM sem fazer chamadas reais à AWS.

    Por que mockar?
    - Testes rápidos (milissegundos vs segundos)
    - Sem custo de API
    - Sem dependência de rede
    - Resultados determinísticos
    """

    def test_process_node_calls_llm(self, base_state):
        """process_node deve chamar o LLM."""
        mock_response = AIMessage(content="O Brasil venceu a Copa de 1970.")

        with patch("app.graph.nodes.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm

            from app.graph.nodes import process_node
            result = process_node(base_state)

            # Verifica que o LLM foi chamado
            assert mock_llm.invoke.called
            assert result["final_response"] == "O Brasil venceu a Copa de 1970."

    def test_process_node_sends_system_prompt(self, base_state):
        """process_node deve incluir o system prompt nas mensagens."""
        mock_response = AIMessage(content="Resposta de teste")

        with patch("app.graph.nodes.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm

            from app.graph.nodes import process_node
            process_node(base_state)

            # Pega as mensagens enviadas ao LLM
            call_args = mock_llm.invoke.call_args[0][0]

            # Primeira mensagem deve ser o SystemMessage
            assert isinstance(call_args[0], SystemMessage)

    def test_process_node_sends_history(self, base_state):
        """process_node deve incluir o histórico completo."""
        # Adiciona histórico ao estado
        base_state["messages"] = [
            HumanMessage(content="Pergunta 1"),
            AIMessage(content="Resposta 1"),
            HumanMessage(content="Pergunta 2"),
        ]

        mock_response = AIMessage(content="Resposta 2")

        with patch("app.graph.nodes.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm

            from app.graph.nodes import process_node
            process_node(base_state)

            call_args = mock_llm.invoke.call_args[0][0]
            # SystemMessage + 3 mensagens do histórico = 4 total
            assert len(call_args) == 4

    def test_process_node_handles_error(self, base_state):
        """process_node deve tratar erros da API graciosamente."""
        with patch("app.graph.nodes.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.invoke.side_effect = Exception("ThrottlingException")
            mock_get_llm.return_value = mock_llm

            from app.graph.nodes import process_node
            result = process_node(base_state)

            # Não deve lançar exceção — deve registrar no estado
            assert result["error"] is not None
            assert "ThrottlingException" in result["error"]
            assert result["final_response"] is None

    def test_output_node_handles_throttling_error(self, base_state):
        """output_node deve dar mensagem específica para throttling."""
        base_state["error"] = "ThrottlingException: Rate exceeded"
        base_state["final_response"] = None

        from app.graph.nodes import output_node
        result = output_node(base_state)

        assert "aguarde" in result["final_response"].lower() or \
               "momento" in result["final_response"].lower()

    def test_output_node_handles_auth_error(self, base_state):
        """output_node deve dar mensagem específica para erro de auth."""
        base_state["error"] = "AuthenticationError: Invalid credentials"
        base_state["final_response"] = None

        from app.graph.nodes import output_node
        result = output_node(base_state)

        assert "autenticação" in result["final_response"].lower() or \
               "credenciais" in result["final_response"].lower()


# ─────────────────────────────────────────────────────────────────────────────
# Testes de Integração — Chamada REAL ao Bedrock
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestBedrockIntegration:
    """
    Testes que fazem chamadas reais ao Amazon Bedrock.
    Requerem credenciais AWS válidas no .env.
    """

    def test_bedrock_connection(self):
        """Deve conseguir conectar ao Bedrock."""
        from app.config.aws_config import create_bedrock_session
        session = create_bedrock_session()
        client = session.client("bedrock-runtime")
        assert client is not None

    def test_llm_responds(self):
        """Claude deve responder uma pergunta simples."""
        from app.config.aws_config import get_llm
        from langchain_core.messages import HumanMessage

        llm = get_llm()
        response = llm.invoke([HumanMessage(content="Olá, tudo bem? Responda em uma frase.")])

        assert response is not None
        assert len(response.content) > 0

    def test_full_graph_with_bedrock(self):
        """Fluxo completo com LLM real."""
        from app.graph.builder import get_graph

        graph = get_graph()
        state: GraphState = {
            "messages": [],
            "user_input": "Quantos títulos o Brasil tem na Copa do Mundo?",
            "intent": None,
            "retrieved_context": None,
            "final_response": None,
            "error": None,
            "metadata": None,
        }

        result = graph.invoke(state)

        assert result["final_response"] is not None
        assert result["error"] is None
        # Brasil tem 5 títulos — o modelo deve mencionar isso
        assert "5" in result["final_response"] or \
               "cinco" in result["final_response"].lower()
