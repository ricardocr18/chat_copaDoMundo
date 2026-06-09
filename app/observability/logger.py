"""
Configuração de Logging Estruturado com structlog.

Por que logging estruturado em vez de print()?

print("Usuário perguntou: " + pergunta)
→ Difícil de filtrar, buscar e analisar em produção.

logger.info("user.query", question=pergunta, user_id=uid, session=sid)
→ JSON estruturado, filtrável, indexável, compatível com CloudWatch/Datadog.

Exemplo de output em desenvolvimento:
2024-01-15 10:30:00 [info] input_processor_node.start user_input=Quem ganhou? iteration=1

Exemplo de output em produção (JSON):
{"timestamp":"2024-01-15T10:30:00Z","level":"info","event":"input_processor_node.start",
 "user_input":"Quem ganhou?","iteration":1,"service":"world-cup-agent","env":"production"}
"""

import logging
import sys

import structlog


def configure_logging(log_level: str = "DEBUG", is_production: bool = False) -> None:
    """
    Configura o sistema de logging para a aplicação.

    Em desenvolvimento: output legível e colorido no terminal.
    Em produção: output JSON estruturado para ingestão por ferramentas.

    Args:
        log_level: Nível de logging (DEBUG, INFO, WARNING, ERROR).
        is_production: Se True, usa formato JSON para produção.
    """
    # Configura o logging padrão do Python (usado por bibliotecas terceiras)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )

    # Processadores compartilhados entre dev e prod
    shared_processors = [
        structlog.contextvars.merge_contextvars,        # Adiciona contexto global
        structlog.stdlib.add_log_level,                  # Adiciona campo "level"
        structlog.stdlib.add_logger_name,                # Adiciona campo "logger"
        structlog.processors.TimeStamper(fmt="iso"),     # Timestamp ISO 8601
        structlog.processors.StackInfoRenderer(),        # Stack trace se houver
    ]

    if is_production:
        # Produção: JSON para ingestão por CloudWatch, Datadog, etc.
        processors = shared_processors + [
            structlog.processors.format_exc_info,        # Exceções como JSON
            structlog.processors.JSONRenderer(),         # Output: {"key": "value"}
        ]
    else:
        # Desenvolvimento: output bonito e colorido no terminal
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),  # Output legível
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,                  # Otimização de performance
    )
