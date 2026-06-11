"""
Guardrails — Validação de Entrada e Saída.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONCEITO: O QUE SÃO GUARDRAILS?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Guardrails são "trilhos de segurança" — filtros que
protegem o sistema em duas direções:

  ENTRADA (Input Guardrails):
    Analisa a pergunta ANTES de processar
    Bloqueia: conteúdo inadequado, jailbreak attempts,
              perguntas maliciosas

  SAÍDA (Output Guardrails):
    Analisa a resposta ANTES de entregar ao usuário
    Bloqueia: respostas muito curtas, conteúdo inadequado,
              respostas que fogem do escopo

Analogia: são como os seguranças na entrada e saída
de um evento — verificam quem entra e o que sai.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import re
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """
    Resultado de uma validação de guardrail.

    Attributes:
        is_valid: True se passou na validação
        reason:   Motivo do bloqueio (se não passou)
        severity: "low" | "medium" | "high"
    """
    is_valid: bool
    reason: str = ""
    severity: str = "low"


# ── Padrões de conteúdo inadequado ────────────────────────────────────────────
# Lista de padrões que indicam tentativas de manipulação ou conteúdo indevido
JAILBREAK_PATTERNS = [
    r"ignore (all |previous |your )?(instructions|rules|guidelines)",
    r"you are now",
    r"pretend (you are|to be)",
    r"act as (if you are|a)",
    r"forget (everything|all|your training)",
    r"bypass (your|the) (rules|restrictions|filters)",
    r"do anything now",
    r"jailbreak",
    r"prompt injection",
]

INAPPROPRIATE_PATTERNS = [
    r"\b(hack|exploit|vulnerabilit)",
    r"\b(senha|password|credential)",
    r"(dados pessoais|cpf|rg|cartão de crédito)",
]

# Comprimento mínimo para uma resposta ser considerada válida
MIN_RESPONSE_LENGTH = 5
MAX_RESPONSE_LENGTH = 5000


# ─────────────────────────────────────────────────────────────────────────────
# GUARDRAIL DE ENTRADA
# ─────────────────────────────────────────────────────────────────────────────

def validate_input(user_input: str) -> ValidationResult:
    """
    Valida a pergunta do usuário antes de processar.

    Verificações:
    1. Comprimento mínimo e máximo
    2. Tentativas de jailbreak
    3. Conteúdo claramente inadequado

    Args:
        user_input: Pergunta do usuário.

    Returns:
        ValidationResult com is_valid=True se passou.
    """
    # ── Verificação 1: Comprimento ────────────────────────────────────
    if not user_input or len(user_input.strip()) < 2:
        return ValidationResult(
            is_valid=False,
            reason="Pergunta muito curta. Por favor, elabore sua questão.",
            severity="low",
        )

    if len(user_input) > 1000:
        return ValidationResult(
            is_valid=False,
            reason="Pergunta muito longa. Por favor, seja mais conciso.",
            severity="low",
        )

    # ── Verificação 2: Tentativas de Jailbreak ────────────────────────
    input_lower = user_input.lower()
    for pattern in JAILBREAK_PATTERNS:
        if re.search(pattern, input_lower, re.IGNORECASE):
            return ValidationResult(
                is_valid=False,
                reason="Não consigo processar este tipo de solicitação.",
                severity="high",
            )

    # ── Verificação 3: Conteúdo Inadequado ───────────────────────────
    for pattern in INAPPROPRIATE_PATTERNS:
        if re.search(pattern, input_lower, re.IGNORECASE):
            return ValidationResult(
                is_valid=False,
                reason="Este tipo de conteúdo não é suportado.",
                severity="medium",
            )

    return ValidationResult(is_valid=True)


# ─────────────────────────────────────────────────────────────────────────────
# GUARDRAIL DE SAÍDA
# ─────────────────────────────────────────────────────────────────────────────

def validate_output(response: str, intent: str = "rag") -> ValidationResult:
    """
    Valida a resposta gerada antes de entregar ao usuário.

    Verificações:
    1. Resposta não pode ser vazia ou muito curta
    2. Resposta não pode ser excessivamente longa
    3. Resposta deve estar em português (verificação básica)

    Args:
        response: Resposta gerada pelo LLM.
        intent:   Rota que gerou a resposta.

    Returns:
        ValidationResult com is_valid=True se passou.
    """
    # Off-topic tem resposta estática — sempre válida
    if intent == "off_topic":
        return ValidationResult(is_valid=True)

    # ── Verificação 1: Resposta vazia ─────────────────────────────────
    if not response or len(response.strip()) < MIN_RESPONSE_LENGTH:
        return ValidationResult(
            is_valid=False,
            reason="Resposta gerada foi insuficiente. Tentando novamente.",
            severity="medium",
        )

    # ── Verificação 2: Resposta excessivamente longa ──────────────────
    if len(response) > MAX_RESPONSE_LENGTH:
        # Trunca em vez de bloquear
        return ValidationResult(
            is_valid=True,  # válida, mas será truncada
            reason=f"Resposta truncada de {len(response)} para {MAX_RESPONSE_LENGTH} chars",
            severity="low",
        )

    # ── Verificação 3: Detecta se está em português ───────────────────
    # Palavras comuns em português que devem aparecer em respostas normais
    portuguese_indicators = [
        "que", "de", "do", "da", "em", "para", "com", "uma", "foi",
        "são", "pela", "pelo", "como", "mais", "também", "esse", "essa"
    ]
    response_lower = response.lower()
    portuguese_count = sum(
        1 for word in portuguese_indicators
        if f" {word} " in response_lower
    )

    if portuguese_count < 2 and len(response) > 100:
        return ValidationResult(
            is_valid=False,
            reason="Resposta não parece estar em português.",
            severity="medium",
        )

    return ValidationResult(is_valid=True)


# ─────────────────────────────────────────────────────────────────────────────
# FUNÇÃO PRINCIPAL — usada pelos nós do grafo
# ─────────────────────────────────────────────────────────────────────────────

def run_input_guardrail(user_input: str) -> tuple[bool, str]:
    """
    Interface simplificada para o guardrail de entrada.

    Returns:
        (passou, mensagem_de_erro)
    """
    result = validate_input(user_input)
    if not result.is_valid:
        print(f"   🛡️  [guardrail_input] BLOQUEADO ({result.severity}): {result.reason}")
    return result.is_valid, result.reason


def run_output_guardrail(response: str, intent: str = "rag") -> tuple[bool, str, str]:
    """
    Interface simplificada para o guardrail de saída.

    Returns:
        (passou, resposta_possivelmente_truncada, motivo)
    """
    result = validate_output(response, intent)

    # Trunca se necessário
    if result.is_valid and len(response) > MAX_RESPONSE_LENGTH:
        response = response[:MAX_RESPONSE_LENGTH] + "..."
        print(f"   🛡️  [guardrail_output] Resposta truncada")

    if not result.is_valid:
        print(f"   🛡️  [guardrail_output] BLOQUEADO ({result.severity}): {result.reason}")

    return result.is_valid, response, result.reason
