"""
Cliente HTTP para APIs Externas — Fase 4.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
API ESCOLHIDA: api-football.com (via RapidAPI)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Usamos a API do api-football.com que oferece:
- Dados de todas as competições mundiais
- Classificações, resultados e estatísticas
- Plano gratuito: 100 requisições/dia
- Não precisa de cadastro complexo

Endpoint principal que usaremos:
  GET /teams/standings?league=1&season=2026
  (Copa do Mundo = league ID 1)

FALLBACK: Se não tiver API key configurada,
retorna dados mockados para não bloquear o aprendizado.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import httpx
from app.config.settings import settings


# ── URLs e configuração da API ─────────────────────────────────────────────────
RAPIDAPI_BASE_URL = "https://api-football-v1.p.rapidapi.com/v3"
WORLD_CUP_LEAGUE_ID = 1      # Copa do Mundo FIFA
TIMEOUT_SECONDS = 10


def _get_headers() -> dict:
    """Monta os headers de autenticação da RapidAPI."""
    return {
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
        "x-rapidapi-key": settings.rapidapi_key or "",
    }


def fetch_world_cup_standings(season: int = 2026) -> dict:
    """
    Busca a classificação/standings da Copa do Mundo.

    Args:
        season: Ano da Copa (ex: 2026).

    Returns:
        Dicionário com dados da classificação ou mock se sem API key.
    """
    # ── Fallback: retorna mock se não tiver API key ────────────────────
    if not settings.rapidapi_key:
        print("   ℹ️  API key não configurada — usando dados de exemplo")
        return _mock_standings()

    try:
        with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
            response = client.get(
                f"{RAPIDAPI_BASE_URL}/standings",
                headers=_get_headers(),
                params={
                    "league": WORLD_CUP_LEAGUE_ID,
                    "season": season,
                },
            )
            response.raise_for_status()
            data = response.json()
            return {"source": "api", "data": data, "season": season}

    except httpx.TimeoutException:
        return {"source": "error", "error": "Timeout na API", "data": _mock_standings()}
    except httpx.HTTPStatusError as e:
        return {"source": "error", "error": str(e), "data": _mock_standings()}
    except Exception as e:
        return {"source": "error", "error": str(e), "data": _mock_standings()}


def fetch_team_info(team_name: str) -> dict:
    """
    Busca informações de uma seleção específica.

    Args:
        team_name: Nome da seleção (ex: "Brazil", "France").

    Returns:
        Dicionário com informações da seleção.
    """
    if not settings.rapidapi_key:
        return _mock_team_info(team_name)

    try:
        with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
            response = client.get(
                f"{RAPIDAPI_BASE_URL}/teams",
                headers=_get_headers(),
                params={"name": team_name},
            )
            response.raise_for_status()
            return {"source": "api", "data": response.json()}

    except Exception as e:
        return {"source": "error", "error": str(e)}


def fetch_world_cup_news() -> dict:
    """
    Busca notícias recentes sobre Copa do Mundo.
    Usa a API de notícias como alternativa gratuita.

    Returns:
        Dicionário com notícias recentes.
    """
    # NewsAPI gratuita (1000 req/dia no plano free)
    news_api_key = settings.news_api_key if hasattr(settings, 'news_api_key') else None

    if not news_api_key:
        return _mock_news()

    try:
        with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
            response = client.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": "Copa do Mundo FIFA 2026",
                    "language": "pt",
                    "sortBy": "publishedAt",
                    "pageSize": 5,
                    "apiKey": news_api_key,
                },
            )
            response.raise_for_status()
            return {"source": "api", "data": response.json()}

    except Exception as e:
        return {"source": "error", "error": str(e), "data": _mock_news()}


# ── Dados Mock — usados quando não há API key ──────────────────────────────────

def _mock_standings() -> dict:
    """Dados de exemplo para classificação."""
    return {
        "source": "mock",
        "description": "Dados de exemplo (configure RAPIDAPI_KEY no .env para dados reais)",
        "standings": [
            {"position": 1, "team": "Brasil",   "points": 18, "played": 6, "won": 6},
            {"position": 2, "team": "França",    "points": 15, "played": 6, "won": 5},
            {"position": 3, "team": "Argentina", "points": 13, "played": 6, "won": 4},
            {"position": 4, "team": "Alemanha",  "points": 12, "played": 6, "won": 4},
            {"position": 5, "team": "Espanha",   "points": 10, "played": 6, "won": 3},
        ]
    }


def _mock_team_info(team_name: str) -> dict:
    """Dados de exemplo para informações de seleção."""
    return {
        "source": "mock",
        "team": team_name,
        "description": f"Informações sobre {team_name} (dados de exemplo)",
        "titles": "Consulte os documentos históricos para títulos oficiais",
    }


def _mock_news() -> dict:
    """Dados de exemplo para notícias."""
    return {
        "source": "mock",
        "description": "Notícias de exemplo (configure NEWS_API_KEY no .env para notícias reais)",
        "articles": [
            {
                "title": "Copa do Mundo 2026: preparativos avançam nos EUA, México e Canadá",
                "description": "Os três países-sede intensificam os preparativos para o maior torneio da história.",
                "publishedAt": "2026-01-15",
            },
            {
                "title": "FIFA confirma formato com 48 seleções para 2026",
                "description": "A Copa de 2026 será a primeira com 48 participantes, expandindo o torneio.",
                "publishedAt": "2026-01-10",
            },
        ]
    }
