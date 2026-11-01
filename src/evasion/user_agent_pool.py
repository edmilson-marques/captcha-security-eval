"""Pool de user-agents / randomização de fingerprint (Atividade 3, Ago-Set/2026).

Fornece um conjunto de strings de user-agent realistas e um gerador de perfis
simples de fingerprint (user-agent + viewport + locale) para diversificar o
navegador controlado por `src/harness/browser.py`.
"""

from __future__ import annotations

import random

DEFAULT_USER_AGENTS: list[str] = [
    # Chrome desktop (Windows, macOS, Linux)
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    ),
    # Firefox desktop (Windows, macOS)
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:131.0) Gecko/20100101 Firefox/131.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:130.0) Gecko/20100101 Firefox/130.0",
    # Edge desktop (Windows)
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36 Edg/129.0.0.0"
    ),
    # Safari desktop (macOS)
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/17.6 Safari/605.1.15"
    ),
    # Chrome mobile (Android)
    (
        "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/129.0.0.0 Mobile Safari/537.36"
    ),
    # Safari mobile (iOS)
    (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_6 like Mac OS X) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/17.6 Mobile/15E148 Safari/604.1"
    ),
]

# Subconjunto só de desktop — os 5 primeiros itens de DEFAULT_USER_AGENTS.
# Importante escolher a partir daqui (não de DEFAULT_USER_AGENTS) sempre que
# o resto do fingerprint (ex. tamanho de janela) for fixo e de desktop: um
# user-agent mobile (Android/iOS) combinado com uma janela 1366x900 é, ele
# mesmo, uma inconsistência detectável — pior que não randomizar nada.
DESKTOP_USER_AGENTS: list[str] = DEFAULT_USER_AGENTS[:5]

_VIEWPORTS: list[tuple[int, int]] = [
    (1366, 768),
    (1920, 1080),
    (1440, 900),
]

_LOCALES: list[str] = ["pt-BR", "en-US"]


def random_user_agent(pool: list[str] | None = None) -> str:
    """Retorna um user-agent aleatório de `pool` (ou de `DEFAULT_USER_AGENTS`)."""
    candidates = pool if pool is not None else DEFAULT_USER_AGENTS
    if not candidates:
        raise ValueError("O pool de user-agents não pode ser vazio.")
    return random.choice(candidates)


def random_fingerprint_profile(pool: list[str] | None = None) -> dict:
    """Gera um perfil básico de fingerprint: user-agent, viewport e locale.

    Útil para diversificar as sessões de navegador controladas via
    `src/harness/browser.py` em experimentos futuros (Atividades 4-6).
    """
    return {
        "user_agent": random_user_agent(pool),
        "viewport": random.choice(_VIEWPORTS),
        "locale": random.choice(_LOCALES),
    }
