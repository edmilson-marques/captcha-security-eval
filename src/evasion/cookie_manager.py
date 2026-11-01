"""Ciclo de vida de sessão/cookies por identidade simulada (Atividade 3, Ago-Set/2026).

Armazenamento em memória, agnóstico de Selenium: apenas guarda e recupera
listas de cookies associadas a uma "identidade" (sessão simulada). A
integração real com `driver.get_cookies()` / `driver.add_cookie()` fica a
cargo do runner de experimentos (Atividades 4-6).
"""

from __future__ import annotations

import uuid


class CookieManager:
    """Mantém cookies por identidade simulada, em memória."""

    def __init__(self) -> None:
        self._store: dict[str, list[dict]] = {}

    def new_identity(self) -> str:
        """Cria e registra uma nova identidade (sem cookies) e retorna seu id."""
        identity = uuid.uuid4().hex
        self._store[identity] = []
        return identity

    def save_cookies(self, identity: str, cookies: list[dict]) -> None:
        """Substitui os cookies associados a `identity` pelos informados."""
        self._store[identity] = list(cookies)

    def load_cookies(self, identity: str) -> list[dict]:
        """Retorna os cookies de `identity`, ou `[]` se desconhecida/sem cookies."""
        return list(self._store.get(identity, []))

    def clear(self, identity: str) -> None:
        """Remove a identidade e seus cookies. No-op se ela não existir."""
        self._store.pop(identity, None)
