"""Rotação de proxies autorizados (Atividade 3, Ago-Set/2026).

**Diretriz ética do projeto**: usar exclusivamente proxies contratados/autorizados
pelo próprio aluno para os experimentos. Nunca utilizar proxy aberto, gratuito ou
de terceiros sem autorização explícita — isso não é apenas uma boa prática, é uma
condição do escopo ético descrito no README e no TCC1 (seção "Motivações").
"""

from __future__ import annotations


class ProxyRotator:
    """Alterna entre uma lista de proxies autorizados em round-robin.

    Garante que o mesmo proxy não seja usado duas vezes seguidas quando há mais
    de uma opção disponível — reduz a chance de padrões de requisição triviais
    de serem atribuídos a uma única identidade de rede.
    """

    def __init__(self, proxies: list[str]) -> None:
        if not proxies:
            raise ValueError("A lista de proxies não pode ser vazia.")
        self._proxies = list(proxies)
        self._last_index: int | None = None

    def next(self) -> str:
        """Retorna o próximo proxy, evitando repetir o último retornado."""
        if len(self._proxies) == 1:
            self._last_index = 0
            return self._proxies[0]

        if self._last_index is None:
            next_index = 0
        else:
            next_index = (self._last_index + 1) % len(self._proxies)

        self._last_index = next_index
        return self._proxies[next_index]
