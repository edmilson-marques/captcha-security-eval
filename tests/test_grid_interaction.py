"""Teste do clique robusto em grade (Atividade 2, Nov/2026) -- sem navegador real.

Diferente de `experiments/live.py`/`harness/humanized_actions.py` (que
dependem de um `<canvas>` renderizado de verdade e por isso não têm suíte),
`click_grid_tiles` só orquestra chamadas de `driver.find_elements`/
`.click()` -- dá pra testar com um driver falso simples, sem Selenium real.
"""

from __future__ import annotations

from selenium.common.exceptions import ElementClickInterceptedException

from attacks.grid_interaction import click_grid_tiles


class _FakeElement:
    def __init__(self, index: int, fail_times: int = 0):
        self.index = index
        self._fail_times = fail_times
        self.clicked = False

    def click(self):
        if self._fail_times > 0:
            self._fail_times -= 1
            raise ElementClickInterceptedException("Element is not clickable at point (0, -9458)")
        self.clicked = True


class _FakeDriver:
    """Re-cria os elementos a cada `find_elements` -- simula o DOM re-renderizando."""

    def __init__(self, size: int, fail_once_at: int | None = None):
        self.size = size
        self.fail_once_at = fail_once_at
        self._already_failed = False
        self.find_elements_calls = 0
        self.last_elements: list[_FakeElement] = []

    def find_elements(self, _by, _selector):
        self.find_elements_calls += 1
        should_fail = self.fail_once_at is not None and not self._already_failed
        self.last_elements = [
            _FakeElement(i, fail_times=(1 if should_fail and i == self.fail_once_at else 0))
            for i in range(self.size)
        ]
        if should_fail:
            self._already_failed = True
        return self.last_elements


def test_requeries_elements_before_each_click():
    driver = _FakeDriver(size=5)

    click_grid_tiles(driver, [1, 3], settle_seconds=0)

    # uma busca nova por clique, não uma lista reaproveitada pra todos
    assert driver.find_elements_calls == 2
    assert driver.last_elements[3].clicked


def test_retries_once_on_click_intercepted_then_succeeds():
    driver = _FakeDriver(size=3, fail_once_at=1)

    click_grid_tiles(driver, [1], settle_seconds=0, max_retries=2)

    assert driver.last_elements[1].clicked
    # 1 busca que falhou + 1 busca de retry que teve sucesso
    assert driver.find_elements_calls == 2
