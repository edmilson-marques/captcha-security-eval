"""Clique robusto em grade de `.task` do hCaptcha/reCAPTCHA (correção de bug real, 2026-11-04).

**Bug encontrado ao vivo**: na primeira tentativa real de grade de imagens do
hCaptcha com `hcaptcha_combined_attempt`, o clique em sequência travou com
`ElementClickInterceptedException: Element is not clickable at point
(469, -9458)` -- uma coordenada Y impossível numa tela real. Causa provável:
o loop de refinamento por rodadas (`experiments.live.hcaptcha_classical_cv_attempt`/
`hcaptcha_combined_attempt`) busca `current_tasks` **uma vez** antes do laço
`for tile_index in new_indices: current_tasks[tile_index].click()` -- se
clicar o primeiro índice faz a grade re-renderizar/substituir algum tile
(comportamento comum de grade de CAPTCHA, que troca a imagem do tile
clicado), as referências restantes em `current_tasks` ficam presas a nós
removidos do DOM, cujo retângulo de layout pode relatar qualquer coisa,
inclusive um Y absurdo.

Correção: buscar os elementos de novo **a cada clique**, não uma vez por
rodada.
"""

from __future__ import annotations

import time

from selenium.common.exceptions import (
    ElementClickInterceptedException,
    StaleElementReferenceException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver

TASK_SELECTOR = ".task"


def click_grid_tiles(
    driver: WebDriver,
    indices: list[int],
    task_selector: str = TASK_SELECTOR,
    settle_seconds: float = 0.3,
    max_retries: int = 2,
) -> None:
    """Clica cada índice de `indices` numa grade de `.task`, tolerando re-render.

    Rebusca a lista de elementos imediatamente antes de cada clique (não
    reaproveita uma lista buscada antes do laço, que é a causa raiz do bug
    documentado acima) e tenta de novo, com uma nova busca, se o elemento já
    estiver obsoleto (`StaleElementReferenceException`) -- mesmo sintoma que
    o `ElementClickInterceptedException` com coordenada impossível observado
    ao vivo, então os dois disparam o mesmo retry.
    """
    for tile_index in indices:
        for attempt in range(max_retries + 1):
            try:
                tasks = driver.find_elements(By.CSS_SELECTOR, task_selector)
                tasks[tile_index].click()
                break
            except (StaleElementReferenceException, ElementClickInterceptedException):
                if attempt == max_retries:
                    raise
                time.sleep(settle_seconds)
        time.sleep(settle_seconds)
