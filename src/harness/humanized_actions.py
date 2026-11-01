"""Execução de clique/arrasto humanizado sobre um `<canvas>` (Atividade 3 ligada de verdade).

Usado pelos desafios do hCaptcha renderizados inteiramente em `<canvas>`
(sem elemento DOM por item clicável — ver `harness/collectors/hcaptcha_collector.py`):
"ícone diferente", "laços fechados" e o quebra-cabeça de arrastar-e-encaixar.

**Por que não um clique/arrasto direto via JS**: pesquisa conceitual em
frameworks de automação (não copiados, só arquitetura) mostra que motores de
risco de CAPTCHA podem rejeitar eventos de mouse sintetizados via JS puro
(`isTrusted=false`) mesmo quando a resposta é visualmente correta. Por isso
aqui a interação sempre passa por `ActionChains` (entrada via protocolo do
navegador, `isTrusted=true`) e percorre uma trajetória não-linear
(`evasion.human_behavior.canvas_offset_path`, curva de Bézier) com pausas
(`human_delay`) em vez de teleportar direto ao alvo — mesmo princípio já
aplicado ao user-agent em `browser.py`: um teleporte instantâneo de mouse é,
por si só, um sinal de automação.

Não tem suíte pytest (depende de um `<canvas>` real renderizado num
navegador) — mesma categoria de `experiments/live.py`, validado manualmente
contra a demo oficial do hCaptcha.
"""

from __future__ import annotations

from selenium.webdriver import ActionChains
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement

from evasion.human_behavior import canvas_offset_path, human_delay


def _walk_bezier(
    actions: ActionChains,
    canvas: WebElement,
    canvas_size: tuple[int, int],
    start_point: tuple[float, float],
    end_point: tuple[float, float],
    n_points: int = 14,
) -> ActionChains:
    path = canvas_offset_path(canvas_size, start_point, end_point, n_points=n_points)
    first_x, first_y = round(path[0][0]), round(path[0][1])
    actions.move_to_element_with_offset(canvas, first_x, first_y)
    prev_x, prev_y = first_x, first_y
    for x, y in path[1:]:
        xi, yi = round(x), round(y)
        if (xi, yi) != (prev_x, prev_y):
            actions.move_by_offset(xi - prev_x, yi - prev_y)
        prev_x, prev_y = xi, yi
    return actions


def click_canvas_point(
    driver: WebDriver,
    canvas: WebElement,
    point: tuple[float, float],
    canvas_size: tuple[int, int],
) -> None:
    """Clica em `point` (pixel, canto superior esquerdo do canvas) de forma humanizada.

    Parte sempre do centro do canvas (não há como saber a posição real prévia
    do cursor do SO) e percorre uma curva de Bézier até o alvo antes de clicar.
    """
    canvas_width, canvas_height = canvas_size
    center_point = (canvas_width / 2, canvas_height / 2)
    actions = ActionChains(driver)
    _walk_bezier(actions, canvas, canvas_size, center_point, point)
    actions.pause(human_delay(mean=0.15, std=0.05, minimum=0.05))
    actions.click()
    actions.perform()


def drag_canvas_point(
    driver: WebDriver,
    canvas: WebElement,
    piece_point: tuple[float, float],
    target_point: tuple[float, float],
    canvas_size: tuple[int, int],
) -> None:
    """Arrasta de `piece_point` até `target_point` (pixels, canto superior esquerdo do canvas).

    Sequência: move até a peça (curva de Bézier a partir do centro) -> pausa
    -> pressiona e segura -> move até o alvo (curva de Bézier) -> pausa ->
    solta. Tudo via `ActionChains` (entrada confiável), nunca via evento de
    mouse sintetizado em JS.
    """
    canvas_width, canvas_height = canvas_size
    center_point = (canvas_width / 2, canvas_height / 2)
    actions = ActionChains(driver)
    _walk_bezier(actions, canvas, canvas_size, center_point, piece_point)
    actions.pause(human_delay(mean=0.2, std=0.05, minimum=0.05))
    actions.click_and_hold()
    actions.pause(human_delay(mean=0.1, std=0.03, minimum=0.03))
    _walk_bezier(actions, canvas, canvas_size, piece_point, target_point, n_points=18)
    actions.pause(human_delay(mean=0.15, std=0.05, minimum=0.05))
    actions.release()
    actions.perform()
