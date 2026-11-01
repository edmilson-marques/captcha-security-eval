"""Simulação de comportamento humano — trajetória de mouse e timing (Atividade 3, Ago-Set/2026).

Alimenta diretamente o sinal que o motor de risco do reCAPTCHA/Turnstile avalia.
"""

from __future__ import annotations

import random


def bezier_mouse_path(
    start: tuple[float, float],
    end: tuple[float, float],
    n_points: int = 20,
    jitter: float = 15.0,
) -> list[tuple[float, float]]:
    """Gera uma trajetória de mouse não-linear entre `start` e `end`.

    Usa uma curva de Bézier quadrática cujo ponto de controle é o ponto médio
    da reta `start`-`end` deslocado aleatoriamente em até `jitter` pixels (em
    x e em y), simulando a curvatura natural de um movimento humano de mouse
    (em oposição a uma linha reta, característica de automação ingênua).

    Retorna uma lista de `n_points` pontos `(x, y)` amostrados uniformemente
    em t ∈ [0, 1], com o primeiro ponto exatamente igual a `start` e o último
    exatamente igual a `end`.

    Com `jitter=0.0` o ponto de controle coincide com o ponto médio exato da
    reta e a trajetória resultante é determinística (mesma entrada sempre
    produz a mesma saída) — útil para testes reprodutíveis.
    """
    if n_points < 2:
        raise ValueError("n_points deve ser >= 2 (inclui start e end).")

    x0, y0 = start
    x2, y2 = end

    mid_x = (x0 + x2) / 2.0
    mid_y = (y0 + y2) / 2.0
    control_x = mid_x + random.uniform(-jitter, jitter)
    control_y = mid_y + random.uniform(-jitter, jitter)

    path: list[tuple[float, float]] = []
    for i in range(n_points):
        t = i / (n_points - 1)
        x = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * control_x + t**2 * x2
        y = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * control_y + t**2 * y2
        path.append((x, y))

    # Garante igualdade exata nas extremidades, sem depender de precisão de ponto flutuante.
    path[0] = (x0, y0)
    path[-1] = (x2, y2)
    return path


def human_delay(mean: float = 0.3, std: float = 0.1, minimum: float = 0.05) -> float:
    """Amostra um delay (segundos) de uma normal truncada em `minimum`.

    Evita valores negativos ou absurdamente pequenos que denunciariam
    automação (ex.: cliques/digitação com latência zero entre ações).
    """
    sample = random.gauss(mean, std)
    return max(sample, minimum)
