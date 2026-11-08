"""Testes do solver de "elementos idênticos" (hCaptcha, Atividade 2, Nov/2026).

Imagens sintéticas (fundo liso + N cópias de um ícone + 1 ou mais
distratores de forma/cor diferente) -- a resposta certa (quais pontos são
o grupo repetido) é conhecida por construção, então dá pra afirmar acerto
real, não só contrato de saída.
"""

from __future__ import annotations

import cv2
import numpy as np

from attacks.identical_elements_solver import solve_identical_elements_classical


def _background(size: int = 400, color: tuple[int, int, int] = (120, 120, 120)) -> np.ndarray:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:, :] = color
    return image


def _draw_star(image: np.ndarray, center: tuple[int, int], color: tuple[int, int, int]) -> None:
    cv2.drawMarker(image, center, color, markerType=cv2.MARKER_STAR, markerSize=30, thickness=6)


def _draw_circle(image: np.ndarray, center: tuple[int, int], color: tuple[int, int, int]) -> None:
    cv2.circle(image, center, 20, color, thickness=6)


def _draw_triangle(
    image: np.ndarray,
    center: tuple[int, int],
    color: tuple[int, int, int],
    angle_deg: float = 0.0,
    scale: float = 1.0,
) -> None:
    pts = np.array([[0, -20], [17, 12], [-17, 12]], dtype=np.float32) * scale
    theta = np.radians(angle_deg)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pts = (pts @ rotation.T) + np.array(center)
    cv2.polylines(image, [pts.astype(np.int32)], isClosed=True, color=color, thickness=5)


IDENTICAL_COLOR = (0, 200, 255)
IDENTICAL_POSITIONS = [(70, 70), (330, 70), (200, 330)]


def test_finds_the_repeated_group_and_skips_the_odd_one():
    image = _background()
    for center in IDENTICAL_POSITIONS:
        _draw_star(image, center, IDENTICAL_COLOR)
    _draw_circle(image, (330, 330), (255, 50, 0))  # distrator: forma diferente

    points = solve_identical_elements_classical(image)

    assert len(points) == len(IDENTICAL_POSITIONS)
    for px, py in points:
        closest = min(IDENTICAL_POSITIONS, key=lambda c: abs(c[0] - px) + abs(c[1] - py))
        assert abs(closest[0] - px) < 25
        assert abs(closest[1] - py) < 25


def test_same_shape_different_color_is_not_treated_as_identical():
    # Mesma forma (estrela), mas cores bem diferentes -- não deve virar um
    # grupo "idêntico" só por silhueta, já que a cor também importa.
    image = _background()
    _draw_star(image, (70, 70), (0, 200, 255))
    _draw_star(image, (330, 70), (255, 0, 120))
    _draw_star(image, (200, 330), (0, 255, 0))

    points = solve_identical_elements_classical(image)

    assert points == []


def test_returns_empty_when_everything_is_unique():
    image = _background()
    _draw_star(image, (70, 70), (0, 200, 255))
    _draw_circle(image, (330, 70), (255, 50, 0))

    assert solve_identical_elements_classical(image) == []


def test_returns_empty_with_fewer_than_two_regions():
    image = _background()

    assert solve_identical_elements_classical(image) == []


def test_finds_repeated_shape_rotated_and_scaled_differently():
    # Achado ao vivo em 2026-11-04: o desafio real mostra cópias idênticas
    # giradas/escaladas de forma diferente (ver docstring de
    # `solve_identical_elements_classical`) -- a versão anterior (IoU de
    # silhueta bbox-relativa) não tolerava isso.
    image = _background()
    _draw_triangle(image, (80, 80), IDENTICAL_COLOR, angle_deg=0)
    _draw_triangle(image, (320, 80), IDENTICAL_COLOR, angle_deg=47, scale=1.3)
    _draw_triangle(image, (200, 320), (255, 50, 0), angle_deg=10)  # distrator: cor diferente

    points = solve_identical_elements_classical(image)

    assert len(points) == 2
    xs = sorted(px for px, _py in points)
    assert abs(xs[0] - 80) < 25
    assert abs(xs[1] - 320) < 25
