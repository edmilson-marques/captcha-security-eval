"""Testes do solver temporal de "ícone diferente" (Atividade 2, Nov/2026).

Usa uma rajada sintética de frames (fundo estático + um ícone comum que
aparece na maioria dos frames + um ícone raro que aparece em poucos) --
não depende de captura real do navegador. Mesma filosofia dos outros
testes: aqui dá pra afirmar o acerto de verdade (não só o contrato de
saída), já que a posição do ícone raro é conhecida por construção.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from attacks.temporal_canvas_solver import solve_icon_outlier_temporal

COMMON_ICON_CENTER = (150, 150)
RARE_ICON_CENTER = (400, 300)


def _background(size: tuple[int, int] = (500, 600)) -> np.ndarray:
    h, w = size
    image = np.full((h, w, 3), (120, 110, 100), dtype=np.uint8)
    cv2.rectangle(image, (0, 0), (w, h // 2), (100, 130, 110), -1)
    return image


def _draw_icon(image: np.ndarray, center: tuple[int, int], color: tuple[int, int, int]) -> None:
    cv2.circle(image, center, 25, color, thickness=6)


def _make_burst(common_frames: int = 6, rare_frames: int = 2) -> list[np.ndarray]:
    frames = []
    for _ in range(common_frames):
        frame = _background()
        _draw_icon(frame, COMMON_ICON_CENTER, (0, 200, 255))
        frames.append(frame)
    for _ in range(rare_frames):
        frame = _background()
        _draw_icon(frame, RARE_ICON_CENTER, (255, 50, 0))
        frames.append(frame)
    return frames


def test_identifies_the_rare_icon_position():
    frames = _make_burst()

    point = solve_icon_outlier_temporal(frames)

    assert point is not None
    px, py = point
    assert abs(px - RARE_ICON_CENTER[0]) < 30
    assert abs(py - RARE_ICON_CENTER[1]) < 30


def test_returns_none_with_too_few_frames():
    frames = _make_burst()[:2]

    assert solve_icon_outlier_temporal(frames) is None


def test_returns_none_for_fully_static_frames():
    frames = [_background() for _ in range(5)]

    assert solve_icon_outlier_temporal(frames) is None


@pytest.mark.parametrize("rare_frames", [1, 3])
def test_identifies_rare_icon_regardless_of_exact_count(rare_frames: int):
    frames = _make_burst(common_frames=6, rare_frames=rare_frames)

    point = solve_icon_outlier_temporal(frames)

    assert point is not None
    px, py = point
    assert abs(px - RARE_ICON_CENTER[0]) < 30
    assert abs(py - RARE_ICON_CENTER[1]) < 30


def test_tolerates_black_header_detected_at_slightly_different_rows():
    # Achado ao vivo em 2026-11-04: detectar a faixa preta do cabeçalho
    # frame a frame (em vez de uma vez só) podia cortar cada frame numa
    # altura diferente e quebrar np.stack em _temporal_std_map ("all input
    # arrays must have the same shape"). Reproduz isso com uma faixa preta
    # cuja última linha varia de brilho entre frames, perto do limiar.
    header_height = 40
    frames = _make_burst()
    for index, frame in enumerate(frames):
        frame[:header_height] = 0
        frame[header_height - 1] = 10 if index % 2 == 0 else 20

    point = solve_icon_outlier_temporal(frames)

    assert point is not None
