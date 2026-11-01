"""Testes da segmentação de caracteres (attacks.classical_cv.segment_characters) — Atividade 2."""

from __future__ import annotations

import cv2
import numpy as np

from attacks import classical_cv


def _synthetic_binary_with_blobs(
    x_positions: list[int], width: int = 15, height: int = 20
) -> np.ndarray:
    """Imagem binária 40x200 com um "blob" retangular branco em cada x de
    `x_positions`, simulando caracteres espaçados horizontalmente sobre fundo
    preto."""
    image = np.zeros((40, 200), dtype=np.uint8)
    for x in x_positions:
        cv2.rectangle(image, (x, 10), (x + width, 10 + height), 255, thickness=-1)
    return image


def test_segment_characters_returns_expected_count_and_left_to_right_order():
    # Posições fora de ordem para garantir que a ordenação por x funciona de fato.
    positions = [120, 10, 60, 90]
    binary = _synthetic_binary_with_blobs(positions)

    crops = classical_cv.segment_characters(binary, min_area=20)

    assert len(crops) == len(positions)

    expected_order = sorted(positions)
    for crop, x in zip(crops, expected_order):
        assert crop.shape == (21, 16)  # altura/largura do retângulo (inclusive) desenhado
        assert np.mean(crop) > 200  # recorte deve ser majoritariamente o "caractere" branco


def test_segment_characters_ignores_components_below_min_area():
    binary = _synthetic_binary_with_blobs([10, 60, 110])
    # Ruído minúsculo (2x2 = área 4) que deve ser descartado pelo limiar de área mínima.
    binary[2:4, 2:4] = 255

    crops = classical_cv.segment_characters(binary, min_area=20)

    assert len(crops) == 3


def test_segment_characters_empty_image_returns_empty_list():
    binary = np.zeros((40, 200), dtype=np.uint8)

    crops = classical_cv.segment_characters(binary, min_area=20)

    assert crops == []


def test_segment_characters_four_blobs_in_natural_order():
    positions = [5, 40, 75, 110]
    binary = _synthetic_binary_with_blobs(positions)

    crops = classical_cv.segment_characters(binary, min_area=20)

    assert len(crops) == 4
    for crop in crops:
        assert crop.shape == (21, 16)
