"""Testes dos solvers clássicos de clique-sobre-canvas (attacks.classical_canvas_solver).

`solve_icon_outlier_classical`/`solve_closed_loop_classical` usam só OpenCV,
testados com imagens sintéticas. `solve_digit_membership_classical` depende
do binário `tesseract-ocr` (via `attacks.ocr`); os testes que exercitam OCR
de verdade são pulados automaticamente se o binário não estiver instalado
(mesmo padrão de `tests/test_ocr.py`) — `_parse_target_digits` é testada à
parte, por ser pura (sem OCR).
"""

from __future__ import annotations

import shutil

import cv2
import numpy as np
import pytest

from attacks import classical_canvas_solver as solver

requires_tesseract = pytest.mark.skipif(
    shutil.which("tesseract") is None, reason="binário tesseract-ocr não encontrado no PATH"
)


def _flat_background(size: int = 300, color: tuple[int, int, int] = (120, 120, 120)) -> np.ndarray:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:, :] = color
    return image


def _background_with_black_header(header_height: int = 60, size: int = 300) -> np.ndarray:
    image = _flat_background(size=size)
    image[:header_height, :] = (0, 0, 0)
    return image


def test_crop_black_header_removes_header_and_reports_offset():
    image = _background_with_black_header(header_height=60)

    cropped, y_offset = solver._crop_black_header(image)

    assert y_offset == 60
    assert cropped.shape[0] == image.shape[0] - 60
    # Não deve sobrar nenhuma linha preta no recorte.
    assert cropped[:5, :].mean() > 15


def test_crop_black_header_no_header_returns_unchanged():
    image = _flat_background()

    cropped, y_offset = solver._crop_black_header(image)

    assert y_offset == 0
    assert cropped.shape == image.shape


def test_solve_icon_outlier_classical_adjusts_for_black_header():
    image = _background_with_black_header(header_height=60, size=300)
    cv2.circle(image, (80, 140), 25, color=(60, 60, 60), thickness=10)  # camuflado
    cv2.circle(image, (220, 220), 25, color=(0, 0, 255), thickness=10)  # destoante

    result = solver.solve_icon_outlier_classical(image)

    assert result is not None
    x, y = result
    # Coordenada deve ser relativa à imagem ORIGINAL (com cabeçalho), não ao recorte.
    assert abs(x - 220) < 40
    assert abs(y - 220) < 40


def test_solve_icon_outlier_classical_picks_the_more_contrasting_icon():
    image = _flat_background(color=(130, 130, 130))
    # Ícone "camuflado": cor próxima do fundo.
    cv2.circle(image, (80, 80), 25, color=(60, 60, 60), thickness=10)
    # Ícone "destoante": cor bem diferente do fundo.
    cv2.circle(image, (220, 220), 25, color=(0, 0, 255), thickness=10)

    result = solver.solve_icon_outlier_classical(image)

    assert result is not None
    x, y = result
    # Deve escolher o ícone destoante (perto de (220,220)), não o camuflado.
    assert abs(x - 220) < 40
    assert abs(y - 220) < 40


def test_solve_icon_outlier_classical_ignores_frame_border_artifact():
    # Achado ao vivo (2026-11-02): sem o filtro de bbox em detect_stroke_regions,
    # a borda/moldura do desafio (alto contraste, bbox gigante) "vencia" como
    # resposta em praticamente todo desafio real, mascarando os ícones de
    # verdade — reprodução mínima com uma borda branca sobre fundo escuro.
    image = _flat_background(color=(30, 30, 30))
    cv2.rectangle(image, (2, 2), (297, 297), color=(255, 255, 255), thickness=3)
    cv2.circle(image, (150, 150), 20, color=(0, 0, 255), thickness=8)

    result = solver.solve_icon_outlier_classical(image)

    assert result is not None
    x, y = result
    # Deve escolher o ícone de verdade (centro da imagem), não um ponto na borda.
    assert abs(x - 150) < 40
    assert abs(y - 150) < 40


def test_solve_icon_outlier_classical_returns_none_without_regions():
    image = _flat_background()
    assert solver.solve_icon_outlier_classical(image) is None


def test_solve_closed_loop_classical_returns_only_closed_curves():
    image = _flat_background()
    cv2.circle(image, (80, 80), 40, color=(0, 200, 255), thickness=12)  # fechado
    cv2.ellipse(image, (220, 220), (40, 40), 0, 0, 270, color=(255, 100, 0), thickness=12)  # aberto

    points = solver.solve_closed_loop_classical(image)

    assert len(points) >= 1
    # O ponto retornado deve estar perto do círculo fechado, não do arco aberto.
    closest_to_circle = min(points, key=lambda p: abs(p[0] - 80) + abs(p[1] - 80))
    assert abs(closest_to_circle[0] - 80) < 40
    assert abs(closest_to_circle[1] - 80) < 40


def test_solve_closed_loop_classical_empty_when_nothing_closed():
    image = _flat_background()
    cv2.ellipse(image, (150, 150), (40, 40), 0, 0, 270, color=(255, 100, 0), thickness=12)

    points = solver.solve_closed_loop_classical(image)

    assert points == []


def test_parse_target_digits_extracts_individual_characters():
    assert solver._parse_target_digits("Clique no número que pertence a este grupo: 789") == {
        "7",
        "8",
        "9",
    }


def test_parse_target_digits_handles_no_digits():
    assert solver._parse_target_digits("Clique no ícone que é diferente") == set()


def test_solve_digit_membership_classical_returns_none_without_target_digits():
    image = _flat_background()
    result = solver.solve_digit_membership_classical(image, "Clique no ícone que é diferente")
    assert result is None


@requires_tesseract
def test_solve_digit_membership_classical_finds_matching_digit():
    image = _flat_background(color=(255, 255, 255))
    cv2.putText(image, "8", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), thickness=6)
    cv2.putText(image, "2", (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), thickness=6)

    result = solver.solve_digit_membership_classical(
        image, "Clique no número que pertence a este grupo: 789"
    )

    assert result is not None
    x, _y = result
    # "8" está à esquerda (~x=40-100), "2" à direita (~x=200-260) -> resposta deve ficar à esquerda.
    assert x < 150
