"""Testes das funções de borda/cor pro clique-sobre-canvas (attacks.classical_cv) — Atividade 2.

Usa imagens sintéticas (círculo fechado vs. arco aberto, ícones com cor
contrastante vs. camuflada) — não depende de amostras reais do hCaptcha nem
de rede. A validação do algoritmo contra desafios reais é manual/ao vivo,
documentada na docstring do módulo.
"""

from __future__ import annotations

import cv2
import numpy as np

from attacks import classical_cv


def _flat_background(size: int = 300, color: tuple[int, int, int] = (120, 120, 120)) -> np.ndarray:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:, :] = color
    return image


def _draw_closed_circle(image: np.ndarray, center: tuple[int, int], radius: int = 40) -> None:
    cv2.circle(image, center, radius, color=(0, 200, 255), thickness=12)


def _draw_open_arc(image: np.ndarray, center: tuple[int, int], radius: int = 40) -> None:
    # 270 graus de 360 -> deixa uma lacuna de 90 graus, claramente aberto.
    cv2.ellipse(image, center, (radius, radius), 0, 0, 270, color=(255, 100, 0), thickness=12)


def test_detect_stroke_regions_finds_both_shapes():
    image = _flat_background()
    _draw_closed_circle(image, (80, 80))
    _draw_open_arc(image, (220, 220))

    regions, mask = classical_cv.detect_stroke_regions(image)

    assert len(regions) >= 2
    assert mask.shape == image.shape[:2]
    assert mask.dtype == np.uint8


def test_detect_stroke_regions_ignores_components_above_max_area_ratio():
    image = _flat_background()
    # Um "traço" artificialmente enorme (quase a imagem inteira) deve ser
    # descartado como se fosse textura de fundo, não um ícone/curva individual.
    cv2.rectangle(image, (0, 0), (299, 299), color=(0, 0, 0), thickness=20)
    _draw_closed_circle(image, (150, 150))

    regions, _mask = classical_cv.detect_stroke_regions(image, max_area_ratio=0.3)

    for region in regions:
        assert region["area"] <= 0.3 * image.shape[0] * image.shape[1]


def test_classify_loop_closure_distinguishes_closed_from_open():
    image = _flat_background()
    _draw_closed_circle(image, (80, 80))
    _draw_open_arc(image, (220, 220))

    regions, mask = classical_cv.detect_stroke_regions(image)
    assert len(regions) >= 2

    # Identifica qual região é o círculo (perto de (80,80)) e qual é o arco (perto de (220,220)).
    closed_region = min(
        regions, key=lambda r: abs(r["centroid"][0] - 80) + abs(r["centroid"][1] - 80)
    )
    open_region = min(
        regions, key=lambda r: abs(r["centroid"][0] - 220) + abs(r["centroid"][1] - 220)
    )

    closed_ratio = classical_cv.classify_loop_closure(mask, closed_region["bbox"])
    open_ratio = classical_cv.classify_loop_closure(mask, open_region["bbox"])

    assert closed_ratio > 0.5
    assert open_ratio < closed_ratio


def test_region_mean_color_uses_mask_when_given():
    image = _flat_background(color=(10, 10, 10))
    # Ícone vermelho vivo, só dentro de um pequeno círculo no centro do bbox.
    cv2.circle(image, (50, 50), 20, color=(0, 0, 255), thickness=-1)
    bbox = (20, 20, 60, 60)  # bbox retangular bem maior que o círculo -> cantos são fundo.

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (50, 50), 20, color=255, thickness=-1)

    color_with_mask = classical_cv.region_mean_color(image, bbox, mask=mask)
    color_without_mask = classical_cv.region_mean_color(image, bbox)

    # Com máscara, a cor média deve ser quase pura vermelho (B,G,R ~ (0,0,255)).
    assert color_with_mask[2] > 200
    assert color_with_mask[0] < 30
    # Sem máscara, os cantos de fundo escuro diluem a média — vermelho bem mais fraco.
    assert color_without_mask[2] < color_with_mask[2]


def test_surrounding_background_color_excludes_bbox_interior():
    image = _flat_background(color=(50, 50, 50))
    cv2.circle(image, (150, 150), 15, color=(0, 255, 0), thickness=-1)
    bbox = (135, 135, 30, 30)

    bg_color = classical_cv.surrounding_background_color(image, bbox, margin=20)

    # O anel ao redor deve refletir o fundo (cinza), não o verde do ícone.
    assert bg_color[1] < 150  # canal G não deve estar dominado pelo verde do ícone


def test_color_contrast_higher_for_more_different_colors():
    background = (120.0, 120.0, 120.0)
    similar = (130.0, 125.0, 118.0)
    very_different = (0.0, 0.0, 255.0)

    low_contrast = classical_cv.color_contrast(background, similar)
    high_contrast = classical_cv.color_contrast(background, very_different)

    assert high_contrast > low_contrast


def test_color_contrast_is_zero_for_identical_colors():
    color = (100.0, 150.0, 200.0)
    assert classical_cv.color_contrast(color, color) == 0.0
