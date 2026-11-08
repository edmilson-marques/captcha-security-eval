"""Solver de "Clique nos elementos que são idênticos" (hCaptcha, achado de 2026-11-04).

8º tipo de desafio do hCaptcha, nunca atacado antes -- descoberto num lote
real de 100 tentativas com `hcaptcha_combined_attempt`: tão comum quanto
"número que pertence a este grupo" (31% vs. 41% da amostra), com zero
solver até agora. Estruturalmente é o inverso de "ícone diferente"
(`solve_icon_outlier_classical`): em vez de achar o único candidato que
destoa dos outros, aqui o alvo é o **grupo que se repete** -- um sinal de
maioria, não de raridade, que é mais fácil de detectar com confiança do que
"o candidato mais raro" (ver a limitação equivalente, nunca resolvida, em
`attacks.temporal_canvas_solver`).

Mesma convenção de `attacks.classical_canvas_solver`: recebe a captura
nativa do canvas (BGR, com a eventual faixa preta do cabeçalho), devolve a
lista de pontos a clicar (um por elemento do grupo idêntico encontrado).
"""

from __future__ import annotations

import cv2
import numpy as np

from attacks.classical_canvas_solver import _crop_black_header
from attacks.classical_cv import (
    color_contrast,
    detect_stroke_regions,
)

SHAPE_DISTANCE_THRESHOLD = 0.05
COLOR_CONTRAST_THRESHOLD = 15.0


def _region_contour(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
    """Maior contorno externo de `mask` dentro de `bbox` (`None` se não houver nenhum)."""
    x, y, w, h = bbox
    sub_mask = mask[y : y + h, x : x + w]
    contours, _ = cv2.findContours(sub_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _shape_distance(contour_a: np.ndarray, contour_b: np.ndarray) -> float:
    """Distância de forma via momentos de Hu (`cv2.matchShapes`) -- 0 = idêntica.

    Invariante a rotação/escala/translação, ao contrário da IoU de silhueta
    bbox-relativa usada antes aqui (ver achado abaixo, em
    `solve_identical_elements_classical`).
    """
    return float(cv2.matchShapes(contour_a, contour_b, cv2.CONTOURS_MATCH_I1, 0.0))


def _filled_mean_color(
    image: np.ndarray, contour: np.ndarray, bbox: tuple[int, int, int, int]
) -> tuple[float, float, float]:
    """Cor média dos pixels DENTRO do contorno preenchido (não só na borda detectada).

    Mais representativo que amostrar só os pixels de borda (`region_mean_color`)
    pra formas com várias facetas coloridas -- ver achado abaixo.
    """
    x, y, w, h = bbox
    filled = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)
    pixels = image[y : y + h, x : x + w][filled > 0]
    b, g, r = pixels.mean(axis=0)
    return (float(b), float(g), float(r))


def solve_identical_elements_classical(image: np.ndarray) -> list[tuple[int, int]]:
    """Agrupa os candidatos detectados por forma + cor e retorna o maior grupo repetido.

    Dois candidatos entram no mesmo grupo quando a forma (`_shape_distance`,
    momentos de Hu, abaixo de `SHAPE_DISTANCE_THRESHOLD`) **e** a cor média
    (`_filled_mean_color`) contrastam pouco entre si (`color_contrast`
    abaixo de `COLOR_CONTRAST_THRESHOLD`) -- exige as duas condições porque
    "idêntico" significa a mesma forma na mesma cor, não só a mesma forma
    (que sozinha confundiria, por exemplo, duas variações de cor do mesmo
    ícone, como no achado de camuflagem de "ícone diferente"). Retorna
    lista vazia se nenhum grupo com 2+ membros for encontrado (nada
    confiável pra clicar).

    **Achado ao vivo em 2026-11-04**: a versão anterior (IoU de silhueta
    redimensionada pro bbox, cor média só dos pixels de borda) nunca
    agrupava nada contra o desafio real -- inspecionando uma amostra real
    (5 formas de cristal multicoloridas, rotacionadas/escaladas de forma
    diferente cada uma), a IoU bbox-relativa deu 0.02-0.07 entre o par
    genuinamente idêntico e os distratores, bem abaixo de qualquer limiar
    razoável, porque o desafio **gira e redimensiona** as cópias idênticas
    de propósito (confirmado: duas formas com distância de Hu de 0.0106,
    quase zero, tinham IoU de silhueta de só 0.06 por estarem em ângulos
    diferentes). Trocado por `cv2.matchShapes` (momentos de Hu), invariante
    a rotação/escala/translação -- na mesma amostra, o par de verdade ficou
    em 0.0106 contra 0.12-88 de todos os outros pares, uma margem de mais
    de 10x. A amostragem de cor de borda (Canny) também sofria o mesmo
    problema indiretamente (qual trecho da borda multifaceta é amostrado
    muda com a rotação): trocada por `_filled_mean_color` (preenche o
    contorno e tira a média dos pixels internos), que deu 8.4 de contraste
    pro par de verdade (era 19.1 com a amostra de borda, acima do limiar).
    """
    image, y_offset = _crop_black_header(image)
    regions, mask = detect_stroke_regions(image)
    if len(regions) < 2:
        return []

    contours = []
    colors = []
    for region in regions:
        contour = _region_contour(mask, region["bbox"])
        contours.append(contour)
        colors.append(
            _filled_mean_color(image, contour, region["bbox"]) if contour is not None else None
        )

    clusters: list[list[int]] = []
    for idx in range(len(regions)):
        if contours[idx] is None:
            continue
        placed = False
        for cluster in clusters:
            rep = cluster[0]
            same_shape = _shape_distance(contours[idx], contours[rep]) <= SHAPE_DISTANCE_THRESHOLD
            same_color = color_contrast(colors[idx], colors[rep]) < COLOR_CONTRAST_THRESHOLD
            if same_shape and same_color:
                cluster.append(idx)
                placed = True
                break
        if not placed:
            clusters.append([idx])

    if not clusters:
        return []
    largest = max(clusters, key=len)
    if len(largest) < 2:
        return []

    points = []
    for idx in largest:
        cx, cy = regions[idx]["centroid"]
        points.append((round(cx), round(cy) + y_offset))
    return points
