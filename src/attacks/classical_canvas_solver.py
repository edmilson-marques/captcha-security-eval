"""Solvers de visão computacional clássica pros desafios de clique-sobre-canvas
do hCaptcha que o solver de visão via LLM não resolveu (0/51 tentativas reais
na sessão de 2026-11-01/02, mesmo depois de self-consistency, captura em
resolução nativa, prompt estruturado por tipo, prompt nomeando o critério
explicitamente e um modelo substancialmente mais forte — ver docstring de
`experiments.live.hcaptcha_llm_vision_attempt` para o histórico completo).

Pedido explícito do usuário: parar de iterar em variações de prompt do mesmo
modelo e tentar uma técnica estruturalmente diferente — segmentação por
borda (OpenCV, `attacks/classical_cv.py`) e OCR (Tesseract,
`attacks/ocr.py`) em vez de pedir um julgamento qualitativo a um modelo de
visão. Compatível com o escopo original do projeto: a Atividade 2 do
cronograma já previa "CV clássica" como uma das técnicas de ataque a
comparar, ao lado do solver via LLM.

Nenhuma das três funções aqui foi validada contra um volume grande de
amostras reais antes de medir ao vivo — a prototipagem manual (ver notas em
`classical_cv.py`) mostrou que um limiar de cor simples não funciona contra
o fundo adversarial do hCaptcha, mas confirmou sinal razoável com
detecção por borda. O resultado real (taxa de sucesso ao vivo) é o que
decide se a técnica vale a pena, não a prototipagem offline — mesma prática
já usada no resto do projeto.
"""

from __future__ import annotations

import re

import cv2
import numpy as np

from attacks.classical_cv import (
    classify_loop_closure,
    color_contrast,
    detect_stroke_regions,
    region_mean_color,
    surrounding_background_color,
)
from attacks.ocr import locate_digit_candidates


def _crop_black_header(
    image: np.ndarray, black_threshold: float = 15.0, max_scan_rows: int = 400
) -> tuple[np.ndarray, int]:
    """Recorta uma faixa preta sólida no topo da imagem, se houver uma.

    **Achado ao vivo em 2026-11-02**: a captura nativa do canvas
    (`experiments.live._canvas_native_screenshot`) inclui uma faixa preta
    sólida no topo correspondente à área do cabeçalho do desafio (não
    desenhada no canvas — é DOM separado). Essa faixa, sem recorte, criava
    uma borda reta de altíssimo contraste que `detect_stroke_regions`
    confundia com um ícone/traço de verdade — descoberto porque dois
    desafios completamente diferentes retornavam exatamente a mesma
    coordenada "detectada" (artefato, não coincidência).

    Detecta a transição automaticamente (em vez de usar uma altura de
    cabeçalho fixa em pixels) escaneando o brilho médio linha a linha —
    mais robusto a pequenas variações de layout: a prototipagem manual
    mostrou que a transição real ficou um pouco diferente da altura de
    cabeçalho documentada no DOM (`challenge-header`, 110px CSS). Se nenhuma
    transição for encontrada dentro de `max_scan_rows`, não recorta nada
    (`y_offset=0`) — mais seguro do que arriscar cortar conteúdo de verdade
    num layout sem essa faixa preta.

    Retorna `(imagem_recortada, y_offset)` — `y_offset` precisa ser somado
    de volta às coordenadas Y detectadas na imagem recortada antes de
    clicar, já que elas passam a ser relativas ao recorte, não à imagem
    original.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    row_means = gray.mean(axis=1)
    limit = min(max_scan_rows, len(row_means))
    for y in range(limit):
        if row_means[y] > black_threshold:
            return image[y:], y
    return image, 0


def solve_icon_outlier_classical(image: np.ndarray) -> tuple[int, int] | None:
    """Escolhe o ícone cuja cor mais contrasta com o fundo imediato ao redor dele.

    Implementa a hipótese de "quebra de camuflagem" levantada ao inspecionar
    amostras reais (ver `experiments.live.hcaptcha_llm_vision_attempt`):
    entre os candidatos detectados por `detect_stroke_regions`, calcula a
    distância de cor (`color_contrast`) entre a cor média de cada um e a cor
    média do fundo que o envolve, e escolhe o de maior contraste. Retorna
    `None` se nenhuma região for detectada.
    """
    image, y_offset = _crop_black_header(image)
    regions, mask = detect_stroke_regions(image)
    if not regions:
        return None

    best_region = None
    best_contrast = -1.0
    for region in regions:
        icon_color = region_mean_color(image, region["bbox"], mask=mask)
        background_color = surrounding_background_color(image, region["bbox"])
        contrast = color_contrast(icon_color, background_color)
        if contrast > best_contrast:
            best_contrast = contrast
            best_region = region

    if best_region is None:
        return None
    cx, cy = best_region["centroid"]
    return (round(cx), round(cy) + y_offset)


def solve_closed_loop_classical(
    image: np.ndarray, closure_threshold: float = 0.5
) -> list[tuple[int, int]]:
    """Retorna o centro de cada traço classificado como laço fechado.

    Usa `detect_stroke_regions` pra achar candidatos e
    `classify_loop_closure` pra decidir, por candidato, se ele cerca uma
    área interna (fração >= `closure_threshold`) ou não. Lista vazia se
    nenhum traço for classificado como fechado.
    """
    image, y_offset = _crop_black_header(image)
    regions, mask = detect_stroke_regions(image)
    closed_points = []
    for region in regions:
        closure_fraction = classify_loop_closure(mask, region["bbox"])
        if closure_fraction >= closure_threshold:
            cx, cy = region["centroid"]
            closed_points.append((round(cx), round(cy) + y_offset))
    return closed_points


def _parse_target_digits(instruction: str) -> set[str]:
    """Extrai o conjunto de dígitos-alvo do texto da instrução (ex. "789" -> {"7","8","9"})."""
    digits_found = re.findall(r"\d", instruction)
    return set(digits_found)


def solve_digit_membership_classical(image: np.ndarray, instruction: str) -> tuple[int, int] | None:
    """Acha, via OCR, um número visível que pertença ao grupo citado na instrução.

    `instruction` é o texto bruto do desafio (ex. "Clique no número que
    pertence a este grupo: 789") — `_parse_target_digits` extrai o conjunto
    de dígitos-alvo dele. Usa `locate_digit_candidates` (OCR) pra ler os
    números visíveis na imagem e retorna o centro do primeiro cujo caractere
    reconhecido esteja no grupo-alvo (maior confiança primeiro). `None` se
    nenhum dígito do grupo for encontrado.

    **Achado ao vivo em 2026-11-02, corrige uma suposição anterior**: antes
    deste achado, este desafio parecia "o mais fácil" dos três tipos canvas
    (leitura de dígito costuma ser ponto forte tanto de OCR quanto de modelo
    de visão). Inspecionando amostras reais capturadas direto do canvas
    (sem custo de API), os dígitos aparecem **deliberadamente ocultos atrás
    de um padrão denso de listras verticais** sobrepostas — a mesma técnica
    clássica de ofuscação usada em CAPTCHAs de texto distorcido desde os
    primeiros desenhos (von Ahn et al.), não um texto limpo e legível. Isso
    explica por que tanto o solver via LLM quanto este OCR clássico tiveram
    dificuldade aqui: não é um teto de capacidade de leitura de dígito em
    geral, é um desafio especificamente projetado pra derrotar reconhecimento
    de caractere. Tesseract chegou a reconhecer um dígito correto numa
    amostra (confiança 28, baixa mas válida); na outra, não reconheceu nada.
    """
    target_digits = _parse_target_digits(instruction)
    if not target_digits:
        return None

    image, y_offset = _crop_black_header(image)
    candidates = locate_digit_candidates(image)
    matching = [c for c in candidates if c["char"] in target_digits]
    if not matching:
        return None

    best = max(matching, key=lambda c: c["confidence"])
    cx, cy = best["centroid"]
    return (round(cx), round(cy) + y_offset)
