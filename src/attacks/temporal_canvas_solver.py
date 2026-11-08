"""Solver de "ícone diferente" do hCaptcha via análise temporal (Nov/2026).

**Contexto do achado que motivou este módulo**: observando o navegador aberto
ao vivo, o usuário notou que o desafio "ícone diferente" não é uma imagem
estática -- é uma animação contínua em que os ícones desenhados no
`<canvas>` aparecem, desaparecem (fade) e se deslocam ao longo do tempo.
Confirmado capturando 8 screenshots nativos do mesmo desafio ao longo de
~4 segundos (antes de qualquer clique): em metade das capturas os dois
ícones visíveis eram **idênticos** (a mesma flor decorativa, sem nenhum
"diferente" presente naquele instante), e um terceiro ícone (visualmente
bem distinto) só apareceu em ~1 segundo dentro desse ciclo de 4s.

Isso refina a causa-raiz do 0% documentado em `attacks/classical_cv.py` e
`attacks/classical_canvas_solver.py` (segmentação por borda + julgamento de
contraste de cor): aquelas técnicas processam **um único screenshot**, que
pode capturar um instante em que não há de fato dois ícones diferentes pra
comparar. Nenhuma técnica anterior (LLM de visão, CV clássica, CNN) levou
isso em conta.

**Técnica**: em vez de analisar um frame só, captura uma rajada de `N`
screenshots nativos ao longo de alguns segundos e usa a variação temporal
pra separar ícone de fundo decorativo (que é majoritariamente estático,
mesmo com leve "respiração" de cor):

1. Pilha os `N` frames e calcula o desvio-padrão por pixel ao longo do
   tempo (`_temporal_std_map`) -- regiões onde algum ícone aparece/some/
   desloca têm desvio alto; fundo decorativo estático tem desvio baixo.
2. Binariza esse mapa (limiar relativo ao próprio máximo) e agrupa em
   componentes conexos -- cada componente é um "slot" espacial onde algum
   ícone esteve presente em pelo menos um frame.
3. Para cada slot, em cada frame, decide se há ícone presente **comparando
   o interior do slot com um anel imediatamente ao redor dele no mesmo
   frame** (`_slot_has_content`, mesmo princípio de
   `attacks.classical_cv.surrounding_background_color`/`color_contrast` --
   não uma estatística agregada ao longo do tempo). Contar presença dessa
   forma, por frame e por comparação espacial local, evita um problema real
   encontrado durante o desenvolvimento: qualquer estimativa de "fundo" que
   dependa só de estatística temporal por pixel (média, mediana, mínimo)
   erra quando o ícone comum fica parado numa posição fixa na maioria dos
   frames -- estatisticamente, a cor dele *é* o valor mais comum ali, então
   a estimativa de fundo acaba "aprendendo" a cor do ícone em vez do fundo
   real, invertendo a contagem (ver `tests/test_temporal_canvas_solver.py`
   para os dois casos sintéticos que expuseram isso). Comparação espacial
   local não sofre desse problema porque não depende de quão frequente o
   conteúdo é ao longo do tempo, só de como ele se destaca da vizinhança
   *naquele instante*.
4. O slot com a **menor frequência de presença** é o ícone raro -- a
   hipótese testada aqui é que esse é o ícone-alvo ("o diferente"), já que
   o ambiente decorativo se repete mais que o ícone que o desafio pede pra
   achar.
5. Retorna o centroide do slot raro no frame onde ele teve sinal mais
   forte (melhor estimativa de posição, menos afetada por frames de
   transição/fade parcial).

**Validado offline** contra 1 captura real (8 frames, ver `scratchpad`
desta sessão) e contra dois casos sintéticos adversariais (ícone comum
parado numa posição fixa; ícone raro com cor mais escura que o fundo --
ambos quebravam versões anteriores desta técnica baseadas em estatística
temporal por pixel, ver `tests/test_temporal_canvas_solver.py`). **Ainda
não validado ao vivo contra um clique real** -- as amostras offline só
provam que a lógica acha o candidato certo, não que clicar nele resolve o
desafio.
"""

from __future__ import annotations

import cv2
import numpy as np

from attacks.classical_canvas_solver import _crop_black_header
from attacks.classical_cv import color_contrast, region_mean_color, surrounding_background_color

MIN_SLOT_AREA_RATIO = 0.01
MAX_SLOT_AREA_RATIO = 0.15
CONTRAST_PRESENCE_THRESHOLD = 10.0
RING_MARGIN = 12


def _temporal_std_map(frames: list[np.ndarray]) -> np.ndarray:
    """Desvio-padrão por pixel (média dos 3 canais) ao longo da pilha de frames."""
    stack = np.stack(frames, axis=0).astype(np.float32)
    return stack.std(axis=0).mean(axis=2)


def _find_slots(
    std_map: np.ndarray, threshold_ratio: float = 90 / 255
) -> list[tuple[int, int, int, int]]:
    """Agrupa regiões de alta variância temporal em "slots" candidatos (bboxes).

    Descarta componentes muito pequenos (ruído de textura do fundo) ou
    muito grandes relativos à imagem inteira (`MIN_SLOT_AREA_RATIO`/
    `MAX_SLOT_AREA_RATIO`) -- um componente gigante normalmente é a fusão de
    duas ou mais ocorrências de ícone próximas no tempo/posição (mesma
    limitação de componentes conexos já documentada em
    `attacks.classical_cv.detect_stroke_regions`), não um ícone individual,
    e inflaria a contagem de presença de forma enganosa.
    """
    if std_map.max() <= 0:
        return []
    img_area = std_map.shape[0] * std_map.shape[1]
    std_norm = (255 * std_map / std_map.max()).astype(np.uint8)
    threshold = round(255 * threshold_ratio)
    _, mask = cv2.threshold(std_norm, threshold, 255, cv2.THRESH_BINARY)

    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    slots = []
    for i in range(1, num_labels):
        x, y, w, h, _area = stats[i]
        bbox_ratio = (w * h) / img_area
        if not (MIN_SLOT_AREA_RATIO < bbox_ratio < MAX_SLOT_AREA_RATIO):
            continue
        slots.append((int(x), int(y), int(w), int(h)))
    return slots


def _slot_has_content(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[bool, float]:
    """Decide, só a partir deste frame, se há algo desenhado dentro de `bbox`.

    Compara a cor média do interior de `bbox` com a cor média de um anel
    imediatamente ao redor dele **no mesmo frame** (mesmas funções de
    `attacks.classical_cv` usadas por `solve_icon_outlier_classical`) --
    uma comparação espacial local, não uma estatística ao longo do tempo
    (ver docstring do módulo sobre por que isso importa). Retorna
    `(presente, contraste)`.
    """
    icon_color = region_mean_color(frame, bbox)
    background_color = surrounding_background_color(frame, bbox, margin=RING_MARGIN)
    contrast = color_contrast(icon_color, background_color)
    return contrast > CONTRAST_PRESENCE_THRESHOLD, contrast


def _slot_presence(
    frames: list[np.ndarray], bbox: tuple[int, int, int, int]
) -> tuple[int, int | None, tuple[float, float] | None]:
    """Conta em quantos frames o slot `bbox` tem conteúdo (ver `_slot_has_content`).

    Retorna `(contagem_de_presença, frame_de_pico, centroide_no_frame_de_pico)`,
    onde o "pico" é o frame de maior contraste local -- melhor estimativa de
    posição, menos afetada por frames de transição/fade parcial.
    """
    presence_count = 0
    peak_contrast = -1.0
    peak_frame = None

    for frame_index, frame in enumerate(frames):
        present, contrast = _slot_has_content(frame, bbox)
        if not present:
            continue
        presence_count += 1
        if contrast > peak_contrast:
            peak_contrast = contrast
            peak_frame = frame_index

    if peak_frame is None:
        return presence_count, None, None

    x, y, w, h = bbox
    return presence_count, peak_frame, (x + w / 2, y + h / 2)


def solve_icon_outlier_temporal(frames: list[np.ndarray]) -> tuple[int, int] | None:
    """Acha o ícone raro (candidato a "o diferente") a partir de uma rajada de frames.

    `frames`: lista de capturas nativas do mesmo desafio ao longo do tempo
    (BGR, mesmo formato de `_canvas_native_screenshot`, com a eventual faixa
    preta do cabeçalho ainda presente -- recortada aqui, mesma convenção dos
    outros solvers clássicos). Requer pelo menos 3 frames pra ter algum
    sinal temporal; retorna `None` se não houver frames suficientes ou
    nenhum slot candidato for encontrado.
    """
    if len(frames) < 3:
        return None

    # Acha o recorte a partir só do primeiro frame e aplica a mesma faixa a
    # todos -- achado ao vivo em 2026-11-04 (`testa`, erro real em produção):
    # rodar `_crop_black_header` por frame detecta a transição em linhas
    # ligeiramente diferentes entre frames (ruído de renderização perto do
    # limiar de brilho), gerando recortes de altura diferente e quebrando
    # `np.stack` em `_temporal_std_map` ("all input arrays must have the
    # same shape"). O cabeçalho é DOM estático, não deveria se mover dentro
    # de uma mesma rajada -- um único recorte aplicado a todos é correto e
    # resolve o crash.
    _, y_offset = _crop_black_header(frames[0])
    cropped_frames = [frame[y_offset:] for frame in frames]

    std_map = _temporal_std_map(cropped_frames)
    slots = _find_slots(std_map)
    if not slots:
        return None

    candidates = []
    for bbox in slots:
        presence_count, _peak_frame, peak_centroid = _slot_presence(cropped_frames, bbox)
        if presence_count == 0 or peak_centroid is None:
            continue
        candidates.append((presence_count, peak_centroid))

    if not candidates:
        return None

    _rarest_count, (cx, cy) = min(candidates, key=lambda c: c[0])
    return (round(cx), round(cy) + y_offset)
