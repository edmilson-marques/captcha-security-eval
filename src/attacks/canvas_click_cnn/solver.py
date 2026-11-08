"""Solver de "laços fechados" do hCaptcha via classificador CNN treinado (Nov/2026).

Alternativa a `attacks.classical_canvas_solver.solve_closed_loop_classical`
(heurística de fração de enclausuramento via flood-fill): aqui, cada região
candidata (mesma detecção por borda, `attacks.classical_cv.detect_stroke_regions`)
é classificada por uma CNN pequena treinada sobre um dataset real rotulado
manualmente — ver `build_dataset.py` sobre a origem e a limitação séria do
rótulo (20 exemplos, rotulagem assistida por IA, nunca validada contra uma
tentativa real ao vivo antes desta técnica existir).

Mesma assinatura/semântica de `solve_closed_loop_classical` (recebe a
imagem do desafio, devolve a lista de pontos classificados como fechados,
vazia se nenhum) — intercambiável como `closed_loop_solver` em
`experiments.live.hcaptcha_classical_cv_attempt`.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from attacks.canvas_click_cnn.dataset_prep import _closed_loop_mask
from attacks.canvas_click_cnn.train import CHECKPOINT_PATH
from attacks.classical_canvas_solver import _crop_black_header
from attacks.classical_cv import detect_stroke_regions
from attacks.image_grid_cnn.infer import predict
from attacks.image_grid_cnn.model import build_model

_MODEL_CACHE: dict[str, tuple[torch.nn.Module, list[str]]] = {}


def _load_model(checkpoint_path: Path = CHECKPOINT_PATH) -> tuple[torch.nn.Module, list[str]]:
    """Carrega (e cacheia em memória do processo) o checkpoint treinado por `train.py`."""
    key = str(checkpoint_path)
    if key not in _MODEL_CACHE:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"{checkpoint_path} não existe -- rode "
                "`python -m attacks.canvas_click_cnn.train` antes de usar este solver."
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = build_model(num_classes=len(checkpoint["classes"]))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        _MODEL_CACHE[key] = (model, checkpoint["classes"])
    return _MODEL_CACHE[key]


def solve_closed_loop_cnn(
    image: np.ndarray, checkpoint_path: Path = CHECKPOINT_PATH
) -> list[tuple[int, int]]:
    """Classifica cada região candidata (fechada/aberta) e retorna o centro das fechadas.

    `image` é a mesma captura nativa do canvas (BGR, bruta, com a eventual
    faixa preta do cabeçalho) recebida pelos outros solvers clássicos de
    clique-sobre-canvas -- mesma assinatura de `solve_closed_loop_classical`.
    Recorta o cabeçalho internamente (`_crop_black_header`) antes de
    detectar regiões, pela mesma razão documentada em
    `classical_canvas_solver.py`: sem o recorte, a faixa confunde a
    detecção de borda. Esse recorte precisa ser idêntico ao usado em
    `dataset_prep.extract_candidates` (treino) -- sem ele aqui, os
    candidatos vistos em inferência vêm de um espaço de coordenada
    diferente do usado no treino. Mesmo motivo pra usar `_closed_loop_mask`
    (Canny sobre o canal de saturação, não luminância -- ver sua docstring,
    achado ao vivo em 2026-11-04) em vez da máscara default: tem que ser a
    mesma segmentação usada em `dataset_prep.extract_candidates` ao montar
    o dataset de treino, senão os recortes vistos aqui vêm de uma
    distribuição diferente da aprendida pelo modelo.
    """
    model, classes = _load_model(checkpoint_path)
    image, y_offset = _crop_black_header(image)
    regions, _mask = detect_stroke_regions(image, mask=_closed_loop_mask(image))

    closed_points: list[tuple[int, int]] = []
    for region in regions:
        x, y, w, h = region["bbox"]
        crop = image[y : y + h, x : x + w]
        if crop.size == 0:
            continue
        # Achado ao vivo em 2026-11-04 (bug pré-existente, independente da
        # segmentação): `crop` vem de `cv2.imread`/`imdecode` (BGR), mas o
        # dataset de treino passa por PNG + PIL (`ImageGridDataset`, RGB) --
        # sem esta conversão, o modelo via em produção os canais R/B
        # trocados em relação ao que aprendeu no treino, o que sozinho já
        # bastava pra destruir qualquer sinal de cor aprendido numa CNN tão
        # pequena (42 exemplos).
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        class_idx, _confidence = predict(model, crop_rgb)
        if classes[class_idx] == "closed":
            cx, cy = region["centroid"]
            closed_points.append((round(cx), round(cy) + y_offset))
    return closed_points
