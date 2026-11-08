"""Solver de grid do reCAPTCHA via CNN treinado, por categoria -- Nov/2026.

Alternativa ao solver de visão via LLM (`attacks.llm_vision_solver.
solve_recaptcha_grid_adaptive_vote`): aqui cada célula do grid é recortada
(divisão regular, sem segmentação -- ver `dataset_prep.extract_cells`) e
classificada por uma CNN pequena treinada sobre células reais rotuladas
manualmente (ver `build_dataset.py`/`category_models.py` sobre origem e
limitação do rótulo).

**Escopo explícito, não contorná-lo em silêncio**: a categoria da instrução
é resolvida via `dataset_prep.CATEGORY_KEYWORDS`; `solve_recaptcha_grid_cnn`
só tenta resolver quando existe um checkpoint treinado pra essa categoria em
`data/models/` (ver `category_models.checkpoint_path_for`) -- categoria sem
dado de treino suficiente (pontes, faixas de pedestre, escadas, montanhas,
no momento) retorna `None` (fora de escopo), nunca uma classificação
arriscada fora do que foi medido.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from attacks.image_grid_cnn.category_models import checkpoint_path_for
from attacks.image_grid_cnn.dataset_prep import CATEGORY_KEYWORDS
from attacks.image_grid_cnn.infer import predict_proba
from attacks.image_grid_cnn.model import build_model

_MODEL_CACHE: dict[str, tuple[torch.nn.Module, list[str]]] = {}

# ver a explicação completa e a calibração em `solve_recaptcha_grid_cnn`.
RESCUE_FACTOR = 0.3


def _load_model(checkpoint_path: Path) -> tuple[torch.nn.Module, list[str]]:
    """Carrega (e cacheia em memória do processo) um checkpoint treinado por `category_models.train`."""
    key = str(checkpoint_path)
    if key not in _MODEL_CACHE:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = build_model(num_classes=len(checkpoint["classes"]))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        _MODEL_CACHE[key] = (model, checkpoint["classes"])
    return _MODEL_CACHE[key]


def resolve_checkpoint(instruction: str) -> Path | None:
    """Mapeia a instrução do desafio pra um checkpoint treinado via
    `CATEGORY_KEYWORDS`, ou `None` se a categoria não tiver modelo treinado
    (fora de escopo)."""
    lowered = instruction.lower()
    for keyword, category in CATEGORY_KEYWORDS.items():
        if keyword in lowered:
            checkpoint_path = checkpoint_path_for(category)
            return checkpoint_path if checkpoint_path.exists() else None
    return None


def solve_recaptcha_grid_cnn(
    screenshot_png: bytes,
    instruction: str,
    grid_rows: int,
    grid_cols: int,
    checkpoint_path: Path | None = None,
) -> list[int] | None:
    """Classifica cada célula do grid; retorna os índices do objeto-alvo ou
    `None` fora de escopo.

    Mesma assinatura de entrada que o solver via LLM (screenshot do grid +
    instrução + dimensões), pra ser um substituto direto em
    `experiments.live`. `grid_rows`/`grid_cols` precisam ser iguais (grids
    quadrados, única forma observada ao vivo neste estudo). `checkpoint_path`
    permite forçar um modelo específico; por padrão é resolvido a partir da
    instrução (ver `resolve_checkpoint`).
    """
    resolved_checkpoint = checkpoint_path or resolve_checkpoint(instruction)
    if resolved_checkpoint is None:
        return None

    # cv2.imdecode devolve BGR (convenção do OpenCV); o treino usa imagens RGB
    # (extract_cells/cv2.imwrite fazem essa troca de forma transparente ao
    # salvar em disco) -- sem este cvtColor, todo recorte ia pro modelo com
    # vermelho/azul trocados (achado ao vivo em 2026-11-06, revendo por que a
    # acurácia real ficava tão abaixo da validação offline).
    image = cv2.cvtColor(
        cv2.imdecode(np.frombuffer(screenshot_png, dtype=np.uint8), cv2.IMREAD_COLOR),
        cv2.COLOR_BGR2RGB,
    )
    height, width = image.shape[:2]
    cell_h, cell_w = height // grid_rows, width // grid_cols

    model, classes = _load_model(resolved_checkpoint)
    positive_class = next(c for c in classes if not c.startswith("nao_"))
    positive_idx = classes.index(positive_class)

    positive_probs: dict[int, float] = {}
    for index in range(grid_rows * grid_cols):
        row, col = divmod(index, grid_cols)
        y, x = row * cell_h, col * cell_w
        crop = image[y : y + cell_h, x : x + cell_w]
        positive_probs[index] = predict_proba(model, crop)[positive_idx]

    confident = [prob for prob in positive_probs.values() if prob > 0.5]
    if not confident:
        return []

    # Um desafio real dificilmente tem só uma célula com o objeto-alvo --
    # um corte fixo de 0,5 por célula isolada ignora isso e perde células
    # genuínas que ficaram um pouco abaixo do limiar (achado do usuário,
    # 2026-11-08). Resgata células com confiança perto da média das já
    # confiantes, em vez de decidir cada célula sozinha.
    # RESCUE_FACTOR calibrado contra dado real held-out (2026-11-08,
    # scratchpad/calibrate_rescue_factor.py): 2832 células em 259 grids
    # nunca vistos pelos checkpoints atuais, varrendo 0.3-1.0. 0.3 deu o
    # melhor exact-match (22.4% vs 21.6% do baseline sem resgate) e o
    # melhor F1 (61.9% vs 61.0%) -- 0.5 (chute inicial) na real ficava
    # ABAIXO do baseline (20.8%). Diferenças pequenas (~1pp, perto do
    # ruído com esse n), mas 0.3 é o ponto com mais apoio no dado real.
    rescue_threshold = (sum(confident) / len(confident)) * RESCUE_FACTOR
    return [index for index, prob in positive_probs.items() if prob >= rescue_threshold]
