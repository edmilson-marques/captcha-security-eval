"""Extração de recortes de candidato a partir de amostras reais salvas (Atividade 2).

Reaproveita a mesma segmentação clássica já usada pelos solvers heurísticos
(`attacks/classical_canvas_solver.py`) para achar os candidatos de cada
desafio de clique-sobre-`canvas` do hCaptcha — não decide qual é a resposta
certa, só isola as regiões candidatas pra rotulagem manual (ground truth) e
posterior treino do classificador em `canvas_click_cnn/model.py`.

Amostras de origem: `data/raw/hcaptcha/<timestamp>/` persistidas por
`harness.storage.save_attempt_capture` (ver `experiments/live.py`). Mesma
limitação de origem dos dados já documentada em
`attacks/image_grid_cnn/dataset.py`: a rotulagem (qual candidato é a
resposta certa) é um passo manual, feito uma vez via inspeção visual direta
das amostras (ver `data/raw/hcaptcha/_labeled/` e a docstring de
`LabeledExample` logo abaixo) — não há como inferir automaticamente, já que
nenhuma tentativa real até agora teve sucesso (nenhum rótulo de verdade
vem do próprio pipeline de experimento).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from attacks.classical_canvas_solver import _crop_black_header
from attacks.classical_cv import detect_stroke_regions
from attacks.ocr import locate_digit_candidates

RAW_HCAPTCHA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "hcaptcha"


def _closed_loop_mask(
    image: np.ndarray, sat_low: int = 40, sat_high: int = 120, kernel_size: int = 9
) -> np.ndarray:
    """Máscara de borda sobre o canal de saturação (HSV), não luminância.

    **Achado ao vivo em 2026-11-04** (usuário observou "laços fechados"
    marcando errado de novo, desta vez via CNN): `detect_stroke_regions`
    default (`_stroke_edge_mask`, Canny sobre luminância) funde a maior
    parte do desafio numa região só sempre que o fundo decorativo tem
    textura de alto contraste (padrões de círculos/quadrados concêntricos,
    comuns nestas amostras) -- confirmado em 4 amostras reais de hoje,
    incluindo um caso de **zero candidatos** (a imagem inteira virou 1 blob
    só, descartado pelo filtro de área máxima de `detect_stroke_regions`).

    Isso NÃO é a mesma ideia já documentada e descartada em
    `classical_cv.py` ("um limiar global de saturação não funciona nesse
    fundo adversarial") -- aquela usava o VALOR absoluto de saturação por
    pixel (um traço pastel tem saturação baixa igual o fundo, falso
    negativo). Aqui é detecção de BORDA (Canny) sobre o canal de saturação
    em vez do canal de luminância -- o que importa é a *transição* de
    saturação entre traço e fundo, não o valor absoluto de nenhum dos dois,
    então funciona mesmo quando o fundo tem saturação alta em algumas
    amostras (confirmado: um pixel de fundo com saturação 88 e um pixel de
    traço com saturação 22 na mesma amostra real -- um limiar absoluto não
    distinguiria os dois, mas a borda da transição entre eles continua
    detectável). Testado contra as mesmas 4 amostras reais: elimina a falha
    catastrófica de 0 candidatos em todas elas (viram 5-11 candidatos), sem
    piorar nenhuma pra pior que antes -- não é segmentação perfeita (ainda
    funde ou perde traço isolado ocasionalmente), mas é uma melhora líquida
    reproduzível, diferente do ajuste de limiar da investigação anterior
    (ver README, seção "Laços fechados") que trocava uma amostra por outra
    sem ganho líquido.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    blurred = cv2.GaussianBlur(saturation, (3, 3), 0)
    edges = cv2.Canny(blurred, sat_low, sat_high)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)


@dataclass
class Candidate:
    """Um recorte de região candidata, com posição relativa à imagem já sem cabeçalho."""

    sample_name: str
    index: int
    bbox: tuple[int, int, int, int]
    crop: np.ndarray
    ocr_char_guess: str | None = None


def extract_candidates(sample_dir: Path) -> tuple[str | None, list[Candidate]]:
    """Extrai os recortes de candidato de uma amostra salva por `save_attempt_capture`.

    Retorna `(criterion_type, candidatos)`. `criterion_type` vem direto da
    `metadata.json` da amostra (`icon_outlier`/`closed_loop`/`digit_membership`);
    `None` se a amostra não for desse tipo (ex. grade, arrastar-e-encaixar —
    `extract_candidates` retorna lista vazia nesses casos, nada a extrair).
    """
    metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
    criterion_type = metadata.get("extra", {}).get("criterion_type")
    if criterion_type not in ("icon_outlier", "closed_loop", "digit_membership"):
        return criterion_type, []

    image = cv2.imread(str(sample_dir / "challenge.png"))
    image, _y_offset = _crop_black_header(image)

    candidates: list[Candidate] = []
    if criterion_type in ("icon_outlier", "closed_loop"):
        mask = _closed_loop_mask(image) if criterion_type == "closed_loop" else None
        regions, _mask = detect_stroke_regions(image, mask=mask)
        for i, region in enumerate(regions):
            x, y, w, h = region["bbox"]
            candidates.append(
                Candidate(
                    sample_name=sample_dir.name,
                    index=i,
                    bbox=region["bbox"],
                    crop=image[y : y + h, x : x + w].copy(),
                )
            )
    else:  # digit_membership
        for i, digit in enumerate(locate_digit_candidates(image)):
            x, y, w, h = digit["bbox"]
            candidates.append(
                Candidate(
                    sample_name=sample_dir.name,
                    index=i,
                    bbox=digit["bbox"],
                    crop=image[y : y + h, x : x + w].copy(),
                    ocr_char_guess=digit["char"],
                )
            )

    return criterion_type, candidates


def extract_all(raw_dir: Path = RAW_HCAPTCHA_DIR) -> dict[str, list[Candidate]]:
    """Roda `extract_candidates` sobre toda amostra salva em `raw_dir`.

    Retorna um dict `{criterion_type: [candidatos de todas as amostras desse tipo]}`,
    pulando amostras sem `metadata.json` (ex. formato antigo da Atividade 1) ou
    sem desafio de clique-sobre-canvas aplicável.
    """
    grouped: dict[str, list[Candidate]] = {}
    for sample_dir in sorted(raw_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
        metadata_path = sample_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        criterion_type, candidates = extract_candidates(sample_dir)
        if criterion_type is None or not candidates:
            continue
        grouped.setdefault(criterion_type, []).extend(candidates)
    return grouped
