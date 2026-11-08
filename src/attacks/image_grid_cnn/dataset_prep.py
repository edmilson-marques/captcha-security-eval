"""Extração de células de grid a partir de amostras reais salvas (Atividade 2).

Diferente do recorte de candidato em `attacks/canvas_click_cnn/dataset_prep.py`
(que precisa de segmentação por borda, já que os candidatos têm posição
irregular), uma célula de grid do reCAPTCHA/hCaptcha já vem em posição
regular (`grid_size` x `grid_size`, lado a lado) -- bastaria dividir a
imagem em partes iguais. Nada aqui decide qual célula é a resposta certa,
só isola os recortes pra rotulagem manual (ver
`image_grid_cnn/build_dataset.py`).

Amostras de origem: `data/raw/recaptcha/<timestamp>/` persistidas por
`harness.storage.save_attempt_capture` (ver
`experiments.live.recaptcha_baseline_attempt`, que captura o grid real sem
nenhum custo de API -- a técnica em si não tenta resolver nada).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

RAW_RECAPTCHA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "recaptcha"

LABELS_CSV_DIR = Path(__file__).resolve().parents[3] / "data" / "labels" / "recaptcha_grid_labels"

# palavra-chave (minúscula, como aparece na instrução real) -> categoria/classe
# positiva usada em `category_models.py` e `solver.py`. Ordem não importa --
# checada por substring contra a instrução em minúsculas.
CATEGORY_KEYWORDS: dict[str, str] = {
    "carro": "carros",
    "ônibus": "onibus",
    "semáforo": "semaforos",
    "motocicleta": "motocicletas",
    "bicicleta": "bicicletas",
    "ponte": "pontes",
    "faixa": "faixas_pedestre",
    "escada": "escadas",
    "montanha": "montanhas",
}


def categorize_raw_samples(raw_dir: Path = RAW_RECAPTCHA_DIR) -> dict[str, list[str]]:
    """Agrupa as amostras em `raw_dir` por categoria, a partir da instrução salva
    em cada `metadata.json["extra"]["instruction"]` (ver `CATEGORY_KEYWORDS`).

    Amostras sem `grid_size`/instrução registrados (formato antigo, ou outro
    tipo de desafio) ou cuja instrução não bate com nenhuma palavra-chave
    conhecida ficam de fora do resultado.
    """
    groups: dict[str, list[str]] = {}
    for sample_dir in sorted(raw_dir.iterdir()):
        metadata_path = sample_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        instruction = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            .get("extra", {})
            .get("instruction")
        )
        if not instruction:
            continue
        lowered = instruction.lower()
        for keyword, category in CATEGORY_KEYWORDS.items():
            if keyword in lowered:
                groups.setdefault(category, []).append(sample_dir.name)
                break
    return groups


def load_labels_csv(category: str, csv_dir: Path = LABELS_CSV_DIR) -> dict[tuple[str, int], str]:
    """Carrega `csv_dir/<category>.csv` (colunas `sample,index,label`) como o
    dict `(amostra, índice) -> classe` usado por `build()`.

    Rótulos viraram CSV (em vez de dict literal no código) a partir de
    ~10 mil células rotuladas (Nov/2026, lote de coleta de madrugada) --
    um dict hardcoded desse tamanho vira um arquivo .py de dezenas de
    milhares de linhas, pior em todo aspecto que um arquivo de dados.
    """
    path = csv_dir / f"{category}.csv"
    with path.open(newline="", encoding="utf-8") as f:
        return {(row["sample"], int(row["index"])): row["label"] for row in csv.DictReader(f)}


def extract_cells(sample_dir: Path) -> tuple[str | None, list[tuple[int, np.ndarray]]]:
    """Divide o grid salvo em `sample_dir` em células `(índice, recorte)`.

    `índice` segue a mesma convenção 0-based, esquerda->direita/cima->baixo
    usada pelos solvers reais (`td.rc-imageselect-tile[id='<índice>']`).
    Retorna `(instrução, [])` se a amostra não tiver `grid_size` registrado
    em `metadata.json["extra"]` (ex. formato antigo, ou amostra de outro
    tipo de desafio) -- ou se o grid inteiro for uma captura em branco
    (`is_blank_capture`), achado em capturas tomadas com um texto-d'água do
    reCAPTCHA sobreposto em vez do desafio real.
    """
    metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
    extra = metadata.get("extra", {})
    grid_size = extra.get("grid_size")
    instruction = extra.get("instruction")
    if not grid_size:
        return instruction, []

    image = cv2.imread(str(sample_dir / "challenge.png"))
    if is_blank_capture(image):
        return instruction, []
    height, width = image.shape[:2]
    cell_h, cell_w = height // grid_size, width // grid_size

    cells: list[tuple[int, np.ndarray]] = []
    for index in range(grid_size * grid_size):
        row, col = divmod(index, grid_size)
        y, x = row * cell_h, col * cell_w
        cells.append((index, image[y : y + cell_h, x : x + cell_w].copy()))
    return instruction, cells


def is_blank_capture(
    image: np.ndarray, std_threshold: float = 10.0, mean_threshold: float = 240.0
) -> bool:
    """`True` se `image` (grid inteiro ou célula) for quase uniforme e muito
    clara -- fundo de formulário/placeholder em vez de foto real, achado em
    algumas capturas de 2026-11-05/06 onde o Google sobrepôs um texto-d'água
    translúcido ("Não sou um robô"/reCAPTCHA) por cima do desafio. Calibrado
    com exemplos reais: capturas em branco confirmadas tinham desvio-padrão
    0,95-3,57 (grid inteiro) ou 0-6 (célula), bem abaixo de qualquer foto de
    rua real (desvio 30+).

    shortcut: só pega o caso extremo (quase branco uniforme). Um texto-d'água
    translúcido sobre uma foto que ainda mantém contraste visível (ex. amostra
    `20261105T173200765839`, onde o topo do grid tem a marca-d'água mas
    continua com desvio-padrão 30-40, igual a uma foto limpa) não é pego por
    este limiar -- ainda precisa de revisão visual se aparecer de novo.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return bool(gray.std() < std_threshold and gray.mean() > mean_threshold)


def has_checkmark_overlay(cell: np.ndarray, blue_fraction_threshold: float = 0.015) -> bool:
    """`True` se `cell` tiver uma fração relevante de pixel azul saturado --
    sinal POSSÍVEL do selo de "marcado" deixado por um clique de uma
    tentativa de resolução anterior (contaminação real, encontrada em
    capturas feitas DEPOIS de clicar via `save_attempt_capture`, nunca em
    capturas pré-clique).

    **Não usar como filtro automático** (testado em 2026-11-06 contra ~10 mil
    células reais: ~5-10% marcadas, muito acima da meia dúzia de contaminações
    reais já confirmadas na sessão -- falso-positivo claro em parede/carro/
    placa azul saturado comuns em foto de rua, ex. amostra
    `20261105T101526036171` célula 7). Serve só pra PRÉ-TRIAGEM: reduz a lista
    de candidatos que um revisor (humano ou fork) confere visualmente antes de
    descartar, nunca decide sozinho.
    """
    hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (95, 120, 120), (125, 255, 255))
    return bool((mask.mean() / 255) > blue_fraction_threshold)


def plan_labeling_chunks(sample_ids: list[str], max_chunk_size: int = 60) -> list[list[str]]:
    """Divide `sample_ids` em lotes de no máximo `max_chunk_size`, prontos
    pra despachar um fork de rotulagem por lote.

    Achado ao vivo em 2026-11-06: um único fork tentando rotular uma lista
    grande (260 grids) esgotou o orçamento de contexto na metade (37/260,
    14%); dividir o restante em 4 lotes de ~56 e rodar em paralelo completou
    os 4 por inteiro. `max_chunk_size=60` é o tamanho que funcionou nesse
    caso -- não validado acima disso.
    """
    return [sample_ids[i : i + max_chunk_size] for i in range(0, len(sample_ids), max_chunk_size)]


def validate_and_merge_labels(
    existing_labels: dict[tuple[str, int], str],
    new_labels: dict[tuple[str, int], str],
    raw_dir: Path = RAW_RECAPTCHA_DIR,
) -> tuple[dict[tuple[str, int], str], list[tuple[str, int, str]]]:
    """Mescla `new_labels` em `existing_labels`, descartando (sample, index)
    cuja amostra não existe em `raw_dir` ou cujo índice está fora do grid
    real de `extract_cells` -- validação obrigatória pós-rotulagem (achado ao
    vivo em 2026-11-06: rotulagem em paralelo por sub-agentes produziu 23
    entradas assim, por transcrição/alucinação de tamanho de grid).

    Retorna `(mesclado, descartados)`, onde `descartados` é
    `[(sample, index, label), ...]` pra quem chamar poder reportar o que foi
    perdido em vez de descartar em silêncio.
    """
    merged = dict(existing_labels)
    dropped: list[tuple[str, int, str]] = []
    valid_indices_cache: dict[str, set[int] | None] = {}

    for (sample, index), label in new_labels.items():
        if sample not in valid_indices_cache:
            sample_dir = raw_dir / sample
            if not (sample_dir / "metadata.json").exists():
                valid_indices_cache[sample] = None
            else:
                _instruction, cells = extract_cells(sample_dir)
                valid_indices_cache[sample] = {i for i, _ in cells}

        valid_indices = valid_indices_cache[sample]
        if valid_indices is None or index not in valid_indices:
            dropped.append((sample, index, label))
            continue
        merged[(sample, index)] = label

    return merged, dropped
