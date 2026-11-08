"""Reforço com dataset de terceiros pras categorias fracas (Nov/2026).

Depois do segundo lote de coleta ao vivo, "onibus"/"escadas"/"faixas_pedestre"
continuaram abaixo do chute trivial mesmo com todo o dado real já coletado e
rotulado (sem backlog -- ver `category_models.py`). A alternativa a coletar
mais ao vivo (lento, martela a demo do Google) é usar um dataset de terceiros
já composto de imagens de tile do reCAPTCHA v2, não fotos genéricas: Danny
Luna, "recaptcha-57k-images-dataset" (Hugging Face, licença MIT,
https://huggingface.co/datasets/DannyLuna/recaptcha-57k-images-dataset).
Mesmo domínio visual do que coletamos (tile real do reCAPTCHA), não é o
mesmo risco de mismatch de um dataset genérico tipo COCO/Open Images.

**Origem e proporção, pra não esconder**: isso é dado de um terceiro, não
coletado pelo autor do TCC -- precisa ser citado como fonte externa no
relatório. Pras categorias usadas aqui, o volume externo costuma superar
o real coletado (ex. "Car": ~10 mil imagens externas) -- parte do dado de
treino passa a vir do terceiro, não da coleta ao vivo própria. Decisão
explícita do usuário, registrada aqui pra não virar um "detalhe perdido"
na metodologia.

**Resultado honesto nas 3 categorias originais (onibus/escadas/
faixas_pedestre, medido em 2026-11-06 após corrigir o vazamento de
validação -- ver `category_models.cross_validate_category`)**: o reforço
externo NÃO trouxe generalização real nenhuma -- a CV corrigida de todas
as 3 ficou praticamente idêntica ao número medido antes de qualquer dado
externo entrar (78%, 76%, 76%, respectivamente). O único efeito
mensurável do dataset externo foi inflar artificialmente a CV quando a
validação ainda vazava pra imagem externa. Mesmo assim, estendido às
demais categorias (carros/semaforos/motocicletas/bicicletas) em
2026-11-07 a pedido do usuário, agora já com a correção de validação
embutida desde o início -- ver o resultado real de cada uma antes de
assumir que ajudou.

Classes extraídas do dataset completo (`Bus`, `Stair`, `Crosswalk`,
`Other`, `Bicycle`, `Car`, `Motorcycle`, `Traffic Light`) e copiadas pra
`data/external/recaptcha_57k/<classe>/` (os splits train/val originais do
Hugging Face foram mesclados num único pool por classe -- a separação
treino/validação deste projeto é feita por `train.cross_validate`, não
pela do dataset de origem).
"""

from __future__ import annotations

import shutil
from pathlib import Path

EXTERNAL_DIR = Path(__file__).resolve().parents[3] / "data" / "external" / "recaptcha_57k"

# categoria do projeto -> classe correspondente no dataset externo
CATEGORY_TO_EXTERNAL_CLASS: dict[str, str] = {
    "onibus": "Bus",
    "escadas": "Stair",
    "faixas_pedestre": "Crosswalk",
    "carros": "Car",
    "semaforos": "Traffic Light",
    "motocicletas": "Motorcycle",
    "bicicletas": "Bicycle",
}

# classe "Other" do dataset externo serve de negativo extra pras 3 categorias
EXTERNAL_NEGATIVE_CLASS = "Other"

EXTERNAL_FILENAME_PREFIX = "ext57k_"


def is_external_grid_id(grid_id: str) -> bool:
    """`True` se `grid_id` (ver `ImageGridDataset.grid_ids`) vier deste dataset
    externo, não de uma coleta real -- usado como `validation_eligible` em
    `train.cross_validate` pra nunca deixar a validação cair em dado de
    terceiro (achado ao vivo em 2026-11-06: sem essa restrição, a validação
    das categorias aqui media principalmente "acerta o dataset externo")."""
    return grid_id.startswith(EXTERNAL_FILENAME_PREFIX)


def add_external_images(category: str, output_dir: Path) -> dict[str, int]:
    """Copia as imagens externas da classe de `category` (positivo) e de
    `Other` (negativo) pra dentro de `output_dir/<classe>/`, já no layout
    `ImageFolder` usado por `category_models.build`.

    Prefixo `ext57k_` no nome do arquivo evita colisão com células reais
    (`<amostra>__<índice>.png`) e também faz cada imagem externa contar
    como seu próprio "grid" isolado no split por grid de
    `ImageGridDataset.grid_ids` (correto -- são fotos independentes, não
    células de um mesmo grid).

    No-op (retorna `{}`) se `category` não tiver mapeamento em
    `CATEGORY_TO_EXTERNAL_CLASS`.
    """
    if category not in CATEGORY_TO_EXTERNAL_CLASS:
        return {}

    labels = _category_labels(category)
    counts: dict[str, int] = {}

    external_class = CATEGORY_TO_EXTERNAL_CLASS[category]
    for source_class, label in [
        (external_class, labels.positive),
        (EXTERNAL_NEGATIVE_CLASS, labels.negative),
    ]:
        source_dir = EXTERNAL_DIR / source_class
        dest_dir = output_dir / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for image_path in source_dir.iterdir():
            shutil.copyfile(image_path, dest_dir / f"{EXTERNAL_FILENAME_PREFIX}{image_path.name}")
            count += 1
        counts[label] = count
    return counts


class _Labels:
    def __init__(self, positive: str, negative: str) -> None:
        self.positive = positive
        self.negative = negative


def _category_labels(category: str) -> _Labels:
    # mesma convenção singular "<objeto>"/"nao_<objeto>" de category_models.py
    singular = {
        "onibus": "onibus",
        "escadas": "escada",
        "faixas_pedestre": "faixa_pedestre",
        "carros": "carro",
        "semaforos": "semaforo",
        "motocicletas": "motocicleta",
        "bicicletas": "bicicleta",
    }[category]
    return _Labels(positive=singular, negative=f"nao_{singular}")
