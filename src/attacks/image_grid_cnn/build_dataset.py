"""Monta o dataset rotulado de "carros" (grid do reCAPTCHA) a partir de
amostras reais salvas em `data/raw/recaptcha/` (Atividade 2, Nov/2026).

**Por que "carros" e não todas as categorias observadas**: o grid do
reCAPTCHA pede por uma categoria diferente a cada desafio (carros,
motocicletas, semáforos, ônibus, bicicletas, faixas de pedestre, pontes,
escadas, montanhas -- 9 categorias distintas vistas num lote real de 30
amostras coletadas via `experiments.live.recaptcha_baseline_attempt`, zero
custo de API). Treinar um classificador que generalize pra qualquer
categoria exigiria dado rotulado em volume pra cada uma -- fora de escopo
aqui. "Carros" foi a categoria com mais amostras reais nesse lote (7 de
30), então é o recorte tratável: um classificador binário carro/não-carro,
mesmo espírito do classificador fechado/aberto de
`attacks/canvas_click_cnn/`. As outras 8 categorias ficam fora deste
treino -- um solver baseado nisto só se aplica quando o desafio real pedir
"carros" especificamente, outra categoria cai fora de escopo (ver
`attacks/image_grid_cnn/solver.py`).

**Origem e limitação do rótulo**: rotulagem assistida por IA (Claude), não
pelo autor do TCC -- mesma ressalva metodológica já registrada em
`attacks/canvas_click_cnn/build_dataset.py`. Diferente das amostras de
clique-sobre-canvas, aqui a rotulagem é bem mais simples (uma célula
contém ou não um carro, julgamento direto sobre uma foto real, sem
ambiguidade de segmentação) -- mas ainda assim não é o aluno quem rotulou.

**Rótulos em CSV, não mais dict literal** (Nov/2026, lote de coleta de
madrugada): o dataset cresceu de 60 pra ~1,4 mil células rotuladas --
um dict Python hardcoded desse tamanho vira um arquivo-fonte de milhares
de linhas só de dados, sem ganho nenhum sobre um arquivo de dados de
verdade. `LABELS` agora é carregado de
`data/labels/recaptcha_grid_labels/carros.csv` via
`dataset_prep.load_labels_csv` -- mesma estrutura `(amostra, índice) ->
classe`, só a origem física mudou. Ver esse CSV pra conferir/editar
rótulos individuais em vez de procurar no código.

**Reforço com dataset externo** (2026-11-07, mesma fonte/ressalva de
`external_recaptcha57k.py`): soma classe `Car` do Hugging Face aos carros
reais coletados. Checar `train_recaptcha_carros.cross_validate_carros()`
pelo número honesto (validação restrita a grids reais) antes de assumir
que ajudou -- nas 3 categorias onde esse reforço já foi medido antes,
não trouxe generalização real nenhuma.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2

from attacks.image_grid_cnn.dataset_prep import RAW_RECAPTCHA_DIR, extract_cells, load_labels_csv
from attacks.image_grid_cnn.external_recaptcha57k import add_external_images

OUTPUT_DIR = Path(__file__).resolve().parents[3] / "data" / "labels" / "recaptcha_grid_carros"

LABELS: dict[tuple[str, int], str] = load_labels_csv("carros")


def build(output_dir: Path = OUTPUT_DIR, raw_dir: Path = RAW_RECAPTCHA_DIR) -> dict[str, int]:
    """Recorta e salva, em `output_dir/<classe>/*.png` (layout `ImageFolder`),
    só as células presentes em `LABELS`. Retorna a contagem por classe.

    Idempotente: limpa `output_dir` a cada chamada antes de gravar (mesma
    correção já aplicada em `canvas_click_cnn/build_dataset.py`, pra nunca
    deixar PNG de uma rotulagem anterior ao lado dos novos).
    """
    shutil.rmtree(output_dir, ignore_errors=True)

    counts: dict[str, int] = {}
    cache: dict[str, dict[int, object]] = {}

    for (sample_name, index), label in LABELS.items():
        if sample_name not in cache:
            _instruction, cells = extract_cells(raw_dir / sample_name)
            cache[sample_name] = dict(cells)

        crop = cache[sample_name].get(index)
        if crop is None:
            raise ValueError(
                f"Célula {index} não encontrada em {sample_name} -- a extração "
                "mudou desde que os rótulos foram atribuídos?"
            )

        class_dir = output_dir / label
        class_dir.mkdir(parents=True, exist_ok=True)
        out_path = class_dir / f"{sample_name}__{index}.png"
        cv2.imwrite(str(out_path), crop)
        counts[label] = counts.get(label, 0) + 1

    for label, external_count in add_external_images("carros", output_dir).items():
        counts[label] = counts.get(label, 0) + external_count
    return counts


if __name__ == "__main__":
    result = build()
    print(f"Dataset montado em {OUTPUT_DIR}: {result}")
