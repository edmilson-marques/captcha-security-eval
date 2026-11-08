"""Monta o dataset e treina o classificador binário <objeto>/nao_<objeto> pra
categorias do grid do reCAPTCHA além de "carros" (Nov/2026).

"carros" continua em `build_dataset.py`/`train_recaptcha_carros.py` (primeiro
treino, já referenciado por `solver.py`) -- este módulo generaliza o mesmo
recorte binário por célula pras categorias coletadas depois (`onibus`,
`semaforos`, `motocicletas`, `bicicletas`, `escadas`, `faixas_pedestre`),
parametrizado por categoria em vez de duplicar um par de arquivos build/train
por categoria.

**Origem e limitação do rótulo**: rotulagem assistida por IA (Claude), não
pelo autor do TCC -- mesma ressalva metodológica já registrada em
`build_dataset.py`. Categorias ainda sem dado suficiente (pontes, montanhas)
ficam fora deste módulo; um solver só se aplica à categoria quando
`LABELS_BY_CATEGORY` tem entradas pra ela (ver `solver.py`).

**Rótulos em CSV, não mais dict literal** (Nov/2026, lote de coleta de
madrugada): o dataset cresceu de centenas pra ~10 mil células rotuladas --
um dict Python hardcoded desse tamanho vira um arquivo-fonte de dezenas de
milhares de linhas só de dados. Cada categoria agora é carregada de
`data/labels/recaptcha_grid_labels/<categoria>.csv` via
`dataset_prep.load_labels_csv` (mesma estrutura `(amostra, índice) ->
classe`, só a origem física mudou). Ver esses CSVs pra conferir/editar
rótulos individuais em vez de procurar no código.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import cv2
import torch
from torch.utils.data import Subset
from torchvision import transforms

from attacks.image_grid_cnn.dataset import DEFAULT_TRANSFORM, ImageGridDataset
from attacks.image_grid_cnn.dataset_prep import RAW_RECAPTCHA_DIR, extract_cells, load_labels_csv
from attacks.image_grid_cnn.external_recaptcha57k import add_external_images, is_external_grid_id
from attacks.image_grid_cnn.model import INPUT_SIZE, build_model
from attacks.image_grid_cnn.train import (
    cross_validate as cross_validate_loop,
    evaluate,
    train as train_loop,
)

LABELS_DIR = Path(__file__).resolve().parents[3] / "data" / "labels"
MODELS_DIR = Path(__file__).resolve().parents[3] / "data" / "models"

CATEGORY_NAMES = [
    "onibus",
    "semaforos",
    "motocicletas",
    "bicicletas",
    "escadas",
    "faixas_pedestre",
]

LABELS_BY_CATEGORY: dict[str, dict[tuple[str, int], str]] = {
    category: load_labels_csv(category) for category in CATEGORY_NAMES
}

# mesma augmentação leve de `train_recaptcha_carros.py` -- datasets pequenos
# e reais, sem ela o modelo só decora os exemplos em vez de generalizar.
TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


def output_dir_for(category: str) -> Path:
    return LABELS_DIR / f"recaptcha_grid_{category}"


def checkpoint_path_for(category: str) -> Path:
    return MODELS_DIR / f"recaptcha_grid_{category}_cnn.pt"


def build(category: str, raw_dir: Path = RAW_RECAPTCHA_DIR) -> dict[str, int]:
    """Recorta e salva as células rotuladas de `category` em
    `output_dir_for(category)/<classe>/*.png` (layout `ImageFolder`).

    Idempotente: limpa o diretório de saída antes de gravar, mesma correção
    já aplicada em `build_dataset.py`. Pra categorias ainda abaixo do chute
    trivial com todo o dado real coletado (`external_recaptcha57k.
    CATEGORY_TO_EXTERNAL_CLASS`), soma também imagens de um dataset de
    terceiros feito de tiles reais de reCAPTCHA v2 -- ver docstring desse
    módulo pra origem, licença e proporção real/externo.

    Não filtra selo de "check" automaticamente -- `dataset_prep.
    has_checkmark_overlay` teve falso-positivo grave em teste real (parede/
    carro azul saturado confundido com o selo, ~5-10% das células descartadas
    por engano); serve só pra PRÉ-TRIAGEM de candidatos pra revisão manual
    (ver checklist no relatório), não como filtro automático aqui.
    """
    labels = LABELS_BY_CATEGORY[category]
    output_dir = output_dir_for(category)
    shutil.rmtree(output_dir, ignore_errors=True)

    counts: dict[str, int] = {}
    cache: dict[str, dict[int, object]] = {}
    for (sample_name, index), label in labels.items():
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
        cv2.imwrite(str(class_dir / f"{sample_name}__{index}.png"), crop)
        counts[label] = counts.get(label, 0) + 1

    for label, external_count in add_external_images(category, output_dir).items():
        counts[label] = counts.get(label, 0) + external_count
    return counts


def train(category: str, epochs: int = 30, val_fraction: float = 0.2) -> dict:
    """Treina o classificador binário de `category` sobre
    `output_dir_for(category)` e salva o checkpoint em `checkpoint_path_for(category)`.

    Separa `val_fraction` dos GRIDS de origem (não das células soltas, sem
    augmentação, embaralhados com seed fixa pra reprodutibilidade) como
    validação -- datasets deste porte (dezenas de exemplos) convergem pra
    loss de treino perto de zero mesmo decorando, então só a loss de treino
    não diz nada sobre generalização (achado ao vivo em 2026-11-05: cliques
    errados na demo real apesar de loss baixa). Retorna também
    `val_accuracy`/`val_size` pra essa checagem mínima existir.

    **Split por grid, não por célula** (corrigido em 2026-11-05): um split
    aleatório sobre o índice de CADA CÉLULA deixava células do MESMO grid
    (mesma foto) caírem uma no treino e outra na validação -- conferido ao
    vivo: ~100% dos grids de validação tinham outra célula sua no treino,
    vazando contexto da cena (luz, local, enquadramento) e inflando a
    acurácia reportada. Agora o split separa por `grid_ids` inteiro antes de
    expandir pras células.

    **Validação nunca cai em dataset externo** (corrigido em 2026-11-06): pra
    categorias com `external_recaptcha57k.add_external_images`, a validação
    só sorteia entre grids reais (`is_external_grid_id` filtra o resto) --
    sem isso, como cada imagem externa conta como seu próprio "grid"
    isolado e há muito mais imagem externa que real, a validação acabava
    dominada por dataset de terceiro (achado ao vivo: CV de 92-96% medindo
    "acerta o dataset externo", não "resolve o desafio real" -- acerto real
    medido depois foi de 0-8%). O treino continua usando tudo, externo e
    real.
    """
    output_dir = output_dir_for(category)
    if not output_dir.exists():
        raise FileNotFoundError(f"{output_dir} não existe -- rode build({category!r}) antes.")

    train_view = ImageGridDataset(output_dir, transform=TRAIN_TRANSFORM)
    eval_view = ImageGridDataset(output_dir, transform=DEFAULT_TRANSFORM)

    grid_ids = train_view.grid_ids
    all_grids = set(grid_ids)
    eligible_grids = sorted(g for g in all_grids if not is_external_grid_id(g))
    if not eligible_grids:
        raise ValueError(f"{category!r} não tem nenhum grid real pra validação")
    random.Random(42).shuffle(eligible_grids)
    val_grid_count = max(1, round(len(eligible_grids) * val_fraction))
    val_grids = set(eligible_grids[:val_grid_count])

    val_indices = [i for i, grid_id in enumerate(grid_ids) if grid_id in val_grids]
    train_indices = [i for i, grid_id in enumerate(grid_ids) if grid_id not in val_grids]

    model = build_model(num_classes=len(train_view.classes))
    losses = train_loop(
        Subset(train_view, train_indices), model, epochs=epochs, batch_size=8, lr=1e-3
    )
    val_accuracy = evaluate(model, Subset(eval_view, val_indices))

    checkpoint_path = checkpoint_path_for(category)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": train_view.classes}, checkpoint_path)
    return {"losses": losses, "val_accuracy": val_accuracy, "val_size": len(val_indices)}


def cross_validate_category(category: str, k: int = 5, epochs: int = 30) -> dict:
    """`train.cross_validate` sobre `output_dir_for(category)`, sempre
    excluindo dataset externo da validação (`is_external_grid_id`) -- mesma
    razão do bloco "Validação nunca cai em dataset externo" na docstring de
    `train()` acima. Wrapper fino pra não repetir esse filtro em todo
    chamador."""
    return cross_validate_loop(
        output_dir_for(category),
        k=k,
        epochs=epochs,
        train_transform=TRAIN_TRANSFORM,
        eval_transform=DEFAULT_TRANSFORM,
        validation_eligible=lambda grid_id: not is_external_grid_id(grid_id),
    )


if __name__ == "__main__":
    for category in LABELS_BY_CATEGORY:
        counts = build(category)
        result = train(category)
        print(
            f"{category}: dataset={counts} loss_final={result['losses'][-1]:.4f} "
            f"val_accuracy={result['val_accuracy']:.2f} (n={result['val_size']})"
        )
