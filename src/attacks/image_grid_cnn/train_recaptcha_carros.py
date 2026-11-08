"""Treino do classificador carro/não-carro pra célula de grid do reCAPTCHA (Nov/2026).

Ponto de entrada específico (dataset pequeno e real, ver
`build_dataset.py`) sobre a arquitetura/loop de treino genéricos já
existentes neste mesmo pacote (`model.py`/`train.py`) -- nada duplicado,
mesmo espírito de `attacks/canvas_click_cnn/train.py`.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import Subset
from torchvision import transforms

from attacks.image_grid_cnn.build_dataset import OUTPUT_DIR
from attacks.image_grid_cnn.dataset import DEFAULT_TRANSFORM, ImageGridDataset
from attacks.image_grid_cnn.external_recaptcha57k import is_external_grid_id
from attacks.image_grid_cnn.model import INPUT_SIZE, build_model
from attacks.image_grid_cnn.train import (
    cross_validate as cross_validate_loop,
    evaluate,
    train as train_loop,
)

CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "models" / "recaptcha_grid_carros_cnn.pt"
)

# Augmentação leve -- dataset pequeno (60 exemplos), sem ela o modelo só
# decora as 60 fotos em vez de aprender qualquer invariância real. Sem
# flip vertical (um carro de cabeça pra baixo não ocorre numa foto de rua
# real) nem rotação forte (perspectiva de rua, não objeto centralizado).
TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


def main(
    epochs: int = 30, checkpoint_path: Path = CHECKPOINT_PATH, val_fraction: float = 0.2
) -> dict:
    """Treina o classificador sobre `data/labels/recaptcha_grid_carros/` e salva o checkpoint.

    Separa `val_fraction` dos GRIDS de origem (não das células soltas, sem
    augmentação) como validação -- mesma correção aplicada em
    `category_models.train`: splitar por célula solta deixava células do
    MESMO grid caírem uma no treino e outra na validação, vazando contexto
    da cena e inflando a acurácia (achado ao vivo em 2026-11-05).

    **Validação nunca cai em dataset externo** (2026-11-07, mesma correção de
    `category_models.train`): com `external_recaptcha57k.add_external_images`
    somando a classe `Car`, a validação só sorteia entre grids reais.
    """
    if not OUTPUT_DIR.exists():
        raise FileNotFoundError(
            f"{OUTPUT_DIR} não existe -- rode "
            "`python -m attacks.image_grid_cnn.build_dataset` antes de treinar."
        )

    train_view = ImageGridDataset(OUTPUT_DIR, transform=TRAIN_TRANSFORM)
    eval_view = ImageGridDataset(OUTPUT_DIR, transform=DEFAULT_TRANSFORM)

    grid_ids = train_view.grid_ids
    unique_grids = sorted(g for g in set(grid_ids) if not is_external_grid_id(g))
    random.Random(42).shuffle(unique_grids)
    val_grid_count = max(1, round(len(unique_grids) * val_fraction))
    val_grids = set(unique_grids[:val_grid_count])

    val_indices = [i for i, grid_id in enumerate(grid_ids) if grid_id in val_grids]
    train_indices = [i for i, grid_id in enumerate(grid_ids) if grid_id not in val_grids]

    model = build_model(num_classes=len(train_view.classes))
    losses = train_loop(
        Subset(train_view, train_indices), model, epochs=epochs, batch_size=8, lr=1e-3
    )
    val_accuracy = evaluate(model, Subset(eval_view, val_indices))

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "classes": train_view.classes},
        checkpoint_path,
    )
    return {"losses": losses, "val_accuracy": val_accuracy, "val_size": len(val_indices)}


def cross_validate_carros(k: int = 5, epochs: int = 30) -> dict:
    """`train.cross_validate` sobre `OUTPUT_DIR`, sempre excluindo dataset
    externo da validação (`is_external_grid_id`) -- mesmo motivo do bloco
    "Validação nunca cai em dataset externo" na docstring de `main()`."""
    return cross_validate_loop(
        OUTPUT_DIR,
        k=k,
        epochs=epochs,
        train_transform=TRAIN_TRANSFORM,
        eval_transform=DEFAULT_TRANSFORM,
        validation_eligible=lambda grid_id: not is_external_grid_id(grid_id),
    )


if __name__ == "__main__":
    result = main()
    print(f"Loss por época: {result['losses']}")
    print(f"val_accuracy={result['val_accuracy']:.2f} (n={result['val_size']})")
    print(f"Checkpoint salvo em {CHECKPOINT_PATH}")
