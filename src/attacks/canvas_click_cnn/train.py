"""Treino do classificador fechado/aberto de "laços fechados" (hCaptcha, Nov/2026).

Reaproveita a arquitetura/dataset/loop de treino genéricos já existentes em
`attacks/image_grid_cnn/` (nada ali é de fato específico de grade de
imagem, apesar do nome do módulo — é só uma CNN pequena + um
`ImageFolder` + Adam/CrossEntropy) em vez de duplicar o mesmo código aqui.
A única coisa nova deste módulo é o dataset em si (`build_dataset.py`) e
este ponto de entrada de treino, já configurado pro caso de uso específico
(2 classes, dataset pequeno e desbalanceado — ver `build_dataset.py` sobre
a origem e a limitação dos rótulos).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torchvision import transforms

from attacks.canvas_click_cnn.build_dataset import OUTPUT_DIR
from attacks.image_grid_cnn.dataset import ImageGridDataset
from attacks.image_grid_cnn.model import INPUT_SIZE, build_model
from attacks.image_grid_cnn.train import train as train_loop

CHECKPOINT_PATH = Path(__file__).resolve().parents[3] / "data" / "models" / "closed_loop_cnn.pt"

# Transform de treino com augmentação leve (flip/rotação) -- justificado pelo
# dataset ser muito pequeno (20 exemplos): sem augmentação, o modelo decora
# os próprios 20 exemplos em vez de aprender qualquer invariância real.
# Avaliação/inferência continuam usando o transform default (sem augmentação,
# ver `image_grid_cnn.dataset.DEFAULT_TRANSFORM`), pra não distorcer a
# imagem real recebida em produção.
TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


def main(epochs: int = 30, checkpoint_path: Path = CHECKPOINT_PATH) -> list[float]:
    """Treina o classificador sobre `data/labels/hcaptcha_closed_loop/` e salva o checkpoint.

    Retorna a lista de loss por época (ver `image_grid_cnn.train.train`).
    Levanta `FileNotFoundError` com uma mensagem clara se o dataset ainda
    não foi montado (rodar `python -m attacks.canvas_click_cnn.build_dataset` antes).
    """
    if not OUTPUT_DIR.exists():
        raise FileNotFoundError(
            f"{OUTPUT_DIR} não existe -- rode "
            "`python -m attacks.canvas_click_cnn.build_dataset` antes de treinar."
        )

    dataset = ImageGridDataset(OUTPUT_DIR, transform=TRAIN_TRANSFORM)
    model = build_model(num_classes=len(dataset.classes))

    losses = train_loop(dataset, model, epochs=epochs, batch_size=8, lr=1e-3)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "classes": dataset.classes},
        checkpoint_path,
    )
    return losses


if __name__ == "__main__":
    final_losses = main()
    print(f"Loss por época: {final_losses}")
    print(f"Checkpoint salvo em {CHECKPOINT_PATH}")
