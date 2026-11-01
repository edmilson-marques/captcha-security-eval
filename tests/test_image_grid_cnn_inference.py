"""Teste fim-a-fim do classificador de célula de grid (Atividade 2, Ago-Set/2026).

Não depende de amostras reais (que ainda não existem, ver `dataset.py`) nem de
rede/GPU: gera um dataset sintético em `tmp_path` com duas classes visualmente
bem distintas (vermelho x azul, com ruído), treina a CNN por poucas épocas e
valida apenas o CONTRATO de saída de `infer.predict` — não exige um limiar de
acurácia, já que o treino com poucos dados sintéticos pode não convergir de
forma estável.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from attacks.image_grid_cnn import infer
from attacks.image_grid_cnn.dataset import ImageGridDataset
from attacks.image_grid_cnn.model import build_model
from attacks.image_grid_cnn.train import train

IMAGES_PER_CLASS = 6
IMAGE_SIDE = 48  # imagem "crua" antes do resize para INPUT_SIZE em dataset.py


def _make_class_image(base_rgb: tuple[int, int, int], rng: random.Random) -> Image.Image:
    """Gera uma imagem procedural com ruído em torno de uma cor-base."""
    array = np.zeros((IMAGE_SIDE, IMAGE_SIDE, 3), dtype=np.uint8)
    for channel in range(3):
        noise = rng.randint(-20, 20)
        array[:, :, channel] = max(0, min(255, base_rgb[channel] + noise))
    # Ruído por pixel, para não gerar imagens perfeitamente uniformes.
    pixel_noise = np.array(
        [[rng.randint(-15, 15) for _ in range(IMAGE_SIDE)] for _ in range(IMAGE_SIDE)],
        dtype=np.int16,
    )
    noisy = array.astype(np.int16) + pixel_noise[:, :, None]
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy, mode="RGB")


def _build_synthetic_dataset(root: Path) -> None:
    rng = random.Random(42)
    class_colors = {
        "classe_a": (200, 30, 30),  # predominantemente vermelho
        "classe_b": (30, 30, 200),  # predominantemente azul
    }
    for class_name, base_rgb in class_colors.items():
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(IMAGES_PER_CLASS):
            image = _make_class_image(base_rgb, rng)
            image.save(class_dir / f"{class_name}_{i:02d}.png")


@pytest.fixture
def synthetic_dataset_dir(tmp_path: Path) -> Path:
    dataset_root = tmp_path / "grid_cells"
    _build_synthetic_dataset(dataset_root)
    return dataset_root


def test_pipeline_runs_end_to_end_and_respects_output_contract(synthetic_dataset_dir: Path):
    dataset = ImageGridDataset(synthetic_dataset_dir)
    assert set(dataset.classes) == {"classe_a", "classe_b"}
    assert len(dataset) == 2 * IMAGES_PER_CLASS

    num_classes = len(dataset.classes)
    model = build_model(num_classes=num_classes)

    losses = train(dataset, model, epochs=5, batch_size=4, device="cpu")
    assert len(losses) == 5
    assert all(isinstance(loss, float) for loss in losses)

    test_image = _make_class_image((200, 30, 30), random.Random(7))
    predicted_idx, confidence = infer.predict(model, test_image, device="cpu")

    assert isinstance(predicted_idx, int)
    assert 0 <= predicted_idx < num_classes
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0


def test_predict_accepts_numpy_array_input(synthetic_dataset_dir: Path):
    dataset = ImageGridDataset(synthetic_dataset_dir)
    model = build_model(num_classes=len(dataset.classes))
    train(dataset, model, epochs=2, batch_size=4, device="cpu")

    numpy_image = np.array(_make_class_image((30, 30, 200), random.Random(3)))
    predicted_idx, confidence = infer.predict(model, numpy_image, device="cpu")

    assert isinstance(predicted_idx, int)
    assert 0 <= predicted_idx < len(dataset.classes)
    assert 0.0 <= confidence <= 1.0


def test_softmax_probabilities_sum_to_one(synthetic_dataset_dir: Path):
    import torch

    from attacks.image_grid_cnn.dataset import preprocess_image

    dataset = ImageGridDataset(synthetic_dataset_dir)
    model = build_model(num_classes=len(dataset.classes))
    model.eval()

    test_image = _make_class_image((30, 30, 200), random.Random(11)).convert("RGB")
    input_tensor = preprocess_image(test_image).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.softmax(model(input_tensor), dim=1)

    assert probabilities.sum().item() == pytest.approx(1.0, abs=1e-5)
