"""Inferência do classificador de objetos sobre um desafio coletado (Atividades 4-6)."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from attacks.image_grid_cnn.dataset import preprocess_image


def predict(
    model: torch.nn.Module,
    image: Image.Image | np.ndarray,
    device: str | None = None,
) -> tuple[int, float]:
    """Classifica uma célula de grid já recortada (PIL.Image ou numpy array RGB).

    Aplica o mesmo pré-processamento usado no treinamento (`dataset.preprocess_image`),
    roda o modelo em modo avaliação e retorna `(classe_predita_idx, confianca)`,
    onde `confianca` é a probabilidade softmax da classe escolhida (float em [0, 1]).
    """
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGB")

    input_tensor = preprocess_image(image).unsqueeze(0).to(resolved_device)

    model.to(resolved_device)
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)

    return int(predicted_idx.item()), float(confidence.item())
