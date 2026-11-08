"""Inferência do classificador de objetos sobre um desafio coletado (Atividades 4-6)."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from attacks.image_grid_cnn.dataset import preprocess_image


def predict_proba(
    model: torch.nn.Module,
    image: Image.Image | np.ndarray,
    device: str | None = None,
) -> list[float]:
    """Classifica uma célula de grid já recortada (PIL.Image ou numpy array RGB) e
    retorna o vetor de probabilidades softmax completo, uma por classe (na ordem
    de `checkpoint["classes"]") -- não só a classe escolhida, pra quem chamar
    poder comparar a confiança de uma classe específica entre células (ver
    `solver.solve_recaptcha_grid_cnn`).
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

    return probabilities.squeeze(0).tolist()


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
    probabilities = predict_proba(model, image, device)
    predicted_idx = max(range(len(probabilities)), key=probabilities.__getitem__)
    return predicted_idx, probabilities[predicted_idx]
