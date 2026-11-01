"""Arquitetura do classificador de objetos para desafios em grade (Atividade 2, Ago-Set/2026).

CNN pequena e leve, pensada para rodar em CPU sobre células individuais de um
grid 3x3/4x4 (reCAPTCHA v2 / hCaptcha). Cada entrada é uma imagem RGB já
redimensionada para `INPUT_SIZE` x `INPUT_SIZE` (ver `dataset.py`/`infer.py`
para o pré-processamento compartilhado).
"""

from __future__ import annotations

import torch
from torch import nn

# Tamanho (em pixels, lado x lado) esperado na entrada do modelo. Pequeno o
# suficiente para treinar/inferir em CPU com poucos recursos.
INPUT_SIZE = 64


class GridCellCNN(nn.Module):
    """CNN leve para classificar o conteúdo de uma célula de um grid de CAPTCHA.

    Arquitetura: 3 blocos conv(3x3) + ReLU + maxpool(2x2), seguidos de uma
    camada linear final. Com `INPUT_SIZE = 64`, o mapa de features após os
    três poolings tem resolução 8x8.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16 -> 8
        )
        feature_map_size = INPUT_SIZE // 8
        self.classifier = nn.Linear(64 * feature_map_size * feature_map_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_model(num_classes: int) -> GridCellCNN:
    """Instancia o classificador de célula de grid com `num_classes` saídas."""
    return GridCellCNN(num_classes=num_classes)
