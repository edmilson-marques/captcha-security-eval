"""Treinamento do classificador de objetos (Atividade 2, Ago-Set/2026)."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset


def train(
    dataset: Dataset,
    model: torch.nn.Module,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str | None = None,
) -> list[float]:
    """Treina `model` sobre `dataset` com Adam + CrossEntropyLoss.

    Retorna a lista com a loss média de cada época (útil para inspeção ou
    plot posterior, ex. nos notebooks de análise exploratória).
    """
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(resolved_device)
    model.train()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    epoch_losses: list[float] = []
    for _ in range(epochs):
        running_loss = 0.0
        num_batches = 0
        for inputs, labels in dataloader:
            inputs = inputs.to(resolved_device)
            labels = labels.to(resolved_device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        epoch_losses.append(running_loss / max(num_batches, 1))

    return epoch_losses
