"""Treinamento do classificador de objetos (Atividade 2, Ago-Set/2026)."""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset


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


def evaluate(model: torch.nn.Module, dataset: Dataset, device: str | None = None) -> float:
    """Acurácia de `model` sobre `dataset` (fração de acertos, sem augmentação).

    Dataset pequeno demais pra não ter nenhum sinal de generalização só com
    a loss de treino (ver `category_models.train`) -- este é o cálculo
    mínimo pra responder "o modelo acerta em exemplos que não viu".
    """
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(resolved_device)
    model.eval()

    if len(dataset) == 0:
        return 0.0

    correct = 0
    with torch.no_grad():
        for image, label in dataset:
            logits = model(image.unsqueeze(0).to(resolved_device))
            predicted = int(torch.argmax(logits, dim=1).item())
            correct += int(predicted == label)
    return correct / len(dataset)


def cross_validate(
    samples_dir: str | Path,
    k: int = 5,
    epochs: int = 30,
    train_transform=None,
    eval_transform=None,
    seed: int = 42,
    validation_eligible=None,
) -> dict:
    """Validação cruzada em k dobras, separadas por GRID de origem (não por
    célula solta -- mesma correção de `category_models.train`/
    `train_recaptcha_carros.main`, pra não vazar contexto da mesma foto
    entre treino e validação).

    Um único split treino/validação tem variância alta em datasets deste
    porte (dezenas de grids): o resultado muda bastante dependendo de qual
    fatia caiu na validação (achado ao vivo em 2026-11-05 -- "bicicletas"
    foi de 85% pra 63% só trocando a semente do split). Treina um modelo do
    zero por dobra e reporta a média e o desvio-padrão entre as k dobras,
    em vez de um único número.

    `validation_eligible` (opcional, `(grid_id) -> bool`) restringe quais
    grids podem CAIR NA VALIDAÇÃO -- todo grid continua disponível pro
    treino de cada dobra, elegível ou não. Existe pra categorias misturadas
    com dataset externo (ver `category_models.py`): sem essa restrição, cada
    imagem externa conta como seu próprio "grid" isolado no split, e como
    há muito mais imagem externa que real, a fatia de validação acaba
    dominada por dataset de terceiro -- a acurácia reportada então mede
    "acerta o dataset externo", não "resolve o desafio real" (achado ao
    vivo em 2026-11-06, ao comparar a validação cruzada com uma avaliação
    real: onibus/escadas/faixas_pedestre tinham CV de 92-96% mas acerto
    real de 0-8%). Passar `validation_eligible=lambda g: not is_external_grid_id(g)`
    força a validação a usar só grids reais, mantendo o externo só pro treino.
    """
    # import local evita ciclo (dataset.py/model.py não importam este módulo,
    # mas mantém o padrão já usado em outros pontos do pacote)
    from attacks.image_grid_cnn.dataset import ImageGridDataset
    from attacks.image_grid_cnn.model import build_model

    train_view = ImageGridDataset(samples_dir, transform=train_transform)
    eval_view = ImageGridDataset(samples_dir, transform=eval_transform)

    grid_ids = train_view.grid_ids
    all_grids = set(grid_ids)
    eligible_grids = (
        sorted(all_grids)
        if validation_eligible is None
        else sorted(g for g in all_grids if validation_eligible(g))
    )
    if not eligible_grids:
        raise ValueError("nenhum grid elegível pra validação (validation_eligible excluiu todos)")
    random.Random(seed).shuffle(eligible_grids)
    folds = [eligible_grids[i::k] for i in range(k)]

    fold_accuracies: list[float] = []
    fold_sizes: list[int] = []
    for fold_grids in folds:
        val_grids = set(fold_grids)
        train_grids = all_grids - val_grids
        val_indices = [i for i, grid_id in enumerate(grid_ids) if grid_id in val_grids]
        train_indices = [i for i, grid_id in enumerate(grid_ids) if grid_id in train_grids]
        if not val_indices or not train_indices:
            continue

        model = build_model(num_classes=len(train_view.classes))
        train(Subset(train_view, train_indices), model, epochs=epochs, batch_size=8, lr=1e-3)
        fold_accuracies.append(evaluate(model, Subset(eval_view, val_indices)))
        fold_sizes.append(len(val_indices))

    mean_accuracy = sum(fold_accuracies) / len(fold_accuracies)
    variance = sum((a - mean_accuracy) ** 2 for a in fold_accuracies) / len(fold_accuracies)
    return {
        "fold_accuracies": fold_accuracies,
        "fold_sizes": fold_sizes,
        "mean_accuracy": mean_accuracy,
        "std": variance**0.5,
    }
