"""Teste do `validation_eligible` de `train.cross_validate` (2026-11-06):
garante que grids marcados como não-elegíveis (ex. dataset externo) nunca
caem na validação, mesmo continuando disponíveis pro treino de cada dobra.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image

from attacks.image_grid_cnn.train import cross_validate

EXTERNAL_PREFIX = "ext_"


def _make_dataset(root: Path) -> None:
    rng = random.Random(0)
    for cls, base_rgb in [("a", (200, 30, 30)), ("b", (30, 30, 200))]:
        class_dir = root / cls
        class_dir.mkdir(parents=True)
        # 6 grids "reais" (nome com "__", um por amostra -- grid_id = prefixo antes do "__")
        for i in range(6):
            array = np.full((48, 48, 3), base_rgb, dtype=np.int16) + rng.randint(-10, 10)
            Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).save(
                class_dir / f"real_{cls}_{i}__0.png"
            )
        # 6 imagens "externas" (sem "__" -- cada uma é seu próprio grid isolado)
        for i in range(6):
            array = np.full((48, 48, 3), base_rgb, dtype=np.int16) + rng.randint(-10, 10)
            Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).save(
                class_dir / f"{EXTERNAL_PREFIX}{cls}_{i}.png"
            )


def test_cross_validate_never_puts_ineligible_grids_in_validation(tmp_path: Path):
    _make_dataset(tmp_path)

    result = cross_validate(
        tmp_path,
        k=3,
        epochs=1,
        validation_eligible=lambda grid_id: not grid_id.startswith(EXTERNAL_PREFIX),
    )

    # só 12 células "reais" (6 por classe) são elegíveis pra validação; as 12
    # "externas" nunca devem aparecer numa dobra de validação.
    assert sum(result["fold_sizes"]) == 12
    assert len(result["fold_accuracies"]) == 3


def test_cross_validate_without_filter_uses_all_grids(tmp_path: Path):
    _make_dataset(tmp_path)

    result = cross_validate(tmp_path, k=3, epochs=1)

    assert sum(result["fold_sizes"]) == 24
