"""Testes do pipeline de qualidade/rotulagem adicionado em 2026-11-06:
detecção de captura em branco, de selo de "check" contaminado, divisão de
lote de rotulagem e validação/mesclagem de rótulos novos contra o grid real.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from attacks.image_grid_cnn.dataset_prep import (
    has_checkmark_overlay,
    is_blank_capture,
    plan_labeling_chunks,
    validate_and_merge_labels,
)


def test_is_blank_capture_flags_uniform_white_image():
    blank = np.full((300, 300, 3), 254, dtype=np.uint8)
    assert is_blank_capture(blank) is True


def test_is_blank_capture_does_not_flag_real_photo_like_noise():
    rng = np.random.default_rng(0)
    photo_like = rng.integers(0, 255, size=(300, 300, 3), dtype=np.uint8)
    assert is_blank_capture(photo_like) is False


def test_has_checkmark_overlay_flags_saturated_blue_circle():
    cell = np.full((130, 130, 3), 200, dtype=np.uint8)  # fundo claro, neutro
    # círculo azul saturado (BGR) cobrindo boa parte da célula, como o selo real
    cell[30:100, 30:100] = (220, 80, 20)  # B alto, G/R baixos -> azul saturado em HSV
    assert has_checkmark_overlay(cell) is True


def test_has_checkmark_overlay_does_not_flag_plain_gray_cell():
    cell = np.full((130, 130, 3), 150, dtype=np.uint8)
    assert has_checkmark_overlay(cell) is False


def test_plan_labeling_chunks_splits_into_bounded_batches():
    ids = [f"sample_{i}" for i in range(223)]
    chunks = plan_labeling_chunks(ids, max_chunk_size=60)
    assert [len(c) for c in chunks] == [60, 60, 60, 43]
    assert sum(chunks, []) == ids


def test_plan_labeling_chunks_empty_list():
    assert plan_labeling_chunks([], max_chunk_size=60) == []


def _make_raw_sample(raw_dir: Path, name: str, grid_size: int = 3) -> None:
    sample_dir = raw_dir / name
    sample_dir.mkdir(parents=True)
    (sample_dir / "metadata.json").write_text(
        json.dumps({"extra": {"grid_size": grid_size, "instruction": "teste"}}),
        encoding="utf-8",
    )
    import cv2

    side = grid_size * 40
    image = np.full((side, side, 3), 120, dtype=np.uint8)
    cv2.imwrite(str(sample_dir / "challenge.png"), image)


def test_validate_and_merge_labels_keeps_valid_entries(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    _make_raw_sample(raw_dir, "sample_a", grid_size=3)

    merged, dropped = validate_and_merge_labels(
        existing_labels={("sample_a", 0): "carro"},
        new_labels={("sample_a", 1): "nao_carro", ("sample_a", 8): "carro"},
        raw_dir=raw_dir,
    )

    assert merged == {
        ("sample_a", 0): "carro",
        ("sample_a", 1): "nao_carro",
        ("sample_a", 8): "carro",
    }
    assert dropped == []


def test_validate_and_merge_labels_drops_out_of_range_index(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    _make_raw_sample(raw_dir, "sample_a", grid_size=3)  # só índices 0-8 existem

    merged, dropped = validate_and_merge_labels(
        existing_labels={},
        new_labels={("sample_a", 0): "carro", ("sample_a", 15): "carro"},
        raw_dir=raw_dir,
    )

    assert merged == {("sample_a", 0): "carro"}
    assert dropped == [("sample_a", 15, "carro")]


def test_validate_and_merge_labels_drops_missing_sample(tmp_path: Path):
    raw_dir = tmp_path / "raw"

    merged, dropped = validate_and_merge_labels(
        existing_labels={},
        new_labels={("amostra_inexistente", 0): "carro"},
        raw_dir=raw_dir,
    )

    assert merged == {}
    assert dropped == [("amostra_inexistente", 0, "carro")]
