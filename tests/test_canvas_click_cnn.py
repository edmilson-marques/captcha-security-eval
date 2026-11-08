"""Testes do módulo canvas_click_cnn (Atividade 2, Nov/2026).

Usa amostras sintéticas (metadata.json + challenge.png fabricados em
`tmp_path`) e um dataset/checkpoint sintético pro solver — não depende das
amostras reais em `data/raw/hcaptcha/` (gitignored, efêmeras, podem não
existir num checkout novo) nem do checkpoint treinado em `data/models/`
(também gitignored). Mesma filosofia de teste de
`tests/test_image_grid_cnn_inference.py`: valida o CONTRATO de
entrada/saída, não uma acurácia específica.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
import pytest

from attacks.canvas_click_cnn import build_dataset, dataset_prep
from attacks.canvas_click_cnn.solver import solve_closed_loop_cnn
from attacks.image_grid_cnn.dataset import ImageGridDataset
from attacks.image_grid_cnn.model import build_model
from attacks.image_grid_cnn.train import train


def _flat_background(size: int = 200, color: tuple[int, int, int] = (120, 120, 120)) -> np.ndarray:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:, :] = color
    return image


def _draw_closed_circle(image: np.ndarray, center: tuple[int, int], radius: int = 30) -> None:
    cv2.circle(image, center, radius, color=(0, 200, 255), thickness=10)


def _draw_open_arc(image: np.ndarray, center: tuple[int, int], radius: int = 30) -> None:
    cv2.ellipse(image, center, (radius, radius), 0, 0, 270, color=(255, 100, 0), thickness=10)


def _write_sample(
    sample_dir: Path, image: np.ndarray, criterion_type: str | None, prompt: str = ""
) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(sample_dir / "challenge.png"), image)
    metadata = {
        "target": "hcaptcha",
        "technique": "classical_cv",
        "outcome": "failure",
        "extra": {"criterion_type": criterion_type, "prompt": prompt},
    }
    (sample_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


class TestExtractCandidates:
    def test_closed_loop_sample_returns_region_candidates(self, tmp_path: Path):
        image = _flat_background()
        _draw_closed_circle(image, (60, 60))
        _draw_open_arc(image, (140, 140))
        sample_dir = tmp_path / "sample_closed_loop"
        _write_sample(sample_dir, image, "closed_loop")

        criterion_type, candidates = dataset_prep.extract_candidates(sample_dir)

        assert criterion_type == "closed_loop"
        assert len(candidates) >= 2
        assert all(c.crop.size > 0 for c in candidates)

    def test_unsupported_criterion_type_returns_empty(self, tmp_path: Path):
        image = _flat_background()
        sample_dir = tmp_path / "sample_grid"
        _write_sample(sample_dir, image, "grid")

        criterion_type, candidates = dataset_prep.extract_candidates(sample_dir)

        assert criterion_type == "grid"
        assert candidates == []

    def test_digit_membership_without_digits_returns_empty_candidates(self, tmp_path: Path):
        image = _flat_background()
        sample_dir = tmp_path / "sample_digits"
        _write_sample(sample_dir, image, "digit_membership", prompt="Clique no número: 789")

        criterion_type, candidates = dataset_prep.extract_candidates(sample_dir)

        assert criterion_type == "digit_membership"
        assert candidates == []  # fundo liso não tem dígito nenhum pro OCR achar


class TestBuildDataset:
    def test_build_writes_imagefolder_layout_from_labels(self, tmp_path: Path, monkeypatch):
        image = _flat_background()
        _draw_closed_circle(image, (60, 60))
        _draw_open_arc(image, (140, 140))
        raw_dir = tmp_path / "raw"
        _write_sample(raw_dir / "sample_a", image, "closed_loop")

        criterion_type, candidates = dataset_prep.extract_candidates(raw_dir / "sample_a")
        assert criterion_type == "closed_loop"
        assert len(candidates) >= 2
        fake_labels = {
            ("sample_a", candidates[0].index): "closed",
            ("sample_a", candidates[1].index): "open",
        }
        monkeypatch.setattr(build_dataset, "LABELS", fake_labels)

        output_dir = tmp_path / "labels_out"
        counts = build_dataset.build(output_dir=output_dir, raw_dir=raw_dir)

        assert counts == {"closed": 1, "open": 1}
        assert len(list((output_dir / "closed").glob("*.png"))) == 1
        assert len(list((output_dir / "open").glob("*.png"))) == 1

    def test_build_clears_stale_files_from_a_previous_labels_set(self, tmp_path: Path, monkeypatch):
        # Achado ao vivo em 2026-11-04: build() dizia recriar output_dir do
        # zero mas só sobrescrevia arquivos de mesmo nome -- quando LABELS
        # muda (ex. segmentação nova, índices de candidato diferentes), os
        # PNGs da rotulagem antiga ficavam no diretório ao lado dos novos.
        image = _flat_background()
        _draw_closed_circle(image, (60, 60))
        _draw_open_arc(image, (140, 140))
        raw_dir = tmp_path / "raw"
        _write_sample(raw_dir / "sample_a", image, "closed_loop")
        output_dir = tmp_path / "labels_out"

        _criterion_type, candidates = dataset_prep.extract_candidates(raw_dir / "sample_a")
        monkeypatch.setattr(build_dataset, "LABELS", {("sample_a", candidates[0].index): "closed"})
        build_dataset.build(output_dir=output_dir, raw_dir=raw_dir)
        assert len(list(output_dir.glob("*/*.png"))) == 1

        monkeypatch.setattr(build_dataset, "LABELS", {("sample_a", candidates[1].index): "open"})
        counts = build_dataset.build(output_dir=output_dir, raw_dir=raw_dir)

        assert counts == {"open": 1}
        assert len(list(output_dir.glob("*/*.png"))) == 1
        assert not (output_dir / "closed").exists()

    def test_build_raises_on_missing_candidate_index(self, tmp_path: Path, monkeypatch):
        image = _flat_background()
        raw_dir = tmp_path / "raw"
        _write_sample(raw_dir / "sample_b", image, "closed_loop")
        monkeypatch.setattr(build_dataset, "LABELS", {("sample_b", 99): "closed"})

        with pytest.raises(ValueError, match="não encontrado"):
            build_dataset.build(output_dir=tmp_path / "out", raw_dir=raw_dir)


@pytest.fixture
def trained_checkpoint(tmp_path: Path) -> Path:
    """Dataset sintético (fechado x aberto) + treino rápido, salvo como checkpoint."""
    rng = random.Random(42)
    dataset_root = tmp_path / "closed_loop_labels"
    for class_name, draw_fn in (("closed", _draw_closed_circle), ("open", _draw_open_arc)):
        class_dir = dataset_root / class_name
        class_dir.mkdir(parents=True)
        for i in range(6):
            crop = _flat_background(size=64, color=(120 + rng.randint(-10, 10),) * 3)
            draw_fn(crop, (32, 32), radius=20)
            cv2.imwrite(str(class_dir / f"{class_name}_{i}.png"), crop)

    dataset = ImageGridDataset(dataset_root)
    model = build_model(num_classes=len(dataset.classes))
    train(dataset, model, epochs=3, batch_size=4, device="cpu")

    checkpoint_path = tmp_path / "closed_loop_cnn.pt"
    import torch

    torch.save({"state_dict": model.state_dict(), "classes": dataset.classes}, checkpoint_path)
    return checkpoint_path


class TestSolveClosedLoopCnn:
    def test_returns_list_of_int_point_tuples(self, trained_checkpoint: Path):
        image = _flat_background()
        _draw_closed_circle(image, (60, 60))
        _draw_open_arc(image, (140, 140))

        points = solve_closed_loop_cnn(image, checkpoint_path=trained_checkpoint)

        assert isinstance(points, list)
        for point in points:
            assert len(point) == 2
            assert all(isinstance(coord, int) for coord in point)

    def test_raises_file_not_found_without_checkpoint(self, tmp_path: Path):
        missing_checkpoint = tmp_path / "does_not_exist.pt"
        image = _flat_background()

        with pytest.raises(FileNotFoundError):
            solve_closed_loop_cnn(image, checkpoint_path=missing_checkpoint)
