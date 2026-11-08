"""Teste do resgate de células por média dos positivos em
`solve_recaptcha_grid_cnn` (achado do usuário, 2026-11-08): um corte fixo de
0,5 por célula isolada ignora que um desafio real raramente tem só uma célula
certa -- perde células genuínas um pouco abaixo do limiar. `_load_model` e
`predict_proba` são trocados por dublês (nenhum checkpoint/modelo real
necessário) -- só o critério de seleção em `solve_recaptcha_grid_cnn` está
sob teste aqui.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from attacks.image_grid_cnn import solver

CLASSES = ["nao_carro", "carro"]  # positive_idx = 1
FAKE_CHECKPOINT = Path("/fake/checkpoint.pt")


def _fake_screenshot(grid_size: int) -> bytes:
    image = Image.new("RGB", (grid_size * 32, grid_size * 32), color=(128, 128, 128))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _patch_model_and_probs(monkeypatch: pytest.MonkeyPatch, positive_probs: list[float]) -> None:
    monkeypatch.setattr(solver, "_load_model", lambda path: (object(), CLASSES))
    calls = iter(positive_probs)

    def fake_predict_proba(model, crop):
        p = next(calls)
        return [1 - p, p]

    monkeypatch.setattr(solver, "predict_proba", fake_predict_proba)


def _solve(monkeypatch: pytest.MonkeyPatch, positive_probs: list[float], grid_size: int = 3):
    _patch_model_and_probs(monkeypatch, positive_probs)
    return solver.solve_recaptcha_grid_cnn(
        screenshot_png=_fake_screenshot(grid_size),
        instruction="Selecione todos os quadrados com carro",
        grid_rows=grid_size,
        grid_cols=grid_size,
        checkpoint_path=FAKE_CHECKPOINT,
    )


def test_rescues_borderline_cell_near_average_of_confident_positives(monkeypatch):
    # confiantes: índices 0 (0.9) e 4 (0.8) -> média 0.85 -> limiar 0.425.
    # índice 6 (0.45) fica abaixo do corte fixo de 0.5 mas acima do limiar
    # resgatado -- deve ser incluído junto dos dois confiantes.
    probs = [0.9, 0.05, 0.05, 0.05, 0.8, 0.05, 0.45, 0.05, 0.05]
    assert _solve(monkeypatch, probs) == [0, 4, 6]


def test_does_not_rescue_cells_far_below_the_confident_average(monkeypatch):
    # só uma célula confiante (0.9) -> limiar 0.45; nenhuma outra célula
    # (todas em 0.1) chega perto -- resultado igual ao corte fixo de 0.5.
    probs = [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    assert _solve(monkeypatch, probs) == [0]


def test_returns_empty_without_any_confident_cell(monkeypatch):
    # nenhuma célula passa de 0.5 -- sem âncora pra calcular média, não há
    # o que resgatar (mesmo comportamento de antes da mudança).
    probs = [0.3] * 9
    assert _solve(monkeypatch, probs) == []
