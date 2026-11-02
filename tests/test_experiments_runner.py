"""Testes do orquestrador de experimentos (experiments.runner) — Atividades 4-6.

Nenhum teste aqui faz rede ou abre navegador: `attempt_fn` é sempre uma
closure falsa e determinística, injetada no runner.
"""

from __future__ import annotations

import pytest

from attacks.turnstile_behavioral import run_pass_rate_probe
from experiments.metrics import AttemptResult
from experiments.runner import run_experiment


def test_run_experiment_persists_and_returns_results(tmp_path):
    def attempt_fn() -> dict:
        return {"outcome": "success", "confidence": 0.9}

    results = run_experiment(
        target="recaptcha",
        technique="classical_cv_ocr",
        evasion_profile="none",
        n_attempts=5,
        attempt_fn=attempt_fn,
        results_dir=tmp_path,
    )

    assert len(results) == 5
    assert all(isinstance(r, AttemptResult) for r in results)
    assert all(r.outcome == "success" for r in results)
    assert all(r.confidence == 0.9 for r in results)
    assert all(r.target == "recaptcha" for r in results)
    assert all(r.technique == "classical_cv_ocr" for r in results)
    assert all(r.evasion_profile == "none" for r in results)

    # attempt_ids devem ser únicos.
    assert len({r.attempt_id for r in results}) == 5

    target_dir = tmp_path / "recaptcha"
    assert target_dir.is_dir()

    jsonl_files = list(target_dir.glob("classical_cv_ocr__none__*.jsonl"))
    assert len(jsonl_files) == 1

    lines = jsonl_files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5

    restored = [AttemptResult.from_json_line(line) for line in lines]
    assert restored == results


def test_run_experiment_handles_exceptions_per_attempt(tmp_path):
    calls = {"n": 0}

    def attempt_fn() -> dict:
        idx = calls["n"]
        calls["n"] += 1
        if idx % 2 == 0:
            raise RuntimeError(f"falha simulada na tentativa {idx}")
        return {"outcome": "success"}

    results = run_experiment(
        target="hcaptcha",
        technique="image_grid_cnn",
        evasion_profile="stealth_v1",
        n_attempts=5,
        attempt_fn=attempt_fn,
        results_dir=tmp_path,
    )

    assert len(results) == 5

    failed = [r for r in results if r.outcome == "failure"]
    succeeded = [r for r in results if r.outcome == "success"]

    # índices 0, 2, 4 levantam RuntimeError -> 3 falhas; índices 1, 3 -> sucesso.
    assert len(failed) == 3
    assert len(succeeded) == 2

    for r in failed:
        assert "error" in r.extra
        assert "falha simulada" in r.extra["error"]


def test_run_experiment_invalid_outcome_raises_value_error(tmp_path):
    def attempt_fn() -> dict:
        return {"outcome": "nao_existe"}

    with pytest.raises(ValueError):
        run_experiment(
            target="turnstile",
            technique="behavioral_probe",
            evasion_profile="none",
            n_attempts=1,
            attempt_fn=attempt_fn,
            results_dir=tmp_path,
        )


def test_run_experiment_same_round_shares_one_file(tmp_path):
    def attempt_fn() -> dict:
        return {"outcome": "blocked"}

    run_experiment(
        target="recaptcha",
        technique="classical_cv_ocr",
        evasion_profile="none",
        n_attempts=3,
        attempt_fn=attempt_fn,
        results_dir=tmp_path,
    )

    jsonl_files = list((tmp_path / "recaptcha").glob("*.jsonl"))
    assert len(jsonl_files) == 1
    lines = jsonl_files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3


def test_run_experiment_pauses_after_every_n_attempts(tmp_path):
    def attempt_fn() -> dict:
        return {"outcome": "success"}

    sleep_calls = []

    run_experiment(
        target="hcaptcha",
        technique="classical_cv",
        evasion_profile="none",
        n_attempts=25,
        attempt_fn=attempt_fn,
        results_dir=tmp_path,
        pause_every=10,
        pause_seconds=5.0,
        sleep_fn=sleep_calls.append,
    )

    # 25 tentativas, pausa a cada 10 completadas -> pausa depois da 10ª e da
    # 20ª, mas NÃO depois da 25ª (última tentativa, nada a esperar).
    assert sleep_calls == [5.0, 5.0]


def test_run_experiment_pause_every_none_never_pauses(tmp_path):
    def attempt_fn() -> dict:
        return {"outcome": "success"}

    sleep_calls = []

    run_experiment(
        target="hcaptcha",
        technique="classical_cv",
        evasion_profile="none",
        n_attempts=30,
        attempt_fn=attempt_fn,
        results_dir=tmp_path,
        pause_every=None,
        sleep_fn=sleep_calls.append,
    )

    assert sleep_calls == []


def test_run_experiment_pause_applies_even_after_failed_attempt(tmp_path):
    def attempt_fn() -> dict:
        raise RuntimeError("falha simulada")

    sleep_calls = []

    run_experiment(
        target="hcaptcha",
        technique="classical_cv",
        evasion_profile="none",
        n_attempts=10,
        attempt_fn=attempt_fn,
        results_dir=tmp_path,
        pause_every=5,
        pause_seconds=2.0,
        sleep_fn=sleep_calls.append,
    )

    # Pausa conta tentativas COMPLETADAS (sucesso ou falha), não só sucessos.
    assert sleep_calls == [2.0]


def test_run_experiment_on_pause_called_before_each_pause(tmp_path):
    def attempt_fn() -> dict:
        return {"outcome": "success"}

    pause_events = []

    run_experiment(
        target="hcaptcha",
        technique="classical_cv",
        evasion_profile="none",
        n_attempts=22,
        attempt_fn=attempt_fn,
        results_dir=tmp_path,
        pause_every=10,
        sleep_fn=lambda _seconds: None,
        on_pause=lambda done, total: pause_events.append((done, total)),
    )

    assert pause_events == [(10, 22), (20, 22)]


def test_run_experiment_n_attempts_below_pause_every_never_pauses(tmp_path):
    def attempt_fn() -> dict:
        return {"outcome": "success"}

    sleep_calls = []

    run_experiment(
        target="hcaptcha",
        technique="classical_cv",
        evasion_profile="none",
        n_attempts=3,
        attempt_fn=attempt_fn,
        results_dir=tmp_path,
        pause_every=10,
        sleep_fn=sleep_calls.append,
    )

    assert sleep_calls == []


def test_run_pass_rate_probe_without_navigate_fn_does_not_raise(tmp_path):
    results = run_pass_rate_probe(
        evasion_profile="none",
        n_attempts=3,
        results_dir=tmp_path,
    )

    assert len(results) == 3
    assert all(r.outcome == "failure" for r in results)
    assert all(r.target == "turnstile" for r in results)
    assert all(r.technique == "behavioral_probe" for r in results)
    assert all("Integração real" in r.extra.get("error", "") for r in results)
