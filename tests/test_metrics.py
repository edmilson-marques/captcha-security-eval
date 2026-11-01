"""Testes das métricas de experimento (experiments.metrics) — Atividades 4-7."""

from __future__ import annotations

import numpy as np

from experiments.metrics import (
    AttemptResult,
    accuracy,
    bootstrap_confidence_interval,
    mean_duration,
    outcome_breakdown,
)


def _make_result(outcome: str, duration: float, **kwargs) -> AttemptResult:
    return AttemptResult(
        target="recaptcha",
        technique="classical_cv_ocr",
        evasion_profile="none",
        attempt_id="abc123",
        started_at="2026-10-01T10:00:00+00:00",
        finished_at="2026-10-01T10:00:01+00:00",
        outcome=outcome,
        duration_seconds=duration,
        **kwargs,
    )


def test_attempt_result_json_round_trip():
    result = AttemptResult(
        target="hcaptcha",
        technique="image_grid_cnn",
        evasion_profile="stealth_v1",
        attempt_id="deadbeef",
        started_at="2026-10-01T10:00:00+00:00",
        finished_at="2026-10-01T10:00:05.5+00:00",
        outcome="success",
        duration_seconds=5.5,
        confidence=0.87,
        sub_challenges=2,
        detected_as_bot=False,
        extra={"notes": "ok", "n": 1},
    )

    line = result.to_json_line()
    assert "\n" not in line

    restored = AttemptResult.from_json_line(line)
    assert restored == result


def test_attempt_result_json_round_trip_with_defaults():
    result = AttemptResult(
        target="turnstile",
        technique="behavioral_probe",
        evasion_profile="none",
        attempt_id="feedface",
        started_at="2026-10-01T10:00:00+00:00",
        finished_at="2026-10-01T10:00:01+00:00",
        outcome="failure",
        duration_seconds=1.0,
    )

    restored = AttemptResult.from_json_line(result.to_json_line())
    assert restored == result
    assert restored.confidence is None
    assert restored.sub_challenges == 0
    assert restored.detected_as_bot is None
    assert restored.extra == {}


def _sample_results() -> list[AttemptResult]:
    # 4 resultados: 2 sucessos, 1 falha, 1 bloqueio.
    return [
        _make_result("success", 1.0),
        _make_result("success", 3.0),
        _make_result("failure", 2.0),
        _make_result("blocked", 2.0),
    ]


def test_accuracy_computed_by_hand():
    results = _sample_results()
    assert accuracy(results) == 0.5  # 2 de 4


def test_accuracy_empty_list_is_zero():
    assert accuracy([]) == 0.0


def test_mean_duration_computed_by_hand():
    results = _sample_results()
    # (1.0 + 3.0 + 2.0 + 2.0) / 4 = 2.0
    assert mean_duration(results) == 2.0


def test_mean_duration_empty_list_is_zero():
    assert mean_duration([]) == 0.0


def test_outcome_breakdown_computed_by_hand():
    results = _sample_results()
    breakdown = outcome_breakdown(results)
    assert breakdown == {
        "success": 0.5,
        "failure": 0.25,
        "blocked": 0.25,
    }
    assert sum(breakdown.values()) == 1.0


def test_outcome_breakdown_empty_list_is_empty_dict():
    assert outcome_breakdown([]) == {}


def test_bootstrap_confidence_interval_is_plausible_and_deterministic():
    results = _sample_results()
    rng = np.random.default_rng(42)

    low, high = bootstrap_confidence_interval(
        results, accuracy, n_resamples=500, alpha=0.05, rng=rng
    )

    assert isinstance(low, float)
    assert isinstance(high, float)
    assert low <= high
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0


def test_bootstrap_confidence_interval_is_reproducible_with_same_seed():
    results = _sample_results()

    low1, high1 = bootstrap_confidence_interval(
        results, accuracy, n_resamples=300, rng=np.random.default_rng(7)
    )
    low2, high2 = bootstrap_confidence_interval(
        results, accuracy, n_resamples=300, rng=np.random.default_rng(7)
    )

    assert low1 == low2
    assert high1 == high2


def test_bootstrap_confidence_interval_empty_list_does_not_raise():
    low, high = bootstrap_confidence_interval([], accuracy, n_resamples=100)
    assert (low, high) == (0.0, 0.0)
