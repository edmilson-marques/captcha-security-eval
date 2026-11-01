"""Testes de agregação e ranking de técnicas (analysis.compare) — Atividade 7."""

from __future__ import annotations

from analysis.compare import load_results, rank_techniques
from experiments.metrics import AttemptResult


def _make_result(
    technique: str,
    outcome: str,
    *,
    attempt_id: str,
    duration: float = 1.0,
    confidence: float | None = None,
    target: str = "recaptcha",
) -> AttemptResult:
    return AttemptResult(
        target=target,
        technique=technique,
        evasion_profile="none",
        attempt_id=attempt_id,
        started_at="2026-10-01T10:00:00+00:00",
        finished_at="2026-10-01T10:00:01+00:00",
        outcome=outcome,
        duration_seconds=duration,
        confidence=confidence,
    )


def _populate_recaptcha_results(base_dir) -> None:
    """Cria base_dir/recaptcha/ com 2 arquivos .jsonl para 2 técnicas.

    Técnica A ("classical_cv_ocr"): 3 sucessos + 1 falha -> success_rate = 0.75
    Técnica B ("image_grid_cnn"): 1 sucesso + 3 falhas -> success_rate = 0.25
    """
    target_dir = base_dir / "recaptcha"
    target_dir.mkdir(parents=True)

    technique_a_results = [
        _make_result("classical_cv_ocr", "success", attempt_id="a1", duration=1.0, confidence=0.9),
        _make_result("classical_cv_ocr", "success", attempt_id="a2", duration=2.0, confidence=0.8),
        _make_result("classical_cv_ocr", "success", attempt_id="a3", duration=3.0, confidence=None),
        _make_result("classical_cv_ocr", "failure", attempt_id="a4", duration=4.0, confidence=0.5),
    ]
    technique_b_results = [
        _make_result("image_grid_cnn", "success", attempt_id="b1", duration=5.0, confidence=0.6),
        _make_result("image_grid_cnn", "failure", attempt_id="b2", duration=6.0),
        _make_result("image_grid_cnn", "failure", attempt_id="b3", duration=7.0),
        _make_result("image_grid_cnn", "failure", attempt_id="b4", duration=8.0),
    ]

    file_a = target_dir / "classical_cv_ocr__none__20261001T100000000000.jsonl"
    file_a.write_text(
        "\n".join(r.to_json_line() for r in technique_a_results) + "\n",
        encoding="utf-8",
    )

    file_b = target_dir / "image_grid_cnn__none__20261001T100100000000.jsonl"
    file_b.write_text(
        "\n".join(r.to_json_line() for r in technique_b_results) + "\n",
        encoding="utf-8",
    )

    return technique_a_results, technique_b_results


def test_load_results_with_target_filter(tmp_path):
    technique_a_results, technique_b_results = _populate_recaptcha_results(tmp_path)

    loaded = load_results(tmp_path, target="recaptcha")

    assert len(loaded) == len(technique_a_results) + len(technique_b_results)
    assert all(isinstance(r, AttemptResult) for r in loaded)
    loaded_ids = {r.attempt_id for r in loaded}
    expected_ids = {r.attempt_id for r in technique_a_results + technique_b_results}
    assert loaded_ids == expected_ids


def test_load_results_without_target_scans_all_subdirs(tmp_path):
    technique_a_results, technique_b_results = _populate_recaptcha_results(tmp_path)

    # Popula também um alvo "hcaptcha" distinto para confirmar que, sem
    # filtro de target, load_results combina todos os subdiretórios.
    hcaptcha_dir = tmp_path / "hcaptcha"
    hcaptcha_dir.mkdir()
    hcaptcha_result = _make_result(
        "behavioral_probe", "success", attempt_id="h1", target="hcaptcha"
    )
    (hcaptcha_dir / "behavioral_probe__none__20261001T100200000000.jsonl").write_text(
        hcaptcha_result.to_json_line() + "\n", encoding="utf-8"
    )

    loaded = load_results(tmp_path, target=None)

    expected_total = len(technique_a_results) + len(technique_b_results) + 1
    assert len(loaded) == expected_total
    loaded_ids = {r.attempt_id for r in loaded}
    assert "h1" in loaded_ids


def test_load_results_skips_blank_lines(tmp_path):
    target_dir = tmp_path / "recaptcha"
    target_dir.mkdir()
    result = _make_result("classical_cv_ocr", "success", attempt_id="x1")
    (target_dir / "classical_cv_ocr__none__ts.jsonl").write_text(
        "\n" + result.to_json_line() + "\n\n", encoding="utf-8"
    )

    loaded = load_results(tmp_path, target="recaptcha")

    assert len(loaded) == 1
    assert loaded[0].attempt_id == "x1"


def test_load_results_nonexistent_dir_returns_empty_list(tmp_path):
    missing_dir = tmp_path / "does_not_exist"

    assert load_results(missing_dir) == []
    assert load_results(missing_dir, target="recaptcha") == []


def test_load_results_nonexistent_target_subdir_returns_empty_list(tmp_path):
    _populate_recaptcha_results(tmp_path)

    assert load_results(tmp_path, target="turnstile") == []


def test_rank_techniques_orders_by_success_rate_desc(tmp_path):
    _populate_recaptcha_results(tmp_path)

    ranking = rank_techniques(tmp_path, "recaptcha")

    assert [row["technique"] for row in ranking] == ["classical_cv_ocr", "image_grid_cnn"]


def test_rank_techniques_values_match_manual_calculation(tmp_path):
    _populate_recaptcha_results(tmp_path)

    ranking = rank_techniques(tmp_path, "recaptcha")
    by_technique = {row["technique"]: row for row in ranking}

    # Técnica A: 3 sucessos de 4 -> 0.75; duração média (1+2+3+4)/4 = 2.5;
    # confiança média entre as 3 não-None: (0.9+0.8+0.5)/3 = 0.7333...
    technique_a = by_technique["classical_cv_ocr"]
    assert technique_a["n_attempts"] == 4
    assert technique_a["success_rate"] == 0.75
    assert technique_a["mean_duration_seconds"] == 2.5
    assert technique_a["mean_confidence"] == (0.9 + 0.8 + 0.5) / 3

    # Técnica B: 1 sucesso de 4 -> 0.25; duração média (5+6+7+8)/4 = 6.5;
    # confiança média entre a única não-None: 0.6.
    technique_b = by_technique["image_grid_cnn"]
    assert technique_b["n_attempts"] == 4
    assert technique_b["success_rate"] == 0.25
    assert technique_b["mean_duration_seconds"] == 6.5
    assert technique_b["mean_confidence"] == 0.6


def test_rank_techniques_mean_confidence_is_none_when_no_confidence_values(tmp_path):
    target_dir = tmp_path / "turnstile"
    target_dir.mkdir()
    results = [
        _make_result("behavioral_probe", "success", attempt_id="t1", target="turnstile"),
        _make_result("behavioral_probe", "failure", attempt_id="t2", target="turnstile"),
    ]
    (target_dir / "behavioral_probe__none__ts.jsonl").write_text(
        "\n".join(r.to_json_line() for r in results) + "\n", encoding="utf-8"
    )

    ranking = rank_techniques(tmp_path, "turnstile")

    assert len(ranking) == 1
    assert ranking[0]["mean_confidence"] is None


def test_rank_techniques_empty_for_target_without_results(tmp_path):
    _populate_recaptcha_results(tmp_path)

    assert rank_techniques(tmp_path, "turnstile") == []


def test_rank_techniques_empty_for_nonexistent_results_dir(tmp_path):
    missing_dir = tmp_path / "does_not_exist"

    assert rank_techniques(missing_dir, "recaptcha") == []
