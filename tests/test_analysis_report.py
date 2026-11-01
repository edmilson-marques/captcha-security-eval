"""Testes de geração do relatório comparativo (analysis.report) — Atividade 9."""

from __future__ import annotations

import pandas as pd

from analysis.report import build_report
from experiments.metrics import AttemptResult


def _make_result(
    technique: str,
    outcome: str,
    *,
    attempt_id: str,
    target: str,
    duration: float = 1.0,
    confidence: float | None = None,
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


def _populate_recaptcha_results(results_dir) -> None:
    target_dir = results_dir / "recaptcha"
    target_dir.mkdir(parents=True)

    results_a = [
        _make_result(
            "classical_cv_ocr", "success", attempt_id=f"a{i}", target="recaptcha", confidence=0.9
        )
        for i in range(3)
    ] + [_make_result("classical_cv_ocr", "failure", attempt_id="a4", target="recaptcha")]
    results_b = [_make_result("image_grid_cnn", "success", attempt_id="b1", target="recaptcha")] + [
        _make_result("image_grid_cnn", "failure", attempt_id=f"b{i}", target="recaptcha")
        for i in range(2, 5)
    ]

    (target_dir / "classical_cv_ocr__none__ts.jsonl").write_text(
        "\n".join(r.to_json_line() for r in results_a) + "\n", encoding="utf-8"
    )
    (target_dir / "image_grid_cnn__none__ts.jsonl").write_text(
        "\n".join(r.to_json_line() for r in results_b) + "\n", encoding="utf-8"
    )


def test_build_report_generates_files_and_dataframe_for_target_with_data(tmp_path):
    results_dir = tmp_path / "results"
    output_dir = tmp_path / "report"
    _populate_recaptcha_results(results_dir)

    report = build_report(
        results_dir=results_dir,
        output_dir=output_dir,
        targets=["recaptcha", "hcaptcha", "turnstile"],
    )

    assert "recaptcha" in report
    assert isinstance(report["recaptcha"], pd.DataFrame)
    assert not report["recaptcha"].empty
    assert set(report["recaptcha"]["technique"]) == {"classical_cv_ocr", "image_grid_cnn"}

    tex_path = output_dir / "recaptcha_ranking.tex"
    png_path = output_dir / "recaptcha_success_rate.png"

    assert tex_path.is_file()
    assert tex_path.stat().st_size > 0

    assert png_path.is_file()
    assert png_path.stat().st_size > 0


def test_build_report_skips_targets_without_data(tmp_path):
    results_dir = tmp_path / "results"
    output_dir = tmp_path / "report"
    _populate_recaptcha_results(results_dir)

    report = build_report(
        results_dir=results_dir,
        output_dir=output_dir,
        targets=["recaptcha", "hcaptcha", "turnstile"],
    )

    assert "hcaptcha" not in report
    assert "turnstile" not in report

    assert not (output_dir / "hcaptcha_ranking.tex").exists()
    assert not (output_dir / "hcaptcha_success_rate.png").exists()
    assert not (output_dir / "turnstile_ranking.tex").exists()
    assert not (output_dir / "turnstile_success_rate.png").exists()


def test_build_report_creates_output_dir_if_missing(tmp_path):
    results_dir = tmp_path / "results"
    output_dir = tmp_path / "nested" / "report_dir"
    _populate_recaptcha_results(results_dir)

    assert not output_dir.exists()

    build_report(results_dir=results_dir, output_dir=output_dir, targets=["recaptcha"])

    assert output_dir.is_dir()
    assert (output_dir / "recaptcha_ranking.tex").is_file()


def test_build_report_returns_empty_dict_when_no_targets_have_data(tmp_path):
    results_dir = tmp_path / "results"
    output_dir = tmp_path / "report"
    results_dir.mkdir()

    report = build_report(results_dir=results_dir, output_dir=output_dir, targets=["recaptcha"])

    assert report == {}
