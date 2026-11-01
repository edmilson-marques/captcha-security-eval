"""Testes da persistência de amostras (harness.storage) — Atividade 1."""
from __future__ import annotations

import json

from harness import storage


def test_save_sample_writes_expected_files(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DATA_ROOT", tmp_path)

    sample_dir = storage.save_sample(
        target="recaptcha",
        source_url="https://www.google.com/recaptcha/api2/demo",
        screenshot_png=b"\x89PNG\r\n\x1a\n-fake-",
        page_html="<html></html>",
        label="exemplo",
    )

    assert sample_dir.parent.name == "recaptcha"
    assert (sample_dir / "challenge.png").read_bytes().startswith(b"\x89PNG")
    assert (sample_dir / "page.html").read_text() == "<html></html>"

    metadata = json.loads((sample_dir / "metadata.json").read_text())
    assert metadata["target"] == "recaptcha"
    assert metadata["label"] == "exemplo"
    assert metadata["source_url"] == "https://www.google.com/recaptcha/api2/demo"


def test_save_sample_without_label_defaults_to_none(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DATA_ROOT", tmp_path)

    sample_dir = storage.save_sample(
        target="hcaptcha",
        source_url="https://accounts.hcaptcha.com/demo",
        screenshot_png=b"fake",
        page_html="<html></html>",
    )

    metadata = json.loads((sample_dir / "metadata.json").read_text())
    assert metadata["label"] is None
