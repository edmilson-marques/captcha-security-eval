"""Testes de sanidade do config/targets.yaml."""
from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "targets.yaml"


def _load_config():
    return yaml.safe_load(CONFIG_PATH.read_text())


def test_targets_yaml_has_three_targets():
    config = _load_config()
    assert set(config["targets"].keys()) == {"recaptcha", "hcaptcha", "turnstile"}


def test_recaptcha_and_hcaptcha_have_official_demo_urls():
    config = _load_config()
    assert config["targets"]["recaptcha"]["demo_url"].startswith(
        "https://www.google.com/recaptcha"
    )
    assert config["targets"]["hcaptcha"]["demo_url"].startswith(
        "https://accounts.hcaptcha.com"
    )


def test_turnstile_has_no_fixed_official_demo_documented():
    config = _load_config()
    assert config["targets"]["turnstile"]["demo_url"] is None
