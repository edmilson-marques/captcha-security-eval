"""Testes do servidor HTTP local de teste do Cloudflare Turnstile (Atividade 1/6).

Só exercita o servidor HTTP (via `urllib.request`, sem Selenium/navegador) —
o comportamento real do widget (callbacks JS) é validado manualmente ao vivo,
documentado na docstring de `experiments.live.turnstile_behavioral_attempt`.
"""

from __future__ import annotations

import urllib.request

from harness.turnstile_test_server import TurnstileTestServer


def test_server_serves_page_with_given_sitekey():
    server = TurnstileTestServer("1x00000000000000000000AA")
    url = server.start()
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            body = response.read().decode("utf-8")
        assert response.status == 200
        assert 'data-sitekey="1x00000000000000000000AA"' in body
        assert "challenges.cloudflare.com/turnstile/v0/api.js" in body
    finally:
        server.stop()


def test_server_registers_official_turnstile_callbacks():
    server = TurnstileTestServer("2x00000000000000000000AB")
    url = server.start()
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            body = response.read().decode("utf-8")
        assert 'data-callback="__turnstileOnSuccess"' in body
        assert 'data-error-callback="__turnstileOnError"' in body
        assert 'data-expired-callback="__turnstileOnExpired"' in body
    finally:
        server.stop()


def test_server_as_context_manager_starts_and_stops():
    with (
        TurnstileTestServer("3x00000000000000000000FF") as url,
        urllib.request.urlopen(url, timeout=5) as response,
    ):
        assert response.status == 200

    # Depois do bloco `with`, o servidor deve estar desligado (a porta não
    # aceita mais conexões) — `urlopen` deve falhar.
    try:
        urllib.request.urlopen(url, timeout=2)
        raised = False
    except OSError:
        raised = True
    assert raised


def test_each_server_instance_uses_a_different_port():
    with (
        TurnstileTestServer("1x00000000000000000000AA") as url_a,
        TurnstileTestServer("1x00000000000000000000AA") as url_b,
    ):
        assert url_a != url_b
