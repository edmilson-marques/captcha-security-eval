"""Testes das técnicas auxiliares de evasão (src/evasion) — Atividade 3."""

from __future__ import annotations

import pytest

from evasion.cookie_manager import CookieManager
from evasion.human_behavior import bezier_mouse_path, canvas_offset_path, human_delay
from evasion.proxy_rotation import ProxyRotator
from evasion.user_agent_pool import (
    DEFAULT_USER_AGENTS,
    DESKTOP_USER_AGENTS,
    random_fingerprint_profile,
    random_user_agent,
)

# --------------------------------------------------------------------------
# ProxyRotator
# --------------------------------------------------------------------------


def test_proxy_rotator_never_repeats_consecutively():
    proxies = ["http://p1:8080", "http://p2:8080", "http://p3:8080"]
    rotator = ProxyRotator(proxies)

    previous = rotator.next()
    assert previous in proxies
    for _ in range(20):
        current = rotator.next()
        assert current in proxies
        assert current != previous
        previous = current


def test_proxy_rotator_single_proxy_always_same():
    rotator = ProxyRotator(["http://only:8080"])
    for _ in range(5):
        assert rotator.next() == "http://only:8080"


def test_proxy_rotator_empty_list_raises():
    with pytest.raises(ValueError):
        ProxyRotator([])


# --------------------------------------------------------------------------
# user_agent_pool
# --------------------------------------------------------------------------


def test_random_user_agent_returns_string_from_default_pool():
    for _ in range(20):
        ua = random_user_agent()
        assert isinstance(ua, str)
        assert ua in DEFAULT_USER_AGENTS


def test_random_user_agent_uses_custom_pool():
    pool = ["custom-agent-1", "custom-agent-2"]
    for _ in range(10):
        assert random_user_agent(pool) in pool


def test_random_user_agent_empty_pool_raises():
    with pytest.raises(ValueError):
        random_user_agent([])


def test_random_fingerprint_profile_shape():
    for _ in range(20):
        profile = random_fingerprint_profile()
        assert isinstance(profile, dict)
        assert set(profile.keys()) == {"user_agent", "viewport", "locale"}
        assert isinstance(profile["user_agent"], str)
        assert profile["user_agent"] in DEFAULT_USER_AGENTS

        viewport = profile["viewport"]
        assert isinstance(viewport, tuple)
        assert len(viewport) == 2
        width, height = viewport
        assert isinstance(width, int)
        assert isinstance(height, int)

        assert isinstance(profile["locale"], str)
        assert profile["locale"] in {"pt-BR", "en-US"}


def test_random_fingerprint_profile_empty_pool_raises():
    with pytest.raises(ValueError):
        random_fingerprint_profile([])


def test_desktop_user_agents_excludes_mobile_entries():
    assert len(DESKTOP_USER_AGENTS) > 0
    assert len(DESKTOP_USER_AGENTS) < len(DEFAULT_USER_AGENTS)
    for ua in DESKTOP_USER_AGENTS:
        assert "Mobile" not in ua
        assert "Android" not in ua
        assert "iPhone" not in ua


# --------------------------------------------------------------------------
# CookieManager
# --------------------------------------------------------------------------


def test_new_identity_returns_distinct_strings():
    manager = CookieManager()
    identities = {manager.new_identity() for _ in range(50)}
    assert len(identities) == 50
    assert all(isinstance(i, str) for i in identities)


def test_save_and_load_cookies_round_trip():
    manager = CookieManager()
    identity = manager.new_identity()
    cookies = [{"name": "sid", "value": "abc123"}, {"name": "theme", "value": "dark"}]

    manager.save_cookies(identity, cookies)
    loaded = manager.load_cookies(identity)

    assert loaded == cookies


def test_load_cookies_unknown_identity_returns_empty_list():
    manager = CookieManager()
    assert manager.load_cookies("nao-existe") == []


def test_clear_removes_identity():
    manager = CookieManager()
    identity = manager.new_identity()
    manager.save_cookies(identity, [{"name": "sid", "value": "abc123"}])

    manager.clear(identity)

    assert manager.load_cookies(identity) == []
    # Chamar clear de novo (ou load) não deve levantar erro.
    manager.clear(identity)
    assert manager.load_cookies(identity) == []


# --------------------------------------------------------------------------
# human_behavior.bezier_mouse_path
# --------------------------------------------------------------------------


def test_bezier_mouse_path_shape_and_endpoints():
    start = (0.0, 0.0)
    end = (100.0, 50.0)
    path = bezier_mouse_path(start, end, n_points=20)

    assert len(path) == 20
    assert path[0] == pytest.approx(start)
    assert path[-1] == pytest.approx(end)


def test_bezier_mouse_path_is_random_with_jitter():
    start = (0.0, 0.0)
    end = (200.0, 200.0)

    path_a = bezier_mouse_path(start, end, n_points=10, jitter=15.0)
    path_b = bezier_mouse_path(start, end, n_points=10, jitter=15.0)

    # Extremidades sempre coincidem...
    assert path_a[0] == pytest.approx(path_b[0])
    assert path_a[-1] == pytest.approx(path_b[-1])
    # ...mas o meio do caminho difere entre chamadas (jitter real, não determinístico).
    assert path_a[len(path_a) // 2] != pytest.approx(path_b[len(path_b) // 2])


def test_bezier_mouse_path_zero_jitter_is_deterministic():
    start = (0.0, 0.0)
    end = (200.0, 200.0)

    path_a = bezier_mouse_path(start, end, n_points=10, jitter=0.0)
    path_b = bezier_mouse_path(start, end, n_points=10, jitter=0.0)

    assert len(path_a) == len(path_b)
    for point_a, point_b in zip(path_a, path_b, strict=True):
        assert point_a == pytest.approx(point_b)


# --------------------------------------------------------------------------
# human_behavior.human_delay
# --------------------------------------------------------------------------


def test_human_delay_respects_minimum_and_type():
    for _ in range(200):
        delay = human_delay(mean=0.3, std=0.1, minimum=0.05)
        assert isinstance(delay, float)
        assert delay >= 0.05


def test_human_delay_with_aggressive_minimum():
    # Média/desvio que produziriam muitos valores negativos sem o truncamento.
    for _ in range(200):
        delay = human_delay(mean=0.0, std=1.0, minimum=0.1)
        assert delay >= 0.1


# --------------------------------------------------------------------------
# human_behavior.canvas_offset_path
# --------------------------------------------------------------------------


def test_canvas_offset_path_endpoints_are_center_relative():
    canvas_size = (500, 470)
    start_point = (250.0, 235.0)  # centro exato do canvas
    end_point = (100.0, 50.0)

    path = canvas_offset_path(canvas_size, start_point, end_point, n_points=10, jitter=0.0)

    assert len(path) == 10
    # start_point é o centro -> offset (0, 0).
    assert path[0] == pytest.approx((0.0, 0.0))
    # end_point convertido para offset relativo ao centro (250, 235).
    assert path[-1] == pytest.approx((100.0 - 250.0, 50.0 - 235.0))


def test_canvas_offset_path_matches_plain_bezier_after_conversion():
    canvas_size = (500, 470)
    start_point = (300.0, 400.0)
    end_point = (50.0, 50.0)

    path = canvas_offset_path(canvas_size, start_point, end_point, n_points=8, jitter=0.0)
    expected_start_offset = (300.0 - 250.0, 400.0 - 235.0)
    expected_end_offset = (50.0 - 250.0, 50.0 - 235.0)
    expected = bezier_mouse_path(expected_start_offset, expected_end_offset, n_points=8, jitter=0.0)

    assert len(path) == len(expected)
    for actual_point, expected_point in zip(path, expected, strict=True):
        assert actual_point == pytest.approx(expected_point)
