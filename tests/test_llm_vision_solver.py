"""Testes do solver de visão — cliente Anthropic mockado, sem chamadas reais à API."""

from __future__ import annotations

from unittest.mock import MagicMock

from attacks.llm_vision_solver import (
    PuzzlePieceAnswer,
    RecaptchaGridAnswer,
    solve_puzzle_piece,
    solve_recaptcha_grid,
)


def _fake_client(parsed_output):
    client = MagicMock()
    client.messages.parse.return_value = MagicMock(parsed_output=parsed_output)
    return client


def test_solve_recaptcha_grid_returns_parsed_answer_and_builds_correct_request():
    expected = RecaptchaGridAnswer(tile_indices=[0, 3, 7], reasoning="tiles com ônibus")
    client = _fake_client(expected)

    result = solve_recaptcha_grid(
        screenshot_png=b"fake-png-bytes",
        instruction="Selecione todas as imagens com um ônibus",
        grid_rows=3,
        grid_cols=3,
        client=client,
    )

    assert result == expected

    call_kwargs = client.messages.parse.call_args.kwargs
    assert call_kwargs["model"] == "claude-sonnet-5"
    assert call_kwargs["output_format"] is RecaptchaGridAnswer
    assert call_kwargs["thinking"] == {"type": "disabled"}

    content = call_kwargs["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["type"] == "base64"
    assert content[0]["source"]["media_type"] == "image/png"
    assert content[1]["type"] == "text"
    assert "ônibus" in content[1]["text"]
    assert "3x3" in content[1]["text"]


def test_solve_puzzle_piece_returns_parsed_answer_and_builds_correct_request():
    expected = PuzzlePieceAnswer(target_x=120, target_y=80, confidence="high")
    client = _fake_client(expected)

    result = solve_puzzle_piece(
        screenshot_png=b"fake-png-bytes",
        instruction="Arraste o elemento para o local onde se encaixa",
        client=client,
    )

    assert result == expected

    call_kwargs = client.messages.parse.call_args.kwargs
    assert call_kwargs["output_format"] is PuzzlePieceAnswer

    content = call_kwargs["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert "arrastar-e-encaixar" in content[1]["text"]
