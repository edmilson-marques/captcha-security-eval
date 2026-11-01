"""Testes do solver de visão — cliente OpenAI mockado, sem chamadas reais à API."""

from __future__ import annotations

from unittest.mock import MagicMock

from attacks.llm_vision_solver import (
    PuzzlePieceAnswer,
    RecaptchaGridAnswer,
    solve_puzzle_piece,
    solve_recaptcha_grid,
)


def _fake_client(output_parsed):
    client = MagicMock()
    client.responses.parse.return_value = MagicMock(output_parsed=output_parsed)
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

    call_kwargs = client.responses.parse.call_args.kwargs
    assert call_kwargs["model"] == "gpt-6.1-terra"
    assert call_kwargs["text_format"] is RecaptchaGridAnswer

    content = call_kwargs["input"][0]["content"]
    assert content[0]["type"] == "input_text"
    assert "ônibus" in content[0]["text"]
    assert "3x3" in content[0]["text"]
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"].startswith("data:image/png;base64,")


def test_solve_puzzle_piece_returns_parsed_answer_and_builds_correct_request():
    expected = PuzzlePieceAnswer(target_x=120, target_y=80, confidence="high")
    client = _fake_client(expected)

    result = solve_puzzle_piece(
        screenshot_png=b"fake-png-bytes",
        instruction="Arraste o elemento para o local onde se encaixa",
        client=client,
    )

    assert result == expected

    call_kwargs = client.responses.parse.call_args.kwargs
    assert call_kwargs["text_format"] is PuzzlePieceAnswer

    content = call_kwargs["input"][0]["content"]
    assert "arrastar-e-encaixar" in content[0]["text"]
    assert content[1]["type"] == "input_image"
