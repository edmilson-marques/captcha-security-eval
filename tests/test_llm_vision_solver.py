"""Testes do solver de visão — cliente OpenAI mockado, sem chamadas reais à API."""

from __future__ import annotations

from unittest.mock import MagicMock

from attacks.llm_vision_solver import (
    CanvasPoint,
    CanvasPointsAnswer,
    DragPuzzleAnswer,
    RecaptchaGridAnswer,
    _strip_ui_instructions,
    solve_canvas_points,
    solve_puzzle_piece,
    solve_recaptcha_grid,
    solve_recaptcha_grid_by_majority_vote,
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


def test_strip_ui_instructions_removes_button_instructions_only():
    assert (
        _strip_ui_instructions(
            'Selecione todas as imagens com\num ônibus\nClique em "Confirmar" quando não houver mais nenhuma.'
        )
        == "Selecione todas as imagens com\num ônibus"
    )
    assert (
        _strip_ui_instructions(
            'Selecione todos os quadrados com motocicletas\nSe não houver nenhum, clique em "Pular"'
        )
        == "Selecione todos os quadrados com motocicletas"
    )
    assert _strip_ui_instructions("Selecione todas as imagens com bicicletas") == (
        "Selecione todas as imagens com bicicletas"
    )


def test_solve_recaptcha_grid_strips_ui_instructions_from_prompt():
    expected = RecaptchaGridAnswer(tile_indices=[], reasoning="nenhum")
    client = _fake_client(expected)

    solve_recaptcha_grid(
        screenshot_png=b"fake-png-bytes",
        instruction='Selecione todas as imagens com\num ônibus\nClique em "Confirmar" quando não houver mais nenhuma.',
        grid_rows=3,
        grid_cols=3,
        client=client,
    )

    prompt_text = client.responses.parse.call_args.kwargs["input"][0]["content"][0]["text"]
    assert "Confirmar" not in prompt_text
    assert "ônibus" in prompt_text


def test_solve_puzzle_piece_returns_parsed_answer_and_builds_correct_request():
    expected = DragPuzzleAnswer(
        piece_x=40, piece_y=300, target_x=120, target_y=80, confidence="high"
    )
    client = _fake_client(expected)

    result = solve_puzzle_piece(
        screenshot_png=b"fake-png-bytes",
        instruction="Arraste o elemento para o local onde se encaixa",
        client=client,
    )

    assert result == expected

    call_kwargs = client.responses.parse.call_args.kwargs
    assert call_kwargs["text_format"] is DragPuzzleAnswer

    content = call_kwargs["input"][0]["content"]
    assert "arrastar-e-encaixar" in content[0]["text"]
    assert content[1]["type"] == "input_image"


def test_solve_canvas_points_returns_parsed_answer_and_builds_correct_request():
    expected = CanvasPointsAnswer(
        points=[CanvasPoint(x=123, y=45)], reasoning="ícone diferente no canto superior"
    )
    client = _fake_client(expected)

    result = solve_canvas_points(
        screenshot_png=b"fake-png-bytes",
        instruction='Clique no ícone que é diferente\nClique em "Confirmar" quando terminar.',
        client=client,
    )

    assert result == expected

    call_kwargs = client.responses.parse.call_args.kwargs
    assert call_kwargs["text_format"] is CanvasPointsAnswer

    content = call_kwargs["input"][0]["content"]
    assert "ícone que é diferente" in content[0]["text"]
    assert "Confirmar" not in content[0]["text"]
    assert content[1]["type"] == "input_image"


def test_solve_recaptcha_grid_by_majority_vote_keeps_only_majority_indices():
    answers = [
        RecaptchaGridAnswer(tile_indices=[1, 2, 3], reasoning="voto 1"),
        RecaptchaGridAnswer(tile_indices=[1, 2, 9], reasoning="voto 2"),
        RecaptchaGridAnswer(tile_indices=[1, 4], reasoning="voto 3"),
    ]
    client = MagicMock()
    client.responses.parse.side_effect = [MagicMock(output_parsed=a) for a in answers]

    result = solve_recaptcha_grid_by_majority_vote(
        screenshot_png=b"fake-png-bytes",
        instruction="Selecione todas as imagens com ônibus",
        grid_rows=3,
        grid_cols=3,
        n_votes=3,
        client=client,
    )

    # índice 1: 3/3 votos, índice 2: 2/3 votos -> maioria (>1.5). Os demais (3, 9, 4) só 1 voto cada.
    assert result.tile_indices == [1, 2]
    assert client.responses.parse.call_count == 3


def test_solve_recaptcha_grid_by_majority_vote_tolerates_partial_failures():
    good_answer = RecaptchaGridAnswer(tile_indices=[5], reasoning="voto bom")
    client = MagicMock()
    client.responses.parse.side_effect = [
        RuntimeError("falha transitória da API"),
        MagicMock(output_parsed=good_answer),
        MagicMock(output_parsed=good_answer),
    ]

    result = solve_recaptcha_grid_by_majority_vote(
        screenshot_png=b"fake-png-bytes",
        instruction="Selecione todas as imagens com ônibus",
        grid_rows=3,
        grid_cols=3,
        n_votes=3,
        client=client,
    )

    assert result.tile_indices == [5]


def test_solve_recaptcha_grid_by_majority_vote_raises_if_all_calls_fail():
    client = MagicMock()
    client.responses.parse.side_effect = RuntimeError("API fora do ar")

    try:
        solve_recaptcha_grid_by_majority_vote(
            screenshot_png=b"fake-png-bytes",
            instruction="Selecione todas as imagens com ônibus",
            grid_rows=3,
            grid_cols=3,
            n_votes=2,
            client=client,
        )
        raise AssertionError("deveria ter propagado o erro")
    except RuntimeError:
        pass


def test_solve_recaptcha_grid_by_majority_vote_single_vote_matches_plain_call():
    expected = RecaptchaGridAnswer(tile_indices=[0, 1], reasoning="só um voto")
    client = _fake_client(expected)

    result = solve_recaptcha_grid_by_majority_vote(
        screenshot_png=b"fake-png-bytes",
        instruction="Selecione todas as imagens com ônibus",
        grid_rows=3,
        grid_cols=3,
        n_votes=1,
        client=client,
    )

    assert result.tile_indices == [0, 1]
    assert client.responses.parse.call_count == 1
