"""Testes da votação adaptativa do grid do reCAPTCHA — cliente mockado, sem chamada real."""

from __future__ import annotations

from unittest.mock import MagicMock

from attacks.llm_vision_solver import RecaptchaGridAnswer
from attacks.recaptcha_adaptive_vote import solve_recaptcha_grid_adaptive_vote


def test_clear_majority_does_not_escalate():
    answers = [
        RecaptchaGridAnswer(tile_indices=[1, 2], reasoning="voto 1"),
        RecaptchaGridAnswer(tile_indices=[1, 2], reasoning="voto 2"),
        RecaptchaGridAnswer(tile_indices=[1], reasoning="voto 3"),
    ]
    client = MagicMock()
    client.responses.parse.side_effect = [MagicMock(output_parsed=a) for a in answers]

    result = solve_recaptcha_grid_adaptive_vote(
        screenshot_png=b"fake-png-bytes",
        instruction="Selecione todas as imagens com ônibus",
        grid_rows=3,
        grid_cols=3,
        n_votes=3,
        max_extra_votes=3,
        client=client,
    )

    assert result.tile_indices == [1, 2]
    assert client.responses.parse.call_count == 3  # não escalou


def test_refusal_does_not_escalate():
    refusal = RecaptchaGridAnswer(tile_indices=[], reasoning="Não posso ajudar.")
    client = MagicMock()
    client.responses.parse.side_effect = [MagicMock(output_parsed=refusal) for _ in range(3)]

    result = solve_recaptcha_grid_adaptive_vote(
        screenshot_png=b"fake-png-bytes",
        instruction="Selecione todas as imagens com ônibus",
        grid_rows=3,
        grid_cols=3,
        n_votes=3,
        max_extra_votes=3,
        client=client,
    )

    assert result.tile_indices == []
    assert client.responses.parse.call_count == 3  # não gastou voto extra numa recusa


def test_hung_vote_escalates_until_majority_found():
    # 1ª rodada (n_votes=2): [1] e [2] -> empate, sem recusa (há votos, nenhuma maioria).
    # Voto extra: [1] -> agora 1 tem 2/3, vira maioria.
    answers = [
        RecaptchaGridAnswer(tile_indices=[1], reasoning="voto 1"),
        RecaptchaGridAnswer(tile_indices=[2], reasoning="voto 2"),
        RecaptchaGridAnswer(tile_indices=[1], reasoning="voto extra"),
    ]
    client = MagicMock()
    client.responses.parse.side_effect = [MagicMock(output_parsed=a) for a in answers]

    result = solve_recaptcha_grid_adaptive_vote(
        screenshot_png=b"fake-png-bytes",
        instruction="Selecione todas as imagens com ônibus",
        grid_rows=3,
        grid_cols=3,
        n_votes=2,
        max_extra_votes=3,
        client=client,
    )

    assert result.tile_indices == [1]
    assert client.responses.parse.call_count == 3  # 2 iniciais + 1 extra, parou ao resolver


def test_hung_vote_gives_up_after_max_extra_votes():
    # Toda rodada discorda -- nunca atinge maioria, mesmo depois do limite de votos extras.
    answers = [
        RecaptchaGridAnswer(tile_indices=[1], reasoning="voto 1"),
        RecaptchaGridAnswer(tile_indices=[2], reasoning="voto 2"),
        RecaptchaGridAnswer(tile_indices=[3], reasoning="voto extra 1"),
        RecaptchaGridAnswer(tile_indices=[4], reasoning="voto extra 2"),
    ]
    client = MagicMock()
    client.responses.parse.side_effect = [MagicMock(output_parsed=a) for a in answers]

    result = solve_recaptcha_grid_adaptive_vote(
        screenshot_png=b"fake-png-bytes",
        instruction="Selecione todas as imagens com ônibus",
        grid_rows=3,
        grid_cols=3,
        n_votes=2,
        max_extra_votes=2,
        client=client,
    )

    assert result.tile_indices == []
    assert client.responses.parse.call_count == 4  # 2 iniciais + 2 extras (limite), desistiu


def test_max_extra_votes_zero_matches_plain_majority_vote():
    answers = [
        RecaptchaGridAnswer(tile_indices=[1], reasoning="voto 1"),
        RecaptchaGridAnswer(tile_indices=[2], reasoning="voto 2"),
    ]
    client = MagicMock()
    client.responses.parse.side_effect = [MagicMock(output_parsed=a) for a in answers]

    result = solve_recaptcha_grid_adaptive_vote(
        screenshot_png=b"fake-png-bytes",
        instruction="Selecione todas as imagens com ônibus",
        grid_rows=3,
        grid_cols=3,
        n_votes=2,
        max_extra_votes=0,
        client=client,
    )

    assert result.tile_indices == []
    assert client.responses.parse.call_count == 2
