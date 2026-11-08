"""Votação adaptativa pro grid do reCAPTCHA — ataca só a imprecisão visual residual.

`solve_recaptcha_grid_by_majority_vote` (ver `llm_vision_solver.py`) já
elimina a maior parte da imprecisão visual com `n_votes` fixo, mas deixa um
resíduo (~5% medido ao vivo) de casos em que os votos discordam sem
nenhum índice atingir maioria clara — um sinal de ambiguidade visual
pontual, diferente de recusa de segurança (que vota `tile_indices=[]` em
todas as rodadas). `solve_recaptcha_grid_adaptive_vote` reaproveita
`solve_recaptcha_grid` (chamada única) e só gasta chamadas extras quando
detecta esse padrão específico de "votação empatada" — não tenta reduzir
recusa de segurança (fora de escopo: reformular o prompt para contornar o
guardrail de segurança de outro modelo não é uma mudança legítima aqui).
"""

from __future__ import annotations

import collections

from openai import OpenAI

from attacks.llm_vision_solver import RecaptchaGridAnswer, solve_recaptcha_grid


def _tally(answers: list[RecaptchaGridAnswer]) -> tuple[list[int], bool]:
    """Retorna (índices vencedores, votação empatada sem recusa aparente)."""
    votes = collections.Counter(idx for answer in answers for idx in answer.tile_indices)
    majority_threshold = len(answers) / 2
    winning = sorted(idx for idx, count in votes.items() if count > majority_threshold)
    is_hung = bool(votes) and not winning  # há votos, mas nenhum atingiu maioria
    return winning, is_hung


def solve_recaptcha_grid_adaptive_vote(
    screenshot_png: bytes,
    instruction: str,
    grid_rows: int,
    grid_cols: int,
    n_votes: int = 3,
    max_extra_votes: int = 3,
    client: OpenAI | None = None,
) -> RecaptchaGridAnswer:
    """Vota `n_votes` vezes; se a votação empatar (sem recusa), soma até `max_extra_votes` votos.

    Recusa de segurança (todas as rodadas voltam `tile_indices=[]`) não
    aciona a escalada — mais chamadas não mudam uma recusa consistente, só
    custam dinheiro à toa. `max_extra_votes=0` desliga a escalada
    (equivalente a `solve_recaptcha_grid_by_majority_vote`).
    """
    if n_votes < 1:
        raise ValueError("n_votes deve ser >= 1.")

    answers: list[RecaptchaGridAnswer] = []
    last_error: Exception | None = None

    def _vote() -> None:
        nonlocal last_error
        try:
            answers.append(
                solve_recaptcha_grid(
                    screenshot_png=screenshot_png,
                    instruction=instruction,
                    grid_rows=grid_rows,
                    grid_cols=grid_cols,
                    client=client,
                )
            )
        except Exception as exc:  # noqa: BLE001 - qualquer rodada pode falhar isoladamente
            last_error = exc

    for _ in range(n_votes):
        _vote()

    if not answers:
        assert last_error is not None
        raise last_error

    winning, is_hung = _tally(answers)
    extra_used = 0
    while is_hung and extra_used < max_extra_votes:
        _vote()
        extra_used += 1
        winning, is_hung = _tally(answers)

    combined_reasoning = " | ".join(answer.reasoning for answer in answers)
    return RecaptchaGridAnswer(tile_indices=winning, reasoning=combined_reasoning)
