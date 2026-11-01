"""Métricas de experimento: acurácia, tempo, taxa de detecção (Atividades 4-7).

Define o contrato `AttemptResult` — uma tentativa individual de ataque contra
um CAPTCHA-alvo — e funções de agregação estatística sobre listas dele. Nada
aqui faz rede/IO além de (de)serialização JSONL; os valores de entrada vêm do
runner de experimentos (`src/experiments/runner.py`), que por sua vez é
alimentado por coletores/ataques/evasão reais (fora do escopo deste módulo).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field

import numpy as np

VALID_OUTCOMES = ("success", "failure", "blocked", "challenge_not_presented")


@dataclass
class AttemptResult:
    """Registro de uma única tentativa de ataque contra um CAPTCHA-alvo.

    Este dataclass é um contrato consumido por análises futuras (Atividades
    7-9) — não renomeie campos sem atualizar quem os lê.
    """

    target: str  # "recaptcha" | "hcaptcha" | "turnstile"
    technique: str  # ex. "classical_cv_ocr", "image_grid_cnn", "behavioral_probe"
    evasion_profile: str  # rótulo livre, ex. "none", "stealth_v1"
    attempt_id: str
    started_at: str  # ISO 8601
    finished_at: str  # ISO 8601
    outcome: str  # "success" | "failure" | "blocked" | "challenge_not_presented"
    duration_seconds: float
    confidence: float | None = None
    sub_challenges: int = 0
    detected_as_bot: bool | None = None
    extra: dict = field(default_factory=dict)

    def to_json_line(self) -> str:
        """Serializa este resultado como uma única linha JSON (formato JSONL)."""
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json_line(cls, line: str) -> AttemptResult:
        """Desserializa uma linha JSONL de volta para `AttemptResult`."""
        return cls(**json.loads(line))


def accuracy(results: Sequence[AttemptResult]) -> float:
    """Fração de tentativas com `outcome == "success"`.

    Retorna `0.0` para lista vazia (não lança exceção).
    """
    if not results:
        return 0.0
    return sum(1 for r in results if r.outcome == "success") / len(results)


def mean_duration(results: Sequence[AttemptResult]) -> float:
    """Média de `duration_seconds` entre as tentativas.

    Retorna `0.0` para lista vazia (não lança exceção).
    """
    if not results:
        return 0.0
    return sum(r.duration_seconds for r in results) / len(results)


def outcome_breakdown(results: Sequence[AttemptResult]) -> dict[str, float]:
    """Fração de cada valor distinto de `outcome` presente na lista.

    As frações somam 1.0 quando a lista não é vazia. Retorna `{}` para lista
    vazia (não lança exceção).
    """
    if not results:
        return {}
    counts: dict[str, int] = {}
    for r in results:
        counts[r.outcome] = counts.get(r.outcome, 0) + 1
    total = len(results)
    return {outcome: count / total for outcome, count in counts.items()}


def bootstrap_confidence_interval(
    results: Sequence[AttemptResult],
    metric_fn: Callable[[Sequence[AttemptResult]], float],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Intervalo de confiança de `metric_fn(results)` via bootstrap não-paramétrico.

    Reamostra `results` com reposição `n_resamples` vezes (mesmo tamanho da
    lista original), aplica `metric_fn` em cada reamostra e retorna os
    percentis `alpha/2` e `1 - alpha/2` das `n_resamples` estimativas, como
    `(low, high)`.

    `rng` é opcional (numpy `Generator`) para tornar o resultado determinístico
    em testes — ex. `numpy.random.default_rng(42)`. Se omitido, usa
    `numpy.random.default_rng()`.

    Para lista vazia, retorna `(0.0, 0.0)` sem lançar exceção (não há o que
    reamostrar).
    """
    if not results:
        return (0.0, 0.0)

    if rng is None:
        rng = np.random.default_rng()

    n = len(results)
    results_arr = np.asarray(results, dtype=object)
    estimates = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        resample = list(results_arr[indices])
        estimates[i] = metric_fn(resample)

    low, high = np.percentile(estimates, [100 * (alpha / 2), 100 * (1 - alpha / 2)])
    return (float(low), float(high))
