"""Avaliação comportamental do Cloudflare Turnstile (Atividades 2 e 6).

Turnstile raramente apresenta desafio visual — não há imagem para "resolver".
Esta técnica dispara `experiments.runner` repetidamente, sob diferentes
perfis de evasão (ver `src/evasion/`), e mede se foi emitido token sem
desafio, desafio interativo, ou bloqueio. Ver nota em config/targets.yaml.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from experiments.metrics import AttemptResult
from experiments.runner import run_experiment


def _unavailable_navigate_fn() -> dict:
    """Função default usada quando nenhum `navigate_fn` é injetado.

    Ainda não há integração ao vivo com o navegador/Cloudflare Turnstile
    (isso é uma etapa humana deliberada posterior, fora do escopo desta
    tarefa). `experiments.runner.run_experiment` captura esta exceção por
    tentativa e registra `outcome="failure"` com o erro em `extra`, então
    chamar `run_pass_rate_probe` sem `navigate_fn` roda sem travar — só
    produz resultados todos de falha.
    """
    raise NotImplementedError(
        "Integração real com o navegador/Cloudflare Turnstile ainda não "
        "implementada — passe navigate_fn para testes ou uso real."
    )


def run_pass_rate_probe(
    evasion_profile: str,
    n_attempts: int,
    navigate_fn: Callable[[], dict] | None = None,
    results_dir: str | Path | None = None,
) -> list[AttemptResult]:
    """Mede a taxa de pass/challenge/block do Turnstile sob um perfil de evasão.

    `navigate_fn` é uma função sem argumentos, injetada pelo chamador, que
    executa uma navegação de verdade contra a página-alvo (ligando coletor,
    evasão e leitura do resultado do Turnstile) e retorna um `dict` no
    formato esperado por `experiments.runner.run_experiment` (chave
    `"outcome"` obrigatória). Se omitido, usa `_unavailable_navigate_fn`.
    """
    return run_experiment(
        target="turnstile",
        technique="behavioral_probe",
        evasion_profile=evasion_profile,
        n_attempts=n_attempts,
        attempt_fn=navigate_fn or _unavailable_navigate_fn,
        results_dir=results_dir,
    )
