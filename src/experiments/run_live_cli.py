"""CLI para rodar a técnica "baseline_no_select" de verdade (Atividades 4-6).

Exemplo:
    python -m experiments.run_live_cli --target recaptcha --n 3
    python -m experiments.run_live_cli --target hcaptcha --n 3 --headed
"""

from __future__ import annotations

import argparse
from functools import partial

from experiments.live import hcaptcha_baseline_attempt, recaptcha_baseline_attempt
from experiments.metrics import accuracy, outcome_breakdown
from experiments.runner import run_experiment

ATTEMPT_FNS = {
    "recaptcha": recaptcha_baseline_attempt,
    "hcaptcha": hcaptcha_baseline_attempt,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=sorted(ATTEMPT_FNS))
    parser.add_argument("--n", type=int, default=3, help="Número de tentativas")
    parser.add_argument("--headed", action="store_true", help="Roda com navegador visível (debug)")
    args = parser.parse_args()

    attempt_fn = partial(ATTEMPT_FNS[args.target], headless=not args.headed)
    results = run_experiment(
        target=args.target,
        technique="baseline_no_select",
        evasion_profile="none",
        n_attempts=args.n,
        attempt_fn=attempt_fn,
    )

    print(f"{len(results)} tentativas registradas para {args.target!r}.")
    print("accuracy (taxa de sucesso):", accuracy(results))
    print("outcome_breakdown:", outcome_breakdown(results))


if __name__ == "__main__":
    main()
