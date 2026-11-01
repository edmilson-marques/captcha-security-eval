"""CLI para rodar uma técnica de verdade contra um alvo real (Atividades 4-6).

Exemplo:
    python -m experiments.run_live_cli --target recaptcha --technique baseline_no_select --n 3
    python -m experiments.run_live_cli --target recaptcha --technique llm_vision --n 1
    python -m experiments.run_live_cli --target hcaptcha --n 3 --headed
"""

from __future__ import annotations

import argparse
from functools import partial

from experiments.live import (
    hcaptcha_baseline_attempt,
    recaptcha_baseline_attempt,
    recaptcha_llm_vision_attempt,
)
from experiments.metrics import accuracy, outcome_breakdown
from experiments.runner import run_experiment

ATTEMPT_FNS = {
    "recaptcha": {
        "baseline_no_select": recaptcha_baseline_attempt,
        "llm_vision": recaptcha_llm_vision_attempt,
    },
    "hcaptcha": {
        "baseline_no_select": hcaptcha_baseline_attempt,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=sorted(ATTEMPT_FNS))
    parser.add_argument(
        "--technique", default="baseline_no_select", help="Técnica a usar (ver ATTEMPT_FNS)"
    )
    parser.add_argument("--n", type=int, default=3, help="Número de tentativas")
    parser.add_argument("--headed", action="store_true", help="Roda com navegador visível (debug)")
    args = parser.parse_args()

    techniques = ATTEMPT_FNS[args.target]
    if args.technique not in techniques:
        parser.error(
            f"Técnica {args.technique!r} não implementada para {args.target!r}. "
            f"Opções: {sorted(techniques)}"
        )

    attempt_fn = partial(techniques[args.technique], headless=not args.headed)
    results = run_experiment(
        target=args.target,
        technique=args.technique,
        evasion_profile="none",
        n_attempts=args.n,
        attempt_fn=attempt_fn,
    )

    print(
        f"{len(results)} tentativas registradas para {args.target!r} (técnica={args.technique!r})."
    )
    print("accuracy (taxa de sucesso):", accuracy(results))
    print("outcome_breakdown:", outcome_breakdown(results))


if __name__ == "__main__":
    main()
