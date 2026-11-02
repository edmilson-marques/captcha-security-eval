"""Orquestrador de experimentos (Atividades 4-6, Set-Out/2026).

(alvo, técnica, perfil de evasão, n_tentativas) -> log estruturado em
data/results/<alvo>/<técnica>__<perfil>__<timestamp>.jsonl

A execução de cada tentativa é delegada a `attempt_fn`, uma função sem
argumentos injetada pelo chamador. Isto mantém este módulo 100%
testável sem rede/Selenium: nos testes, `attempt_fn` é uma closure falsa;
em uso real (fora do escopo desta tarefa), `attempt_fn` vai ligar os
coletores (`src/harness/collectors/`), os ataques (`src/attacks/`) e as
técnicas de evasão (`src/evasion/`).
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from experiments.metrics import VALID_OUTCOMES, AttemptResult

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[2] / "data" / "results"


def run_experiment(
    target: str,
    technique: str,
    evasion_profile: str,
    n_attempts: int,
    attempt_fn: Callable[[], dict],
    results_dir: str | Path | None = None,
    pause_every: int | None = None,
    pause_seconds: float = 0.0,
    sleep_fn: Callable[[float], None] = time.sleep,
    on_pause: Callable[[int, int], None] | None = None,
) -> list[AttemptResult]:
    """Executa `n_attempts` tentativas de `attempt_fn` e persiste o resultado.

    `attempt_fn` é chamada sem argumentos uma vez por tentativa e deve
    retornar um `dict` com ao menos a chave `"outcome"` (um dos valores de
    `experiments.metrics.VALID_OUTCOMES`) e, opcionalmente, `"confidence"`,
    `"sub_challenges"`, `"detected_as_bot"` e `"extra"`.

    Se `attempt_fn()` lançar uma exceção numa tentativa específica, essa
    exceção É CAPTURADA — a rodada inteira não é derrubada. A tentativa em
    questão é registrada com `outcome="failure"` e
    `extra={"error": str(exceção)}`, e a execução segue para a próxima
    tentativa.

    Se `attempt_fn()` retornar um `outcome` fora de `VALID_OUTCOMES`, uma
    `ValueError` é levantada (isso não é capturado como falha de tentativa,
    pois indica um bug no `attempt_fn` em si, não uma tentativa malsucedida).

    Todas as `n_attempts` tentativas desta chamada são gravadas (append,
    uma linha JSONL por tentativa) no mesmo arquivo
    `results_dir/<target>/<technique>__<evasion_profile>__<timestamp>.jsonl`,
    onde `timestamp` é gerado uma única vez no início desta função — não a
    cada tentativa — para que todas as tentativas da rodada compartilhem o
    arquivo. Os diretórios necessários são criados automaticamente.

    `pause_every` (ex. `10`) pausa a execução por `pause_seconds` depois de
    cada bloco de `pause_every` tentativas COMPLETADAS — pensado pra lotes
    grandes (ex. 500 tentativas), pra não martelar a demo ao vivo do
    fornecedor continuamente por um período longo e ininterrupto (ver
    "Escopo ético" no README). Não pausa depois da última tentativa (não
    há nada a esperar). `pause_every=None` (padrão) ou `pause_every<=0`
    desliga a pausa — comportamento idêntico ao de antes desta extensão.
    `sleep_fn` é injetável (padrão `time.sleep`) pra permitir testar a
    lógica de pausa sem esperar de verdade. `on_pause` (opcional), se
    passado, é chamado como `on_pause(tentativas_completadas, n_attempts)`
    logo antes de cada pausa — útil pra imprimir progresso em lotes longos
    (ver `run_live_cli.py`), já que esta função só retorna no final.

    Retorna a lista de `AttemptResult` gerados (já persistidos em disco).
    """
    base_dir = Path(results_dir) if results_dir is not None else DEFAULT_RESULTS_DIR
    target_dir = base_dir / target
    target_dir.mkdir(parents=True, exist_ok=True)

    round_timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    output_path = target_dir / f"{technique}__{evasion_profile}__{round_timestamp}.jsonl"

    results: list[AttemptResult] = []
    with output_path.open("a", encoding="utf-8") as fh:
        for attempt_number in range(1, n_attempts + 1):
            attempt_id = uuid.uuid4().hex
            started_at_dt = datetime.now(UTC)

            try:
                outcome_data = attempt_fn()
            except Exception as exc:  # noqa: BLE001 - captura intencional por tentativa
                finished_at_dt = datetime.now(UTC)
                result = AttemptResult(
                    target=target,
                    technique=technique,
                    evasion_profile=evasion_profile,
                    attempt_id=attempt_id,
                    started_at=started_at_dt.isoformat(),
                    finished_at=finished_at_dt.isoformat(),
                    outcome="failure",
                    duration_seconds=(finished_at_dt - started_at_dt).total_seconds(),
                    extra={"error": str(exc)},
                )
                results.append(result)
                fh.write(result.to_json_line() + "\n")
            else:
                finished_at_dt = datetime.now(UTC)
                outcome = outcome_data.get("outcome")
                if outcome not in VALID_OUTCOMES:
                    raise ValueError(
                        f"outcome inválido retornado por attempt_fn: {outcome!r} "
                        f"(esperado um de {VALID_OUTCOMES!r})"
                    )

                result = AttemptResult(
                    target=target,
                    technique=technique,
                    evasion_profile=evasion_profile,
                    attempt_id=attempt_id,
                    started_at=started_at_dt.isoformat(),
                    finished_at=finished_at_dt.isoformat(),
                    outcome=outcome,
                    duration_seconds=(finished_at_dt - started_at_dt).total_seconds(),
                    confidence=outcome_data.get("confidence"),
                    sub_challenges=outcome_data.get("sub_challenges", 0),
                    detected_as_bot=outcome_data.get("detected_as_bot"),
                    extra=outcome_data.get("extra", {}),
                )
                results.append(result)
                fh.write(result.to_json_line() + "\n")

            is_last_attempt = attempt_number == n_attempts
            if (
                pause_every
                and pause_every > 0
                and attempt_number % pause_every == 0
                and not is_last_attempt
            ):
                if on_pause is not None:
                    on_pause(attempt_number, n_attempts)
                sleep_fn(pause_seconds)

    return results
