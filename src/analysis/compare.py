"""Agregação e ranking de técnicas por CAPTCHA-alvo (Atividade 7, Out-Nov/2026)."""

from __future__ import annotations

from pathlib import Path

from experiments.metrics import AttemptResult, accuracy, mean_duration


def load_results(results_dir: str | Path, target: str | None = None) -> list[AttemptResult]:
    """Carrega todos os `AttemptResult` persistidos em `results_dir`.

    Varre `results_dir` recursivamente por arquivos `*.jsonl`. Se `target`
    for informado, a busca é restrita a `results_dir/<target>/`; caso
    contrário, percorre todos os subdiretórios de `results_dir`.

    Cada linha não-vazia de cada arquivo é desserializada via
    `AttemptResult.from_json_line`. Se `results_dir` (ou
    `results_dir/<target>`) não existir, retorna `[]` sem lançar exceção.
    """
    base_dir = Path(results_dir)
    search_dir = base_dir / target if target is not None else base_dir

    if not search_dir.exists():
        return []

    results: list[AttemptResult] = []
    for jsonl_path in sorted(search_dir.rglob("*.jsonl")):
        with jsonl_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                results.append(AttemptResult.from_json_line(line))

    return results


def rank_techniques(results_dir: str | Path, target: str) -> list[dict]:
    """Classifica as técnicas usadas contra `target` por taxa de sucesso.

    Agrupa os resultados de `load_results(results_dir, target=target)` por
    `technique` (ignorando `evasion_profile` — esta é uma visão agregada por
    técnica) e calcula, para cada uma, `n_attempts`, `success_rate`,
    `mean_duration_seconds` e `mean_confidence`.

    Retorna a lista ordenada por `success_rate` decrescente, com empate
    desfeito por `technique` em ordem alfabética. Retorna `[]` se não houver
    resultados para o alvo.
    """
    results = load_results(results_dir, target=target)
    if not results:
        return []

    by_technique: dict[str, list[AttemptResult]] = {}
    for result in results:
        by_technique.setdefault(result.technique, []).append(result)

    ranking: list[dict] = []
    for technique, technique_results in by_technique.items():
        confidences = [r.confidence for r in technique_results if r.confidence is not None]
        mean_confidence = sum(confidences) / len(confidences) if confidences else None

        ranking.append(
            {
                "technique": technique,
                "n_attempts": len(technique_results),
                "success_rate": accuracy(technique_results),
                "mean_duration_seconds": mean_duration(technique_results),
                "mean_confidence": mean_confidence,
            }
        )

    ranking.sort(key=lambda row: (-row["success_rate"], row["technique"]))
    return ranking
