"""Geração de tabelas/figuras do relatório comparativo final (Atividade 9, Nov/2026)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis import compare

DEFAULT_TARGETS = ("recaptcha", "hcaptcha", "turnstile")


def build_report(
    results_dir: str | Path,
    output_dir: str | Path,
    targets: list[str] = DEFAULT_TARGETS,
) -> dict[str, pd.DataFrame]:
    """Gera a tabela LaTeX e o gráfico de barras de cada alvo com dados.

    Para cada alvo em `targets`, calcula `compare.rank_techniques` e, se
    houver resultados, escreve `output_dir/<target>_ranking.tex` (tabela
    LaTeX via `DataFrame.to_latex`) e `output_dir/<target>_success_rate.png`
    (gráfico de barras de `success_rate` por `technique`, eixo y em
    [0, 1]). Alvos sem dados são pulados silenciosamente — não geram arquivo
    nem entram no dict retornado.

    Retorna um `dict` mapeando `target -> DataFrame` para os alvos com dados.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataframes: dict[str, pd.DataFrame] = {}

    for target in targets:
        ranking = compare.rank_techniques(results_dir, target)
        if not ranking:
            continue

        df = pd.DataFrame(ranking)
        dataframes[target] = df

        tex_path = output_path / f"{target}_ranking.tex"
        tex_path.write_text(df.to_latex(index=False), encoding="utf-8")

        fig, ax = plt.subplots()
        ax.bar(df["technique"], df["success_rate"])
        ax.set_ylim(0, 1)
        ax.set_xlabel("technique")
        ax.set_ylabel("success_rate")
        ax.set_title(f"Taxa de sucesso por técnica — {target}")
        fig.autofmt_xdate(rotation=45)
        fig.tight_layout()
        fig.savefig(output_path / f"{target}_success_rate.png")
        plt.close(fig)

    return dataframes
