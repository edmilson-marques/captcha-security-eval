"""Avaliação comportamental do Cloudflare Turnstile (Atividades 2 e 6).

Turnstile raramente apresenta desafio visual — não há imagem para "resolver".
Esta técnica dispara `experiments.runner` repetidamente, sob diferentes
perfis de evasão (ver `src/evasion/`), e mede se foi emitido token sem
desafio, desafio interativo, ou bloqueio. Ver nota em config/targets.yaml.
"""
from __future__ import annotations


def run_pass_rate_probe(profile, n_attempts: int):
    raise NotImplementedError("Atividade 6 — taxa de pass/challenge/block (Out/2026)")
