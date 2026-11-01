"""Pré-processamento clássico de imagem — baseline/fallback textual (Atividade 2, Ago-Set/2026).

Binarização, remoção de ruído e segmentação por projeção/contornos
(ver Cap. 3 do TCC1). Útil para qualquer elemento textual residual nos
desafios, não para os grids de imagem do reCAPTCHA/hCaptcha.
"""
from __future__ import annotations


def binarize(image):
    raise NotImplementedError("Atividade 2 — pré-processamento clássico (Ago-Set/2026)")


def segment_characters(image):
    raise NotImplementedError("Atividade 2 — segmentação por projeção/contornos (Ago-Set/2026)")
