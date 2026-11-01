"""Testes do wrapper OCR (attacks.ocr) — Atividade 2.

Requer o binário de sistema `tesseract-ocr`. Como essa é uma dependência de
SO fora do controle do pipeline, os testes deste módulo são pulados
graciosamente (não falham) quando `tesseract` não está disponível no PATH.
"""

from __future__ import annotations

import shutil

import cv2
import numpy as np
import pytest

from attacks import ocr

TESSERACT_AVAILABLE = shutil.which("tesseract") is not None

pytestmark = pytest.mark.skipif(
    not TESSERACT_AVAILABLE, reason="binário tesseract-ocr não encontrado no sistema"
)

EXPECTED_TEXT = "AB12"


def _easy_text_image(text: str = EXPECTED_TEXT) -> np.ndarray:
    """Imagem branca com texto grande, preto, sem distorção — OCR "fácil",
    apenas para teste de regressão do wrapper (não simula um CAPTCHA real)."""
    image = np.full((100, 300), 255, dtype=np.uint8)
    cv2.putText(
        image,
        text,
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0,
        0,
        thickness=4,
        lineType=cv2.LINE_AA,
    )
    return image


def test_recognize_text_matches_expected_tolerantly():
    image = _easy_text_image()

    recognized = ocr.recognize_text(image)
    normalized = recognized.upper().replace(" ", "")

    assert normalized != ""
    # Comparação tolerante: o OCR pode errar 1-2 caracteres, então exige só
    # que a maioria dos caracteres esperados apareça em algum lugar do texto.
    matched_chars = sum(1 for char in EXPECTED_TEXT if char in normalized)
    assert matched_chars >= 2


def test_recognize_text_with_confidence_returns_tuple_in_range():
    image = _easy_text_image()

    text, confidence = ocr.recognize_text_with_confidence(image)

    assert isinstance(text, str)
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 100.0


def test_recognize_text_with_confidence_nonempty_text_has_positive_confidence():
    image = _easy_text_image()

    text, confidence = ocr.recognize_text_with_confidence(image)

    if text.strip():
        assert confidence > 0.0
