"""Pipeline OCR (Tesseract) como baseline textual (Atividade 2, Ago-Set/2026).

Wrapper fino sobre `pytesseract`. Requer o binário de sistema `tesseract-ocr`
instalado (dependência de SO, fora do controle deste pipeline). Se o binário
não estiver disponível, `pytesseract.TesseractNotFoundError` propaga
normalmente a partir das funções abaixo — não é capturada aqui de propósito,
para que o chamador decida como lidar com a ausência da dependência (ver
`tests/test_ocr.py`, que pula os testes quando `tesseract` não está no PATH).
"""

from __future__ import annotations

import numpy as np
import pytesseract


def recognize_text(image: np.ndarray, lang: str = "eng") -> str:
    """Roda o OCR sobre `image` e retorna o texto reconhecido, já stripado."""
    text = pytesseract.image_to_string(image, lang=lang)
    return text.strip()


def recognize_text_with_confidence(image: np.ndarray, lang: str = "eng") -> tuple[str, float]:
    """Roda o OCR e retorna `(texto_concatenado, confiança_média)`.

    A confiança média é calculada sobre as entradas de `image_to_data` que o
    Tesseract de fato reportou (ignora entradas com confiança -1, usadas pelo
    Tesseract para níveis de layout — página/bloco/parágrafo — sem texto
    detectado). Se nenhuma entrada tiver confiança válida, retorna 0.0.
    """
    data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)

    words: list[str] = []
    confidences: list[float] = []
    for text, raw_conf in zip(data["text"], data["conf"]):
        confidence = float(raw_conf)
        if confidence < 0:
            continue
        confidences.append(confidence)
        stripped = text.strip()
        if stripped:
            words.append(stripped)

    recognized_text = " ".join(words)
    mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return recognized_text, mean_confidence
