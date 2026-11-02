"""Pipeline OCR (Tesseract) como baseline textual (Atividade 2, Ago-Set/2026).

Wrapper fino sobre `pytesseract`. Requer o binário de sistema `tesseract-ocr`
instalado (dependência de SO, fora do controle deste pipeline). Se o binário
não estiver disponível, `pytesseract.TesseractNotFoundError` propaga
normalmente a partir das funções abaixo — não é capturada aqui de propósito,
para que o chamador decida como lidar com a ausência da dependência (ver
`tests/test_ocr.py`, que pula os testes quando `tesseract` não está no PATH).

**Extensão de 2026-11-02**: `locate_digit_candidates` aplica o mesmo OCR ao
desafio "número que pertence a um grupo" do hCaptcha, como alternativa
estrutural ao solver de visão via LLM (que não acertou nenhuma tentativa
real desse tipo) — ver `attacks/classical_canvas_solver.py`.
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


def locate_digit_candidates(image: np.ndarray, lang: str = "eng") -> list[dict]:
    """Localiza dígitos individuais via OCR, com posição (não só o texto concatenado).

    Extensão de 2026-11-02 (Atividade 2) para o desafio "número que pertence
    a um grupo" do hCaptcha (ver
    `attacks.llm_vision_solver._CANVAS_CRITERION_PROMPTS["digit_membership"]`
    e `attacks.classical_canvas_solver.solve_digit_membership_classical`) —
    o solver de visão via LLM não acertou nenhuma tentativa real desse tipo;
    esta função tenta a mesma tarefa (ler o dígito e saber onde ele está) por
    OCR clássico em vez de julgamento de um modelo de visão.

    Usa `--psm 11` (texto esparso, sem ordem de leitura assumida — os
    números aparecem soltos no canvas, não em frases) e restringe o
    reconhecimento a dígitos (`tessedit_char_whitelist`), já que não há
    texto não-numérico relevante nesse tipo de desafio. Retorna uma lista de
    `{"char": "8", "bbox": (x,y,w,h), "centroid": (cx,cy), "confidence": float}`,
    uma entrada por dígito reconhecido com confiança >= 0 (mesmo critério de
    `recognize_text_with_confidence` pra descartar entradas de layout sem
    texto).
    """
    config = "--psm 11 -c tessedit_char_whitelist=0123456789"
    data = pytesseract.image_to_data(
        image, lang=lang, config=config, output_type=pytesseract.Output.DICT
    )

    candidates: list[dict] = []
    for i, text in enumerate(data["text"]):
        stripped = text.strip()
        if not stripped or not stripped.isdigit():
            continue
        confidence = float(data["conf"][i])
        if confidence < 0:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        candidates.append(
            {
                "char": stripped,
                "bbox": (x, y, w, h),
                "centroid": (x + w / 2, y + h / 2),
                "confidence": confidence,
            }
        )
    return candidates
