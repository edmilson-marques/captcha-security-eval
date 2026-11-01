"""Pré-processamento clássico de imagem — baseline/fallback textual (Atividade 2, Ago-Set/2026).

Binarização, remoção de ruído e segmentação por componentes conexos
(ver Cap. 3.2 do TCC1). Útil para qualquer elemento textual residual nos
desafios, não para os grids de imagem do reCAPTCHA/hCaptcha.

Todas as funções operam sobre `numpy.ndarray` no formato que `cv2.imread`/
screenshots já fornecem (BGR, 8 bits por canal).
"""

from __future__ import annotations

import cv2
import numpy as np


def to_grayscale(image: np.ndarray, color_order: str = "bgr") -> np.ndarray:
    """Converte uma imagem colorida (BGR ou RGB) para escala de cinza.

    Se `image` já for 2D (já em escala de cinza), retorna uma cópia inalterada.
    `color_order` aceita `"bgr"` (padrão do OpenCV/cv2.imread) ou `"rgb"`.
    """
    if image.ndim == 2:
        return image.copy()

    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"Formato de imagem não suportado: shape={image.shape!r}")

    color_order = color_order.lower()
    has_alpha = image.shape[2] == 4
    if color_order == "bgr":
        code = cv2.COLOR_BGRA2GRAY if has_alpha else cv2.COLOR_BGR2GRAY
    elif color_order == "rgb":
        code = cv2.COLOR_RGBA2GRAY if has_alpha else cv2.COLOR_RGB2GRAY
    else:
        raise ValueError(f"color_order desconhecido: {color_order!r} (use 'bgr' ou 'rgb')")

    return cv2.cvtColor(image, code)


def binarize(
    image: np.ndarray,
    method: str = "otsu",
    *,
    block_size: int = 35,
    c: int = 5,
) -> np.ndarray:
    """Binariza uma imagem em escala de cinza (ou colorida, convertida internamente).

    - `"otsu"`: limiar global automático (cv2.threshold + THRESH_OTSU).
    - `"adaptive"`: limiar local (cv2.adaptiveThreshold, gaussiano), útil quando
      a iluminação/fundo do desafio não é uniforme. `block_size` (vizinhança,
      forçado a ímpar) e `c` (constante subtraída da média) são ajustáveis.

    Retorna uma imagem binária com valores apenas em {0, 255}.
    """
    gray = image if image.ndim == 2 else to_grayscale(image)

    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive":
        if block_size % 2 == 0:
            block_size += 1
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c,
        )
    else:
        raise ValueError(f"Método de binarização desconhecido: {method!r}")

    return binary


def denoise(image: np.ndarray, method: str = "median", *, ksize: int = 3) -> np.ndarray:
    """Remove ruído (tipicamente "sal e pimenta") de uma imagem.

    - `"median"`: cv2.medianBlur — bom para ruído impulsivo, preserva bordas.
    - `"gaussian"`: cv2.GaussianBlur — suaviza ruído gaussiano/compressão.

    `ksize` é forçado a ser ímpar (exigência do OpenCV).
    """
    if ksize % 2 == 0:
        ksize += 1

    if method == "median":
        return cv2.medianBlur(image, ksize)
    if method == "gaussian":
        return cv2.GaussianBlur(image, (ksize, ksize), 0)

    raise ValueError(f"Método de remoção de ruído desconhecido: {method!r}")


def segment_characters(binary_image: np.ndarray, min_area: int = 20) -> list[np.ndarray]:
    """Segmenta caracteres individuais de uma imagem binarizada (fundo preto,
    caracteres/traços em branco — ver `binarize`).

    Usa componentes conexos (`cv2.connectedComponentsWithStats`) para isolar
    cada blob, descarta componentes com área menor que `min_area` (ruído
    residual) e retorna os recortes ordenados da esquerda para a direita
    (por coordenada x do bounding box), para preservar a ordem do texto.
    """
    if binary_image.ndim != 2:
        raise ValueError("segment_characters espera uma imagem binária 2D (ver `binarize`).")

    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary_image, connectivity=8
    )

    boxes: list[tuple[int, int, int, int]] = []
    for label in range(1, num_labels):  # label 0 é o fundo
        x, y, w, h, area = stats[label]
        if area < min_area:
            continue
        boxes.append((int(x), int(y), int(w), int(h)))

    boxes.sort(key=lambda box: box[0])

    return [binary_image[y : y + h, x : x + w] for (x, y, w, h) in boxes]
