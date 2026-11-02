"""Pré-processamento clássico de imagem — baseline/fallback textual (Atividade 2, Ago-Set/2026).

Binarização, remoção de ruído e segmentação por componentes conexos
(ver Cap. 3.2 do TCC1). Útil para qualquer elemento textual residual nos
desafios, não para os grids de imagem do reCAPTCHA/hCaptcha.

Todas as funções operam sobre `numpy.ndarray` no formato que `cv2.imread`/
screenshots já fornecem (BGR, 8 bits por canal).

**Extensão de 2026-11-02** (Atividade 2 — técnica alternativa pros desafios
de clique-sobre-canvas do hCaptcha que o solver de visão via LLM não
resolveu nem uma vez em 51 tentativas reais — ver docstring de
`experiments.live.hcaptcha_llm_vision_attempt`): `detect_stroke_regions`,
`classify_loop_closure`, `region_mean_color`,
`surrounding_background_color` e `color_contrast` dão suporte a uma
abordagem estrutural diferente (segmentação por borda + geometria/cor, não
julgamento qualitativo de um modelo) — ver `attacks/classical_canvas_solver.py`
para os solvers de alto nível que usam essas peças.

**Prototipagem manual contra amostras reais (2026-11-02), achado
importante**: um limiar global de saturação (abordagem óbvia inicial) NÃO
funciona nesse fundo adversarial — traços diferentes do mesmo desafio têm
saturação de cor bem diferente entre si (alguns vívidos, outros em tom
pastel), então um único limiar global não separa nenhum dos dois de forma
confiável sem também capturar textura do fundo. Detecção por borda (Canny)
seguida de fechamento morfológico mostrou sinal bem mais forte e é a base
usada aqui. Limitação conhecida, não resolvida: quando dois traços ficam
muito próximos no desafio, a detecção por componentes conexos pode
fundi-los numa região só ou fragmentá-los — funciona bem para traços bem
separados, degrada em imagens mais aglomeradas. Isso não foi contornado;
documentado para análise honesta do resultado real medido ao vivo.
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


def _stroke_edge_mask(image: np.ndarray) -> np.ndarray:
    """Máscara binária de bordas (Canny + fechamento morfológico) usada para
    isolar traços/ícones desenhados sobre um fundo decorativo — ver nota no
    topo do módulo sobre por que threshold de saturação não funciona aqui.
    """
    gray = image if image.ndim == 2 else to_grayscale(image)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 30, 90)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)


def detect_stroke_regions(
    image: np.ndarray, min_area: int = 150, max_area_ratio: float = 0.3
) -> tuple[list[dict], np.ndarray]:
    """Detecta regiões de traço (ícones/curvas) via borda + fechamento morfológico.

    Descarta componentes menores que `min_area` (ruído residual) ou maiores
    que `max_area_ratio` da imagem inteira (textura de fundo conectada em
    uma massa só, não um traço individual — ver achado no topo do módulo).

    Retorna `(regions, mask)`: `regions` é uma lista de dicts
    `{"bbox": (x,y,w,h), "centroid": (cx,cy), "area": int}`, e `mask` é a
    máscara binária usada (precisa ser reaproveitada por
    `classify_loop_closure`, que opera sobre a mesma máscara).
    """
    mask = _stroke_edge_mask(image)
    total_area = mask.shape[0] * mask.shape[1]

    num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    regions = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < min_area or area > total_area * max_area_ratio:
            continue
        regions.append(
            {
                "bbox": (int(x), int(y), int(w), int(h)),
                "centroid": (float(centroids[i][0]), float(centroids[i][1])),
                "area": int(area),
            }
        )
    return regions, mask


def classify_loop_closure(
    mask: np.ndarray, bbox: tuple[int, int, int, int], padding: int = 5
) -> float:
    """Fração da área de `bbox` (mais `padding`) que fica cercada pelo traço.

    Recorta `mask` (a mesma máscara binária devolvida por
    `detect_stroke_regions`) ao redor de `bbox`, preenche a partir da borda
    do recorte (`cv2.floodFill` a partir do canto) e marca como "cercado"
    qualquer pixel de fundo não alcançado por esse preenchimento — ou seja,
    uma região interna que só existe porque o traço a envolve completamente.
    Fração alta (ex. > 0.5) indica um laço fechado; fração baixa indica uma
    curva aberta (o preenchimento a partir da borda alcança quase tudo).
    """
    x, y, w, h = bbox
    img_h, img_w = mask.shape
    x1, y1 = max(0, x - padding), max(0, y - padding)
    x2, y2 = min(img_w, x + w + padding), min(img_h, y + h + padding)
    region = mask[y1:y2, x1:x2]

    flood_fill_mask = np.zeros((region.shape[0] + 2, region.shape[1] + 2), dtype=np.uint8)
    flooded = region.copy()
    cv2.floodFill(flooded, flood_fill_mask, (0, 0), 255)
    enclosed = cv2.bitwise_and(cv2.bitwise_not(region), cv2.bitwise_not(flooded))
    return float(np.count_nonzero(enclosed)) / float(region.size)


def region_mean_color(
    image: np.ndarray, bbox: tuple[int, int, int, int], mask: np.ndarray | None = None
) -> tuple[float, float, float]:
    """Cor média (B, G, R) dos pixels de `bbox`.

    `bbox` é um retângulo que envolve uma região possivelmente curva/irregular
    (ex. um ícone ou traço) — os cantos do retângulo costumam cair fora da
    forma de verdade. Se `mask` for passada (ex. a máscara devolvida por
    `detect_stroke_regions`), usa só os pixels onde `mask > 0` dentro de
    `bbox` para a média, evitando que a cor do fundo nos cantos do retângulo
    contamine o resultado. Sem `mask`, usa todos os pixels do retângulo
    (comportamento mais simples, usado como fallback).
    """
    x, y, w, h = bbox
    crop = image[y : y + h, x : x + w]
    if mask is not None:
        mask_crop = mask[y : y + h, x : x + w] > 0
        if np.any(mask_crop):
            pixels = crop[mask_crop]
            mean = pixels.mean(axis=0)
            return (float(mean[0]), float(mean[1]), float(mean[2]))
    mean = crop.reshape(-1, crop.shape[-1]).mean(axis=0)
    return (float(mean[0]), float(mean[1]), float(mean[2]))


def surrounding_background_color(
    image: np.ndarray, bbox: tuple[int, int, int, int], margin: int = 20
) -> tuple[float, float, float]:
    """Cor média (B, G, R) de um anel ao redor de `bbox`, excluindo o próprio `bbox`.

    Usado para comparar a cor de um ícone/traço com o fundo imediatamente
    ao redor dele (não a cor média da imagem inteira, que incluiria outras
    partes do desafio).
    """
    x, y, w, h = bbox
    img_h, img_w = image.shape[:2]
    ox1, oy1 = max(0, x - margin), max(0, y - margin)
    ox2, oy2 = min(img_w, x + w + margin), min(img_h, y + h + margin)

    ring_mask = np.ones((oy2 - oy1, ox2 - ox1), dtype=bool)
    inner_x1, inner_y1 = x - ox1, y - oy1
    ring_mask[max(0, inner_y1) : inner_y1 + h, max(0, inner_x1) : inner_x1 + w] = False

    crop = image[oy1:oy2, ox1:ox2]
    pixels = crop[ring_mask]
    if len(pixels) == 0:
        return region_mean_color(image, bbox)
    mean = pixels.mean(axis=0)
    return (float(mean[0]), float(mean[1]), float(mean[2]))


def color_contrast(
    color_a: tuple[float, float, float], color_b: tuple[float, float, float]
) -> float:
    """Distância de cor perceptual aproximada entre duas cores BGR (formato OpenCV).

    Usa a fórmula "redmean" (aproximação de baixo custo, sem precisar
    converter pra Lab/CIEDE2000): pondera o canal verde mais que os outros
    dois, com um peso que varia conforme o nível médio de vermelho. Boa o
    bastante para comparar contraste *relativo* entre candidatos (qual
    contrasta mais com o fundo ao redor), não para medir diferença de cor
    com precisão absoluta.
    """
    b1, g1, r1 = color_a
    b2, g2, r2 = color_b
    rmean = (r1 + r2) / 2
    dr, dg, db = r1 - r2, g1 - g2, b1 - b2
    return ((2 + rmean / 256) * dr**2 + 4 * dg**2 + (2 + (255 - rmean) / 256) * db**2) ** 0.5
