"""Testes do pré-processamento clássico de imagem (attacks.classical_cv) — Atividade 2."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from attacks import classical_cv


def _synthetic_bgr_image() -> np.ndarray:
    """Canvas preto 60x120 BGR com um retângulo branco (simula um glifo) e
    ruído "sal e pimenta" injetado manualmente."""
    image = np.zeros((60, 120, 3), dtype=np.uint8)
    cv2.rectangle(image, (20, 15), (90, 45), (255, 255, 255), thickness=-1)

    rng = np.random.default_rng(42)
    salt_rows, salt_cols = rng.integers(0, 60, size=50), rng.integers(0, 120, size=50)
    pepper_rows, pepper_cols = rng.integers(0, 60, size=50), rng.integers(0, 120, size=50)
    image[salt_rows, salt_cols] = (255, 255, 255)
    image[pepper_rows, pepper_cols] = (0, 0, 0)
    return image


def test_to_grayscale_converts_bgr_to_2d_array():
    image = _synthetic_bgr_image()

    gray = classical_cv.to_grayscale(image)

    assert gray.ndim == 2
    assert gray.shape == image.shape[:2]
    assert gray.dtype == np.uint8


def test_to_grayscale_accepts_rgb_order():
    image = _synthetic_bgr_image()

    gray_bgr = classical_cv.to_grayscale(image, color_order="bgr")
    gray_rgb = classical_cv.to_grayscale(image, color_order="rgb")

    assert gray_bgr.shape == gray_rgb.shape == image.shape[:2]


def test_to_grayscale_passthrough_when_already_gray():
    gray_input = np.zeros((10, 10), dtype=np.uint8)
    gray_input[2:5, 2:5] = 200

    result = classical_cv.to_grayscale(gray_input)

    assert result.ndim == 2
    np.testing.assert_array_equal(result, gray_input)


def test_to_grayscale_rejects_unsupported_shape():
    bad_image = np.zeros((10, 10, 2), dtype=np.uint8)

    with pytest.raises(ValueError):
        classical_cv.to_grayscale(bad_image)


@pytest.mark.parametrize("method", ["otsu", "adaptive"])
def test_binarize_produces_only_0_or_255(method):
    gray = classical_cv.to_grayscale(_synthetic_bgr_image())

    binary = classical_cv.binarize(gray, method=method)

    assert binary.shape == gray.shape
    assert binary.dtype == np.uint8
    unique_values = set(np.unique(binary).tolist())
    assert unique_values <= {0, 255}


def test_binarize_otsu_separates_bright_rectangle_from_background():
    gray = classical_cv.to_grayscale(_synthetic_bgr_image())

    binary = classical_cv.binarize(gray, method="otsu")

    # O centro do retângulo branco deve virar foreground (255); um canto
    # distante do retângulo e do ruído deve permanecer background (0).
    assert binary[30, 55] == 255
    assert binary[5, 5] == 0


def test_binarize_unknown_method_raises():
    gray = classical_cv.to_grayscale(_synthetic_bgr_image())

    with pytest.raises(ValueError):
        classical_cv.binarize(gray, method="nao-existe")


@pytest.mark.parametrize("method", ["median", "gaussian"])
def test_denoise_preserves_shape_and_dtype(method):
    gray = classical_cv.to_grayscale(_synthetic_bgr_image())

    denoised = classical_cv.denoise(gray, method=method)

    assert denoised.shape == gray.shape
    assert denoised.dtype == gray.dtype


def test_denoise_median_removes_isolated_salt_pepper_pixel():
    gray = np.full((20, 20), 50, dtype=np.uint8)
    gray[10, 10] = 255  # pixel isolado de "sal" em meio a fundo uniforme

    denoised = classical_cv.denoise(gray, method="median", ksize=3)

    assert denoised[10, 10] != 255


def test_denoise_forces_even_ksize_to_odd():
    gray = classical_cv.to_grayscale(_synthetic_bgr_image())

    # ksize par não deve estourar (a função deve corrigir internamente para ímpar).
    denoised = classical_cv.denoise(gray, method="median", ksize=4)

    assert denoised.shape == gray.shape


def test_denoise_unknown_method_raises():
    gray = classical_cv.to_grayscale(_synthetic_bgr_image())

    with pytest.raises(ValueError):
        classical_cv.denoise(gray, method="nao-existe")
