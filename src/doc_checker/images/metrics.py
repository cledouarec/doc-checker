# Copyright 2026 Christophe Le Douarec
"""Quality measures of a pair of images: scale, SSIM, sharpness, colour."""

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from doc_checker.images.pixels import flatten, gray

COMPARE_SIDE_MAX: Final = 1024
SSIM_RADIUS: Final = 3
TILE: Final = 32
TEXTURE_MIN: Final = 25.0
STEP_MIN: Final = 1.0
COLOR_SIDE: Final = 64

_SSIM_C1: Final = (0.01 * 255) ** 2
_SSIM_C2: Final = (0.03 * 255) ** 2
_LAPLACIAN_MIN_SIDE: Final = 3
_FULL_RANGE: Final = 255

type Gray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Measures:
    """Quality measures of an image compared with its reference.

    Attributes:
        identical: Whether both images have the same pixels.
        scale: Square root of the ratio of the pixel counts.
        effective: Same ratio, from the effective sizes.
        ssim: Mean structural similarity of the gray images.
        sharpness: Sharpness of the new image over the reference's.
        ref_color: Colorfulness of the reference, from 0 to 1.
        new_color: Colorfulness of the new image, from 0 to 1.
    """

    identical: bool
    scale: float
    effective: float
    ssim: float
    sharpness: float
    ref_color: float
    new_color: float


def common_gray(ref: Image.Image, new: Image.Image) -> tuple[Gray, Gray]:
    """Put two images in gray at the same, limited, size.

    Args:
        ref: Reference image.
        new: New image.

    Returns:
        The two gray arrays, of the smaller width and the smaller height,
        reduced so that the longest side is at most `COMPARE_SIDE_MAX`.
    """
    width = min(ref.width, new.width)
    height = min(ref.height, new.height)
    longest = max(width, height)
    if longest > COMPARE_SIDE_MAX:
        ratio = COMPARE_SIDE_MAX / longest
        width = max(1, round(width * ratio))
        height = max(1, round(height * ratio))
    size = (width, height)
    return (
        np.asarray(
            gray(ref).resize(size, Image.Resampling.LANCZOS),
            dtype=np.float64,
        ),
        np.asarray(
            gray(new).resize(size, Image.Resampling.LANCZOS),
            dtype=np.float64,
        ),
    )


def _box_mean(values: Gray, height: int, width: int) -> Gray:
    """Average over every full window, by cumulative sums.

    Args:
        values: A 2D array at least as large as the window.
        height: Height of the window.
        width: Width of the window.

    Returns:
        The means of the windows fully inside the array.
    """
    total = np.zeros((values.shape[0] + 1, values.shape[1] + 1))
    total[1:, 1:] = values.cumsum(axis=0).cumsum(axis=1)
    sums = (
        total[height:, width:]
        - total[:-height, width:]
        - total[height:, :-width]
        + total[:-height, :-width]
    )
    return sums / (height * width)


def ssim(first: Gray, second: Gray) -> float:
    """Compute the mean structural similarity of two gray images.

    Args:
        first: Gray array.
        second: Gray array of the same shape.

    Returns:
        The mean SSIM over square windows; one window, the whole image,
        when a side is smaller than a window.
    """
    side = 2 * SSIM_RADIUS + 1
    height = min(side, first.shape[0])
    width = min(side, first.shape[1])
    mean_a = _box_mean(first, height, width)
    mean_b = _box_mean(second, height, width)
    var_a = _box_mean(first * first, height, width) - mean_a**2
    var_b = _box_mean(second * second, height, width) - mean_b**2
    cov = _box_mean(first * second, height, width) - mean_a * mean_b
    score = ((2 * mean_a * mean_b + _SSIM_C1) * (2 * cov + _SSIM_C2)) / (
        (mean_a**2 + mean_b**2 + _SSIM_C1) * (var_a + var_b + _SSIM_C2)
    )
    return float(score.mean())


def _tile_stat(values: Gray, stat: str) -> Gray:
    """Compute the variance or deviation of every whole tile.

    Args:
        values: A 2D array.
        stat: "var" or "std".

    Returns:
        One value per whole `TILE` x `TILE` tile, row by row.
    """
    rows, cols = values.shape[0] // TILE, values.shape[1] // TILE
    blocks = values[: rows * TILE, : cols * TILE].reshape(
        rows,
        TILE,
        cols,
        TILE,
    )
    result = blocks.var(axis=(1, 3))
    if stat == "std":
        result = np.sqrt(result)
    return result.ravel()


def sharpness(values: Gray) -> float:
    """Measure the sharpness as the variance of the Laplacian.

    Args:
        values: Gray array.

    Returns:
        The median variance of the textured tiles, or the variance of the
        whole image without such tile; 0 under 3 x 3 pixels.
    """
    if min(values.shape) < _LAPLACIAN_MIN_SIDE:
        return 0.0
    centre = values[1:-1, 1:-1]
    laplacian = (
        values[:-2, 1:-1]
        + values[2:, 1:-1]
        + values[1:-1, :-2]
        + values[1:-1, 2:]
        - 4 * centre
    )
    textured = _tile_stat(centre, "std") >= TEXTURE_MIN
    if textured.any():
        return float(np.median(_tile_stat(laplacian, "var")[textured]))
    return float(laplacian.var())


def effective_size(image: Image.Image) -> tuple[float, float]:
    """Count the columns and rows that differ from their predecessor.

    Args:
        image: Gray image, at its own size.

    Returns:
        The effective width and height: 1 plus the number of columns (rows)
        whose mean absolute gap to the previous one is at least `STEP_MIN`.
    """
    values = np.asarray(image, dtype=np.int16)
    columns = np.abs(np.diff(values, axis=1)).mean(axis=0)
    rows = np.abs(np.diff(values, axis=0)).mean(axis=1)
    return (
        1.0 + float((columns >= STEP_MIN).sum()),
        1.0 + float((rows >= STEP_MIN).sum()),
    )


def colorfulness(image: Image.Image) -> float:
    """Measure how far the colours are from gray.

    Args:
        image: Any decoded image.

    Returns:
        The mean gap between the largest and the smallest channel, from 0
        to 1, on an image reduced to `COLOR_SIDE` pixels.
    """
    small = flatten(image).resize(
        (COLOR_SIDE, COLOR_SIDE),
        Image.Resampling.BILINEAR,
    )
    values = np.asarray(small, dtype=np.int16)
    spread = values.max(axis=2) - values.min(axis=2)
    return float(spread.mean() / _FULL_RANGE)


def measure(ref: Image.Image, new: Image.Image) -> Measures:
    """Measure the quality of a new image against its reference.

    Args:
        ref: Reference image.
        new: New image.

    Returns:
        The measures; all ratios are 1 when the pixels are identical.
    """
    ref_color = colorfulness(ref)
    new_color = colorfulness(new)
    if ref.size == new.size and np.array_equal(
        np.asarray(ref.convert("RGBA")),
        np.asarray(new.convert("RGBA")),
    ):
        return Measures(
            identical=True,
            scale=1.0,
            effective=1.0,
            ssim=1.0,
            sharpness=1.0,
            ref_color=ref_color,
            new_color=new_color,
        )
    ref_w, ref_h = effective_size(gray(ref))
    new_w, new_h = effective_size(gray(new))
    first, second = common_gray(ref, new)
    ref_sharp = sharpness(first)
    return Measures(
        identical=False,
        scale=math.sqrt(new.width * new.height / (ref.width * ref.height)),
        effective=math.sqrt(new_w * new_h / (ref_w * ref_h)),
        ssim=ssim(first, second),
        sharpness=sharpness(second) / ref_sharp if ref_sharp else 1.0,
        ref_color=ref_color,
        new_color=new_color,
    )
