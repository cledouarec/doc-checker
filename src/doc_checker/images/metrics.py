# Copyright 2026 Christophe Le Douarec
"""Quality measures of images: scale, SSIM, sharpness, blur, blocks, colour."""

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
BLUR_SIDE_MAX: Final = 1024
BLUR_TILES_MIN: Final = 4
BLOCK: Final = 8
GRID_SIDE_MAX: Final = 2048
PIXEL_CHANGES_MIN: Final = 16
PIXEL_LEVELS_MIN: Final = 32
PIXELATED_MAX: Final = 0.10
COLOR_COUNT_SIDE: Final = 512

_SSIM_C1: Final = (0.01 * 255) ** 2
_SSIM_C2: Final = (0.03 * 255) ** 2
_LAPLACIAN_MIN_SIDE: Final = 3
_FULL_RANGE: Final = 255

type Gray = NDArray[np.float64]
type Grid = NDArray[np.uint8]


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
        ref_blockiness: Blockiness of the reference, 1 when no block shows.
        new_blockiness: Blockiness of the new image.
    """

    identical: bool
    scale: float
    effective: float
    ssim: float
    sharpness: float
    ref_color: float
    new_color: float
    ref_blockiness: float = 1.0
    new_blockiness: float = 1.0


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


def tile_stat(values: Gray, stat: str) -> Gray:
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
    textured = tile_stat(centre, "std") >= TEXTURE_MIN
    if textured.any():
        return float(np.median(tile_stat(laplacian, "var")[textured]))
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
        ref_blockiness=blockiness(ref),
        new_blockiness=blockiness(new),
    )


@dataclass(frozen=True, slots=True)
class ImageQuality:
    """Quality measures of one image, without a reference.

    Attributes:
        blur: Normalized blur score (high is sharp), or None when too few
            tiles are textured to tell.
        blockiness: Ratio of the gaps at 8 x 8 block borders to the others;
            1 when no block shows.
        adjacent: Share of changing columns or rows that touch another
            changing one; low for repeated columns and rows.
        levels: Number of distinct gray levels.
        colors: Number of distinct colours of the reduced image.
        block_grid_share: Share of the runs of 8 columns, or rows, whose
            largest step is on their block border, the smaller of both
            axes; near 1 for the blocks of a JPEG, low for a few edges that
            fall on the borders, 0 when it cannot be told.
    """

    blur: float | None
    blockiness: float
    adjacent: float
    levels: int
    colors: int
    block_grid_share: float

    @property
    def pixelated(self) -> bool:
        """Whether the image looks like an enlarged, pixelated one."""
        return (
            self.adjacent <= PIXELATED_MAX and self.levels >= PIXEL_LEVELS_MIN
        )


def _resized(
    image: Image.Image,
    side: int,
    method: Image.Resampling,
) -> Image.Image:
    """Reduce an image so that its longest side is at most `side`.

    Args:
        image: Any image.
        side: Largest side allowed.
        method: Resampling filter.

    Returns:
        The image itself when it fits, else a reduced copy.
    """
    longest = max(image.size)
    if longest <= side:
        return image
    ratio = side / longest
    size = (
        max(1, round(image.width * ratio)),
        max(1, round(image.height * ratio)),
    )
    return image.resize(size, method)


def _blur(values: Gray) -> float | None:
    """Compute the blur score of a gray array.

    Args:
        values: Gray array.

    Returns:
        The median of the Laplacian deviation over the tile deviation, on
        textured tiles; None under `BLUR_TILES_MIN` of them.
    """
    if min(values.shape) < _LAPLACIAN_MIN_SIDE:
        return None
    centre = values[1:-1, 1:-1]
    laplacian = (
        values[:-2, 1:-1]
        + values[2:, 1:-1]
        + values[1:-1, :-2]
        + values[1:-1, 2:]
        - 4 * centre
    )
    deviations = tile_stat(centre, "std")
    textured = deviations >= TEXTURE_MIN
    if int(textured.sum()) < BLUR_TILES_MIN:
        return None
    ratios = tile_stat(laplacian, "std")[textured] / deviations[textured]
    return float(np.median(ratios))


def blur_score(image: Image.Image) -> float | None:
    """Measure the blur of an image, independently of its contrast.

    This is the public form of one measure of `assess`, its ``blur``.

    Args:
        image: Any decoded image.

    Returns:
        The median, over the textured tiles, of the deviation of the
        Laplacian over the deviation of the tile, on an image reduced to
        `BLUR_SIDE_MAX`; None when it cannot be told.
    """
    small = _resized(gray(image), BLUR_SIDE_MAX, Image.Resampling.LANCZOS)
    return _blur(np.asarray(small, dtype=np.float64))


def _grid(values: Grid) -> Grid:
    """Crop a gray array to the centre, on the block alignment.

    Args:
        values: Gray array.

    Returns:
        At most `GRID_SIDE_MAX` per side; the top and left edges of the crop
        are multiples of `BLOCK`.
    """
    height, width = values.shape
    top = (height - min(height, GRID_SIDE_MAX)) // 2 // BLOCK * BLOCK
    left = (width - min(width, GRID_SIDE_MAX)) // 2 // BLOCK * BLOCK
    return values[top : top + GRID_SIDE_MAX, left : left + GRID_SIDE_MAX]


def grid(image: Image.Image) -> Grid:
    """Return the gray pixels on which blocks and repetitions are measured.

    Args:
        image: Any decoded image.

    Returns:
        The gray image at its own size, cropped to the centre to
        `GRID_SIDE_MAX` per side, aligned on the 8 x 8 blocks.
    """
    return _grid(np.asarray(gray(image), dtype=np.uint8))


def _steps(values: Grid) -> tuple[Gray, Gray]:
    """Compute the mean gaps between neighbouring columns and rows.

    Args:
        values: Gray array.

    Returns:
        The mean absolute gap of every column to the previous one, and of
        every row to the previous one.
    """
    wide = values.astype(np.int16)
    columns = np.abs(np.diff(wide, axis=1)).mean(axis=0)
    rows = np.abs(np.diff(wide, axis=0)).mean(axis=1)
    return columns, rows


def _blockiness(values: Grid) -> float:
    """Compute the blockiness of a grid.

    Args:
        values: Gray grid.

    Returns:
        The mean, over both axes, of the gaps at block borders over the
        other gaps; 1 under four blocks a side.
    """
    if min(values.shape) < 4 * BLOCK:
        return 1.0
    ratios = []
    for steps in _steps(values):
        border = np.arange(len(steps)) % BLOCK == BLOCK - 1
        ratios.append(
            (steps[border].mean() + 0.5) / (steps[~border].mean() + 0.5),
        )
    return float(sum(ratios) / 2)


def blockiness(image: Image.Image) -> float:
    """Measure the 8 x 8 blocks of a JPEG on the pixel grid.

    Args:
        image: Any decoded image.

    Returns:
        About 1 without blocks, above with them; 1 under four blocks a side.
    """
    return _blockiness(grid(image))


def _block_grid(values: Grid) -> float:
    """Compute the share of block runs that step most on their border.

    Args:
        values: Gray grid.

    Returns:
        The smaller, over both axes, of the share of the runs of `BLOCK`
        steps, each ending on a block border, whose border step is larger
        than each of the others; 0 under four blocks a side.
    """
    if min(values.shape) < 4 * BLOCK:
        return 0.0
    shares = []
    for steps in _steps(values):
        runs = steps[: len(steps) // BLOCK * BLOCK].reshape(-1, BLOCK)
        wins = runs[:, -1] > runs[:, :-1].max(axis=1)
        shares.append(float(wins.mean()))
    return min(shares)


def block_grid_share(image: Image.Image) -> float:
    """Measure whether the steps of an image follow the 8 x 8 grid.

    A JPEG steps most on the borders of its blocks nearly everywhere; a
    drawing or a screenshot whose few edges fall on the borders does not
    between them, where it is flat. This is the public form of one measure
    of `assess`, its ``block_grid_share``.

    Args:
        image: Any decoded image.

    Returns:
        The share of block runs whose largest step is on their border, the
        smaller of both axes; 0 under four blocks a side.
    """
    return _block_grid(grid(image))


def _adjacent(values: Grid) -> float:
    """Compute the share of adjacent changes of a grid.

    Args:
        values: Gray grid.

    Returns:
        The larger, over both axes, of the share of changing columns (rows)
        that directly follow a changing one; 1 when an axis has fewer than
        `PIXEL_CHANGES_MIN` changes.
    """
    shares = []
    for steps in _steps(values):
        changed = np.flatnonzero(steps >= STEP_MIN)
        if len(changed) < PIXEL_CHANGES_MIN:
            return 1.0
        shares.append(float((np.diff(changed) == 1).mean()))
    return max(shares)


def adjacent_changes(image: Image.Image) -> float:
    """Measure whether the columns and rows are repeated.

    This is the public form of one measure of `assess`, its ``adjacent``.

    Args:
        image: Any decoded image.

    Returns:
        The share of adjacent changes; near 0 for an image enlarged by
        nearest neighbour, 1 when it cannot be told.
    """
    return _adjacent(grid(image))


def _colors(flat: Image.Image) -> int:
    """Count the distinct colours of a flattened image.

    Args:
        flat: RGB image.

    Returns:
        The number of distinct RGB triplets, after a nearest-neighbour
        reduction to `COLOR_COUNT_SIDE` when larger.
    """
    small = _resized(flat, COLOR_COUNT_SIDE, Image.Resampling.NEAREST)
    pixels = np.asarray(small).reshape(-1, 3).astype(np.int32)
    packed = pixels[:, 0] << 16 | pixels[:, 1] << 8 | pixels[:, 2]
    return int(np.unique(packed).size)


def color_count(image: Image.Image) -> int:
    """Count the distinct colours of an image.

    This is the public form of one measure of `assess`, its ``colors``.

    Args:
        image: Any decoded image.

    Returns:
        The number of distinct RGB triplets, after a nearest-neighbour
        reduction to `COLOR_COUNT_SIDE` when larger.
    """
    return _colors(flatten(image))


def assess(image: Image.Image) -> ImageQuality:
    """Measure the quality of one image.

    Args:
        image: Any decoded image.

    Returns:
        Its blur, blockiness, repetition, levels, colours and the share of
        its block runs that step most on their border.
    """
    flat = flatten(image)
    mono = flat.convert("L")
    pixels = _grid(np.asarray(mono, dtype=np.uint8))
    small = _resized(mono, BLUR_SIDE_MAX, Image.Resampling.LANCZOS)
    return ImageQuality(
        blur=_blur(np.asarray(small, dtype=np.float64)),
        blockiness=_blockiness(pixels),
        adjacent=_adjacent(pixels),
        levels=int(np.unique(pixels).size),
        colors=_colors(flat),
        block_grid_share=_block_grid(pixels),
    )
