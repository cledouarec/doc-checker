# Copyright 2026 Christophe Le Douarec
"""Perceptual hashes of images and the distances between them.

Both hashes read the luminance of the image put on white, made once for the
two of them. A large image is first reduced by a whole factor, as
`Image.reduce` does, then resampled with LANCZOS, so that the cost of a hash
does not grow with the size of the image while it stays that of a direct
LANCZOS resampling, but for rounding.
"""

import math
from typing import TYPE_CHECKING, Final

import numpy as np
from PIL import Image

from doc_checker.images.pixels import gray

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

_DHASH_SIZE: Final = (9, 8)
_PHASH_SIDE: Final = 32
_PHASH_BLOCK: Final = 8
_PHASH_DECIMALS: Final = 9
# The least ratio of the LANCZOS resampling after the reduction by a whole
# factor: Pillow's own advice for a result that matches a direct one.
_REDUCING_GAP: Final = 3.0


def _bits(flags: NDArray[np.bool_]) -> int:
    """Pack booleans into an integer, the first one as the top bit.

    Args:
        flags: The bits, in order.

    Returns:
        The integer.
    """
    value = 0
    for flag in flags.ravel().tolist():
        value = (value << 1) | int(flag)
    return value


def _small(luminance: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Resample a gray image to a small size.

    Args:
        luminance: An `L` image.
        size: The width and the height to resample to.

    Returns:
        The image reduced by a whole factor, then resampled with LANCZOS.
    """
    return luminance.resize(
        size,
        Image.Resampling.LANCZOS,
        reducing_gap=_REDUCING_GAP,
    )


def _dhash_of(luminance: Image.Image) -> int:
    """Hash the horizontal gradients of a gray image on 64 bits.

    Args:
        luminance: The image put on white, in gray.

    Returns:
        The difference hash.
    """
    pixels = np.asarray(_small(luminance, _DHASH_SIZE), dtype=np.int16)
    return _bits(pixels[:, 1:] > pixels[:, :-1])


def dhash(image: Image.Image) -> int:
    """Hash the horizontal gradients of an image on 64 bits.

    Args:
        image: Any decoded image.

    Returns:
        The difference hash.
    """
    return _dhash_of(gray(image))


def _dct_matrix(side: int) -> NDArray[np.float64]:
    """Build the orthonormal DCT-II matrix of a side.

    Args:
        side: Number of samples.

    Returns:
        The matrix, so that `C @ X @ C.T` is the 2D transform.
    """
    k = np.arange(side)[:, None]
    n = np.arange(side)[None, :]
    matrix = np.cos(math.pi * (2 * n + 1) * k / (2 * side))
    matrix *= math.sqrt(2 / side)
    matrix[0] /= math.sqrt(2)
    return matrix


_DCT: Final = _dct_matrix(_PHASH_SIDE)


def _phash_of(luminance: Image.Image) -> int:
    """Hash the low frequencies of a gray image on 64 bits.

    Args:
        luminance: The image put on white, in gray.

    Returns:
        The perceptual hash.
    """
    small = _small(luminance, (_PHASH_SIDE, _PHASH_SIDE))
    pixels = np.asarray(small, dtype=np.float64)
    coefficients = _DCT @ pixels @ _DCT.T
    block = coefficients[:_PHASH_BLOCK, :_PHASH_BLOCK]
    block = np.round(block, _PHASH_DECIMALS)
    return _bits(block > np.median(block))


def phash(image: Image.Image) -> int:
    """Hash the low frequencies of an image on 64 bits.

    Args:
        image: Any decoded image.

    Returns:
        The perceptual hash.
    """
    return _phash_of(gray(image))


def hashes(image: Image.Image) -> tuple[int, int]:
    """Hash an image both ways, from one gray image.

    Args:
        image: Any decoded image.

    Returns:
        Its difference hash and its perceptual hash, as `dhash` and `phash`
        give them.
    """
    luminance = gray(image)
    return _dhash_of(luminance), _phash_of(luminance)


def hamming(first: int, second: int) -> int:
    """Count the bits that differ between two hashes.

    Args:
        first: A hash.
        second: Another hash.

    Returns:
        The Hamming distance.
    """
    return (first ^ second).bit_count()


def hamming_matrix(
    rows: Sequence[int],
    columns: Sequence[int],
) -> NDArray[np.uint8]:
    """Compute the Hamming distance of every pair of hashes.

    Args:
        rows: Hashes of the rows.
        columns: Hashes of the columns.

    Returns:
        A matrix of `len(rows)` by `len(columns)` distances.
    """
    left = np.array(rows, dtype=np.uint64)[:, None]
    right = np.array(columns, dtype=np.uint64)[None, :]
    return np.bitwise_count(left ^ right)
