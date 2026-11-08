# Copyright 2026 Christophe Le Douarec
"""Safe opening of raster images and the facts read from their pixels."""

import io
import warnings
from typing import TYPE_CHECKING, Final

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path

MAX_PIXELS: Final = 40_000_000
MAX_FILE_BYTES: Final = 50_000_000
# The raster formats of the web, the only ones opened. Any other format, and
# above all PostScript, which Pillow would hand to Ghostscript, is refused.
# A multi-picture JPEG (MPO), as phones write, opens as a JPEG does: Pillow
# reads it with its JPEG opener and has no opener of its own for it.
ALLOWED_FORMATS: Final = ("PNG", "JPEG", "GIF", "WEBP", "BMP", "ICO", "TIFF")
ALPHA_OPAQUE: Final = 250
# Why the pixels of an image are not read: the `problem` of its `ImageRef`.
TOO_LARGE: Final = "too large"
CANNOT_DECODE: Final = "cannot decode"
NOT_FOUND: Final = "not found"
OUTSIDE: Final = "outside the site"
EXTERNAL: Final = "external"  # Never fetched, compared by its URL.
# A vector figure of a PDF page past the figures rendered on it, which a
# warning of its page counts.
PAGE_LIMIT: Final = "past the page limit"
# A vector figure of a PDF page that `/Rotate` turns, which is not rendered.
NOT_RENDERED: Final = "not rendered"
# The problems of the images left unread on purpose: an image that is never
# fetched, or a vector figure that its page draws but that is not rendered.
# Such an image is not broken.
NOT_READ: Final = frozenset({EXTERNAL, PAGE_LIMIT, NOT_RENDERED})
LOSSLESS_FORMATS: Final = frozenset(
    {"png", "gif", "bmp", "tiff", "flate", "raw", "ccitt", "jbig2"},
)

_ALPHA_MODES: Final = frozenset({"RGBA", "LA", "PA"})
_WHITE: Final = (255, 255, 255, 255)
_FULL_SCALE: Final = 100
_IJG_LUMINANCE: Final = (
    *(16, 11, 10, 16, 24, 40, 51, 61),
    *(12, 12, 14, 19, 26, 58, 60, 55),
    *(14, 13, 16, 24, 40, 57, 69, 56),
    *(14, 17, 22, 29, 51, 87, 80, 62),
    *(18, 22, 37, 56, 68, 109, 103, 77),
    *(24, 35, 55, 64, 81, 104, 113, 92),
    *(49, 64, 78, 87, 103, 121, 120, 101),
    *(72, 92, 95, 98, 112, 100, 103, 99),
)
_IJG_SUM: Final = sum(_IJG_LUMINANCE)


class UnreadableImage(Exception):  # ruff: ignore[error-suffix-on-exception-name] - public name set by the plan
    """An image whose pixels cannot be read.

    Attributes:
        reason: Short reason, such as `TOO_LARGE` or `CANNOT_DECODE`.
    """

    def __init__(self, reason: str) -> None:
        """Keep the reason.

        Args:
            reason: Why the image is unreadable.
        """
        super().__init__(reason)
        self.reason = reason


def read_image_file(file: Path, limit: int = MAX_FILE_BYTES) -> bytes:
    """Read an image file, its size checked before it is read.

    Args:
        file: The file.
        limit: The most bytes it may have.

    Returns:
        Its bytes.

    Raises:
        UnreadableImage: `NOT_FOUND` if it is not a file or cannot be read,
            `TOO_LARGE` beyond the limit.
    """
    try:
        size = file.stat().st_size if file.is_file() else None
    except OSError:
        size = None
    if size is None:
        raise UnreadableImage(NOT_FOUND)
    if size > limit:
        raise UnreadableImage(TOO_LARGE)
    try:
        with file.open("rb") as stream:
            data = stream.read(limit + 1)
    except OSError:
        raise UnreadableImage(NOT_FOUND) from None
    if len(data) > limit:
        raise UnreadableImage(TOO_LARGE)
    return data


def _refuse_large(image: Image.Image) -> None:
    """Refuse an image that announces too many pixels, before decoding.

    Args:
        image: An image opened, not loaded.

    Raises:
        UnreadableImage: If the image has more than `MAX_PIXELS`.
    """
    if image.width * image.height > MAX_PIXELS:
        raise UnreadableImage(TOO_LARGE)


def open_raster(data: bytes) -> Image.Image:
    """Decode a raster image, refusing too large images before decoding.

    Only the `ALLOWED_FORMATS` are opened. The warnings of Pillow about the
    file are silenced, except the one about too many pixels, which refuses
    it.

    Args:
        data: Encoded image, from an untrusted document.

    Returns:
        The loaded image (the first frame of an animation), its format kept.

    Raises:
        UnreadableImage: If the image is too large, or if it cannot be
            decoded, which an image of another format cannot.
    """
    try:
        return _decoded(data)
    except Image.DecompressionBombError, Image.DecompressionBombWarning:
        raise UnreadableImage(TOO_LARGE) from None
    except UnreadableImage:
        raise  # `_refuse_large` already tells why, before any decoding.
    except Exception:  # ruff: ignore[blind-except] - any decoder failure on an untrusted file means it cannot be decoded
        raise UnreadableImage(CANNOT_DECODE) from None


def _decoded(data: bytes) -> Image.Image:
    """Open and load an image of `ALLOWED_FORMATS`, its warnings silenced.

    Args:
        data: Encoded image.

    Returns:
        The loaded image. Too many pixels raise `UnreadableImage` before
        decoding, or Pillow's `DecompressionBombWarning` as an error.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        image = Image.open(io.BytesIO(data), formats=ALLOWED_FORMATS)
        _refuse_large(image)
        image.load()
    return image


def _has_transparency(image: Image.Image) -> bool:
    """Tell whether the image has an alpha channel or a transparent colour.

    Args:
        image: Any decoded image.

    Returns:
        True if a transparency is declared.
    """
    return image.mode in _ALPHA_MODES or "transparency" in image.info


def _to_eight_bits(image: Image.Image) -> Image.Image:
    """Scale a 16 or 32 bit gray image to 8 bits instead of clamping it.

    Args:
        image: An image of mode `I;16*`, `I` or `F`.

    Returns:
        An `L` image.
    """
    values = np.asarray(image, dtype=np.float64)
    if image.mode.startswith("I;16"):
        return Image.fromarray((values // 256).astype(np.uint8), "L")
    low, high = float(values.min()), float(values.max())
    if high <= low:
        return Image.new("L", image.size, 0)
    scaled = (values - low) * (255 / (high - low))
    return Image.fromarray(scaled.astype(np.uint8), "L")


def flatten(image: Image.Image) -> Image.Image:
    """Put the image on a white page, as RGB.

    Args:
        image: Any decoded image.

    Returns:
        An RGB image without transparency.
    """
    if image.mode.startswith(("I", "F")):
        image = _to_eight_bits(image)
    if not _has_transparency(image):
        return image.convert("RGB")
    rgba = image.convert("RGBA")
    page = Image.new("RGBA", rgba.size, _WHITE)
    return Image.alpha_composite(page, rgba).convert("RGB")


def gray(image: Image.Image) -> Image.Image:
    """Return the luminance of the image put on white.

    Args:
        image: Any decoded image.

    Returns:
        An `L` image.
    """
    return flatten(image).convert("L")


def uses_alpha(image: Image.Image) -> bool:
    """Whether some pixel of the image is not opaque.

    Args:
        image: Any decoded image.

    Returns:
        True if there is a transparency and a pixel has an alpha below
        `ALPHA_OPAQUE`.
    """
    if not _has_transparency(image):
        return False
    low, _high = image.convert("RGBA").getchannel("A").getextrema()
    return low < ALPHA_OPAQUE


def jpeg_quality(image: Image.Image) -> int | None:
    """Estimate the quality of a JPEG from its luminance table.

    Args:
        image: A decoded image.

    Returns:
        The IJG quality from 1 to 100, or None without quantization tables.
    """
    tables = getattr(image, "quantization", None)
    if not tables or 0 not in tables:
        return None
    scale = _FULL_SCALE * sum(tables[0]) / _IJG_SUM
    if scale <= _FULL_SCALE:
        quality = (2 * _FULL_SCALE - scale) / 2
    else:
        quality = 50 * _FULL_SCALE / scale
    return max(1, min(_FULL_SCALE, round(quality)))
