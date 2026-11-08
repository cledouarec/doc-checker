# Copyright 2026 Christophe Le Douarec
"""WebP thumbnails of the images that the reports show.

A thumbnail is the image put on white, shrunk to fit a square of
`THUMB_SIDE` pixels, and encoded again by Pillow with fixed settings and no
metadata: the same pixels give the same bytes, and no byte of the document
comes through. Its name is a digest of these bytes.
"""

import base64
import io
from dataclasses import dataclass, field
from hashlib import blake2b
from typing import Final

from PIL import Image

from doc_checker.images.pixels import flatten

THUMB_SIDE: Final = 240
WEBP_QUALITY: Final = 75
WEBP_METHOD: Final = 4
_NAME_BYTES: Final = 8


@dataclass(frozen=True, slots=True)
class Thumbnail:
    """A WebP thumbnail.

    Attributes:
        data: The WebP file.
        width: Its width in pixels.
        height: Its height in pixels.
    """

    data: bytes = field(repr=False)
    width: int
    height: int

    @property
    def name(self) -> str:
        """File name of the thumbnail: 16 hexadecimal digits of its digest."""
        digest = blake2b(self.data, digest_size=_NAME_BYTES).hexdigest()
        return f"{digest}.webp"

    @property
    def data_uri(self) -> str:
        """Data URL of the thumbnail, in base64."""
        encoded = base64.b64encode(self.data).decode("ascii")
        return f"data:image/webp;base64,{encoded}"


def make_thumbnail(image: Image.Image) -> Thumbnail:
    """Make the thumbnail of an image.

    Args:
        image: Decoded pixels of any mode, left unchanged.

    Returns:
        The image put on white, shrunk with LANCZOS to fit `THUMB_SIDE`
        pixels on both sides, but never enlarged, and encoded in lossy WebP
        at `WEBP_QUALITY` with `WEBP_METHOD`.
    """
    small = flatten(image)  # A new image: `thumbnail` changes it in place.
    small.thumbnail((THUMB_SIDE, THUMB_SIDE), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    small.save(
        buffer,
        format="WEBP",
        quality=WEBP_QUALITY,
        method=WEBP_METHOD,
    )
    return Thumbnail(buffer.getvalue(), small.width, small.height)
