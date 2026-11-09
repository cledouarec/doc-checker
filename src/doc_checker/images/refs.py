# Copyright 2026 Christophe Le Douarec
"""Builders of the `ImageRef` of an image, and its description.

The facts that the pixels of an encoded image tell are kept by its content
digest, for the documents that a process reads: the same bytes on both sides
of a comparison, or on many pages, are decoded once.
"""

from dataclasses import dataclass
from hashlib import blake2b
from typing import TYPE_CHECKING, Final

from doc_checker.images.hashing import hashes
from doc_checker.images.pixels import (
    UnreadableImage,
    jpeg_quality,
    open_raster,
    uses_alpha,
    web_signature,
)
from doc_checker.model.document import ImageRef, PixelSource

if TYPE_CHECKING:
    from PIL import Image

_DIGEST_BYTES = 16
# The most facts kept; the oldest goes first.
FACTS_KEPT: Final = 65_536
# The formats that a reference names otherwise: a multi-picture JPEG, as
# phones write, is a JPEG.
_FORMAT_NAMES = {"mpo": "jpeg"}


def content_digest(data: bytes) -> str:
    """Digest the bytes of an image.

    Args:
        data: Encoded image.

    Returns:
        32 hexadecimal characters.
    """
    return blake2b(data, digest_size=_DIGEST_BYTES).hexdigest()


def _suffix(name: str) -> str:
    """Take the lower case suffix of a file name, without its dot.

    Args:
        name: File name or path.

    Returns:
        The suffix, or "".
    """
    base = name.rsplit("/", 1)[-1]
    return base.rsplit(".", 1)[-1].lower() if "." in base else ""


@dataclass(frozen=True, slots=True)
class _Facts:
    """What the pixels of an encoded raster image tell.

    Attributes:
        format: Lower case format, as a reference names it.
        width: Width in pixels.
        height: Height in pixels.
        phash: Perceptual hash.
        dhash: Difference hash.
        quality: JPEG quality, if known.
        alpha: Whether a pixel is not opaque.
    """

    format: str
    width: int
    height: int
    phash: int
    dhash: int
    quality: int | None
    alpha: bool


# The facts, or the reason why the pixels cannot be read, by content digest.
_known: dict[str, _Facts | str] = {}


def _facts(data: bytes, digest: str) -> _Facts | str:
    """Decode an image once for its digest, and tell what its pixels say.

    Args:
        data: Encoded image.
        digest: Its content digest.

    Returns:
        The facts, or the reason why the pixels cannot be read.
    """
    found = _known.get(digest)
    if found is not None:
        return found
    try:
        image = open_raster(data)
    except UnreadableImage as error:
        found = error.reason
    else:
        name = (image.format or "").lower()
        dhash, phash = hashes(image)
        found = _Facts(
            format=_FORMAT_NAMES.get(name, name),
            width=image.width,
            height=image.height,
            phash=phash,
            dhash=dhash,
            quality=jpeg_quality(image),
            alpha=uses_alpha(image),
        )
    if len(_known) >= FACTS_KEPT:
        del _known[next(iter(_known))]
    _known[digest] = found
    return found


def raster_ref(
    data: bytes,
    *,
    name: str,
    source: PixelSource | None,
    alt: str | None = None,
    shown: tuple[float, float] | None = None,
) -> ImageRef:
    """Build the `ImageRef` of an encoded raster image.

    The image is decoded only the first time that its content digest comes.

    Args:
        data: Encoded image.
        name: Name of the image in the document.
        source: Where the pixels are read again.
        alt: Alternative text.
        shown: Displayed size.

    Returns:
        The reference, with a `problem` and no hash if unreadable, and the
        format of the web that the signature of its bytes tells, if any.
    """
    digest = content_digest(data)
    facts = _facts(data, digest)
    if isinstance(facts, str):
        return unreadable_ref(
            name=name,
            format=_suffix(name),
            digest=digest,
            problem=facts,
            alt=alt,
            shown=shown,
            size=len(data),
            web_format=web_signature(data),
        )
    return ImageRef(
        name=name,
        format=facts.format,
        digest=digest,
        width=facts.width,
        height=facts.height,
        shown=shown,
        alt=alt,
        phash=facts.phash,
        dhash=facts.dhash,
        quality=facts.quality,
        alpha=facts.alpha,
        size=len(data),
        source=source,
    )


def pixels_ref(  # ruff: ignore[too-many-arguments] - the fields of a decoded PDF bitmap
    image: Image.Image,
    *,
    format: str,  # ruff: ignore[builtin-argument-shadowing] - field name of ImageRef
    digest: str,
    source: PixelSource,
    size: int | None = None,
    quality: int | None = None,
    shown: tuple[float, float] | None = None,
    dpi: float | None = None,
    vector: bool = False,
) -> ImageRef:
    """Build the `ImageRef` of a bitmap decoded by PDFium.

    Args:
        image: The decoded bitmap.
        format: Lower case encoding of the PDF object.
        digest: Content digest.
        source: Where the pixels are read again.
        size: Size of the encoded image.
        quality: JPEG quality, if known.
        shown: Displayed size in points.
        dpi: Effective resolution.
        vector: Whether the bitmap is the render of a vector figure.

    Returns:
        The reference.
    """
    dhash, phash = hashes(image)
    return ImageRef(
        name="",
        format=format,
        digest=digest,
        width=image.width,
        height=image.height,
        shown=shown,
        dpi=dpi,
        vector=vector,
        phash=phash,
        dhash=dhash,
        quality=quality,
        alpha=None,
        size=size,
        source=source,
    )


def unreadable_ref(  # ruff: ignore[too-many-arguments] - the fields of an image without pixels
    *,
    name: str,
    format: str,  # ruff: ignore[builtin-argument-shadowing] - field name of ImageRef
    digest: str,
    problem: str,
    alt: str | None = None,
    shown: tuple[float, float] | None = None,
    size: int | None = None,
    vector: bool = False,
    web_format: str | None = None,
) -> ImageRef:
    """Build the `ImageRef` of an image without pixels.

    Args:
        name: Name of the image in the document.
        format: Lower case encoding, or "".
        digest: Content digest.
        problem: Why the pixels cannot be read.
        alt: Alternative text.
        shown: Displayed size.
        size: Size of the encoded image.
        vector: Whether the image is a drawing.
        web_format: Format of the web that the signature of its bytes
            tells, as `web_signature` gives it.

    Returns:
        The reference, without hash or source.
    """
    return ImageRef(
        name=name,
        format=format,
        digest=digest,
        shown=shown,
        alt=alt,
        vector=vector,
        size=size,
        problem=problem,
        web_format=web_format,
    )


def describe(ref: ImageRef) -> str:
    """Describe an image for a report.

    Args:
        ref: The image.

    Returns:
        Comma separated name, size, format, resolution, alt and problem.
    """
    parts = [ref.name or ("vector figure" if ref.vector else "image")]
    if ref.format == "vector" and ref.shown is not None:
        parts.append(f"{ref.shown[0]:.0f}x{ref.shown[1]:.0f} pt")
    elif ref.width is not None and ref.height is not None:
        parts.append(f"{ref.width}x{ref.height} px")
    if ref.format and ref.format != "vector":
        parts.append(ref.format.upper())
    if ref.dpi is not None:
        parts.append(f"{ref.dpi:.0f} dpi")
    if ref.alt is not None:
        parts.append(f'alt "{ref.alt}"')
    if ref.problem is not None:
        parts.append(f"unreadable: {ref.problem}")
    return ", ".join(parts)
