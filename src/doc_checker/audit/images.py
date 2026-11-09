# Copyright 2026 Christophe Le Douarec
"""Check the images of a document: weight, resolution, quality, pixels.

An image whose file is missing or out of the site, or that cannot be
decoded, a problem of `BROKEN_PROBLEMS`, is broken, and nothing else is told
of it; one too large to be read, external or left unrendered is not: a
browser or a viewer may still show it. Nor is an image of a format that
browsers show though it is not decoded here, AVIF or JPEG XL, as its
signature tells, nor a vector figure of a PDF, whose paths are drawn even
when an image inside it is refused, an image reported as its own block. An
image file is heavy beyond ``heavy_minor`` bytes, and very heavy beyond
``heavy_major``.

A raster image of a PDF whose longest side is shown on `ICON_SHOWN_MIN`
points or more is judged by its effective resolution, against the limits of
the profile of the audit. An image of an HTML page that its ``width`` and
``height`` show more than ``upscale`` times its size is upscaled. A JPEG is
judged by its quality, as its luminance table tells.

The pixels of a readable raster image are read again, by a `PixelLoader`,
and measured by `assess`, once for each content digest: the image is
blurry below ``blur``; blocky from ``blockiness``, unless its JPEG quality
is already reported; pixelated when its rows and columns repeat. A PNG of
`PHOTO_PNG_MIN_BYTES` or more, of `PHOTO_BPP_MIN` bits per pixel or more and
`PHOTO_COLORS_MIN` colours or more is a photograph that a JPEG would hold in
fewer bytes. A vector figure has no pixels to judge, and an image whose
pixels can no longer be read is not judged by them. The blocks of a JPEG
show only where nearly every run of 8 pixels steps most on its block
border, `BLOCK_GRID_MIN` of them on each axis: a screenshot whose few edges
fall on the borders has none. Nor has an image of fewer than
`PIXEL_LEVELS_MIN` gray levels, such as a checkerboard on the grid, or a
pixelated one, whose repeated pixels make the steps.

The weight, the JPEG quality and the pixels belong to a file: they are
reported once for each file, at its first use in reading order, a file
being its name and its content digest in an HTML page, its content digest
alone in a PDF, which names its images by their place. A broken image, a
low resolution and an upscaled image belong to each use, and are reported
at each.

A detail never prints a measure below its limit as the limit itself: the
sharpness is rounded down, and the resolution, rounded as `describe` shows
it, is rounded down where that would reach the limit.
"""

import math
from typing import TYPE_CHECKING, Final

from doc_checker.audit.sides import image_side
from doc_checker.compare.drafts import Draft
from doc_checker.images.metrics import PIXEL_LEVELS_MIN, assess
from doc_checker.images.pixels import CANNOT_DECODE, NOT_FOUND, OUTSIDE
from doc_checker.model.document import BlockKind
from doc_checker.model.findings import Severity

if TYPE_CHECKING:
    from doc_checker.config import AuditThresholds
    from doc_checker.images.load import PixelLoader
    from doc_checker.images.metrics import ImageQuality
    from doc_checker.model.document import Document, ImageRef
    from doc_checker.model.findings import Side

ICON_SHOWN_MIN: Final = 36.0
PHOTO_PNG_MIN_BYTES: Final = 102_400
PHOTO_BPP_MIN: Final = 6.0
PHOTO_COLORS_MIN: Final = 10_000
BROKEN_PROBLEMS: Final = frozenset({NOT_FOUND, OUTSIDE, CANNOT_DECODE})
MEGABYTE: Final = 1_000_000
BLOCK_GRID_MIN: Final = 0.7

_BROKEN: Final = "image.broken"
_HEAVY: Final = "image.heavy"
_LOW_DPI: Final = "image.low_dpi"
_UPSCALED: Final = "image.upscaled"
_JPEG_QUALITY: Final = "image.jpeg_quality"
_BLURRY: Final = "image.blurry"
_BLOCKY: Final = "image.blocky"
_PIXELATED: Final = "image.pixelated"
_PHOTO_PNG: Final = "image.photo_png"
_RULE_PREFIX: Final = "image."
_JPEG: Final = "jpeg"
_PNG: Final = "png"
_BITS_PER_BYTE: Final = 8
_HUNDREDTHS: Final = 100


def _pattern(rule: str) -> str:
    """Return the pattern of a rule that has one problem: its own name."""
    return rule.removeprefix(_RULE_PREFIX)


def _shortfall[N: (int, float)](
    value: N,
    limits: tuple[N, N],
) -> tuple[Severity | None, N] | None:
    """Return how far below its major and minor limits a value falls.

    Args:
        value: The value, such as a resolution.
        limits: The major limit, then the minor one.

    Returns:
        `Severity.MAJOR` and the major limit below it; ``None``, for the
        severity of the rule, and the minor limit below that one; else
        ``None``.
    """
    major, minor = limits
    if value < major:
        return Severity.MAJOR, major
    if value < minor:
        return None, minor
    return None


def _broken(image: ImageRef, side: Side, *, pdf: bool) -> Draft | None:
    """Return the draft of an image that cannot be shown, if it cannot.

    Args:
        image: The image.
        side: Its side.
        pdf: Whether the document is a PDF.

    Returns:
        The draft of a problem of `BROKEN_PROBLEMS`, but for bytes whose
        signature tells a format of the web that browsers show, and for a
        vector figure of a PDF; else ``None``.
    """
    problem = image.problem
    if (
        problem not in BROKEN_PROBLEMS
        or (problem == CANNOT_DECODE and image.web_format is not None)
        or (pdf and image.vector)
    ):
        return None
    detail = f"the image cannot be shown: {problem}"
    return Draft(_BROKEN, None, side, None, detail, problem)


def _heavy(
    image: ImageRef,
    side: Side,
    thresholds: AuditThresholds,
) -> Draft | None:
    """Return the draft of an image file that weighs too much.

    Returns:
        The draft of a file beyond ``heavy_minor`` bytes, major beyond
        ``heavy_major``; else ``None``.
    """
    size = image.size
    if size is None or size <= thresholds.heavy_minor:
        return None
    major = size > thresholds.heavy_major
    limit = thresholds.heavy_major if major else thresholds.heavy_minor
    return Draft(
        _HEAVY,
        Severity.MAJOR if major else None,
        side,
        None,
        f"{size / MEGABYTE:.1f} MB",
        f"over {limit / MEGABYTE:g} MB",
    )


def _low_dpi(
    image: ImageRef,
    side: Side,
    thresholds: AuditThresholds,
    profile: str,
) -> Draft | None:
    """Return the draft of a raster image of a PDF of a low resolution.

    Returns:
        For a raster image whose longest side is shown on `ICON_SHOWN_MIN`
        points or more, the draft of a resolution below the limits of the
        profile; else ``None``.
    """
    dpi, shown = image.dpi, image.shown
    if (
        image.vector
        or dpi is None
        or shown is None
        or max(shown) < ICON_SHOWN_MIN
    ):
        return None
    found = _shortfall(dpi, thresholds.dpi(profile))
    if found is None:
        return None
    severity, limit = found
    # As `describe` rounds it, but rounded down where that would reach the
    # limit.
    shown_dpi = round(dpi)
    if shown_dpi >= limit:
        shown_dpi = math.floor(dpi)
    detail = f"{shown_dpi} dpi, below {limit:g} for the {profile} profile"
    pattern = f"below {limit:g} dpi"
    return Draft(_LOW_DPI, severity, side, None, detail, pattern)


def _upscaled(
    image: ImageRef,
    side: Side,
    thresholds: AuditThresholds,
) -> Draft | None:
    """Return the draft of an HTML image shown larger than its size.

    Returns:
        The draft of a raster image that its ``width`` and ``height`` show
        more than ``upscale`` times its size on either side; else ``None``.
    """
    shown, width, height = image.shown, image.width, image.height
    if image.vector or shown is None or not width or not height:
        return None
    if max(shown[0] / width, shown[1] / height) <= thresholds.upscale:
        return None
    detail = f"{width}x{height} px shown at {shown[0]:.0f}x{shown[1]:.0f}"
    return Draft(_UPSCALED, None, side, None, detail, _pattern(_UPSCALED))


def _jpeg_quality(
    image: ImageRef,
    side: Side,
    thresholds: AuditThresholds,
) -> Draft | None:
    """Return the draft of a JPEG of a low quality.

    Returns:
        The draft of a JPEG whose quality is below ``jpeg_quality_minor``,
        major below ``jpeg_quality_major``; else ``None``.
    """
    quality = image.quality
    if image.format != _JPEG or quality is None:
        return None
    limits = (thresholds.jpeg_quality_major, thresholds.jpeg_quality_minor)
    found = _shortfall(quality, limits)
    if found is None:
        return None
    severity, limit = found
    detail = f"JPEG quality {quality}, below {limit}"
    return Draft(_JPEG_QUALITY, severity, side, None, detail, f"below {limit}")


def _photo_png(image: ImageRef, side: Side, colors: int) -> Draft | None:
    """Return the draft of a photograph kept as a PNG.

    Args:
        image: The image.
        side: Its side.
        colors: The colours of its pixels, as `assess` counts them.

    Returns:
        The draft of a PNG of `PHOTO_PNG_MIN_BYTES` or more, of
        `PHOTO_BPP_MIN` bits per pixel or more and of `PHOTO_COLORS_MIN`
        colours or more; else ``None``.
    """
    size, width, height = image.size, image.width, image.height
    if (
        image.format != _PNG
        or size is None
        or size < PHOTO_PNG_MIN_BYTES
        or not width
        or not height
    ):
        return None
    bpp = _BITS_PER_BYTE * size / (width * height)
    if bpp < PHOTO_BPP_MIN or colors < PHOTO_COLORS_MIN:
        return None
    detail = f"{colors} colours at {bpp:.1f} bits per pixel"
    return Draft(_PHOTO_PNG, None, side, None, detail, _pattern(_PHOTO_PNG))


def _pixel_drafts(
    image: ImageRef,
    side: Side,
    quality: ImageQuality,
    thresholds: AuditThresholds,
    *,
    blocks: bool,
) -> list[Draft]:
    """Return the drafts that the pixels of a raster image tell.

    Args:
        image: The image.
        side: Its side.
        quality: The measures of its pixels.
        thresholds: Thresholds of the audit.
        blocks: Whether its JPEG blocks are judged: not when its JPEG
            quality is reported.

    Returns:
        Its blur; its blocks, when its blockiness reaches ``blockiness``,
        its block runs `BLOCK_GRID_MIN`, its gray levels `PIXEL_LEVELS_MIN`
        and it is not pixelated; its pixelation; and a photograph kept as a
        PNG; in order.
    """
    drafts: list[Draft] = []
    blur = quality.blur
    if blur is not None and blur < thresholds.blur:
        floor = math.floor(blur * _HUNDREDTHS) / _HUNDREDTHS
        detail = f"sharpness {floor:.2f}, below {thresholds.blur:.2f}"
        drafts.append(
            Draft(_BLURRY, None, side, None, detail, _pattern(_BLURRY)),
        )
    if (
        blocks
        and quality.blockiness >= thresholds.blockiness
        and quality.block_grid_share >= BLOCK_GRID_MIN
        and quality.levels >= PIXEL_LEVELS_MIN
        and not quality.pixelated
    ):
        detail = (
            f"8x8 blocks {quality.blockiness:.2f},"
            f" from {thresholds.blockiness:.2f}"
        )
        drafts.append(
            Draft(_BLOCKY, None, side, None, detail, _pattern(_BLOCKY)),
        )
    if quality.pixelated:
        detail = "repeated rows and columns"
        drafts.append(
            Draft(_PIXELATED, None, side, None, detail, _pattern(_PIXELATED)),
        )
    photo = _photo_png(image, side, quality.colors)
    if photo is not None:
        drafts.append(photo)
    return drafts


def _measured(
    image: ImageRef,
    loader: PixelLoader,
    known: dict[str, ImageQuality | None],
) -> ImageQuality | None:
    """Return the measures of the pixels of an image, once for its digest.

    Args:
        image: A readable raster image.
        loader: What reads its pixels again.
        known: The measures already taken, by content digest; ``None`` for
            pixels that cannot be read again.

    Returns:
        The measures, or ``None`` if its pixels cannot be read again. The
        pixels are dropped once measured.
    """
    digest = image.digest
    if digest not in known:
        pixels = loader.load(image)
        known[digest] = None if pixels is None else assess(pixels)
    return known[digest]


def image_quality_drafts(
    doc: Document,
    loader: PixelLoader,
    thresholds: AuditThresholds,
    profile: str,
) -> list[Draft]:
    """Return the drafts of the images of a document.

    Args:
        doc: The audited document.
        loader: What reads the pixels of its images again.
        thresholds: Thresholds of the audit.
        profile: Profile of the audit, ``"screen"`` or ``"print"``.

    Returns:
        For each image block, in order: a broken image, and nothing else of
        it; then a heavy file, a low resolution in a PDF, an upscaled image
        in an HTML page and a low JPEG quality; then, for a readable raster
        image, its blur, its blocks, its pixelation and a photograph kept as
        a PNG. The weight, the JPEG quality and the pixels are told at the
        first use of a file only: its first block of its name and digest,
        or of its digest alone in a PDF.
    """
    pdf = doc.format == "pdf"
    html = doc.format == "html"
    known: dict[str, ImageQuality | None] = {}
    files: set[tuple[str, str]] = set()
    drafts: list[Draft] = []
    for block in doc.blocks:
        image = block.image
        if block.kind is not BlockKind.IMAGE or image is None:
            continue
        side = image_side(block)
        broken = _broken(image, side, pdf=pdf)
        if broken is not None:
            drafts.append(broken)
            continue
        file = ("" if pdf else image.name, image.digest)
        first = file not in files
        files.add(file)
        jpeg = _jpeg_quality(image, side, thresholds) if first else None
        found = (
            _heavy(image, side, thresholds) if first else None,
            _low_dpi(image, side, thresholds, profile) if pdf else None,
            _upscaled(image, side, thresholds) if html else None,
            jpeg,
        )
        drafts += [draft for draft in found if draft is not None]
        if not first or not image.readable or image.vector:
            continue
        quality = _measured(image, loader, known)
        if quality is not None:
            drafts += _pixel_drafts(
                image,
                side,
                quality,
                thresholds,
                blocks=jpeg is None,
            )
    return drafts
