# Copyright 2026 Christophe Le Douarec
"""Tests for the quality measures and the verdict of a pair of images."""

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from doc_checker.compare.verdicts import Verdict, judge
from doc_checker.images.metrics import measure
from doc_checker.images.pixels import open_raster
from doc_checker.images.refs import raster_ref
from doc_checker.model.findings import Severity
from tests.image_builders import (
    blurred,
    drawing,
    encode,
    grayed,
    pixelated,
    scaled,
)

DEGRADED = "image.degraded"
REENCODED = "image.reencoded"


def _verdict(
    ref_data: bytes,
    new_data: bytes,
    *,
    pdf: bool = False,
    new_dpi: float | None = None,
) -> Verdict:
    """Judge two encoded images as the comparison does."""
    ref = raster_ref(ref_data, name="a", source=None)
    new = replace(raster_ref(new_data, name="b", source=None), dpi=new_dpi)
    measures = measure(open_raster(ref_data), open_raster(new_data))
    return judge(ref, new, measures, pdf=pdf)


def _png(image: Image.Image) -> bytes:
    """Encode an image as PNG."""
    return encode(image)


ORIGINAL = drawing("setup")

PLANTED = [
    ("jpeg10", encode(ORIGINAL, "JPEG", quality=10), "jpeg quality"),
    ("x0.4", _png(scaled(ORIGINAL, 0.4)), "resolution"),
    ("blur", _png(blurred(ORIGINAL)), "sharpness"),
    ("nearest", _png(pixelated(ORIGINAL)), "sharpness"),
    ("gray", _png(grayed(ORIGINAL)), "color"),
]


@pytest.mark.parametrize(
    ("degraded", "cause"),
    [(data, cause) for _, data, cause in PLANTED],
    ids=[name for name, _, _ in PLANTED],
)
def test_planted_degradations_are_degraded(
    degraded: bytes,
    cause: str,
) -> None:
    """Each planted degradation is reported with its cause, as major."""
    verdict = _verdict(_png(ORIGINAL), degraded)
    assert (verdict.rule, verdict.severity) == (DEGRADED, None)
    assert verdict.pattern is not None
    assert cause in verdict.pattern.split("+")


def test_strong_downscale_is_critical() -> None:
    """Less than four tenths of the pixels in each direction is critical."""
    verdict = _verdict(_png(ORIGINAL), _png(scaled(ORIGINAL, 0.3)))
    assert (verdict.rule, verdict.severity) == (DEGRADED, Severity.CRITICAL)
    assert verdict.detail is not None
    assert verdict.detail == "resolution x0.30 (400x250\U00002192120x75 px)"


def test_effective_resolution_is_named_when_it_decides() -> None:
    """A blocky upscale keeps its pixels but loses its effective resolution."""
    rng = np.random.default_rng(1)
    texture = Image.fromarray(
        rng.integers(0, 256, (250, 400, 3), dtype=np.uint8),
    )
    blocky = texture.resize((160, 100), Image.Resampling.NEAREST).resize(
        texture.size,
        Image.Resampling.NEAREST,
    )
    verdict = _verdict(_png(texture), _png(blocky))
    assert verdict.detail is not None
    assert verdict.detail.startswith(
        "effective resolution x0.40 (400x250\U00002192400x250 px); ",
    )


def test_details_write_their_arrows_as_the_patterns_do() -> None:
    """The formats and the qualities of a detail are joined by an arrow."""
    jpeg = encode(ORIGINAL, "JPEG", quality=10)
    assert "JPEG quality 100\U0000219215" in (
        _verdict(_png(ORIGINAL), jpeg).detail or ""
    )
    verdict = _verdict(_png(ORIGINAL), encode(ORIGINAL, "JPEG", quality=95))
    assert (verdict.detail or "").startswith("PNG\U00002192JPEG, SSIM ")


@pytest.mark.parametrize(
    "degraded",
    [data for _, data, _ in PLANTED],
    ids=[name for name, _, _ in PLANTED],
)
def test_better_image_is_no_degradation(degraded: bytes) -> None:
    """The repaired image of a degraded reference is never degraded."""
    assert _verdict(degraded, _png(ORIGINAL)).rule in {None, REENCODED}


def test_high_quality_reencoding_is_information() -> None:
    """A PNG saved as a JPEG of quality 95 is only re-encoded."""
    verdict = _verdict(_png(ORIGINAL), encode(ORIGINAL, "JPEG", quality=95))
    assert (verdict.rule, verdict.pattern) == (REENCODED, "png\U00002192jpeg")


def test_identical_pixels_give_no_finding() -> None:
    """Another compression of the same pixels changes nothing."""
    other = encode(ORIGINAL, optimize=True, compress_level=1)
    assert _verdict(_png(ORIGINAL), other).rule is None


def test_lost_transparency_is_a_degradation() -> None:
    """A transparent image flattened into a JPEG lost its transparency."""
    clear = ORIGINAL.convert("RGBA")
    clear.putalpha(0)
    verdict = _verdict(_png(clear), encode(clear, "JPEG", quality=95))
    assert verdict.pattern is not None
    assert "transparency" in verdict.pattern.split("+")


def test_libreoffice_like_reencoding_is_no_degradation() -> None:
    """A PDF image halved to 300 DPI and saved at quality 90 is kept."""
    big = drawing("setup", size=(1200, 750))
    half = encode(scaled(big, 0.5), "JPEG", quality=90)
    kept = _verdict(_png(big), half, pdf=True, new_dpi=300.0)
    assert kept.rule == REENCODED
    lowered = _verdict(_png(big), half, pdf=True, new_dpi=150.0)
    assert lowered.rule == DEGRADED
