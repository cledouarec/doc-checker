# Copyright 2026 Christophe Le Douarec
"""Tests for the quality measures and the verdict of a pair of images."""

import time
import warnings
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image, ImageDraw

from doc_checker.compare.verdicts import Verdict, judge
from doc_checker.images.metrics import (
    ImageQuality,
    adjacent_changes,
    assess,
    block_grid_share,
    blockiness,
    blur_score,
    color_count,
    grid,
    measure,
)
from doc_checker.images.pixels import open_raster
from doc_checker.images.refs import raster_ref
from doc_checker.model.findings import Severity
from tests.image_builders import (
    blurred,
    drawing,
    encode,
    grayed,
    photo,
    pixelated,
    scaled,
    screenshot,
    swatches,
)

if TYPE_CHECKING:
    from collections.abc import Callable

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


def test_sharp_pictures_score_above_the_blur_threshold() -> None:
    """Photographs and drawings are far from blurry."""
    assert blur_score(photo("street")) >= 0.3
    assert blur_score(drawing("setup")) >= 0.6


def test_blurred_pictures_score_below_the_blur_threshold() -> None:
    """A Gaussian blur of a few pixels takes the score under 0.10."""
    assert blur_score(blurred(photo("street"), 3.0)) < 0.10
    assert blur_score(blurred(drawing("setup"), 2.0)) < 0.10


def test_flat_picture_has_no_blur_score() -> None:
    """Without textured tiles, the sharpness cannot be told."""
    assert blur_score(Image.new("RGB", (400, 250), "white")) is None


def test_jpeg_blocks_are_measured_on_the_pixel_grid() -> None:
    """A JPEG of quality 8 saved as PNG keeps its 8 x 8 blocks."""
    once = open_raster(encode(photo("street"), "JPEG", quality=8))
    blocky = open_raster(encode(once))
    assert blockiness(blocky) >= 2.0
    assert blockiness(photo("street")) < 1.3
    assert blockiness(drawing("quota")) < 1.3


@pytest.mark.parametrize("quality", [8, 20, 30, 50])
def test_jpeg_blocks_win_on_their_grid(quality: int) -> None:
    """Nearly every 8 pixel run of a JPEG steps most on its block border."""
    once = open_raster(encode(photo("street"), "JPEG", quality=quality))
    picture = open_raster(encode(once))
    assert block_grid_share(picture) >= 0.9
    assert assess(picture).block_grid_share == block_grid_share(picture)


def test_edges_on_the_grid_are_no_jpeg_blocks() -> None:
    """A few strong edges on the block borders, flat between, score low."""
    for picture in (screenshot(), swatches()):
        assert blockiness(picture) >= 2.0
        assert block_grid_share(picture) <= 0.4
    assert block_grid_share(photo("street")) <= 0.4
    assert block_grid_share(drawing("setup")) <= 0.4


def test_nearest_neighbour_upscale_is_pixelated() -> None:
    """Repeated columns and rows make a pixelated picture."""
    assert assess(pixelated(photo("street"))).pixelated
    assert assess(pixelated(drawing("setup"))).pixelated
    assert not assess(photo("street")).pixelated
    assert not assess(drawing("setup")).pixelated


def test_two_colour_modules_are_not_pixelated() -> None:
    """A code of black and white modules has too few gray levels."""
    modules = Image.new("L", (21, 21), 255)
    for x in range(21):
        for y in range(21):
            if (x * 7 + y * 3) % 5 == 0:
                modules.putpixel((x, y), 0)
    code = modules.resize((210, 210), Image.Resampling.NEAREST)
    assert not assess(code).pixelated


def test_small_pictures_have_neutral_measures() -> None:
    """Under four blocks a side, nothing is measured."""
    quality = assess(Image.new("RGB", (20, 20), "red"))
    assert (quality.blur, quality.blockiness, quality.adjacent) == (
        None,
        1.0,
        1.0,
    )
    assert not quality.pixelated
    assert quality.block_grid_share == pytest.approx(0.0)


def test_photographs_have_many_colours() -> None:
    """A photograph has thousands of colours, a drawing a few dozens."""
    assert color_count(photo("street")) >= 10_000
    assert color_count(drawing("setup")) < 1_000


def test_grid_keeps_the_alignment_of_the_blocks() -> None:
    """A wide picture is cut to 2048 columns starting on a multiple of 8."""
    stripes = Image.new("L", (5000, 64), 0)
    draw = ImageDraw.Draw(stripes)
    for x in range(0, 5000, 8):
        draw.line((x, 0, x, 63), fill=255)
    cut = grid(stripes)
    assert cut.shape == (64, 2048)
    assert cut[0, 0] == 255


def test_photo_is_deterministic() -> None:
    """A photograph is the same picture every time."""
    assert photo("street").tobytes() == photo("street").tobytes()


def test_quality_of_many_pictures_is_quick() -> None:
    """A hundred photographs are assessed in seconds."""
    pictures = [photo(f"p{k}") for k in range(100)]
    start = time.perf_counter()
    for picture in pictures:
        assess(picture)
    assert time.perf_counter() - start < 20.0


def _degenerate(kind: str) -> Image.Image:
    """Build a picture too small, flat or exotic to be measured."""
    pictures = {
        "1x1": Image.new("RGB", (1, 1), "red"),
        "1xN": Image.new("L", (1, 300), 7),
        "Nx1": Image.new("L", (300, 1), 7),
        "flat": Image.new("RGB", (200, 200), "white"),
        "transparent": Image.new("RGBA", (64, 64), (0, 0, 0, 0)),
        "palette": Image.new("P", (64, 64)),
        "16-bit": Image.new("I;16", (64, 64)),
    }
    return pictures[kind]


@pytest.mark.parametrize(
    "kind",
    ["1x1", "1xN", "Nx1", "flat", "transparent", "palette", "16-bit"],
)
def test_degenerate_pictures_have_neutral_measures(kind: str) -> None:
    """Every measure says it cannot judge, without error or warning."""
    image = _degenerate(kind)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert blur_score(image) is None
        assert blockiness(image) == pytest.approx(1.0)
        assert adjacent_changes(image) == pytest.approx(1.0)
        assert color_count(image) >= 1
        assert block_grid_share(image) == pytest.approx(0.0)
        quality = assess(image)
    assert (quality.blur, quality.blockiness, quality.adjacent) == (
        None,
        1.0,
        1.0,
    )
    assert quality.levels == 1
    assert quality.block_grid_share == pytest.approx(0.0)
    assert quality.colors == color_count(image)
    assert not quality.pixelated
    assert isinstance(quality, ImageQuality)


def test_adjacent_changes_tell_repeated_columns_from_smooth_ones() -> None:
    """Repeated columns and rows score near 0, a photograph 1."""
    assert adjacent_changes(photo("street")) == pytest.approx(1.0)
    assert adjacent_changes(pixelated(photo("street"))) <= 0.10


def _blocky_and_clean() -> tuple[bytes, bytes]:
    """Build a twice-encoded JPEG and a clean one of the same picture."""
    picture = photo("street")
    once = open_raster(encode(picture, "JPEG", quality=10))
    blocky = encode(once, "JPEG", quality=90)
    return blocky, encode(picture, "JPEG", quality=90)


def test_repaired_image_is_not_degraded() -> None:
    """A blocky reference facing a clean image of its quality: re-encoded."""
    blocky, clean = _blocky_and_clean()
    pair = measure(open_raster(blocky), open_raster(clean))
    assert pair.ref_blockiness >= 1.4
    assert pair.ref_blockiness >= 1.3 * pair.new_blockiness
    assert _verdict(blocky, clean).rule == REENCODED


def test_blocky_new_image_is_still_degraded() -> None:
    """The other way round, the blocks are a degradation."""
    blocky, clean = _blocky_and_clean()
    assert _verdict(clean, blocky).rule == DEGRADED


def test_blur_of_a_clean_reference_is_still_degraded() -> None:
    """A clean reference has no blocks that would excuse a blurred image."""
    picture = drawing("setup")
    verdict = _verdict(encode(picture), encode(blurred(picture)))
    assert (verdict.rule, verdict.pattern) == (DEGRADED, "sharpness")


def test_blur_of_a_blocky_reference_is_still_degraded() -> None:
    """The repair signal excuses a mild sharpness loss only."""
    blocky, _ = _blocky_and_clean()
    new = encode(blurred(photo("street"), 3), "JPEG", quality=90)
    verdict = _verdict(blocky, new)
    assert (verdict.rule, verdict.pattern) == (DEGRADED, "sharpness")


@pytest.mark.parametrize(
    ("change", "cause"),
    [(lambda p: scaled(p, 0.5), "resolution"), (grayed, "color")],
    ids=["downscaled", "grayed"],
)
def test_repair_signal_hides_no_other_cause(
    change: Callable[[Image.Image], Image.Image],
    cause: str,
) -> None:
    """A blocky reference facing a smaller or grayed image is degraded."""
    blocky, _ = _blocky_and_clean()
    new = encode(change(photo("street")), "JPEG", quality=90)
    verdict = _verdict(blocky, new)
    assert (verdict.rule, verdict.pattern) == (DEGRADED, cause)


def test_blocky_flat_drawing_pairs() -> None:
    """A flat drawing repaired is re-encoded, blurred it is degraded."""
    picture = drawing("setup")
    once = open_raster(encode(picture, "JPEG", quality=10))
    blocky = encode(once, "JPEG", quality=90)
    clean = encode(picture, "JPEG", quality=90)
    blur = encode(blurred(picture, 3), "JPEG", quality=90)
    assert _verdict(blocky, clean).rule == REENCODED
    verdict = _verdict(blocky, blur)
    assert (verdict.rule, verdict.pattern) == (DEGRADED, "sharpness")
