# Copyright 2026 Christophe Le Douarec
"""Tests for the safe opening and the signatures of images."""

import warnings
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from doc_checker.images import refs
from doc_checker.images.hashing import (
    dhash,
    hamming,
    hamming_matrix,
    hashes,
    phash,
)
from doc_checker.images.pixels import (
    UnreadableImage,
    flatten,
    gray,
    jpeg_quality,
    open_raster,
    uses_alpha,
)
from doc_checker.images.refs import (
    content_digest,
    describe,
    pixels_ref,
    raster_ref,
    unreadable_ref,
)
from doc_checker.model.document import ImageRef, PixelSource
from tests.image_builders import (
    blurred,
    bomb_png,
    drawing,
    encode,
    pixelated,
    scaled,
)

NAMES = ("setup", "dashboard", "pipeline", "quota", "alpha", "beta")


def test_hashes_survive_planted_degradations() -> None:
    """JPEG at quality 10, a third of the size, blur and pixels keep hashes."""
    image = drawing("setup")
    variants = (
        open_raster(encode(image, "JPEG", quality=10)),
        scaled(image, 0.4),
        blurred(image),
        pixelated(image),
    )
    for variant in variants:
        assert hamming(dhash(image), dhash(variant)) <= 8
        assert hamming(phash(image), phash(variant)) <= 24


def test_hashes_tell_drawings_apart() -> None:
    """Two different drawings are more than eight bits apart in dHash."""
    hashes = [dhash(drawing(name)) for name in NAMES]
    matrix = hamming_matrix(hashes, hashes)
    for i in range(len(NAMES)):
        assert matrix[i, i] == 0
        assert min(matrix[i, j] for j in range(len(NAMES)) if j != i) > 8


def test_transparent_image_hashes_as_on_white() -> None:
    """Transparency is put on a white page before hashing."""
    image = drawing("setup").convert("RGBA")
    clear = image.copy()
    clear.putalpha(0)
    assert dhash(clear) == dhash(drawing("setup").point(lambda _: 255))
    assert flatten(clear).getpixel((0, 0)) == (255, 255, 255)
    assert uses_alpha(clear)
    assert not uses_alpha(image)


def test_decompression_bomb_is_not_decoded() -> None:
    """An image that announces too many pixels is never decoded."""
    ref = raster_ref(bomb_png(60_000, 60_000), name="bomb.png", source=None)
    assert (ref.problem, ref.readable, ref.width) == ("too large", False, None)
    assert ref.format == "png"


def test_image_above_the_pixel_limit_is_too_large() -> None:
    """An image above the limit but below Pillow's own is too large."""
    with pytest.raises(UnreadableImage) as caught:
        open_raster(bomb_png(7_000, 7_000))
    assert caught.value.reason == "too large"


def test_truncated_image_is_unreadable() -> None:
    """A cut file or a file of another kind is unreadable, quietly."""
    data = encode(drawing("setup"))
    cut = raster_ref(data[: len(data) // 2], name="a.png", source=None)
    assert cut.problem == "cannot decode"
    junk = raster_ref(b"not an image", name="b.jpg", source=None)
    assert (junk.problem, junk.format) == ("cannot decode", "jpg")


def test_jpeg_quality_is_read_from_the_tables() -> None:
    """The estimated quality matches the one of the encoder."""
    image = drawing("setup")
    assert jpeg_quality(open_raster(encode(image, "JPEG", quality=95))) == 95
    assert jpeg_quality(open_raster(encode(image, "JPEG", quality=75))) == 75
    low = jpeg_quality(open_raster(encode(image, "JPEG", quality=10)))
    assert low is not None
    assert low <= 20
    assert jpeg_quality(open_raster(encode(image))) is None


def test_raster_ref_reads_the_facts_of_an_image() -> None:
    """Size, format, hashes, transparency and digest come from the bytes."""
    image = drawing("setup").convert("RGBA")
    image.putalpha(128)
    data = encode(image)
    ref = raster_ref(
        data,
        name="img/setup.png",
        source=PixelSource(data=data),
        alt="Setup",
    )
    assert (ref.width, ref.height, ref.format, ref.alpha) == (
        400,
        250,
        "png",
        True,
    )
    assert (ref.digest, ref.size) == (content_digest(data), len(data))
    assert ref.readable
    assert describe(ref) == 'img/setup.png, 400x250 px, PNG, alt "Setup"'


@pytest.mark.parametrize("name", NAMES)
def test_drawings_are_deterministic(name: str) -> None:
    """A drawing is the same image every time."""
    assert drawing(name).tobytes() == drawing(name).tobytes()


def test_sixteen_bit_gray_hashes_as_its_eight_bit_version() -> None:
    """High bit depth images are scaled, not clamped to white."""
    ramp = np.linspace(0, 65535, 64 * 48, dtype=np.uint16).reshape(48, 64)
    ramp[:, ::7] = 0
    wide = Image.fromarray(ramp)
    narrow = Image.fromarray((ramp >> 8).astype(np.uint8))
    assert wide.mode.startswith("I;16")
    assert dhash(wide) == dhash(narrow)
    assert hamming(phash(wide), phash(narrow)) <= 4
    floats = Image.fromarray(ramp.astype(np.float32), "F")
    mean = float(np.asarray(gray(floats)).mean())
    assert 100 < mean < 160


def test_uniform_images_hash_equally() -> None:
    """Solid images of the same color hash the same, without noise bits."""
    for color in ("white", "black", (200, 30, 30)):
        first = Image.new("RGB", (40, 30), color)
        second = Image.new("RGB", (90, 70), color)
        assert phash(first) == phash(second)
        assert dhash(first) == dhash(second)


def test_garbage_that_pillow_rejects_oddly_is_unreadable() -> None:
    """Any decoder failure is reported as cannot decode."""
    for data in (b"", b"II*\x00\x08\x00\x00\x00", b"\x00\x00\x01\x00\x01\x00"):
        with pytest.raises(UnreadableImage, match="cannot decode"):
            open_raster(data)


def test_warnings_about_a_bad_file_stay_quiet() -> None:
    """Pillow's warnings about an untrusted file never reach the user."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(UnreadableImage):
            open_raster(b"II*\x00\x08\x00\x00\x00")
    assert caught == []


@pytest.mark.parametrize("fmt", ["PPM", "PCX", "TGA", "SGI"])
def test_formats_other_than_web_rasters_cannot_be_decoded(fmt: str) -> None:
    """Only the raster formats of the web are opened."""
    data = encode(drawing("setup"), fmt)
    assert raster_ref(data, name="a.png", source=None).problem == (
        "cannot decode"
    )


def test_multi_picture_jpeg_is_a_jpeg() -> None:
    """A phone photo in MPO keeps the format and quality of a JPEG."""
    image = flatten(drawing("setup"))
    data = encode(image, "MPO", quality=80, append_images=[image])
    ref = raster_ref(data, name="photo.jpg", source=None)
    assert (ref.format, ref.quality) == ("jpeg", 80)


@pytest.mark.parametrize("mode", ["P", "LA", "CMYK", "1", "L", "RGBA"])
@pytest.mark.parametrize("size", [(1, 1), (8, 1), (30, 20)])
def test_modes_and_tiny_sizes_are_readable(
    mode: str,
    size: tuple[int, int],
) -> None:
    """Common modes and tiny sizes give hashes."""
    data = encode(Image.new(mode, size), "PNG" if mode != "CMYK" else "TIFF")
    ref = raster_ref(data, name="x", source=None)
    assert ref.readable
    assert (ref.width, ref.height) == size


def test_palette_transparency_uses_alpha() -> None:
    """A palette image with a transparent color uses alpha."""
    image = Image.new("P", (4, 4), 0)
    image.info["transparency"] = 0
    assert uses_alpha(image)
    assert flatten(image).getpixel((0, 0)) == (255, 255, 255)


def test_pixels_ref_and_unreadable_ref_carry_their_fields() -> None:
    """The builders of PDF images keep what they are given."""
    image = drawing("setup")
    source = PixelSource(page=2, index=1)
    ref = pixels_ref(
        image,
        format="flate",
        digest="d" * 32,
        source=source,
        size=10,
        shown=(100.0, 50.0),
        dpi=150.0,
    )
    assert (ref.name, ref.alpha, ref.format, ref.size) == (
        "",
        None,
        "flate",
        10,
    )
    assert (ref.width, ref.height, ref.dpi) == (400, 250, 150.0)
    assert ref.dhash == dhash(image)
    assert ref.source == source
    bad = unreadable_ref(
        name="a.png",
        format="png",
        digest="e" * 32,
        problem="too large",
    )
    assert (bad.readable, bad.source, bad.phash) == (False, None, None)


def test_describe_variants() -> None:
    """The description names what is known and what is not."""
    figure = ImageRef(
        name="",
        format="vector",
        digest="a",
        shown=(120.4, 80.6),
        vector=True,
    )
    assert describe(figure) == "vector figure, 120x81 pt"
    scan = ImageRef(name="", format="jbig2", digest="a", width=3, height=2)
    scan = replace(scan, dpi=300.2, alt="")
    assert describe(scan) == 'image, 3x2 px, JBIG2, 300 dpi, alt ""'
    bad = unreadable_ref(
        name="a.jpg",
        format="jpg",
        digest="a",
        problem="cannot decode",
    )
    assert describe(bad) == "a.jpg, JPG, unreadable: cannot decode"


def test_palette_alpha_mode_is_readable() -> None:
    """A PA image, which no encoder writes, still hashes."""
    image = Image.new("PA", (8, 8))
    assert flatten(image).getpixel((0, 0)) == (255, 255, 255)
    assert isinstance(dhash(image), int)


def test_same_bytes_are_decoded_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bytes of an image already read are not decoded again."""
    decoded: list[int] = []

    def counting(data: bytes) -> Image.Image:
        decoded.append(len(data))
        return open_raster(data)

    monkeypatch.setattr(refs, "open_raster", counting)
    data = encode(drawing("read once"))
    first = raster_ref(data, name="a.png", source=None)
    second = raster_ref(data, name="b.png", source=PixelSource(data=data))
    assert decoded == [len(data)]
    assert replace(second, name="a.png", source=None) == first


def test_hashes_are_both_hashes() -> None:
    """The two hashes read together are those read apart."""
    image = drawing("setup", size=(1600, 1000))
    assert hashes(image) == (dhash(image), phash(image))
