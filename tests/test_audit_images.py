# Copyright 2026 Christophe Le Douarec
"""Tests for the image checks of the audit."""

import io
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

from doc_checker.audit import images as audit_images
from doc_checker.audit.engine import audit_document
from doc_checker.audit.images import BLOCK_GRID_MIN, image_quality_drafts
from doc_checker.config import AuditThresholds, Config
from doc_checker.extract.loader import load_document
from doc_checker.images.load import PixelLoader
from doc_checker.images.metrics import PIXEL_LEVELS_MIN, ImageQuality
from doc_checker.images.pixels import open_raster
from doc_checker.images.refs import describe, raster_ref
from doc_checker.model.document import (
    Block,
    BlockKind,
    ImageRef,
    PixelSource,
    SourceRef,
)
from doc_checker.model.findings import Severity
from tests.builders import make_audit_document, make_image_block
from tests.image_builders import (
    blurred,
    drawing,
    encode,
    photo,
    pixelated,
    screenshot,
    swatches,
)

if TYPE_CHECKING:
    from doc_checker.compare.drafts import Draft

THRESHOLDS = AuditThresholds()
DIGEST = "1" * 32


def _block(data: bytes, name: str = "a.png") -> Block:
    ref = raster_ref(data, name=name, source=PixelSource(data=data))
    return make_image_block(ref)


def _all_drafts(
    *blocks: Block,
    fmt: str = "html",
    profile: str = "screen",
) -> list[Draft]:
    doc = make_audit_document(*blocks, fmt=fmt)
    with PixelLoader(Path("doc")) as loader:
        return image_quality_drafts(doc, loader, THRESHOLDS, profile)


def _drafts(
    *blocks: Block,
    fmt: str = "html",
    profile: str = "screen",
) -> list[tuple[str, Severity | None]]:
    drafts = _all_drafts(*blocks, fmt=fmt, profile=profile)
    return [(d.rule, d.severity) for d in drafts]


def test_broken_images() -> None:
    """Missing or undecodable images; too large or external ones are not."""
    problems = ("not found", "cannot decode", "too large", "external")
    blocks = [
        make_image_block(ImageRef(f"{k}.png", "png", DIGEST, problem=problem))
        for k, problem in enumerate(problems)
    ]
    assert _drafts(*blocks) == [("image.broken", None)] * 2


def test_heavy_images() -> None:
    """Above one megabyte minor, above three major."""
    blocks = [
        make_image_block(ImageRef(f"{size}.jpg", "jpeg", DIGEST, size=size))
        for size in (1_500_000, 3_500_000, 900_000)
    ]
    assert _drafts(*blocks) == [
        ("image.heavy", None),
        ("image.heavy", Severity.MAJOR),
    ]


@pytest.mark.parametrize(
    ("profile", "dpi", "shown", "expected"),
    [
        ("screen", 120.0, (300.0, 200.0), [("image.low_dpi", None)]),
        ("screen", 80.0, (300.0, 200.0), [("image.low_dpi", Severity.MAJOR)]),
        ("screen", 120.0, (30.0, 20.0), []),
        ("screen", 160.0, (300.0, 200.0), []),
        ("print", 200.0, (300.0, 200.0), [("image.low_dpi", None)]),
        ("print", 120.0, (300.0, 200.0), [("image.low_dpi", Severity.MAJOR)]),
    ],
)
def test_low_resolution_in_pdf(
    profile: str,
    dpi: float,
    shown: tuple[float, float],
    expected: list[tuple[str, Severity | None]],
) -> None:
    """The thresholds of the profile; icons are not judged."""
    ref = ImageRef(
        "",
        "flate",
        DIGEST,
        width=400,
        height=250,
        dpi=dpi,
        shown=shown,
    )
    block = make_image_block(ref, file="doc.pdf")
    assert _drafts(block, fmt="pdf", profile=profile) == expected


def test_upscaled_html_images() -> None:
    """Shown at twice its size it is reported; at 1.05 it is not."""
    data = encode(drawing("setup"))
    big = replace(_block(data).image, shown=(800.0, 500.0))
    near = replace(_block(data).image, shown=(420.0, 262.0))
    assert _drafts(make_image_block(big), make_image_block(near)) == [
        ("image.upscaled", None),
    ]


def test_low_jpeg_quality() -> None:
    """Below 50 minor, below 25 major, and no blocks finding with it."""
    picture = photo("street")
    assert _drafts(
        _block(encode(picture, "JPEG", quality=20), "a.jpg"),
        _block(encode(picture, "JPEG", quality=40), "b.jpg"),
        _block(encode(picture, "JPEG", quality=90), "c.jpg"),
    ) == [
        ("image.jpeg_quality", Severity.MAJOR),
        ("image.jpeg_quality", None),
    ]


def test_pixel_checks() -> None:
    """Blur, JPEG blocks, pixelation and a photograph kept as PNG."""
    blocky = encode(
        open_raster(encode(photo("street"), "JPEG", quality=8)),
    )
    assert _drafts(
        _block(encode(blurred(drawing("setup"), 2.0)), "blur.png"),
        _block(blocky, "blocky.png"),
        _block(encode(pixelated(drawing("setup"))), "pix.png"),
        _block(encode(photo("street")), "photo.png"),
        _block(encode(drawing("setup")), "fine.png"),
    ) == [
        ("image.blurry", None),
        ("image.blocky", None),
        ("image.pixelated", None),
        ("image.photo_png", None),
    ]


def test_each_image_is_assessed_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same content on three blocks is measured once."""
    calls: list[Image.Image] = []
    measure = audit_images.assess

    def counted(image: Image.Image) -> ImageQuality:
        calls.append(image)
        return measure(image)

    monkeypatch.setattr(audit_images, "assess", counted)
    block = _block(encode(drawing("setup")))
    _drafts(block, block, block)
    assert len(calls) == 1


def test_vector_figures_are_not_judged_by_their_pixels() -> None:
    """A PDF vector figure has no blur, blocks nor resolution to judge."""
    ref = ImageRef(
        "",
        "vector",
        DIGEST,
        vector=True,
        shown=(300.0, 200.0),
        source=PixelSource(page=0, box=(0.0, 0.0, 300.0, 200.0)),
    )
    assert _drafts(make_image_block(ref, file="doc.pdf"), fmt="pdf") == []


def test_details_and_patterns() -> None:
    """Each finding tells its measure, and groups by its kind of problem."""
    data = encode(drawing("setup"))
    upscaled = replace(_block(data).image, shown=(800.0, 500.0))
    missing = ImageRef("a.png", "png", DIGEST, problem="not found")
    blocks = [
        make_image_block(missing),
        make_image_block(ImageRef("b.jpg", "jpeg", DIGEST, size=3_500_000)),
        make_image_block(upscaled),
        _block(encode(photo("street"), "JPEG", quality=20), "c.jpg"),
        _block(encode(pixelated(drawing("setup"))), "pix.png"),
        _block(encode(blurred(drawing("setup"), 2.0)), "blur.png"),
    ]
    *drafts, blur = _all_drafts(*blocks)
    assert [(d.rule, d.detail, d.pattern) for d in drafts] == [
        (
            "image.broken",
            "the image cannot be shown: not found",
            "not found",
        ),
        ("image.heavy", "3.5 MB", "over 3 MB"),
        ("image.upscaled", "400x250 px shown at 800x500", "upscaled"),
        ("image.jpeg_quality", "JPEG quality 20, below 25", "below 25"),
        ("image.pixelated", "repeated rows and columns", "pixelated"),
    ]
    assert (blur.rule, blur.pattern) == ("image.blurry", "blurry")
    assert blur.detail is not None
    assert blur.detail.startswith("sharpness 0.0")
    assert blur.detail.endswith(", below 0.10")


def test_details_of_resolution_blocks_and_photo() -> None:
    """The resolution, the blocks and the colours of a photo are told."""
    ref = ImageRef("", "flate", DIGEST, width=400, height=250, dpi=80.4)
    pdf = make_image_block(replace(ref, shown=(300.0, 200.0)), file="doc.pdf")
    (low,) = _all_drafts(pdf, fmt="pdf", profile="print")
    assert (low.detail, low.pattern) == (
        "80 dpi, below 150 for the print profile",
        "below 150 dpi",
    )
    blocky = encode(open_raster(encode(photo("street"), "JPEG", quality=8)))
    kept = encode(photo("street"))
    blocks_draft, photo_draft = _all_drafts(
        _block(blocky, "blocky.png"),
        _block(kept, "photo.png"),
    )
    assert blocks_draft.pattern == "blocky"
    assert blocks_draft.detail is not None
    assert blocks_draft.detail.startswith("8x8 blocks ")
    assert blocks_draft.detail.endswith(", from 2.00")
    bpp = 8 * len(kept) / (480 * 320)
    assert photo_draft.pattern == "photo_png"
    assert photo_draft.detail is not None
    assert photo_draft.detail.endswith(f" colours at {bpp:.1f} bits per pixel")


def test_side_shows_the_image() -> None:
    """The side is the block of the image, described, with the image."""
    ref = ImageRef("a.png", "png", DIGEST, problem="not found")
    block = make_image_block(ref, file="guide.html")
    (draft,) = _all_drafts(block)
    assert draft.new is None
    assert draft.ref is not None
    assert draft.ref.source == block.source
    assert (draft.ref.text, draft.ref.kind) == (describe(ref), BlockKind.IMAGE)
    assert draft.ref.image == ref


def test_pixels_that_cannot_be_read_again_are_not_judged(
    tmp_path: Path,
) -> None:
    """An image removed since the extraction gives no pixel finding."""
    file = tmp_path / "gone.png"
    data = encode(blurred(drawing("setup"), 2.0))
    file.write_bytes(data)
    ref = raster_ref(data, name="gone.png", source=PixelSource(file=file))
    file.unlink()
    assert _drafts(make_image_block(ref)) == []


def test_photo_png_needs_its_weight_and_colours() -> None:
    """A light, sparse or few-colour PNG is no photograph kept as PNG."""
    small = encode(photo("street", (160, 100)))
    assert len(small) < audit_images.PHOTO_PNG_MIN_BYTES
    drawn = encode(drawing("setup", (1200, 800)))
    rng = np.random.default_rng(7)
    palette = rng.integers(0, 256, (200, 3), dtype=np.uint8)
    noisy = encode(
        Image.fromarray(palette[rng.integers(0, 200, (320, 480))], "RGB"),
    )
    assert len(noisy) >= audit_images.PHOTO_PNG_MIN_BYTES
    blocks = (
        _block(small, "small.png"),
        _block(drawn, "big.png"),
        _block(noisy, "noisy.png"),
    )
    assert _drafts(*blocks) == []


def test_audit_runs_the_image_family() -> None:
    """The family runs when asked and when the images were read."""
    blur = _block(encode(blurred(drawing("setup"), 2.0)), "blur.png")
    doc = make_audit_document(blur)
    only = Config(checks=("images",))
    assert [f.rule for f in audit_document(doc, only).findings] == [
        "image.blurry",
    ]
    rules = [f.rule for f in audit_document(doc, Config()).findings]
    assert rules[-1] == "image.blurry"
    assert "image.blurry" not in [
        f.rule for f in audit_document(doc, Config(checks=("a11y",))).findings
    ]
    unread = replace(only, images=False)
    assert audit_document(doc, unread).findings == []


AVIF = b"\x00\x00\x00\x1cftypavif\x00\x00\x00\x00avifmif1miaf" + bytes(64)
JXL = b"\xff\x0a" + bytes(64)
# A quality that judges nothing but what a test sets.
NEUTRAL = ImageQuality(
    blur=None,
    blockiness=1.0,
    adjacent=1.0,
    levels=256,
    colors=1,
    block_grid_share=0.0,
)


def _measured_as(
    monkeypatch: pytest.MonkeyPatch,
    **measures: float | None,
) -> Block:
    """Return a readable image block whose pixels measure as told."""
    quality = replace(NEUTRAL, **measures)
    monkeypatch.setattr(audit_images, "assess", lambda _: quality)
    return _block(encode(drawing("setup")))


def test_edges_on_the_block_grid_are_not_blocky() -> None:
    """A screenshot or swatches on the 8 px grid show no JPEG blocks."""
    rules = [
        rule
        for rule, _ in _drafts(
            _block(encode(screenshot()), "window.png"),
            _block(encode(swatches()), "swatches.png"),
        )
    ]
    assert "image.blocky" not in rules
    blocky = encode(open_raster(encode(photo("street"), "JPEG", quality=8)))
    assert _drafts(_block(blocky, "blocky.png")) == [("image.blocky", None)]


@pytest.mark.parametrize(
    ("share", "expected"),
    [(BLOCK_GRID_MIN, [("image.blocky", None)]), (0.69, [])],
)
def test_blocks_need_their_grid(
    monkeypatch: pytest.MonkeyPatch,
    share: float,
    expected: list[tuple[str, Severity | None]],
) -> None:
    """High blockiness alone is no proof of JPEG blocks."""
    block = _measured_as(monkeypatch, blockiness=3.0, block_grid_share=share)
    assert _drafts(block) == expected


def _bad_text_checksum() -> bytes:
    """Return a PNG whose text chunk has a wrong checksum."""
    info = PngInfo()
    info.add_itxt("Comment", "Figure")
    data = bytearray(encode(drawing("setup"), pnginfo=info))
    start = data.index(b"iTXt")
    length = int.from_bytes(data[start - 4 : start])
    data[start + 4 + length] ^= 0xFF
    return bytes(data)


def _checkerboard() -> Image.Image:
    """Draw black and white squares of 8 pixels, on the grid of the blocks."""
    squares = (np.indices((256, 256)) // 8).sum(axis=0) % 2
    return Image.fromarray((squares * 255).astype(np.uint8), "L")


def test_aligned_patterns_are_not_blocky() -> None:
    """A checkerboard or a picture enlarged 8 times shows no JPEG blocks."""
    small = photo("street", (60, 40))
    enlarged = small.resize((480, 320), Image.Resampling.NEAREST)
    assert _drafts(
        _block(encode(_checkerboard()), "board.png"),
        _block(encode(enlarged), "enlarged.png"),
    ) == [("image.pixelated", None)]


@pytest.mark.parametrize(
    ("measures", "expected"),
    [
        ({"levels": PIXEL_LEVELS_MIN}, [("image.blocky", None)]),
        ({"levels": PIXEL_LEVELS_MIN - 1}, []),
        ({"adjacent": 0.0}, [("image.pixelated", None)]),
    ],
)
def test_blocks_need_levels_and_no_pixelation(
    monkeypatch: pytest.MonkeyPatch,
    measures: dict[str, float],
    expected: list[tuple[str, Severity | None]],
) -> None:
    """Few gray levels or repeated pixels tell another cause than a JPEG."""
    block = _measured_as(
        monkeypatch,
        blockiness=3.0,
        block_grid_share=1.0,
        **measures,
    )
    assert _drafts(block) == expected


def test_web_formats_that_browsers_show_are_not_broken() -> None:
    """AVIF and JPEG XL are not decoded here, but browsers show them.

    A PNG with a wrong checksum on a text chunk, which browsers show, is
    still broken: a limit.
    """
    blocks = [
        _block(AVIF, "photo.avif"),
        _block(JXL, "photo.jxl"),
        _block(AVIF, ""),
        _block(_bad_text_checksum(), "damaged.png"),
    ]
    assert [b.image.problem for b in blocks if b.image] == [
        "cannot decode",
    ] * 4
    assert _drafts(*blocks) == [("image.broken", None)]


def test_file_over_the_read_limit_is_heavy(tmp_path: Path) -> None:
    """A file too large to be read is not broken, but very heavy."""
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text(
        '<!doctype html><html lang="en"><head><title>Big</title></head>'
        "<body><main><h1>Big</h1><p>A large picture.</p>"
        '<img src="big.png" alt="Large picture"></main></body></html>',
        encoding="utf-8",
    )
    with (site / "big.png").open("wb") as file:
        file.truncate(51_000_000)
    cfg = Config(checks=("images",))
    findings = audit_document(load_document(site, cfg, audit=True), cfg)
    assert [(f.rule, f.severity, f.detail) for f in findings.findings] == [
        ("image.heavy", Severity.MAJOR, "51.0 MB"),
    ]


def test_pdf_images_are_files_by_their_digest() -> None:
    """A PDF image too large to read, drawn on two pages, is heavy once."""
    blocks = [
        replace(
            make_image_block(
                ImageRef(
                    f"page {page} image 1",
                    "flate",
                    DIGEST,
                    size=1_500_000,
                    problem="too large",
                ),
                file="doc.pdf",
            ),
            source=SourceRef("doc.pdf", page=page),
        )
        for page in (1, 2)
    ]
    assert _drafts(*blocks, fmt="pdf") == [("image.heavy", None)]


def test_file_properties_are_told_once_per_file_in_html() -> None:
    """Weight and quality once for each file; display and breakage per use."""
    data = encode(photo("street"), "JPEG", quality=20)
    ref = raster_ref(data, name="img/a.jpg", source=PixelSource(data=data))
    shown = replace(ref, shown=(960.0, 640.0))
    heavy = ImageRef("img/c.jpg", "jpeg", DIGEST, size=3_500_000)
    missing = ImageRef("img/d.png", "png", "2" * 32, problem="not found")
    blocks = [
        *(make_image_block(shown, file=f"p{k}.html") for k in (1, 2, 2)),
        make_image_block(replace(ref, name="img/b.jpg"), file="p3.html"),
        *(make_image_block(heavy, file=f"p{k}.html") for k in (1, 3)),
        *(make_image_block(missing, file=f"p{k}.html") for k in (1, 3)),
    ]
    drafts = _all_drafts(*blocks)
    assert [(d.rule, d.ref.source.file if d.ref else "") for d in drafts] == [
        ("image.upscaled", "p1.html"),
        ("image.jpeg_quality", "p1.html"),
        ("image.upscaled", "p2.html"),
        ("image.upscaled", "p2.html"),
        ("image.jpeg_quality", "p3.html"),
        ("image.heavy", "p1.html"),
        ("image.broken", "p1.html"),
        ("image.broken", "p3.html"),
    ]


def _repeated_jpeg_pdf(out: Path, jpeg: bytes) -> Path:
    """Write six pages; a JPEG is drawn twice on page 2, once on page 5."""
    canvas = Canvas(str(out), pagesize=A4)
    for number in range(1, 7):
        canvas.setFont("Helvetica", 11)
        canvas.drawString(72, 790, f"Text of page {number}")
        for k in range({2: 2, 5: 1}.get(number, 0)):
            picture = ImageReader(io.BytesIO(jpeg))
            canvas.drawImage(picture, 72, 420 - 360 * k, width=480, height=320)
        canvas.showPage()
    canvas.save()
    return out


def test_file_properties_are_told_once_per_file_in_pdf(
    tmp_path: Path,
) -> None:
    """One JPEG drawn three times: its quality once, its resolution thrice."""
    jpeg = encode(photo("street"), "JPEG", quality=20)
    pdf = _repeated_jpeg_pdf(tmp_path / "repeated.pdf", jpeg)
    cfg = Config(checks=("images",))
    doc = load_document(pdf, cfg, audit=True)
    images = [b.image for b in doc.blocks if b.image is not None]
    assert len(images) == 3
    assert len({(image.name, image.digest) for image in images}) == 1
    findings = audit_document(doc, cfg).findings
    assert [(f.rule, f.severity, f.ref.source.page) for f in findings] == [
        ("image.low_dpi", Severity.MAJOR, 2),
        ("image.jpeg_quality", Severity.MAJOR, 2),
        ("image.low_dpi", Severity.MAJOR, 2),
        ("image.low_dpi", Severity.MAJOR, 5),
    ]


def test_details_never_contradict_their_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A measure just below its limit is never printed at the limit."""
    (blur,) = _all_drafts(_measured_as(monkeypatch, blur=0.0997))
    assert blur.detail == "sharpness 0.09, below 0.10"
    ref = ImageRef("", "flate", DIGEST, dpi=95.6, shown=(300.0, 200.0))
    (low,) = _all_drafts(make_image_block(ref, file="doc.pdf"), fmt="pdf")
    assert low.detail == "95 dpi, below 96 for the screen profile"
    ref = replace(ref, dpi=101.6)
    (low,) = _all_drafts(make_image_block(ref, file="doc.pdf"), fmt="pdf")
    assert low.detail == "102 dpi, below 150 for the screen profile"
    assert "102 dpi" in describe(ref)


def test_vector_figures_are_never_broken() -> None:
    """A PDF figure left unrendered still shows its paths."""
    ref = ImageRef(
        "",
        "vector",
        DIGEST,
        vector=True,
        shown=(300.0, 200.0),
        problem="cannot decode",
    )
    assert _drafts(make_image_block(ref, file="doc.pdf"), fmt="pdf") == []


def _limit_case(case: str) -> tuple[ImageRef, str, dict[str, float]]:
    """Return an image, its format and its measures, at one limit."""
    raster = ImageRef("a.jpg", "jpeg", DIGEST)
    shown = (300.0, 200.0)
    pdf = ImageRef("", "flate", DIGEST, width=400, height=250, shown=shown)
    cases = {
        "heavy": (replace(raster, size=1_000_000), "html", {}),
        "dpi major": (replace(pdf, dpi=96.0), "pdf", {}),
        "dpi minor": (replace(pdf, dpi=150.0), "pdf", {}),
        "quality minor": (replace(raster, quality=50), "html", {}),
        "quality major": (replace(raster, quality=25), "html", {}),
        "upscale": (
            replace(raster, width=400, height=250, shown=(440.0, 275.0)),
            "html",
            {},
        ),
        "blur": (raster, "html", {"blur": 0.10}),
        "blocks": (
            raster,
            "html",
            {"blockiness": 2.0, "block_grid_share": BLOCK_GRID_MIN},
        ),
    }
    return cases[case]


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("heavy", []),
        ("dpi major", [("image.low_dpi", None)]),
        ("dpi minor", []),
        ("quality minor", []),
        ("quality major", [("image.jpeg_quality", None)]),
        ("upscale", []),
        ("blur", []),
        ("blocks", [("image.blocky", None)]),
    ],
)
def test_limits(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    expected: list[tuple[str, Severity | None]],
) -> None:
    """Every limit is strict but the blocks, reported from theirs."""
    ref, fmt, measures = _limit_case(case)
    if measures:
        block = _measured_as(monkeypatch, **measures)
    else:
        block = make_image_block(ref, file="doc.pdf" if fmt == "pdf" else "a")
    assert _drafts(block, fmt=fmt) == expected
