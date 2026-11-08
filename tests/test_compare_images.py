# Copyright 2026 Christophe Le Douarec
"""Tests for the matching of images and their findings."""

import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from doc_checker.compare.engine import compare_documents
from doc_checker.compare.images import (
    PHASH_MAX,
    Placed,
    image_drafts,
    match_images,
    placed_images,
)
from doc_checker.config import Config, HtmlConfig
from doc_checker.extract.html import load_html
from doc_checker.images.hashing import hamming, phash
from doc_checker.images.load import PixelLoader
from doc_checker.images.pixels import EXTERNAL, NOT_FOUND
from doc_checker.images.refs import content_digest, pixels_ref, raster_ref
from doc_checker.images.svg import parse_svg
from doc_checker.model.document import BlockKind, ImageRef, PixelSource
from doc_checker.model.findings import Category, Severity
from tests.builders import (
    PARAGRAPHS,
    make_block,
    make_document,
    make_image_block,
)
from tests.image_builders import (
    blurred,
    drawing,
    encode,
    figure_drawing,
    figure_svg,
    scaled,
    smiley,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from doc_checker.compare.engine import CompareResult
    from doc_checker.model.document import Block

TEXT = [make_block(text) for text in PARAGRAPHS]
CAPTION = make_block("Steps of the installer", BlockKind.CAPTION)


def _ref(image: Image.Image, fmt: str = "PNG", **options: int) -> ImageRef:
    """Return the reference of an encoded image kept in memory."""
    data = encode(image, fmt, **options)
    return raster_ref(data, name="", source=PixelSource(data=data))


def _svg(name: str) -> ImageRef:
    """Return the reference of a drawn SVG file."""
    identity = parse_svg(figure_svg(name).encode())
    return ImageRef(
        name=f"img/{name}.svg",
        format="svg",
        digest=identity.digest,
        vector=True,
        texts=identity.texts,
    )


def _image(ref: ImageRef, *, name: str = "", alt: str | None = None) -> Block:
    """Return an image block between the paragraphs of a test."""
    return make_image_block(replace(ref, name=name or ref.name, alt=alt))


def _compare(ref: list[Block], new: list[Block]) -> CompareResult:
    """Compare two documents made of blocks."""
    return compare_documents(
        make_document(*ref),
        make_document(*new),
        Config(),
    )


def _findings(
    ref: list[Block],
    new: list[Block],
) -> list[tuple[str, Severity]]:
    """Compare two documents and return their image findings."""
    result = _compare(ref, new)
    return [
        (f.rule, f.severity)
        for f in result.findings
        if f.category is Category.IMAGE
    ]


SETUP = _ref(drawing("setup"))
CASES = {
    "degraded": (
        [TEXT[0], _image(SETUP), TEXT[1]],
        [TEXT[0], _image(_ref(scaled(drawing("setup"), 0.4))), TEXT[1]],
        [("image.degraded", Severity.MAJOR)],
    ),
    "replaced": (
        [TEXT[0], _image(SETUP), CAPTION, TEXT[1]],
        [TEXT[0], _image(_ref(drawing("other"))), CAPTION, TEXT[1]],
        [("image.replaced", Severity.MAJOR)],
    ),
    "missing": (
        [TEXT[0], _image(SETUP), TEXT[1]],
        [TEXT[0], TEXT[1]],
        [("image.missing", Severity.CRITICAL)],
    ),
    "added": (
        [TEXT[0], TEXT[1]],
        [TEXT[0], _image(SETUP), TEXT[1]],
        [("image.added", Severity.MINOR)],
    ),
    "rasterized": (
        [TEXT[0], _image(_svg("pipeline"), alt="Pipeline"), TEXT[1]],
        [
            TEXT[0],
            _image(_ref(figure_drawing("pipeline")), alt="Pipeline"),
            TEXT[1],
        ],
        [("image.rasterized", Severity.MINOR)],
    ),
    "vectorized": (
        [
            TEXT[0],
            _image(_ref(figure_drawing("pipeline")), name="pipeline.png"),
        ],
        [TEXT[0], _image(_svg("pipeline"))],
        [("image.vectorized", Severity.INFO)],
    ),
    "alt-lost": (
        [TEXT[0], _image(SETUP, alt="Installer steps"), TEXT[1]],
        [TEXT[0], _image(SETUP, alt=""), TEXT[1]],
        [("image.alt_lost", Severity.MINOR)],
    ),
    "unreadable": (
        [TEXT[0], _image(SETUP, name="img/setup.png"), TEXT[1]],
        [
            TEXT[0],
            _image(
                ImageRef(
                    "img/setup.png",
                    "png",
                    "0" * 32,
                    problem="not found",
                ),
            ),
            TEXT[1],
        ],
        [("image.missing", Severity.CRITICAL)],
    ),
}


@pytest.mark.parametrize("case", list(CASES))
def test_each_change_gives_its_rule(case: str) -> None:
    """Each kind of change of an image gives its own rule."""
    ref, new, expected = CASES[case]
    assert _findings(ref, new) == expected


def test_same_image_under_other_names_gives_nothing() -> None:
    """The same bytes under two names are one image."""
    ref = [TEXT[0], _image(SETUP, name="_images/setup.png"), TEXT[1]]
    new = [TEXT[0], _image(SETUP, name="img/setup-3f2a1b9c.png"), TEXT[1]]
    assert _findings(ref, new) == []


def test_inline_svg_loses_no_alt() -> None:
    """An inline SVG has no alt to lose, unlike an SVG file without one."""
    svg = _svg("pipeline")
    ref = [TEXT[0], _image(svg, alt="Pipeline"), TEXT[1]]
    inline = make_image_block(replace(svg, name="", alt=None))
    assert _findings(ref, [TEXT[0], inline, TEXT[1]]) == []
    assert _findings(ref, [TEXT[0], _image(svg), TEXT[1]]) == [
        ("image.alt_lost", Severity.MINOR),
    ]


def test_unrendered_figure_is_not_missing() -> None:
    """A PDF figure drawn but not rendered is not judged, nor missing."""
    rendered = ImageRef(
        name="",
        format="vector",
        digest="1" * 32,
        width=300,
        height=200,
        vector=True,
        phash=1,
        dhash=1,
        source=PixelSource(page=0, box=(0.0, 0.0, 150.0, 100.0)),
    )
    unrendered = ImageRef(
        name="page 1 figure 1",
        format="vector",
        digest="2" * 32,
        vector=True,
        problem="not rendered",
    )
    result = _compare(
        [TEXT[0], make_image_block(rendered), CAPTION, TEXT[1]],
        [TEXT[0], make_image_block(unrendered), CAPTION, TEXT[1]],
    )
    assert result.findings == []
    assert result.stats.matched_images == 1


def _image_findings(
    ref: list[Block],
    new: list[Block],
) -> list[tuple[str, Severity, str | None, bool, bool]]:
    """Return the rule, severity, detail and sides of the image findings."""
    return [
        (f.rule, f.severity, f.detail, f.ref is not None, f.new is not None)
        for f in _compare(ref, new).findings
        if f.category is Category.IMAGE
    ]


UNREADABLE = {
    "svg-to-png": (
        _svg("pipeline"),
        ImageRef("img/pipeline.png", "png", "0" * 32, problem=NOT_FOUND),
    ),
    "png-to-svg": (
        replace(_ref(figure_drawing("pipeline")), name="pipeline.png"),
        ImageRef(
            "img/pipeline.svg",
            "svg",
            "0" * 32,
            vector=True,
            problem=NOT_FOUND,
        ),
    ),
}


@pytest.mark.parametrize("case", list(UNREADABLE))
def test_unreadable_new_image_is_missing_in_any_format(case: str) -> None:
    """A new image that cannot be read is missing, whatever its format."""
    ref, new = UNREADABLE[case]
    assert _image_findings(
        [TEXT[0], _image(ref), TEXT[1]],
        [TEXT[0], _image(new), TEXT[1]],
    ) == [
        (
            "image.missing",
            Severity.CRITICAL,
            "the new image cannot be read: not found",
            True,
            True,
        ),
    ]


def _external(url: str) -> ImageRef:
    """Return an image that is never read, at an absolute URL."""
    return ImageRef(url, "png", url.encode().hex()[:32], problem=EXTERNAL)


URL = "https://docs.example.com/img/setup.png"
EXTERNALS = {
    "local-to-url": (
        replace(SETUP, name="img/setup.png"),
        _external(URL),
        "Installer steps",
        [],
    ),
    "url-to-local": (
        _external(URL),
        replace(SETUP, name="img/setup.png"),
        "Installer steps",
        [],
    ),
    "url-to-other-url": (
        _external(URL),
        _external("https://cdn.example.com/setup.png"),
        "Installer steps",
        [],
    ),
    "url-without-alt": (
        replace(SETUP, name="img/setup.png"),
        _external(URL),
        "",
        [("image.alt_lost", Severity.MINOR)],
    ),
}


@pytest.mark.parametrize("case", list(EXTERNALS))
def test_external_image_is_not_judged(case: str) -> None:
    """An image at an absolute URL is never read, so only its alt counts."""
    ref, new, new_alt, expected = EXTERNALS[case]
    assert (
        _findings(
            [TEXT[0], _image(ref, alt="Installer steps"), CAPTION, TEXT[1]],
            [TEXT[0], _image(new, alt=new_alt), CAPTION, TEXT[1]],
        )
        == expected
    )


def test_image_chrome_of_either_side_is_left_out() -> None:
    """An image that one side removed as chrome is compared on neither."""
    logo = _ref(drawing("logo", size=(60, 20)))
    ref = make_document(TEXT[0], _image(logo), TEXT[1], _image(SETUP))
    new = replace(
        make_document(TEXT[0], TEXT[1], _image(SETUP)),
        chrome_images=frozenset({logo.digest}),
    )
    for first, second in ((ref, new), (new, ref)):
        result = compare_documents(first, second, Config())
        stats = result.stats
        assert result.findings == []
        assert (stats.ref_images, stats.new_images) == (1, 1)


def _swatch(number: int) -> Image.Image:
    """Return a swatch of one colour."""
    colour = ((number * 53) % 256, (number * 97) % 256, (number * 151) % 256)
    return Image.new("RGB", (120, 80), colour)


def _dialog(number: int) -> Image.Image:
    """Return a mostly white dialog, which only its label tells apart."""
    image = Image.new("RGB", (400, 250), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 390, 240), outline="#888888")
    draw.text((30, 30 + (number % 5) * 3), f"Option {number}", fill="black")
    draw.rectangle((300, 200, 380, 230), outline="#333333")
    return image


@pytest.mark.parametrize(
    ("maker", "count"),
    [(_swatch, 12), (_dialog, 20)],
    ids=["swatches", "dialogs"],
)
def test_near_identical_series_keeps_every_pair(
    maker: Callable[[int], Image.Image],
    count: int,
) -> None:
    """Many near-identical images, re-encoded, are only re-encoded."""
    ref, new = [], []
    for number in range(count):
        text = make_block(f"Paragraph {number} shows the image {number}.")
        image = maker(number)
        ref += [text, _image(_ref(image))]
        new += [text, _image(_ref(image, "JPEG", quality=90))]
    result = _compare(ref, new)
    rules = {
        occurrence.rule
        for finding in result.findings
        for occurrence in finding.occurrences or (finding,)
    }
    assert rules <= {"image.reencoded"}
    assert result.stats.matched_images == count


def test_long_series_of_identical_icons_keeps_every_pair() -> None:
    """Forty identical icons, re-encoded and set elsewhere, all pair."""
    icon = drawing("check", size=(32, 32))
    ref = [_image(_ref(icon)) for _ in range(40)]
    new = [_image(_ref(icon, compress_level=1)) for _ in range(40)]
    texts = [
        make_block(f"Step {number} of the long procedure is done.")
        for number in range(81)
    ]
    # Each reference icon follows an even paragraph, each new one the odd
    # paragraph after it: no icon is at the place of its counterpart.
    first, second = texts[:1], texts[:2]
    for k in range(40):
        first += [ref[k], *texts[2 * k + 1 : 2 * k + 3]]
        second += [new[k], *texts[2 * k + 2 : 2 * k + 4]]
    result = _compare(first, second)
    assert [f for f in result.findings if f.category is Category.IMAGE] == []
    assert result.stats.matched_images == 40


@pytest.mark.parametrize(
    ("changed", "rule"),
    [
        (_ref(drawing("check", size=(48, 48)), "JPEG", quality=5), "degraded"),
        (_ref(drawing("cross", size=(48, 48))), "replaced"),
    ],
    ids=["degraded", "replaced"],
)
def test_one_changed_icon_of_a_long_series_keeps_its_rule(
    changed: ImageRef,
    rule: str,
) -> None:
    """In forty re-encoded icons, a changed one is judged at its place."""
    icon = drawing("check", size=(48, 48))
    ref, new = [TEXT[0]], [TEXT[0]]
    for number in range(40):
        text = make_block(f"Step {number} of the long procedure is done.")
        ref += [_image(_ref(icon)), text]
        image = changed if number == 35 else _ref(icon, compress_level=1)
        new += [_image(image), text]
    result = _compare(ref, new)
    assert [
        (f.rule, f.severity)
        for f in result.findings
        if f.category is Category.IMAGE
    ] == [(f"image.{rule}", Severity.MAJOR)]
    assert result.stats.matched_images == 40


def test_tiny_images_are_not_compared(tmp_path: Path) -> None:
    """Two sites that differ by their spacers and rules give no finding."""
    pages = []
    for name, spacers in (("ref", 3), ("new", 1)):
        site = tmp_path / name
        site.mkdir()
        (site / "spacer.gif").write_bytes(
            encode(Image.new("P", (1, 1)), "GIF"),
        )
        images = '<img src="spacer.gif">' * spacers
        (site / "index.html").write_text(
            f"<html><body><main><p>{PARAGRAPHS[0]}</p>{images}"
            f"<p>{PARAGRAPHS[1]}</p></main></body></html>",
            encoding="utf-8",
        )
        pages.append(load_html(site, HtmlConfig(), lang="en"))
    result = compare_documents(*pages, Config())
    assert (result.findings, result.stats.ref_images) == ([], 0)


def test_close_context_pair_is_judged_by_its_pixels() -> None:
    """Images paired by their context but close in content are measured."""
    small = _ref(scaled(drawing("setup"), 0.4))
    ref = [
        Placed(make_image_block(SETUP), 0, ""),
        Placed(make_image_block(SETUP), 1, ""),
    ]
    new = [
        Placed(make_image_block(small), 0, ""),
        Placed(make_image_block(small), 1, ""),
    ]
    matches = [
        *match_images(ref[:1], new[:1], {0: 0}),
        # A pair that the context matched, as past the rounds of content.
        replace(match_images(ref[:1], new[:1], {0: 0})[0], how="context"),
    ]
    with CountingLoader() as first, CountingLoader() as second:
        drafts = image_drafts(ref, new, matches, (first, second), pdf=False)
    assert [(m.how, d.rule) for m, d in zip(matches, drafts, strict=True)] == [
        ("content", "image.degraded"),
        ("context", "image.degraded"),
    ]


def _figure(image: Image.Image) -> ImageRef:
    """Return a vector figure of a PDF, rendered as the image."""
    data = encode(image)
    return pixels_ref(
        image,
        format="vector",
        digest=content_digest(data),
        source=PixelSource(data=data),
        shown=(image.width / 2, image.height / 2),
        vector=True,
    )


@pytest.mark.parametrize(
    ("maker", "expected"),
    [
        (_figure, []),
        (_ref, [("image.reencoded", Severity.INFO)]),
    ],
    ids=["vector", "raster"],
)
def test_symmetric_drawing_one_pixel_narrower_is_the_same(
    maker: Callable[[Image.Image], ImageRef],
    expected: list[tuple[str, Severity]],
) -> None:
    """A symmetric drawing rendered one pixel narrower is not replaced."""
    wide, narrow = smiley(164, 164), smiley(163, 164)
    # Its perceptual hash flips, so that only its caption pairs it.
    assert hamming(phash(wide), phash(narrow)) > PHASH_MAX
    ref = [TEXT[0], make_image_block(maker(wide)), CAPTION, TEXT[1]]
    new = [TEXT[0], make_image_block(maker(narrow)), CAPTION, TEXT[1]]
    assert _findings(ref, new) == expected


def test_close_context_pair_unlike_in_pixels_is_replaced() -> None:
    """A context pair close in difference hash but not in SSIM is replaced."""
    wide = smiley(164, 164)
    narrow = smiley(163, 164).filter(ImageFilter.GaussianBlur(3))
    assert hamming(phash(wide), phash(narrow)) > PHASH_MAX
    ref = [TEXT[0], make_image_block(_figure(wide)), CAPTION, TEXT[1]]
    new = [TEXT[0], make_image_block(_figure(narrow)), CAPTION, TEXT[1]]
    assert _findings(ref, new) == [("image.replaced", Severity.MAJOR)]


def _bare(
    name: str = "",
    alt: str | None = None,
    digest: str = "0",
) -> ImageRef:
    """Return an image without pixels, told apart by its digest."""
    return ImageRef(name, "png", digest * 32, alt=alt, problem=NOT_FOUND)


@pytest.mark.parametrize(
    ("keys", "expected"),
    [
        ("caption alt name place", 3),
        ("alt name place", 2),
        ("name place", 1),
        ("place", 0),
    ],
)
def test_context_keys_come_in_order(keys: str, expected: int) -> None:
    """Caption, then alt, then file name, then place pair an image."""
    ref = Placed(
        make_image_block(
            _bare(
                "img/setup.png" if "name" in keys else "",
                "Installer steps" if "alt" in keys else None,
            ),
        ),
        0 if "place" in keys else 9,
        "steps" if "caption" in keys else "",
    )
    new = [
        Placed(make_image_block(_bare(digest="1")), 0, ""),
        Placed(
            make_image_block(
                _bare("_images/Setup-0a1b2c3d4e.PNG", digest="2"),
            ),
            5,
            "",
        ),
        Placed(
            make_image_block(_bare(alt=" Installer steps ", digest="3")),
            5,
            "",
        ),
        Placed(make_image_block(_bare(digest="4")), 5, "steps"),
    ]
    (match, *_) = match_images([ref], new, {0: 0})
    assert (match.new, match.how) == (expected, "context")


def test_place_lies_between_the_blocks_around_the_image() -> None:
    """A new image pairs by place only between the counterparts around."""
    ref = Placed(make_image_block(_bare()), 0, "")
    # The block before the image faces new block 0, the block after it new
    # block 3.
    follows, precedes = {0: 0, 1: 3}, {0: 0, 1: 3, 2: 5}
    for slot, paired in ((2, True), (3, False), (-1, False)):
        new = [Placed(make_image_block(_bare(digest="1")), slot, "")]
        (match, *_) = match_images([ref], new, follows, precedes)
        assert (match.new == 0) is paired


def _hashed(dhash: int, *, slot: int, digest: str) -> Placed:
    """Return a placed image that only has hashes."""
    image = ImageRef(
        name="",
        format="png",
        digest=digest * 32,
        width=400,
        height=250,
        phash=0,
        dhash=dhash,
        source=PixelSource(data=b""),
    )
    return Placed(make_image_block(image), slot, "")


def test_place_breaks_ties_of_distance_only() -> None:
    """At one distance the place decides; a nearer image wins over it."""
    ref = [_hashed(0, slot=0, digest="a"), _hashed(0, slot=1, digest="b")]
    swapped = [_hashed(0, slot=1, digest="c"), _hashed(0, slot=0, digest="d")]
    matches = match_images(ref, swapped, {0: 0, 1: 1})
    assert [(m.ref, m.new) for m in matches] == [(0, 1), (1, 0)]
    nearer = [
        _hashed(0, slot=1, digest="c"),
        _hashed(0b11, slot=0, digest="d"),
    ]
    matches = match_images(ref[:1], nearer, {0: 0})
    assert [(m.ref, m.new, m.how) for m in matches] == [
        (0, 0, "content"),
        (None, 1, None),
    ]


def test_findings_of_lookalike_pdf_images_have_their_own_ids() -> None:
    """Two missing PDF images that look alike have two identifiers."""
    image = replace(SETUP, source=PixelSource(page=0, index=0))
    blocks = [
        make_image_block(
            replace(image, source=PixelSource(page=page, index=0)),
        )
        for page in (0, 1)
    ]
    result = _compare(
        [TEXT[0], blocks[0], TEXT[1], blocks[1]],
        [TEXT[0], TEXT[1]],
    )
    ids = [f.id for f in result.findings if f.rule == "image.missing"]
    assert len(set(ids)) == len(ids) == 2


def test_lookalike_images_keep_their_counterparts() -> None:
    """Two near images each find their own counterpart."""
    first = drawing("setup")
    second = first.copy()
    ImageDraw.Draw(second).rectangle((300, 180, 340, 210), fill="#000000")
    ref = [
        TEXT[0],
        _image(_ref(first), name="a.png"),
        TEXT[1],
        _image(_ref(second), name="b.png"),
        TEXT[2],
    ]
    new = [
        TEXT[0],
        _image(_ref(scaled(second, 0.4)), name="b.png"),
        TEXT[1],
        _image(_ref(first, "JPEG", quality=95), name="a.png"),
        TEXT[2],
    ]
    result = _compare(ref, new)
    found = {
        f.rule: (f.ref.text.split(",")[0] if f.ref else "")
        for f in result.findings
        if f.category is Category.IMAGE
    }
    assert found == {"image.reencoded": "a.png", "image.degraded": "b.png"}


def test_improved_image_is_not_degraded() -> None:
    """A blurred reference against its sharp image is no degradation."""
    ref = [TEXT[0], _image(_ref(blurred(drawing("setup")))), TEXT[1]]
    new = [TEXT[0], _image(SETUP), TEXT[1]]
    assert _findings(ref, new) == [("image.reencoded", Severity.INFO)]


def test_svg_with_the_same_texts_is_reencoded() -> None:
    """An SVG whose source changed but not its texts is only information."""
    first = _svg("pipeline")
    other = replace(first, digest="0" * 32)
    assert _findings(
        [TEXT[0], _image(first), TEXT[1]],
        [TEXT[0], _image(other), TEXT[1]],
    ) == [("image.reencoded", Severity.INFO)]


def test_systematic_downscale_is_one_group() -> None:
    """Six images halved the same way are one grouped finding."""
    small = _ref(scaled(drawing("setup"), 0.4))
    ref, new = [], []
    for number in range(6):
        text = make_block(f"Paragraph {number} talks about its own topic.")
        ref += [text, _image(SETUP)]
        new += [text, _image(small)]
    result = _compare(ref, new)
    (group,) = [f for f in result.findings if f.category is Category.IMAGE]
    assert (group.rule, len(group.occurrences)) == ("image.degraded", 6)


def test_image_findings_follow_the_text_findings() -> None:
    """Findings on images come after those on the text."""
    ref = make_document(TEXT[0], _image(SETUP), TEXT[1])
    result = compare_documents(ref, make_document(TEXT[1]), Config())
    assert [f.rule for f in result.findings] == [
        "block.missing",
        "image.missing",
    ]
    stats = result.stats
    assert (stats.ref_images, stats.new_images, stats.matched_images) == (
        1,
        0,
        0,
    )


def test_figure_after_another_float_keeps_its_place() -> None:
    """A figure set after another float's caption pairs by the block after."""
    other = _image(SETUP)
    captions = [
        make_block("Figure 1: Steps of the installer", BlockKind.CAPTION),
        make_block("Figure 2: Pipeline of the service", BlockKind.CAPTION),
    ]
    # The reference floats both figures after the first paragraphs; the new
    # document keeps the first one at its place and sets its captions as
    # paragraphs, without their labels.
    ref = [
        TEXT[0],
        TEXT[1],
        other,
        captions[0],
        _image(_svg("pipeline")),
        captions[1],
        TEXT[2],
    ]
    new = [
        TEXT[0],
        other,
        make_block("Steps of the installer"),
        TEXT[1],
        _image(_ref(figure_drawing("pipeline"), "JPEG", quality=90)),
        make_block("Pipeline of the service"),
        TEXT[2],
    ]
    assert _findings(ref, new) == [("image.rasterized", Severity.MINOR)]


def test_placed_images_know_their_slot_and_caption() -> None:
    """The place of an image is the compared block before it."""
    blocks = [TEXT[0], _image(SETUP), CAPTION, TEXT[1], _image(SETUP)]
    placed = placed_images(blocks, [0, 2, 3])
    assert [(p.slot, p.caption) for p in placed] == [
        (0, "steps of the installer"),
        (2, ""),
    ]


class CountingLoader(PixelLoader):
    """A loader of pixels that keeps the digests of what it loads."""

    def __init__(self) -> None:
        """Load from no document: the test images keep their bytes."""
        super().__init__(Path("doc"))
        self.loaded: list[str] = []

    def load(self, ref: ImageRef) -> Image.Image | None:
        """Keep the digest of the image, then load it."""
        self.loaded.append(ref.digest)
        return super().load(ref)


def test_pixels_are_read_only_for_changed_content() -> None:
    """Only the pairs matched by their content are read, once per pair."""
    dashboard = _ref(drawing("dashboard"))
    small = _ref(scaled(drawing("dashboard"), 0.4))
    pairs = [
        (SETUP, SETUP, ""),
        (dashboard, small, ""),
        (dashboard, small, ""),
        (_ref(drawing("one")), _ref(drawing("two")), "caption"),
    ]
    ref, new = (
        [
            Placed(make_image_block(pair[side]), slot, pair[2])
            for slot, pair in enumerate(pairs)
        ]
        for side in (0, 1)
    )
    follows = {slot: slot for slot in range(len(pairs))}
    matches = match_images(ref, new, follows)
    assert [m.how for m in matches] == [
        "same",
        "content",
        "content",
        "context",
    ]
    with CountingLoader() as first, CountingLoader() as second:
        drafts = image_drafts(ref, new, matches, (first, second), pdf=False)
    assert [d.rule for d in drafts] == [
        "image.degraded",
        "image.degraded",
        "image.replaced",
    ]
    assert (first.loaded, second.loaded) == (
        [dashboard.digest],
        [small.digest],
    )


def _placed(hashes: np.ndarray, side: str) -> list[Placed]:
    """Return placed images that only have hashes."""
    return [
        Placed(
            make_image_block(
                ImageRef(
                    name="",
                    format="png",
                    digest=f"{side}{index:031d}",
                    width=400,
                    height=250,
                    phash=int(value),
                    dhash=int(value),
                    source=PixelSource(data=b""),
                ),
            ),
            slot=-1,
            caption="",
        )
        for index, value in enumerate(hashes)
    ]


def test_many_images_match_quickly() -> None:
    """Three thousand changed images are paired in bounded time."""
    rng = np.random.default_rng(4)
    ref = rng.integers(0, 2**63, size=3000, dtype=np.uint64)
    flips = np.left_shift(
        np.uint64(1),
        rng.integers(0, 63, size=3000, dtype=np.uint64),
    )
    order = rng.permutation(3000)
    new = (ref ^ flips)[order]
    start = time.perf_counter()
    matches = match_images(_placed(ref, "r"), _placed(new, "n"), {})
    assert time.perf_counter() - start < 3.0
    paired = {(m.ref, m.new) for m in matches if m.how == "content"}
    assert paired == {(int(i), j) for j, i in enumerate(order)}


def test_inline_svg_title_is_lost_when_the_image_has_no_alt(
    tmp_path: Path,
) -> None:
    """The title of an inline SVG is its alt: an img without one loses it."""
    drawing_png = encode(drawing("pipeline"))
    bodies = {
        "ref": (
            '<svg width="300" height="200"><title>Build pipeline</title>'
            '<sodipodi:namedview/><rect width="90" height="90"/></svg>'
        ),
        "new": '<img src="p.png">',
    }
    pages = []
    for name, body in bodies.items():
        site = tmp_path / name
        site.mkdir()
        (site / "p.png").write_bytes(drawing_png)
        (site / "index.html").write_text(
            f"<html><body><main><p>{PARAGRAPHS[0]}</p>{body}"
            f"<p>{PARAGRAPHS[1]}</p></main></body></html>",
            encoding="utf-8",
        )
        pages.append(load_html(site, HtmlConfig(), lang="en"))
    result = compare_documents(*pages, Config())
    assert "image.alt_lost" in {f.rule for f in result.findings}
