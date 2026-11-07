# Copyright 2026 Christophe Le Douarec
"""Tests for reading the characters, bookmarks and catalog of PDF files."""

import base64
import binascii
import re
import time
import zlib
from typing import TYPE_CHECKING

import pytest
from pypdf import PdfWriter
from pypdf.generic import (
    BooleanObject,
    ByteStringObject,
    DictionaryObject,
    NameObject,
    TextStringObject,
)
from reportlab.lib import pdfencrypt
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

from doc_checker.errors import InputError
from doc_checker.extract.pdf.catalog import PdfCatalog, read_catalog
from doc_checker.extract.pdf.reader import (
    LABEL_LIMIT,
    METADATA_LIMIT,
    Bookmark,
    char_text,
    font_style,
    open_pdf,
    read_bookmarks,
    read_metadata,
    read_page,
)
from doc_checker.extract.pdf.segments import build_segments
from doc_checker.images.pixels import CANNOT_DECODE, TOO_LARGE
from tests.builders import FIGURES_EN, SAMPLE_EN, SAMPLE_FR
from tests.image_builders import (
    PREDICTOR_COLUMNS,
    ccitt_group4,
    cmyk_encoded,
    drawing,
    encode,
    jpeg_announcing,
    jpeg_behind_a_predictor,
    jpeg_of_components,
    jpx_announcing,
    jpx_of_components,
)
from tests.pdf_builders import (
    build_fpdf2_pdf,
    build_image_stream_pdf,
    build_lines_pdf,
    build_oversized_image_pdf,
    build_raw_pdf,
    build_reportlab_pdf,
    build_single_page_pdf,
    scan_image,
    to_unicode_cmap,
)

if TYPE_CHECKING:
    from pathlib import Path

    from doc_checker.extract.pdf.reader import PdfImage, PdfPageText
    from doc_checker.model.document import ImageRef

_JPEG = encode(drawing("setup"), "JPEG", quality=90)
_PIXELS = drawing("setup").tobytes()  # 400 by 250 RGB pixels.
_HUGE_JPEG = jpeg_announcing(12_000, 12_000)


@pytest.fixture(scope="module")
def manual(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the English sample, with a contents page and roman labels."""
    return build_reportlab_pdf(
        SAMPLE_EN,
        tmp_path_factory.mktemp("pdf") / "en.pdf",
        toc=True,
    )


def _text(page: PdfPageText) -> str:
    """Return the text of the characters of a page."""
    return "".join(char.text for char in page.chars)


def test_fonts_sizes_and_label(manual: Path) -> None:
    """The fonts, the sizes and the label of a page are read."""
    with open_pdf(manual) as pdf:
        page = read_page(pdf, 2)  # The "Installation" page.
    text = _text(page)
    heading = page.chars[text.index("Installation")]
    body = page.chars[text.index("Install the package")]
    code = page.chars[text.index("pip install example")]
    assert (heading.size, heading.bold, heading.mono) == (14.0, True, False)
    assert (body.size, body.bold) == (11.0, False)
    assert (code.size, code.mono) == (9.0, True)
    assert (page.index, page.label, round(page.width, 2)) == (2, "2", 595.28)
    assert page.image_coverage == pytest.approx(0.0)


def test_bookmarks_labels_metadata(manual: Path) -> None:
    """The bookmarks, the page labels and the metadata are read."""
    with open_pdf(manual) as pdf:
        marks = read_bookmarks(pdf)
        labels = [read_page(pdf, i).label for i in range(3)]
        title, generator = read_metadata(pdf)
    assert marks[:3] == [
        Bookmark(0, "Getting started", 1),
        Bookmark(1, "Installation", 2),
        Bookmark(1, "First steps", 3),
    ]
    assert labels == ["i", "1", "2"]
    assert title == "Example"
    assert generator is not None
    assert generator.startswith("ReportLab")


def test_bookmarks_and_metadata_of_other_documents(tmp_path: Path) -> None:
    """Documents without outline or generator give no bookmark or generator."""
    fpdf2 = build_fpdf2_pdf(SAMPLE_EN, tmp_path / "f.pdf")
    with open_pdf(fpdf2) as pdf:
        marks = read_bookmarks(pdf)
        metadata = read_metadata(pdf)
    assert [(mark.level, mark.title) for mark in marks[:3]] == [
        (0, "Getting started"),
        (1, "Installation"),
        (1, "First steps"),
    ]
    assert all(mark.page is not None for mark in marks)
    assert metadata == ("Example", None)
    flat = build_reportlab_pdf(SAMPLE_EN, tmp_path / "n.pdf", outline=False)
    with open_pdf(flat) as pdf:
        assert read_bookmarks(pdf) == []


def test_read_metadata_cleans_values(tmp_path: Path) -> None:
    """Blanks around the metadata are stripped, and empty values vanish."""
    source = build_raw_pdf(tmp_path / "raw.pdf", [])
    padded = tmp_path / "padded.pdf"
    writer = PdfWriter(clone_from=source)
    writer.add_metadata(
        {"/Title": "  Spaced title ", "/Producer": " ", "/Creator": " Tool "},
    )
    writer.write(padded)
    empty = tmp_path / "empty.pdf"
    writer = PdfWriter(clone_from=source)
    writer.add_metadata({"/Title": "", "/Producer": "", "/Creator": ""})
    writer.write(empty)
    with open_pdf(padded) as pdf:
        assert read_metadata(pdf) == ("Spaced title", "Tool")
    with open_pdf(empty) as pdf:
        assert read_metadata(pdf) == (None, None)


def test_labels_and_metadata_are_printable(tmp_path: Path) -> None:
    """Controls in page labels and metadata never come through."""
    source = build_lines_pdf([["Some text here."]], tmp_path / "s.pdf")
    writer = PdfWriter(clone_from=source)
    writer.add_metadata({"/Producer": "Evil\x07Tool\U0000202e"})
    writer.set_page_label(0, 0, prefix="A\U0000202e\x07")
    out = tmp_path / "hostile.pdf"
    writer.write(out)
    with open_pdf(out) as pdf:
        assert read_metadata(pdf)[1] == "Evil Tool"
        assert read_page(pdf, 0).label == "A"


def test_long_labels_and_metadata_are_cut(tmp_path: Path) -> None:
    """A page label and a metadata value stop at their limits."""
    source = build_lines_pdf([["Some text here."]], tmp_path / "s.pdf")
    writer = PdfWriter(clone_from=source)
    writer.add_metadata({"/Producer": "P" * (METADATA_LIMIT + 50)})
    writer.set_page_label(0, 0, prefix="L" * (LABEL_LIMIT + 50))
    out = tmp_path / "long.pdf"
    writer.write(out)
    with open_pdf(out) as pdf:
        assert read_metadata(pdf)[1] == "P" * METADATA_LIMIT + "\U00002026"
        assert read_page(pdf, 0).label == "L" * LABEL_LIMIT + "\U00002026"


def test_a_label_of_controls_is_the_page_number(tmp_path: Path) -> None:
    """A page label with nothing printable left is the number of the page."""
    source = build_lines_pdf([["Some text here."]], tmp_path / "s.pdf")
    writer = PdfWriter(clone_from=source)
    writer.set_page_label(0, 0, prefix="\x07\U0000202e")
    out = tmp_path / "controls.pdf"
    writer.write(out)
    with open_pdf(out) as pdf:
        assert read_page(pdf, 0).label == "1"


def test_kerning_scaled_matrix_and_hyphen(tmp_path: Path) -> None:
    """Kerned words, scaled text and end of line hyphens are read."""
    path = build_raw_pdf(
        tmp_path / "raw.pdf",
        [
            (
                "BT /F1 10 Tf 72 700 Td "
                "[(Install) -333 (the) -333 (pack) -20 (age)] TJ ET"
            ),
            "BT /F1 1 Tf 12 0 0 12 72 650 Tm (Scaled) Tj ET",
            "BT /F1 10 Tf 72 600 Td (configu-) Tj 0 -12 Td (ration) Tj ET",
        ],
    )
    with open_pdf(path) as pdf:
        page = read_page(pdf, 0)
    text = _text(page)
    assert "Install the package" in text
    assert page.chars[text.index("Scaled")].size == pytest.approx(12.0)
    assert "configu-ration" in text


def test_generated_characters_take_the_previous_size(tmp_path: Path) -> None:
    """The spaces that PDFium generates are as big as the text around them."""
    path = build_raw_pdf(
        tmp_path / "raw.pdf",
        [
            # The kerning of a scaled string generates a space.
            "BT /F1 1 Tf 12 0 0 12 72 650 Tm [(Scaled) -333 (text)] TJ ET",
            # So does the gap between two strings of the same line.
            "BT /F1 10 Tf 72 600 Td (Left) Tj ET",
            "BT /F1 10 Tf 200 600 Td (Right) Tj ET",
        ],
    )
    with open_pdf(path) as pdf:
        page = read_page(pdf, 0)
    text = _text(page)
    kerned = page.chars[text.index("Scaled text") + len("Scaled")]
    gap = page.chars[text.index("Left") + len("Left")]
    assert (kerned.text, kerned.size) == (" ", pytest.approx(12.0))
    assert (gap.text, gap.size) == (" ", pytest.approx(10.0))


def test_negative_font_size_is_positive(tmp_path: Path) -> None:
    """A negative font size, which PDF allows, gives a positive size."""
    path = build_raw_pdf(
        tmp_path / "neg.pdf",
        ["BT /F1 -60 Tf 72 630 Td (FLIPPED) Tj ET"],
    )
    with open_pdf(path) as pdf:
        page = read_page(pdf, 0)
    assert page.chars[_text(page).index("F")].size == pytest.approx(60.0)
    assert all(char.size > 0 for char in page.chars)


def test_a_negative_font_size_turns_its_text(tmp_path: Path) -> None:
    """A negative font size sets its text upside down, so it is left out."""
    path = build_raw_pdf(
        tmp_path / "neg.pdf",
        [
            (
                "BT /F1 10 Tf 72 645 Td "
                "(Normal text above the flipped one.) Tj ET"
            ),
            "BT /F1 -60 Tf 72 630 Td (FLIPPED) Tj ET",
        ],
    )
    with open_pdf(path) as pdf:
        page = read_page(pdf, 0)
    assert (_text(page), page.angled) == (
        ".Normal text above the flipped one.",
        7,
    )


def test_unmapped_characters_are_replaced(tmp_path: Path) -> None:
    """A character without Unicode mapping is replaced and counted."""
    path = build_single_page_pdf(
        tmp_path / "unmapped.pdf",
        "BT /F1 12 Tf 20 100 Td (ABC) Tj ET",
        font=(
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
            "/Encoding << /Differences [66 /nosuchglyph] >> >>"
        ),
    )
    with open_pdf(path) as pdf:
        page = read_page(pdf, 0)
    assert _text(page) == "A\U0000fffdC"
    assert page.unmapped == 1


@pytest.mark.parametrize(
    ("shown", "expected", "unmapped"),
    [
        ("xAyBz", "x\U0001f600y\U0001d434z", 0),
        ("CAxDxC", "\U0000fffd\U0001f600x\U0000fffdx\U0000fffd", 3),
    ],
)
def test_surrogate_halves_are_joined(
    tmp_path: Path,
    shown: str,
    expected: str,
    unmapped: int,
) -> None:
    """The halves of an astral character join, and lone halves are replaced."""
    path = build_single_page_pdf(
        tmp_path / "astral.pdf",
        f"BT /F1 12 Tf 20 100 Td ({shown}) Tj ET",
        font=(
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
            "/ToUnicode 6 0 R >>"
        ),
        # A is U+1F600, B is U+1D434, C is a high half and D a low half.
        extra=(
            to_unicode_cmap(
                {
                    "41": "D83DDE00",
                    "42": "D835DC34",
                    "43": "D83D",
                    "44": "DE00",
                },
            ),
        ),
    )
    with open_pdf(path) as pdf:
        page = read_page(pdf, 0)
    assert _text(page) == expected
    assert all(len(char.text) == 1 for char in page.chars)
    assert _text(page).encode("utf-8").decode("utf-8") == expected
    assert page.unmapped == unmapped


def test_geometry_is_relative_to_the_crop_box(tmp_path: Path) -> None:
    """The size and the coordinates are those of the crop box."""
    shown = "BT /F1 12 Tf 60 240 Td (Header) Tj ET"
    whole = build_single_page_pdf(
        tmp_path / "whole.pdf",
        shown,
        page="/MediaBox [0 0 300 300]",
    )
    cropped = build_single_page_pdf(
        tmp_path / "cropped.pdf",
        shown,
        page="/MediaBox [0 0 300 300] /CropBox [50 50 250 250]",
    )
    with open_pdf(whole) as pdf:
        before = read_page(pdf, 0)
    with open_pdf(cropped) as pdf:
        after = read_page(pdf, 0)
    assert (before.width, before.height) == (300, 300)
    assert (after.width, after.height) == (200, 200)
    first, moved = before.chars[0], after.chars[0]
    assert first.baseline == pytest.approx(240)
    assert moved.baseline == pytest.approx(190)
    assert moved.x0 == pytest.approx(first.x0 - 50)
    assert moved.x1 == pytest.approx(first.x1 - 50)


def test_geometry_ignores_the_rotation(tmp_path: Path) -> None:
    """A rotated page has the size and the coordinates of its unrotated box."""
    shown = "BT /F1 12 Tf 60 150 Td (Header) Tj ET"
    upright = build_single_page_pdf(
        tmp_path / "upright.pdf",
        shown,
        page="/MediaBox [0 0 300 200]",
    )
    turned = build_single_page_pdf(
        tmp_path / "turned.pdf",
        shown,
        page="/MediaBox [0 0 300 200] /Rotate 90",
    )
    with open_pdf(upright) as pdf:
        before = read_page(pdf, 0)
    with open_pdf(turned) as pdf:
        after = read_page(pdf, 0)
    assert (before.width, before.height) == (300, 200)
    assert (after.width, after.height) == (300, 200)
    assert after.chars == before.chars


# The matrices that turn the text of a 300 by 400 page by each quarter turn,
# with the rotation that turns the page upright in a viewer.
TURNS = [
    pytest.param("1 0 0 1 0 0", 0, id="upright"),
    pytest.param("0 -1 1 0 0 400", 270, id="quarter"),
    pytest.param("-1 0 0 -1 300 400", 180, id="half"),
    pytest.param("0 1 -1 0 300 0", 90, id="three-quarters"),
]


@pytest.mark.parametrize(("matrix", "rotation"), TURNS)
def test_turned_pages_are_read_upright(
    tmp_path: Path,
    matrix: str,
    rotation: int,
) -> None:
    """Text that runs in one direction is read as if the page were turned."""
    content = (
        f"q {matrix} cm BT /F1 10 Tf 40 250 Td (First line) Tj "
        "0 -12 Td (Second line) Tj ET Q"
    )
    path = build_single_page_pdf(
        tmp_path / "turned.pdf",
        content,
        page=f"/MediaBox [0 0 300 400] /Rotate {rotation}",
    )
    with open_pdf(path) as pdf:
        page = read_page(pdf, 0)
    text = _text(page)
    size = (300, 400) if rotation % 180 == 0 else (400, 300)
    assert (text, (page.width, page.height), page.angled) == (
        "First lineSecond line",
        size,
        0,
    )
    first, second = page.chars[0], page.chars[text.index("Second")]
    assert (first.baseline, second.baseline) == (
        pytest.approx(250),
        pytest.approx(238),
    )
    assert first.x0 == pytest.approx(40, abs=1)


@pytest.mark.parametrize(("matrix", "rotation"), TURNS)
def test_turned_pages_are_read_from_their_crop_box(
    tmp_path: Path,
    matrix: str,
    rotation: int,
) -> None:
    """A turned page is read from the corner of its crop box."""
    # The crop box is the 300 by 400 page of the matrices, moved by 50, 60.
    content = (
        f"q 1 0 0 1 50 60 cm {matrix} cm "
        "BT /F1 10 Tf 40 250 Td (First line) Tj "
        "0 -12 Td (Second line) Tj ET Q"
    )
    path = build_single_page_pdf(
        tmp_path / "cropped.pdf",
        content,
        page=(
            "/MediaBox [0 0 400 500] /CropBox [50 60 350 460] "
            f"/Rotate {rotation}"
        ),
    )
    with open_pdf(path) as pdf:
        page = read_page(pdf, 0)
    text = _text(page)
    size = (300, 400) if rotation % 180 == 0 else (400, 300)
    first, second = page.chars[0], page.chars[text.index("Second")]
    assert (text, (page.width, page.height)) == ("First lineSecond line", size)
    assert (first.x0, first.baseline, second.baseline) == (
        pytest.approx(40, abs=1),
        pytest.approx(250),
        pytest.approx(238),
    )


@pytest.mark.parametrize(("matrix", "rotation"), TURNS)
def test_a_space_between_objects_stays_on_its_line(
    tmp_path: Path,
    matrix: str,
    rotation: int,
) -> None:
    """The space that PDFium puts between two text objects keeps its line."""
    # A line changes text object with no space glyph, as at a change of font.
    content = (
        f"q {matrix} cm "
        "BT /F1 10 Tf 40 250 Td (Hello world and) Tj ET "
        "BT /F1 10 Tf 117 250 Td (more words here) Tj ET Q"
    )
    path = build_single_page_pdf(
        tmp_path / "objects.pdf",
        content,
        page=f"/MediaBox [0 0 300 400] /Rotate {rotation}",
    )
    with open_pdf(path) as pdf:
        page = read_page(pdf, 0)
    segments = [segment.text for segment in build_segments(page)]
    assert segments == ["Hello world and more words here"]
    assert all(char.baseline == pytest.approx(250) for char in page.chars)


def test_a_tie_turns_the_page_the_least(tmp_path: Path) -> None:
    """When as many characters follow two turns, the smaller turn wins."""
    # The text that runs up the page, three quarter turns, comes first.
    content = (
        "q 0 1 -1 0 300 0 cm BT /F1 10 Tf 40 250 Td (Up the page) Tj ET Q "
        "q 0 -1 1 0 0 400 cm BT /F1 10 Tf 40 250 Td (Down a page) Tj ET Q"
    )
    path = build_single_page_pdf(
        tmp_path / "tie.pdf",
        content,
        page="/MediaBox [0 0 300 400]",
    )
    with open_pdf(path) as pdf:
        page = read_page(pdf, 0)
    assert (_text(page), page.angled, (page.width, page.height)) == (
        "Down a page",
        11,
        (400, 300),
    )


def test_text_at_an_angle_is_left_out(tmp_path: Path) -> None:
    """Text off the direction of the page is counted and left out."""
    content = (
        "BT /F1 10 Tf 40 300 Td (Body text of the page.) Tj ET "
        "q 0.7071 0.7071 -0.7071 0.7071 150 100 cm "
        "BT /F1 30 Tf 0 0 Td (DRAFT) Tj ET Q"
    )
    path = build_single_page_pdf(
        tmp_path / "draft.pdf",
        content,
        page="/MediaBox [0 0 300 400]",
    )
    with open_pdf(path) as pdf:
        page = read_page(pdf, 0)
    assert (_text(page), page.angled) == ("Body text of the page.", 5)


@pytest.mark.parametrize(
    ("matrix", "expected"),
    [
        # The slant of synthetic italics leaves the baseline level.
        ("1 0 0.21 1 72 650", (".Install the service.Then run it.", 0)),
        # A rotation of 30 degrees slants the baseline.
        ("0.866 0.5 -0.5 0.866 72 650", (".Install the service.", 12)),
    ],
    ids=["synthetic-italics", "thirty-degrees"],
)
def test_the_baseline_tells_the_angle_of_text(
    tmp_path: Path,
    matrix: str,
    expected: tuple[str, int],
) -> None:
    """Glyphs slanted on a level baseline are kept, a slanted line is not."""
    path = build_raw_pdf(
        tmp_path / "slant.pdf",
        [
            "BT /F1 10 Tf 72 720 Td (Install the service.) Tj ET",
            f"BT /F1 10 Tf {matrix} Tm (Then run it.) Tj ET",
        ],
    )
    with open_pdf(path) as pdf:
        page = read_page(pdf, 0)
    assert (_text(page), page.angled) == expected


def test_artifacts_are_dropped(tmp_path: Path) -> None:
    """The characters marked as artifacts are counted and dropped."""
    path = build_reportlab_pdf(SAMPLE_EN, tmp_path / "a.pdf", artifacts=True)
    with open_pdf(path) as pdf:
        page = read_page(pdf, 0)
    assert "Draft" not in _text(page)
    assert "Getting started" in _text(page)
    assert page.artifacts > 0


def test_scanned_page_has_no_text(tmp_path: Path) -> None:
    """A page that is only an image has no characters and is covered."""
    path = build_reportlab_pdf(
        SAMPLE_FR,
        tmp_path / "s.pdf",
        scanned_page=True,
    )
    with open_pdf(path) as pdf:
        last = read_page(pdf, 5)
    assert last.chars == ()
    assert last.image_coverage == pytest.approx(1.0)


def test_image_coverage_is_cut_to_the_page(tmp_path: Path) -> None:
    """Only the part of an image that is on the page counts as covered."""
    path = tmp_path / "corner.pdf"
    width, height = A4
    canvas = Canvas(str(path), pagesize=A4)
    # Only the corner of a 200 point image is on the page: 100 by 100 points.
    image = ImageReader(scan_image())
    canvas.drawImage(image, width - 100, height - 100, 200, 200)
    canvas.save()
    with open_pdf(path) as pdf:
        page = read_page(pdf, 0)
    expected = 100 * 100 / (width * height)
    assert page.image_coverage == pytest.approx(expected, rel=1e-3)


def test_open_pdf_errors(tmp_path: Path) -> None:
    """Protected and damaged PDFs are reported as input errors."""
    secret = build_raw_pdf(
        tmp_path / "s.pdf",
        ["BT /F1 10 Tf 72 700 Td (Hi) Tj ET"],
        encrypt="secret",
    )
    with pytest.raises(InputError, match="password protected"):
        open_pdf(secret)
    damaged = tmp_path / "bad.pdf"
    damaged.write_bytes(b"%PDF-1.7\n")
    with pytest.raises(InputError, match="invalid or damaged PDF"):
        open_pdf(damaged)


def test_read_page_reports_a_damaged_page(tmp_path: Path) -> None:
    """A page that PDFium cannot load is reported as an input error."""
    good = build_raw_pdf(tmp_path / "good.pdf", [])
    # PDFium opens the file, but it has lost the page tree.
    damaged = tmp_path / "kids.pdf"
    damaged.write_bytes(good.read_bytes().replace(b"/Kids", b"/Kidz", 1))
    with (
        open_pdf(damaged) as pdf,
        pytest.raises(InputError, match="page 1: invalid or damaged PDF"),
    ):
        read_page(pdf, 0)


def test_open_pdf_rejects_what_is_not_a_file(tmp_path: Path) -> None:
    """A missing file and a directory are reported as input errors."""
    for path in (tmp_path / "missing.pdf", tmp_path):
        message = re.escape(f"cannot read {path}: not a file")
        with pytest.raises(InputError, match=message):
            open_pdf(path)


@pytest.mark.parametrize(
    ("name", "flags", "weight", "expected"),
    [
        ("Courier", 32, 0, (True, False)),
        ("Helvetica-Bold", 32, 0, (False, True)),
        ("ABCDEF+DejaVuSansMono-Bold", 32, 400, (True, True)),
        ("NimbusRoman", 1, 400, (True, False)),
        ("Foo", 1 << 18, 400, (False, True)),
        ("Foo", 32, 700, (False, True)),
        ("Times-Roman", 32, 0, (False, False)),
        ("ArialUnicodeMS", 32, 400, (False, False)),
        ("MonotypeCorsiva", 32, 400, (False, False)),
    ],
)
def test_font_style(
    name: str,
    flags: int,
    weight: int,
    expected: tuple[bool, bool],
) -> None:
    """The fixed pitch and bold styles come from the flags and the name."""
    assert font_style(name, flags, weight) == expected


def test_char_text() -> None:
    """Hyphens and unmapped characters have their own text."""
    assert char_text(0x41, hyphen=False, unmapped=False) == "A"
    assert char_text(0x02, hyphen=True, unmapped=False) == "-"
    assert char_text(0x41, hyphen=False, unmapped=True) == "\U0000fffd"


def test_read_catalog(tmp_path: Path) -> None:
    """The language and the tagged flag come from the catalog."""
    reportlab = build_reportlab_pdf(SAMPLE_FR, tmp_path / "r.pdf")
    with open_pdf(reportlab) as pdf:
        assert read_catalog(pdf) == PdfCatalog("fr", tagged=False)
    fpdf2 = build_fpdf2_pdf(SAMPLE_EN, tmp_path / "f.pdf")
    with open_pdf(fpdf2) as pdf:
        assert read_catalog(pdf) == PdfCatalog("en", tagged=True)
    with open_pdf(build_raw_pdf(tmp_path / "raw.pdf", [])) as pdf:
        assert read_catalog(pdf) == PdfCatalog(None, tagged=False)


@pytest.mark.parametrize(
    ("entries", "expected"),
    [
        ({"/Marked": True}, PdfCatalog("en-GB", tagged=True)),
        ({"/Marked": False}, PdfCatalog("en-GB", tagged=False)),
        ({"/UserProperties": True}, PdfCatalog("en-GB", tagged=False)),
    ],
)
def test_read_catalog_tagged_flag(
    tmp_path: Path,
    entries: dict[str, bool],
    expected: PdfCatalog,
) -> None:
    """Only a true Marked entry makes a document tagged."""
    path = tmp_path / "marked.pdf"
    writer = PdfWriter(clone_from=build_raw_pdf(tmp_path / "raw.pdf", []))
    writer.root_object[NameObject("/Lang")] = TextStringObject(" en-GB ")
    writer.root_object[NameObject("/MarkInfo")] = DictionaryObject(
        {
            NameObject(key): BooleanObject(value=value)
            for key, value in entries.items()
        },
    )
    writer.write(path)
    with open_pdf(path) as pdf:
        assert read_catalog(pdf) == expected


def test_read_catalog_of_an_owner_protected_pdf(tmp_path: Path) -> None:
    """The catalog of a PDF encrypted for its owner only is read."""
    path = tmp_path / "owner.pdf"
    encrypt = pdfencrypt.StandardEncryption(
        "",
        ownerPassword="owner",
        canCopy=0,
        strength=128,
    )
    canvas = Canvas(str(path), encrypt=encrypt, lang="fr")
    canvas.setFont("Helvetica", 11)
    canvas.drawString(72, 700, "Le service accepte les demandes.")
    canvas.save()
    with open_pdf(path) as pdf:
        assert read_catalog(pdf) == PdfCatalog("fr", tagged=False)


def test_read_catalog_with_an_unpaired_surrogate(tmp_path: Path) -> None:
    """A language that holds half of a surrogate pair is read, not an error."""
    path = tmp_path / "surrogate.pdf"
    writer = PdfWriter(clone_from=build_raw_pdf(tmp_path / "raw.pdf", []))
    # UTF-16 text with a byte order mark: a high surrogate, then "f".
    writer.root_object[NameObject("/Lang")] = ByteStringObject(
        b"\xfe\xff\xd8\x00\x00f",
    )
    writer.write(path)
    with open_pdf(path) as pdf:
        assert read_catalog(pdf) == PdfCatalog("\U0000fffdf", tagged=False)


def _all_images(path: Path) -> list[PdfImage]:
    """Read the images of every page of a PDF."""
    with open_pdf(path) as pdf:
        return [
            image
            for index in range(len(pdf))
            for image in read_page(pdf, index).images
        ]


def test_page_images_are_read(tmp_path: Path) -> None:
    """Each image object gives its pixels, its format and its resolution."""
    images = [
        image
        for image in _all_images(
            build_reportlab_pdf(FIGURES_EN, tmp_path / "a.pdf"),
        )
        if not image.ref.vector
    ]
    assert [(i.ref.format, i.ref.width, i.ref.height) for i in images] == [
        ("flate", 400, 250),
        ("jpeg", 400, 250),
        ("flate", 400, 250),
    ]
    setup, dashboard, _ = images
    assert setup.ref.readable
    sources = [image.ref.source for image in images]
    assert [s.index for s in sources if s is not None] == [0, 0, 0]
    assert dashboard.ref.quality == 90
    assert setup.x1 - setup.x0 == pytest.approx(100 * mm, abs=0.5)
    assert setup.ref.dpi == pytest.approx(400 / (100 / 25.4), rel=0.02)


def test_images_can_be_skipped(tmp_path: Path) -> None:
    """Without images, no image object is decoded."""
    path = build_reportlab_pdf(FIGURES_EN, tmp_path / "a.pdf")
    with open_pdf(path) as pdf:
        assert all(
            not read_page(pdf, index, images=False).images
            for index in range(len(pdf))
        )


def test_image_in_a_form_has_its_box_on_the_page(tmp_path: Path) -> None:
    """An image drawn in a scaled form has the box and resolution it shows."""
    path = tmp_path / "form.pdf"
    canvas = Canvas(str(path), pagesize=A4)
    canvas.beginForm("figure")
    canvas.drawImage(ImageReader(drawing("setup")), 0, 0, 200, 125)
    canvas.endForm()
    canvas.translate(100, 300)
    canvas.scale(2, 2)
    canvas.doForm("figure")
    canvas.save()
    (image,) = _all_images(path)
    box = (image.x0, image.x1, image.bottom, image.top)
    assert box == pytest.approx((100, 500, 300, 550), abs=0.01)
    assert image.ref.shown == pytest.approx((400, 250), abs=0.01)
    assert image.ref.dpi == pytest.approx(72.0)


@pytest.mark.parametrize("angle", [90, 180, 270])
def test_image_on_a_turned_page_turns_upright(
    tmp_path: Path,
    angle: int,
) -> None:
    """The box of an image turns upright as the text of its page does."""
    path = tmp_path / "turned.pdf"
    width, height = A4
    origins = {90: (width, 0), 180: (width, height), 270: (0, height)}
    canvas = Canvas(str(path), pagesize=A4)
    canvas.translate(*origins[angle])
    canvas.rotate(angle)
    canvas.setFont("Times-Roman", 10)
    canvas.drawString(72, width - 72, "The text above the image.")
    # The image is 200 by 125 points, 28 points right of the text and its
    # top 18 points under the baseline.
    canvas.drawImage(ImageReader(drawing("setup")), 100, width - 215, 200, 125)
    canvas.save()
    with open_pdf(path) as pdf:
        page = read_page(pdf, 0)
    (image,) = page.images
    first = page.chars[0]
    box = (image.x0 - first.x0, image.x1, image.bottom, image.top)
    expected = (28, image.x0 + 200, first.baseline - 143, first.baseline - 18)
    assert box == pytest.approx(expected, abs=0.5)
    assert image.ref.dpi == pytest.approx(144.0)


@pytest.mark.parametrize(
    ("header", "image_format"),
    [(False, "flate"), (True, "jpeg")],
)
def test_image_beyond_the_pixel_limit_is_not_decoded(
    tmp_path: Path,
    header: bool,  # ruff: ignore[boolean-type-hint-positional-argument] - pytest parameter
    image_format: str,
) -> None:
    """An image whose dictionary or header has too many pixels is refused."""
    path = build_oversized_image_pdf(tmp_path / "large.pdf", header=header)
    (image,) = _all_images(path)
    assert (image.ref.problem, image.ref.readable, image.ref.source) == (
        TOO_LARGE,
        False,
        None,
    )
    assert image.ref.format == image_format


_RGB = "/ColorSpace /DeviceRGB /BitsPerComponent 8"
# The predictor of the decoy, a PNG one, of one byte a pixel.
_DECOY_ROWS = (
    f"{_RGB} /DecodeParms [<< /Predictor 12 /Colors 1 /BitsPerComponent 8"
    f" /Columns {PREDICTOR_COLUMNS} >> null]"
)
# Rows of RGB pixels, each after a tag byte, as pdflatex and fpdf2 write.
_PNG_ROWS = (
    f"{_RGB} /DecodeParms << /Predictor 15 /Colors 3 /BitsPerComponent 8"
    " /Columns 400 >>"
)
_CMYK = "/ColorSpace /DeviceCMYK /BitsPerComponent 8"
_GROUP_4 = (
    "/ColorSpace /DeviceGray /BitsPerComponent 1"
    " /DecodeParms << /K -1 /Columns 400 /Rows 250 /BlackIs1 false >>"
)
_JPX = jpx_announcing(400, 250)


def _stream_image(
    tmp_path: Path,
    data: bytes,
    filters: str,
    entries: str = _RGB,
) -> ImageRef:
    """Read the image of a page whose stream and filters are given."""
    path = build_image_stream_pdf(
        tmp_path / "s.pdf",
        data,
        filters=filters,
        entries=entries,
    )
    (image,) = _all_images(path)
    return image.ref


def _run_length_literals(data: bytes) -> bytes:
    """Encode bytes as RunLength runs of one literal byte each."""
    runs = bytearray(2 * len(data))
    runs[1::2] = data
    return bytes(runs) + b"\x80"


@pytest.mark.parametrize(
    ("data", "filters", "entries", "problem"),
    [
        pytest.param(
            b"\x00" * 16 + _HUGE_JPEG,
            "/DCTDecode",
            _RGB,
            TOO_LARGE,
            id="bytes-before-soi",
        ),
        pytest.param(
            _HUGE_JPEG,
            "/DCT",
            _RGB,
            TOO_LARGE,
            id="abbreviated-dct",
        ),
        pytest.param(
            zlib.compress(_HUGE_JPEG),
            "[/FlateDecode /DCTDecode]",
            _RGB,
            TOO_LARGE,
            id="wrapped-huge-sof",
        ),
        pytest.param(
            base64.a85encode(_HUGE_JPEG) + b"~>",
            "[/A85 /DCT]",
            _RGB,
            TOO_LARGE,
            id="ascii85-huge-sof",
        ),
        pytest.param(
            jpeg_announcing(800, 500),
            "/DCTDecode",
            _RGB,
            TOO_LARGE,
            id="sof-above-the-dictionary",
        ),
        pytest.param(
            jpx_announcing(20_000, 20_000),
            "/JPXDecode",
            _RGB,
            TOO_LARGE,
            id="huge-siz",
        ),
        pytest.param(
            zlib.compress(jpx_announcing(20_000, 20_000)),
            "[/FlateDecode /JPXDecode]",
            _RGB,
            TOO_LARGE,
            id="wrapped-huge-siz",
        ),
        pytest.param(
            jpx_of_components(16),
            "/JPXDecode",
            _RGB,
            CANNOT_DECODE,
            id="jpx-of-16-components",
        ),
        pytest.param(
            jpeg_of_components(5),
            "/DCTDecode",
            _RGB,
            CANNOT_DECODE,
            id="jpeg-of-5-components",
        ),
        pytest.param(
            b"no frame",
            "/DCTDecode",
            _RGB,
            CANNOT_DECODE,
            id="no-sof",
        ),
        pytest.param(
            b"no codestream",
            "/JPXDecode",
            _RGB,
            CANNOT_DECODE,
            id="no-siz",
        ),
        pytest.param(
            zlib.compress(_JPEG + bytes(4 << 20)),
            "[/FlateDecode /DCTDecode]",
            _RGB,
            CANNOT_DECODE,
            id="inflates-past-any-jpeg",
        ),
        pytest.param(
            zlib.compress(jpeg_behind_a_predictor(20_000, 20_000)),
            "[/FlateDecode /DCTDecode]",
            _DECOY_ROWS,
            TOO_LARGE,
            id="jpeg-behind-a-predictor",
        ),
        pytest.param(
            zlib.compress(zlib.compress(_PIXELS) + bytes(8 << 20)),
            "[/FlateDecode /FlateDecode]",
            _RGB,
            CANNOT_DECODE,
            id="flate-in-flate",
        ),
        pytest.param(
            b"\x80\x0b\x60\x50\x22\x0c\x0c\x85\x01",
            "[/LZWDecode /DCTDecode]",
            _RGB,
            CANNOT_DECODE,
            id="lzw-wrapped-jpeg",
        ),
        pytest.param(
            b"\x80\x0b\x60\x50\x22\x0c\x0c\x85\x01",
            "/LZWDecode",
            _RGB,
            CANNOT_DECODE,
            id="lzw",
        ),
        pytest.param(
            base64.a85encode(zlib.compress(_JPEG)) + b"~>",
            "[/A85 /Fl /DCT]",
            _RGB,
            CANNOT_DECODE,
            id="three-filters",
        ),
        pytest.param(
            binascii.hexlify(_JPX) + b">",
            "[/AHx /JPXDecode]",
            _RGB,
            CANNOT_DECODE,
            id="ascii-hex-jpx",
        ),
        pytest.param(
            base64.a85encode(_PIXELS) + b"~>",
            "/ASCII85Decode",
            _RGB,
            CANNOT_DECODE,
            id="ascii85-alone",
        ),
        pytest.param(
            b"\xff\xd8" + b"\xff" * 100_000 + b"\x00",
            "/DCTDecode",
            _RGB,
            CANNOT_DECODE,
            id="run-of-ff",
        ),
        pytest.param(
            zlib.compress(b"\xff\xd8" + b"\xff" * 64_000 + b"\x00"),
            "[/FlateDecode /DCTDecode]",
            _RGB,
            CANNOT_DECODE,
            id="wrapped-run-of-ff",
        ),
        pytest.param(
            zlib.compress(b"\xff\xd8" + b"\xff\xd0" * 500_000),
            "[/FlateDecode /DCTDecode]",
            _RGB,
            CANNOT_DECODE,
            id="restart-markers-before-the-frame",
        ),
        pytest.param(
            b"{" + base64.a85encode(_JPEG) + b"~>",
            "[/ASCII85Decode /DCTDecode]",
            _RGB,
            CANNOT_DECODE,
            id="ascii85-out-of-alphabet",
        ),
        pytest.param(
            b"!!z!!!" + base64.a85encode(_JPEG) + b"~>",
            "[/ASCII85Decode /DCTDecode]",
            _RGB,
            CANNOT_DECODE,
            id="ascii85-z-inside-a-group",
        ),
        pytest.param(
            bytes(range(256)) * 20,
            "/FlateDecode",
            _RGB,
            CANNOT_DECODE,
            id="corrupt-flate",
        ),
        pytest.param(
            zlib.compress(_PIXELS)[:200],
            "/FlateDecode",
            _RGB,
            CANNOT_DECODE,
            id="truncated-flate",
        ),
        pytest.param(
            _JPEG[: len(_JPEG) // 3],
            "/DCTDecode",
            _RGB,
            CANNOT_DECODE,
            id="truncated-jpeg",
        ),
        pytest.param(
            zlib.compress(_PIXELS),
            "[/DCTDecode /FlateDecode]",
            _RGB,
            CANNOT_DECODE,
            id="codec-before-a-filter",
        ),
    ],
)
def test_unsound_image_streams_are_refused(
    tmp_path: Path,
    data: bytes,
    filters: str,
    entries: str,
    problem: str,
) -> None:
    """A stream that announces too many pixels, or lacks some, is refused."""
    ref = _stream_image(tmp_path, data, filters, entries)
    assert (ref.problem, ref.readable, ref.source) == (problem, False, None)


@pytest.mark.parametrize(
    ("data", "filters", "entries", "image_format"),
    [
        pytest.param(_PIXELS, "", _RGB, "raw", id="raw"),
        pytest.param(
            zlib.compress(_PIXELS),
            "/FlateDecode",
            _RGB,
            "flate",
            id="flate",
        ),
        pytest.param(
            zlib.compress(
                b"".join(
                    b"\x00" + _PIXELS[row : row + 1200]
                    for row in range(0, len(_PIXELS), 1200)
                ),
            ),
            "/Fl",
            _PNG_ROWS,
            "flate",
            id="flate-with-png-predictor",
        ),
        pytest.param(
            _run_length_literals(_PIXELS),
            "/RunLengthDecode",
            _RGB,
            "runlength",
            id="run-length-literals",
        ),
        pytest.param(
            base64.a85encode(zlib.compress(_PIXELS)) + b"~>",
            "[/ASCII85Decode /FlateDecode]",
            _RGB,
            "flate",
            id="ascii85-flate",
        ),
        pytest.param(
            binascii.hexlify(zlib.compress(_PIXELS), b" ", 32) + b">",
            "[/AHx /Fl]",
            _RGB,
            "flate",
            id="ascii-hex-flate",
        ),
        pytest.param(_JPEG, "/DCT", _RGB, "jpeg", id="dct"),
        pytest.param(
            _JPEG,
            "/DCTDecode",
            _PNG_ROWS,
            "jpeg",
            id="dct-with-ignored-predictor",
        ),
        pytest.param(
            b"\x00" * 16 + _JPEG,
            "/DCTDecode",
            _RGB,
            "jpeg",
            id="bytes-before-soi",
        ),
        pytest.param(
            cmyk_encoded("JPEG"),
            "/DCTDecode",
            _CMYK,
            "jpeg",
            id="cmyk-jpeg",
        ),
        pytest.param(_JPX, "/JPXDecode", _RGB, "jpx", id="jpx"),
        pytest.param(
            cmyk_encoded("JPEG2000"),
            "/JPXDecode",
            _CMYK,
            "jpx",
            id="cmyk-jpx",
        ),
        pytest.param(
            ccitt_group4(drawing("setup")),
            "/CCITTFaxDecode",
            _GROUP_4,
            "ccitt",
            id="ccitt",
        ),
        pytest.param(
            base64.a85encode(_JPEG) + b"~>",
            "[/ASCII85Decode /DCTDecode]",
            _RGB,
            "jpeg",
            id="ascii85-jpeg",
        ),
        pytest.param(
            base64.a85encode(_JPEG, wrapcol=72) + b"\n~>",
            "[/ASCII85Decode /DCTDecode]",
            f"{_RGB} /DecodeParms [<< /Predictor 2 >> null]",
            "jpeg",
            id="ascii85-jpeg-with-ignored-predictor",
        ),
        pytest.param(
            binascii.hexlify(_JPEG) + b">",
            "[/AHx /DCT]",
            _RGB,
            "jpeg",
            id="ascii-hex-jpeg",
        ),
        pytest.param(
            zlib.compress(_JPEG),
            "[/Fl /DCTDecode]",
            _RGB,
            "jpeg",
            id="flate-jpeg",
        ),
        pytest.param(
            base64.a85encode(_JPX) + b"~>",
            "[/A85 /JPXDecode]",
            _RGB,
            "jpx",
            id="ascii85-jpx",
        ),
        pytest.param(
            zlib.compress(_JPX),
            "[/FlateDecode /JPXDecode]",
            _RGB,
            "jpx",
            id="flate-jpx",
        ),
    ],
)
def test_sound_image_streams_are_read(
    tmp_path: Path,
    data: bytes,
    filters: str,
    entries: str,
    image_format: str,
) -> None:
    """Each chain of filters that generators write is read when sound."""
    ref = _stream_image(tmp_path, data, filters, entries)
    assert (ref.format, ref.width, ref.height, ref.problem) == (
        image_format,
        400,
        250,
        None,
    )
    assert ref.quality == (90 if image_format == "jpeg" else None)


def test_a_run_of_ff_is_read_in_linear_time(tmp_path: Path) -> None:
    """A frame header sought past 100,000 bytes 0xFF takes little time."""
    data = b"\xff\xd8" + b"\xff" * 100_000 + b"\x00"
    start = time.perf_counter()
    ref = _stream_image(tmp_path, data, "/DCTDecode")
    assert time.perf_counter() - start < 2.0
    assert ref.problem == CANNOT_DECODE


@pytest.mark.parametrize(
    ("data", "problem"),
    [(_JPEG, None), (_HUGE_JPEG, TOO_LARGE)],
)
def test_inline_wrapped_jpegs_are_checked(
    tmp_path: Path,
    data: bytes,
    problem: str | None,
) -> None:
    """An inline JPEG in an ASCIIHex layer is read, or refused, as others."""
    content = (
        "q 200 0 0 125 72 600 cm"
        " BI /W 400 /H 250 /CS /RGB /BPC 8 /F [/AHx /DCT] ID\n"
        f"{binascii.hexlify(data, b'\n', 64).decode()}>\nEI Q"
    )
    path = build_single_page_pdf(
        tmp_path / "inline.pdf",
        content,
        page="/MediaBox [0 0 595 842]",
    )
    (image,) = _all_images(path)
    assert (image.ref.format, image.ref.problem) == ("jpeg", problem)


@pytest.mark.parametrize(
    ("padding", "expected"),
    [
        (1_000_000, (True, 400, 250, 90, None)),
        (17_000_000, (False, None, None, None, CANNOT_DECODE)),
    ],
)
def test_an_ascii_wrapped_jpeg_is_read_up_to_16_mb(
    tmp_path: Path,
    padding: int,
    expected: tuple[bool, int | None, int | None, int | None, str | None],
) -> None:
    """A JPEG in ASCII85 is read up to 16 MB of stream, and refused past."""
    text = base64.a85encode(_JPEG) + b" " * padding + b"~>"
    ref = _stream_image(tmp_path, text, "[/ASCII85Decode /DCTDecode]")
    assert (
        ref.readable,
        ref.width,
        ref.height,
        ref.quality,
        ref.problem,
    ) == expected


@pytest.mark.parametrize("stop", [b"\x00", b"\x0c"])
@pytest.mark.parametrize(
    ("padding", "problem"),
    [(0, TOO_LARGE), (17_000_000, CANNOT_DECODE)],
)
def test_a_jpeg_after_an_ascii85_layer_ended_at_once_is_refused(
    tmp_path: Path,
    stop: bytes,
    padding: int,
    problem: str,
) -> None:
    """PDFium decodes the raw stream past an empty layer: it is checked."""
    layer = stop + base64.a85encode(_JPEG) + b"~>"
    path = build_image_stream_pdf(
        tmp_path / "s.pdf",
        layer + _HUGE_JPEG + b" " * padding,
        filters="[/A85 /DCT]",
    )
    start = time.perf_counter()
    (image,) = _all_images(path)
    assert time.perf_counter() - start < 1.0
    assert (image.ref.problem, image.ref.readable) == (problem, False)


def test_an_ascii_wrapped_jpeg_drawn_on_many_pages_is_read_quickly(
    tmp_path: Path,
) -> None:
    """A 16 MB stream of few bytes behind a long head, on 20 pages."""
    head = b"\xff\xd8" + b"\xff" * 1_010_000 + b"\xe0\x00\x02"
    text = base64.a85encode(head + b"\xff\xe0\x00\x02" * 5000)
    window = 1 << 14
    text += b" " * (-len(text) % window)
    groups = (15_900_000 - len(text)) // window
    text += (b"!!!!!" + b" " * (window - 5)) * groups + b"~>"
    path = build_image_stream_pdf(
        tmp_path / "s.pdf",
        text,
        filters="[/A85 /DCT]",
        pages=20,
    )
    start = time.perf_counter()
    images = _all_images(path)
    assert time.perf_counter() - start < 10.0
    assert {image.ref.problem for image in images} == {CANNOT_DECODE}
    assert len(images) == 20
