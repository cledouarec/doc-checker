# Copyright 2026 Christophe Le Douarec
"""Tests for the builders of test PDF documents."""

from typing import TYPE_CHECKING

import pytest
from pypdf import PdfReader

from tests.builders import SAMPLE_EN, SAMPLE_FR
from tests.pdf_builders import (
    build_fpdf2_pdf,
    build_linked_pdf,
    build_raw_pdf,
    build_reportlab_pdf,
    build_scanned_pdf,
    make_line,
    make_segment,
    pdf_typography,
    scan_image,
    soft_hyphenate,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pypdf.types import OutlineType


def _outline(path: Path) -> list[tuple[int, str]]:
    """Return the (depth, title) of the outline items, depth first."""
    items: list[tuple[int, str]] = []

    def walk(nodes: OutlineType, depth: int) -> None:
        # A nested list holds the children of the item just before it.
        for node in nodes:
            if isinstance(node, list):
                walk(node, depth + 1)
            else:
                items.append((depth, node.title))

    walk(PdfReader(path).outline, 0)
    return items


def test_soft_hyphenate() -> None:
    """Long words get a soft hyphen every three letters, short ones none."""
    assert soft_hyphenate("the configuration installer 2026") == (
        "the con\xadfig\xadura\xadtion ins\xadtaller 2026"
    )
    assert soft_hyphenate("short words") == "short words"


def test_pdf_typography_avoids_narrow_spaces() -> None:
    """Narrow no-break spaces are replaced, as cp1252 has none."""
    text = pdf_typography(
        'Attention : le mode "strict" ; 1 000 fichiers',
        "fr",
    )
    assert text == (
        "Attention\xa0: le mode \xab\xa0strict\xa0\xbb\xa0; 1\xa0000 fichiers"
    )


def test_scan_image() -> None:
    """The scan is a white sheet crossed by black lines 3 pixels thick."""
    image = scan_image()
    assert (image.mode, image.size) == ("RGB", (400, 560))
    white, black = (255, 255, 255), (0, 0, 0)
    column = [image.getpixel((200, y)) for y in range(560)]
    assert column[18:23] == [white, black, black, black, white]
    assert column[530:535] == [white, black, black, black, white]
    assert column[540] == white
    row = [image.getpixel((x, 20)) for x in range(400)]
    assert row[19:22] == [white, black, black]
    assert row[379:382] == [black, black, white]


def test_reportlab_layout(tmp_path: Path) -> None:
    """The reportlab builder gives one page per section and an outline."""
    en = build_reportlab_pdf(SAMPLE_EN, tmp_path / "en.pdf")
    assert len(PdfReader(en).pages) == 9
    assert _outline(en) == [
        (0, "Getting started"),
        (1, "Installation"),
        (1, "First steps"),
        (0, "User guide"),
        (1, "Projects"),
        (1, "Exports"),
        (0, "Reference"),
        (1, "Command line"),
        (1, "Limits"),
    ]
    root = PdfReader(en).trailer["/Root"]
    assert (root["/Lang"], "/MarkInfo" in root) == ("en", False)
    toc = build_reportlab_pdf(SAMPLE_EN, tmp_path / "t.pdf", toc=True)
    assert len(PdfReader(toc).pages) == 10
    scanned = build_reportlab_pdf(
        SAMPLE_FR,
        tmp_path / "s.pdf",
        scanned_page=True,
    )
    assert len(PdfReader(scanned).pages) == 6
    flat = build_reportlab_pdf(SAMPLE_EN, tmp_path / "n.pdf", outline=False)
    assert _outline(flat) == []


def test_reportlab_toc_and_artifacts(tmp_path: Path) -> None:
    """The contents page is numbered apart and artifacts frame the chrome."""
    path = build_reportlab_pdf(SAMPLE_EN, tmp_path / "t.pdf", toc=True)
    toc = PdfReader(path)
    assert toc.page_labels[:3] == ["i", "1", "2"]
    # Only the contents entry: the header of the first page does not keep the
    # last chapter of the previous pass.
    assert toc.pages[0].extract_text().count("Reference") == 1
    for artifacts in (False, True):
        out = tmp_path / f"{artifacts}.pdf"
        build_reportlab_pdf(SAMPLE_EN, out, artifacts=artifacts)
        content = PdfReader(out).pages[0].get_contents().get_data()
        marked = (b"/Artifact BMC" in content, b"EMC" in content)
        assert marked == (artifacts, artifacts)


def test_fpdf2_layout(tmp_path: Path) -> None:
    """The fpdf2 builder gives a tagged two column document."""
    en = PdfReader(build_fpdf2_pdf(SAMPLE_EN, tmp_path / "en.pdf"))
    fr = PdfReader(build_fpdf2_pdf(SAMPLE_FR, tmp_path / "fr.pdf"))
    assert (len(en.pages), len(fr.pages)) == (2, 1)
    root = fr.trailer["/Root"]
    assert (root["/Lang"], bool(root["/MarkInfo"]["/Marked"])) == ("fr", True)


def test_fpdf2_bookmarks(tmp_path: Path) -> None:
    """The fpdf2 builder adds the section titles as bookmarks."""
    fr = build_fpdf2_pdf(SAMPLE_FR, tmp_path / "fr.pdf")
    assert _outline(fr) == [
        (0, "Prise en main"),
        (1, "Installation"),
        (1, "Premiers pas"),
        (0, "Guide"),
        (1, "Projets"),
    ]


def test_raw_and_scanned_pdfs(tmp_path: Path) -> None:
    """The raw PDF can be encrypted and the scanned one has no text."""
    raw = build_raw_pdf(
        tmp_path / "r.pdf",
        ["BT /F1 10 Tf 72 700 Td (Hi) Tj ET"],
        encrypt="secret",
    )
    assert PdfReader(raw).is_encrypted
    scanned = PdfReader(build_scanned_pdf(tmp_path / "s.pdf", pages=3))
    assert len(scanned.pages) == 3
    assert not scanned.pages[0].extract_text()


def test_raw_pdf_declares_font(tmp_path: Path) -> None:
    """The operators select /F1, a font the page declares."""
    raw = build_raw_pdf(
        tmp_path / "r.pdf",
        ["BT /F1 10 Tf 72 700 Td (Hi) Tj ET"],
    )
    page = PdfReader(raw).pages[0]
    assert "/F1" in page["/Resources"]["/Font"]
    assert "Hi" in page.extract_text()


def test_make_segment() -> None:
    """A segment is as wide as its text, unless it is given a width."""
    segment = make_segment("abcd", 72.0, 700.0, size=8.0)
    assert (segment.x0, segment.x1, segment.baseline, segment.size) == (
        72.0,
        88.0,
        700.0,
        8.0,
    )
    assert (segment.page, segment.mono, segment.bold) == (0, False, False)
    wide = make_segment(
        "abcd",
        72.0,
        700.0,
        width=30.0,
        page=2,
        mono=True,
        bold=True,
    )
    assert (wide.x1, wide.page, wide.mono, wide.bold) == (102.0, 2, True, True)


def test_make_line() -> None:
    """A line is as wide as its text, unless it is given a width."""
    line = make_line("abcd", 700.0, size=8.0)
    assert (line.x0, line.x1, line.baseline, line.size) == (
        72.0,
        88.0,
        700.0,
        8.0,
    )
    assert (line.text, line.page, line.mono, line.bold) == (
        "abcd",
        0,
        False,
        False,
    )
    wide = make_line(
        "abcd",
        700.0,
        x0=10.0,
        width=30.0,
        page=2,
        mono=True,
        bold=True,
    )
    assert (wide.x0, wide.x1, wide.page, wide.mono, wide.bold) == (
        10.0,
        40.0,
        2,
        True,
        True,
    )


def test_linked_pdf_refuses_an_unknown_action(tmp_path: Path) -> None:
    """A link action of no known kind is an error, not a link without one."""
    with pytest.raises(ValueError, match="urI:x"):
        build_linked_pdf(tmp_path / "a.pdf", [("one", "urI:x")])
