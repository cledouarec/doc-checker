# Copyright 2026 Christophe Le Douarec
"""Tests for the vector figures of PDF pages."""

import time
from typing import TYPE_CHECKING

import pytest
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import SimpleDocTemplate, Spacer, Table, TableStyle

from doc_checker.config import PdfConfig
from doc_checker.extract.pdf.figures import (
    CLUSTER_TEXT_MAX,
    Glyph,
    PathFacts,
    are_labels,
    cluster,
    is_drawing,
    vector_figures,
)
from doc_checker.extract.pdf.loader import load_pdf
from doc_checker.extract.pdf.reader import (
    MAX_PAGE_FIGURES,
    NOT_RENDERED,
    open_pdf,
    read_page,
)
from doc_checker.images.load import RENDER_SIDE_MAX, PixelLoader
from doc_checker.images.pixels import PAGE_LIMIT, TOO_LARGE
from doc_checker.model.document import BlockKind
from tests.builders import FIGURES_EN
from tests.pdf_builders import (
    FIGURE_LABEL,
    build_dense_paths_pdf,
    build_figure_pdf,
    build_fpdf2_pdf,
    build_many_figures_pdf,
    build_reportlab_pdf,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from reportlab.lib.colors import Color

BLUE, RED = (68, 119, 170), (238, 102, 119)


def test_boxes_that_touch_make_one_cluster() -> None:
    """Boxes close enough are one group, the others stay apart."""
    boxes = [
        (0.0, 0.0, 10.0, 10.0),
        (14.0, 0.0, 20.0, 10.0),
        (100.0, 0.0, 110.0, 10.0),
        (18.0, 12.0, 30.0, 20.0),
    ]
    assert cluster(boxes, 6.0) == [[0, 1, 3], [2]]


@pytest.mark.parametrize(
    ("paths", "drawing"),
    [
        (
            [
                PathFacts((0, 0, 50, 40), bezier=True, fill=BLUE),
                PathFacts((50, 0, 60, 2), bezier=False, fill=None),
                PathFacts((60, 0, 90, 40), bezier=False, fill=None),
            ],
            True,
        ),
        (
            [
                PathFacts((0, 0, 50, 40), bezier=False, fill=BLUE),
                PathFacts((50, 10, 60, 14), bezier=False, fill=None),
                PathFacts((60, 0, 90, 40), bezier=False, fill=RED),
            ],
            True,
        ),
        (
            [
                PathFacts((0, 0, 200, 12), bezier=False, fill=BLUE),
                PathFacts((0, 12, 200, 24), bezier=False, fill=RED),
                PathFacts((0, 24, 200, 36), bezier=False, fill=BLUE),
            ],
            False,
        ),
        (
            [
                PathFacts((0, 0, 200, 1), bezier=False, fill=None),
                PathFacts((0, 20, 200, 21), bezier=False, fill=None),
                PathFacts((0, 40, 200, 41), bezier=False, fill=None),
            ],
            False,
        ),
        (
            [
                PathFacts((0, 0, 50, 40), bezier=True, fill=BLUE),
                PathFacts((60, 0, 90, 40), bezier=False, fill=RED),
            ],
            False,
        ),
    ],
    ids=["curve", "two-colors", "table-rows", "rules", "too-few"],
)
def test_drawings_are_told_apart(
    paths: list[PathFacts],
    *,
    drawing: bool,
) -> None:
    """Curves or colored shapes make a drawing, rows and rules do not."""
    assert is_drawing(paths) is drawing


def _striped_cells(gap: float) -> list[PathFacts]:
    """Fill two columns of five rows, a colour a row, with the rules."""
    cells = [
        PathFacts(
            (left, bottom, left + 140.0, bottom + 14.0),
            bezier=False,
            fill=BLUE if row % 2 else RED,
        )
        for row, bottom in enumerate(range(0, 70, 14))
        for left in (0.0, 140.0 + gap)
    ]
    width = 280 + gap
    rules = [
        PathFacts((0, bottom, width, bottom + 0.5), bezier=False, fill=None)
        for bottom in range(0, 71, 14)
    ]
    return [*cells, *rules]


@pytest.mark.parametrize(
    ("gap", "drawing"),
    [(0.0, False), (0.4, False), (20.0, True)],
    ids=["abutting", "rounded", "apart"],
)
def test_cells_striped_row_by_row_are_no_drawing(
    gap: float,
    *,
    drawing: bool,
) -> None:
    """Cells that tile each row in one colour are the stripes of a table."""
    assert is_drawing(_striped_cells(gap)) is drawing


def _figures(path: Path) -> list[tuple[int, bool]]:
    """List the page and the kind of each vector figure of a PDF."""
    with open_pdf(path) as pdf:
        return [
            (index, figure.form)
            for index in range(len(pdf))
            for figure in vector_figures(pdf[index])
        ]


def test_figures_are_found_as_paths_or_forms(tmp_path: Path) -> None:
    """The drawn figure is one cluster, or one form, in each generator."""
    paths = _figures(build_reportlab_pdf(FIGURES_EN, tmp_path / "a.pdf"))
    forms = _figures(
        build_reportlab_pdf(FIGURES_EN, tmp_path / "b.pdf", figure_forms=True),
    )
    fpdf2 = _figures(build_fpdf2_pdf(FIGURES_EN, tmp_path / "c.pdf"))
    assert [form for _, form in paths] == [False]
    assert [form for _, form in forms] == [True]
    assert [form for _, form in fpdf2] == [False]


@pytest.mark.parametrize("forms", [False, True])
def test_figure_labels_leave_the_text(tmp_path: Path, *, forms: bool) -> None:
    """The labels drawn in a figure are no paragraph, and are counted."""
    path = build_reportlab_pdf(
        FIGURES_EN,
        tmp_path / "a.pdf",
        figure_forms=forms,
    )
    doc = load_pdf(path, PdfConfig(), lang="en")
    assert not [b.text for b in doc.blocks if "pipeline in" in b.text]
    assert doc.normalization["figure_text"] == len("pipeline in pipeline out")
    figures = [b.image for b in doc.blocks if b.image and b.image.vector]
    assert len(figures) == 1
    assert figures[0].readable


def test_tables_and_rules_are_no_figures(tmp_path: Path) -> None:
    """A table with a grid and striped rows keeps its text."""
    path = tmp_path / "table.pdf"
    rows = [["Option", "Default"], ["timeout", "30"], ["retries", "3"]]
    rows += [[f"key{n}", str(n)] for n in range(12)]
    table = Table(rows, colWidths=[200, 120])
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                (
                    "ROWBACKGROUNDS",
                    (0, 0),
                    (-1, -1),
                    [colors.HexColor("#eeeeee"), colors.HexColor("#cccccc")],
                ),
            ],
        ),
    )
    SimpleDocTemplate(str(path)).build([table])
    assert _figures(path) == []
    doc = load_pdf(path, PdfConfig(), lang="en")
    assert "timeout" in " ".join(b.text for b in doc.blocks)
    assert doc.normalization["figure_text"] == 0


def test_tables_striped_cell_by_cell_keep_their_text(tmp_path: Path) -> None:
    """A table whose cells are filled one by one, as LaTeX does, keeps it."""
    path = tmp_path / "cells.pdf"
    rows = [["Option", "Default"], ["timeout", "30"], ["retries", "3"]]
    rows += [[f"key{n}", str(n)] for n in range(12)]
    stripes = [colors.HexColor("#eeeeee"), colors.HexColor("#cccccc")]
    table = Table(rows, colWidths=[200, 120])
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                *(
                    ("BACKGROUND", (column, row), (column, row), colour)
                    for row in range(len(rows))
                    for column in range(2)
                    for colour in (stripes[row % 2],)
                ),
            ],
        ),
    )
    SimpleDocTemplate(str(path)).build([table])
    assert _figures(path) == []
    doc = load_pdf(path, PdfConfig(), lang="en")
    assert "timeout" in " ".join(b.text for b in doc.blocks)


@pytest.mark.parametrize("paths", [4000, 5000])
def test_dense_vector_page_is_read_quickly(tmp_path: Path, paths: int) -> None:
    """Thousands of small paths on a page are grouped in bounded time."""
    path = build_dense_paths_pdf(tmp_path / "dense.pdf", paths=paths)
    start = time.perf_counter()
    with open_pdf(path) as pdf:
        page = read_page(pdf, 0)
    assert time.perf_counter() - start < 5.0
    assert not [i for i in page.images if i.ref.vector]
    assert page.chars


def test_figure_text_leaves_without_images(tmp_path: Path) -> None:
    """Without images, a figure is not rendered but its text still leaves."""
    path = build_reportlab_pdf(FIGURES_EN, tmp_path / "a.pdf")
    doc = load_pdf(path, PdfConfig(), lang="en", images=False)
    assert not [b for b in doc.blocks if b.kind is BlockKind.IMAGE]
    assert not [b.text for b in doc.blocks if "pipeline in" in b.text]
    assert doc.normalization["figure_text"] == len("pipeline in pipeline out")


@pytest.mark.parametrize("forms", [False, True])
def test_loader_renders_a_figure_again(tmp_path: Path, *, forms: bool) -> None:
    """A figure is rendered again from its box, as at its extraction."""
    path = build_reportlab_pdf(
        FIGURES_EN,
        tmp_path / "a.pdf",
        figure_forms=forms,
    )
    doc = load_pdf(path, PdfConfig(), lang="en")
    (ref,) = [b.image for b in doc.blocks if b.image and b.image.vector]
    with PixelLoader(path) as loader:
        render = loader.load(ref)
    assert render is not None
    assert (render.width, render.height) == (ref.width, ref.height)
    assert ref.format == "vector"
    assert ref.width is not None
    assert ref.height is not None
    # The shapes span 280 by 130 points: at 144 DPI, within rounding.
    assert (ref.width, ref.height) == pytest.approx((560, 260), abs=2)
    assert max(ref.width, ref.height) <= RENDER_SIDE_MAX


@pytest.mark.parametrize("form", [False, True])
def test_figure_hides_its_text_and_the_images_of_its_form(
    tmp_path: Path,
    *,
    form: bool,
) -> None:
    """A form figure gives no image of its own, but its rank still counts."""
    path = build_figure_pdf(tmp_path / "figure.pdf", form=form)
    with open_pdf(path) as pdf:
        page = read_page(pdf, 0)
    text = "".join(char.text for char in page.chars)
    assert FIGURE_LABEL.replace(" ", "") not in text.replace(" ", "")
    assert "figure." in text
    assert page.figure_chars == len(FIGURE_LABEL)
    figures = [i.ref for i in page.images if i.ref.vector]
    rasters = [i.ref for i in page.images if not i.ref.vector]
    assert [ref.readable for ref in figures] == [True]
    indexes = [ref.source.index for ref in rasters if ref.source is not None]
    assert indexes == ([1] if form else [0, 1])


@pytest.mark.parametrize("form", [False, True])
def test_figure_over_a_refused_image_is_not_rendered(
    tmp_path: Path,
    *,
    form: bool,
) -> None:
    """An image that the checks refuse is never rendered in a figure."""
    path = build_figure_pdf(
        tmp_path / "figure.pdf",
        form=form,
        inner=(10_000, 5_000),
    )
    with open_pdf(path) as pdf:
        page = read_page(pdf, 0)
    figures = [i.ref for i in page.images if i.ref.vector]
    rasters = [i.ref for i in page.images if not i.ref.vector]
    assert [(ref.problem, ref.source) for ref in figures] == [
        (TOO_LARGE, None),
    ]
    assert [ref.problem for ref in rasters] == (
        [None] if form else [TOO_LARGE, None]
    )


@pytest.mark.parametrize("form", [False, True])
def test_figure_on_a_rotated_page_is_found_not_rendered(
    tmp_path: Path,
    *,
    form: bool,
) -> None:
    """On a page that /Rotate turns, a figure loses its text, not its place."""
    path = build_figure_pdf(tmp_path / "figure.pdf", form=form, rotate=90)
    with open_pdf(path) as pdf:
        page = read_page(pdf, 0)
    (figure,) = [i for i in page.images if i.ref.vector]
    assert (figure.ref.problem, figure.ref.readable) == (NOT_RENDERED, False)
    assert figure.ref.name == "page 1 figure 1"
    assert figure.ref.shown == pytest.approx((220, 60))
    assert page.figure_chars == len(FIGURE_LABEL)
    # Its text runs left to right on the page, which is not turned.
    assert figure.x1 - figure.x0 == pytest.approx(220)


def _line(
    text: str,
    start: float,
    baseline: float,
    *,
    vertical: bool = False,
) -> list[Glyph]:
    """Set a text of 10 points on a baseline, 5 points a character."""
    glyphs = []
    for k in range(len(text)):
        low, high = start + 5 * k, start + 5 * k + 5
        box = (
            (baseline - 7, low, baseline + 2, high)
            if vertical
            else (low, baseline - 2, high, baseline + 7)
        )
        glyphs.append(Glyph(box, baseline, 10.0, vertical=vertical))
    return glyphs


@pytest.mark.parametrize(
    ("glyphs", "labels"),
    [
        (_line("pipeline in", 20, 50) + _line("pipeline out", 200, 50), True),
        (
            _line("Request flow", 20, 80)
            + _line("axis", 10, 10, vertical=True),
            True,
        ),
        (_line("A paragraph runs across the whole box.", 10, 50), False),
        (_line("x" * 14, 10, 10, vertical=True), False),
        (
            _line("timeout", 10, 80)
            + _line("30", 150, 80)
            + _line("retries", 10, 60)
            + _line("3", 150, 60),
            False,
        ),
        (
            [g for row in range(21) for g in _line("ten chars!", 10, 5 * row)],
            False,
        ),
    ],
    ids=[
        "two-labels",
        "vertical",
        "paragraph",
        "long-vertical",
        "table",
        "many",
    ],
)
def test_labels_are_told_from_text(
    glyphs: list[Glyph],
    *,
    labels: bool,
) -> None:
    """Short scattered lines are labels; paragraphs and tables are not."""
    assert len(glyphs) <= CLUSTER_TEXT_MAX or not labels
    assert are_labels(glyphs, (0, 0, 300, 100)) is labels


def _draw_figure(canvas: Canvas, left: float, bottom: float) -> None:
    """Draw a small diagram: two boxes of two colours, a curve, two labels."""
    for colour, x in ((colors.HexColor("#4477aa"), 0), (colors.red, 80)):
        canvas.setFillColor(colour)
        canvas.rect(left + x, bottom, 50, 40, stroke=0, fill=1)
    canvas.setFillColor(colors.HexColor("#228833"))
    canvas.ellipse(left + 45, bottom - 20, left + 85, bottom + 4, fill=1)
    canvas.setFillColor(colors.black)
    canvas.setFont("Helvetica", 9)
    canvas.drawString(left + 5, bottom + 16, "Reader")
    canvas.drawString(left + 85, bottom + 16, "Writer")


def _page_text(path: Path) -> str:
    """Join the kept characters of the first page of a PDF."""
    with open_pdf(path) as pdf:
        return "".join(char.text for char in read_page(pdf, 0).chars)


def _box(
    canvas: Canvas,
    box: tuple[float, float, float, float],
    *,
    frame: Color,
    title: str,
    body: str,
) -> None:
    """Draw a note box, as Sphinx does: a rounded frame, a title bar, text."""
    left, bottom, width, height = box
    canvas.setFillColor(frame)
    canvas.roundRect(left, bottom, width, height, 3, stroke=0, fill=1)
    canvas.setFillColor(colors.HexColor("#f7f7f7"))
    canvas.roundRect(
        left + 0.5,
        bottom + 0.5,
        width - 1,
        height - 1,
        3,
        fill=1,
    )
    canvas.setFillColor(colors.HexColor("#d0defa"))
    canvas.roundRect(
        left + 0.5,
        bottom + height - 20,
        width - 1,
        19.5,
        3,
        fill=1,
    )
    canvas.setFillColor(colors.black)
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(left + 8, bottom + height - 14, title)
    canvas.setFont("Helvetica", 10)
    canvas.drawString(left + 8, bottom + height - 36, body)


def test_a_paragraph_in_a_rounded_box_keeps_its_text(tmp_path: Path) -> None:
    """A note box with a line across it is no drawing, but body text."""
    path = tmp_path / "note.pdf"
    canvas = Canvas(str(path), pagesize=A4)
    body = "Restart the service after an upgrade, then check its logs."
    _box(
        canvas,
        (72, 600, 451, 46),
        frame=colors.HexColor("#86989b"),
        title="Note",
        body=body,
    )
    canvas.save()
    assert _figures(path) == []
    assert body.replace(" ", "") in _page_text(path).replace(" ", "")


def test_a_table_in_a_box_keeps_its_text(tmp_path: Path) -> None:
    """A striped table in a rounded box reads as a table, not as labels."""
    path = tmp_path / "boxed.pdf"
    canvas = Canvas(str(path), pagesize=A4)
    canvas.setFillColor(colors.HexColor("#86989b"))
    canvas.roundRect(72, 560, 451, 100, 3, stroke=0, fill=1)
    canvas.setFillColor(colors.white)
    canvas.roundRect(72.5, 560.5, 450, 99, 3, stroke=0, fill=1)
    rows = [("Key", "Value"), ("alpha", "one"), ("beta", "two")]
    shades = ["#aabbee", "#eeeeee", "#cccccc"]
    for row, ((key, value), shade) in enumerate(
        zip(rows, shades, strict=True),
    ):
        bottom = 630 - 14 * row
        canvas.setFillColor(colors.HexColor(shade))
        canvas.rect(90, bottom, 60, 14, stroke=0, fill=1)
        canvas.rect(150, bottom, 70, 14, stroke=0, fill=1)
        canvas.setFillColor(colors.black)
        canvas.drawString(94, bottom + 4, key)
        canvas.drawString(154, bottom + 4, value)
    canvas.save()
    assert _figures(path) == []
    assert "alpha" in _page_text(path)


def test_tables_side_by_side_keep_their_text(tmp_path: Path) -> None:
    """Two striped tables of two widths, one under the other, are tables."""
    path = tmp_path / "tables.pdf"
    stripes = [colors.HexColor("#eeeeee"), colors.HexColor("#cccccc")]
    flowables = []
    for widths in ([120, 80], [90, 90, 60]):
        rows = [[f"head{column}" for column in range(len(widths))]]
        rows += [
            [f"r{row}c{column}" for column in range(len(widths))]
            for row in range(4)
        ]
        table = Table(rows, colWidths=widths)
        table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (column, row),
                        (column, row),
                        stripes[row % 2],
                    )
                    for row in range(len(rows))
                    for column in range(len(widths))
                ],
            ),
        )
        flowables += [table, Spacer(1, 1.4)]
    SimpleDocTemplate(str(path)).build(flowables)
    assert _figures(path) == []
    assert "r3c2" in _page_text(path)


def test_a_column_rule_does_not_stretch_a_figure(tmp_path: Path) -> None:
    """A long rule beside a figure joins no cluster; the column keeps text."""
    path = tmp_path / "columns.pdf"
    canvas = Canvas(str(path), pagesize=A4)
    canvas.setLineWidth(0.4)
    canvas.line(297, 72, 297, 770)
    canvas.setFont("Times-Roman", 10)
    lines = [
        (f"Left column line {n} with words.", 760 - 16 * n)
        for n in range(40)
        if not 520 <= 760 - 16 * n <= 680  # Room for the figure.
    ]
    for line, baseline in lines:
        canvas.drawString(72, baseline, line)
    # The figure ends 3 points left of the rule.
    _draw_figure(canvas, 164, 600)
    canvas.save()
    assert [form for _, form in _figures(path)] == [False]
    text = _page_text(path).replace(" ", "")
    assert "Reader" not in text
    assert all(line.replace(" ", "") in text for line, _ in lines)


def test_a_page_background_does_not_hide_a_figure(tmp_path: Path) -> None:
    """A filled page joins no cluster: the drawing on it is still a figure."""
    path = tmp_path / "background.pdf"
    canvas = Canvas(str(path), pagesize=A4)
    canvas.setFillColor(colors.HexColor("#fffdee"))
    canvas.rect(0, 0, *A4, stroke=0, fill=1)
    canvas.setFillColor(colors.black)
    canvas.drawString(72, 770, "The text above the figure.")
    _draw_figure(canvas, 160, 600)
    canvas.save()
    assert [form for _, form in _figures(path)] == [False]
    assert "Reader" not in _page_text(path)


def test_figures_beyond_the_page_limit_are_not_rendered(
    tmp_path: Path,
) -> None:
    """Past the first figures of a page, figures are named, not rendered."""
    count = MAX_PAGE_FIGURES + 6
    path = build_many_figures_pdf(
        tmp_path / "many.pdf",
        figures=count,
        forms=True,
    )
    with open_pdf(path) as pdf:
        page = read_page(pdf, 0)
    refs = [image.ref for image in page.images if image.ref.vector]
    readable = [ref.readable for ref in refs]
    assert readable == [True] * MAX_PAGE_FIGURES + [False] * 6
    assert {ref.problem for ref in refs[MAX_PAGE_FIGURES:]} == {PAGE_LIMIT}
    assert [ref.name for ref in refs[-2:]] == [
        f"page 1 figure {count - 1}",
        f"page 1 figure {count}",
    ]
    assert page.unrendered == 6
    doc = load_pdf(path, PdfConfig(), lang="en")
    # One warning only: the figures past the limit are no unread images.
    warning = (
        f"page 1: 6 vector figure(s) not rendered, beyond the first"
        f" {MAX_PAGE_FIGURES}"
    )
    assert doc.warnings == [warning]


def test_images_and_figures_are_named_by_page_label(tmp_path: Path) -> None:
    """An unread image or figure is named after the label of its page."""
    path = build_figure_pdf(
        tmp_path / "labels.pdf",
        form=False,
        inner=(10_000, 5_000),
        rotate=90,
        pages=2,
        roman=True,
    )
    doc = load_pdf(path, PdfConfig(), lang="en")
    names = [b.image.name for b in doc.blocks if b.image is not None]
    assert "page ii figure 1" in names
    assert "page ii image 1" in names
    assert (
        "4 image(s) could not be read: page i figure 1, page i image 1,"
        " page ii figure 1, page ii image 1"
    ) in doc.warnings


@pytest.mark.parametrize(
    ("figures", "forms", "chars"),
    [(2000, True, 0), (900, False, 20_000)],
    ids=["forms", "clusters"],
)
def test_pages_full_of_figures_are_read_quickly(
    tmp_path: Path,
    figures: int,
    *,
    forms: bool,
    chars: int,
) -> None:
    """Many figures and characters on a page are read in bounded time."""
    path = build_many_figures_pdf(
        tmp_path / "many.pdf",
        figures=figures,
        forms=forms,
        chars=chars,
    )
    start = time.perf_counter()
    with open_pdf(path) as pdf:
        page = read_page(pdf, 0)
    assert time.perf_counter() - start < 5.0
    assert sum(image.ref.readable for image in page.images) == MAX_PAGE_FIGURES


def test_unrendered_figures_of_turned_pages_are_no_chrome(
    tmp_path: Path,
) -> None:
    """Figures at one place of turned pages stay apart: none is chrome."""
    path = build_figure_pdf(
        tmp_path / "turned.pdf",
        form=False,
        rotate=90,
        pages=4,
    )
    doc = load_pdf(path, PdfConfig(), lang="en")
    refs = [
        b.image for b in doc.blocks if b.image is not None and b.image.vector
    ]
    assert [ref.name for ref in refs] == [
        f"page {n} figure 1" for n in range(1, 5)
    ]
    assert len({ref.digest for ref in refs}) == 4


@pytest.mark.parametrize("notes", [1, 2])
def test_a_short_note_keeps_its_text(tmp_path: Path, notes: int) -> None:
    """The rounded corners of a note box are no drawing, whatever its text."""
    path = tmp_path / "note.pdf"
    canvas = Canvas(str(path), pagesize=A4)
    for note in range(notes):
        _box(
            canvas,
            (72, 600 - 60 * note, 451, 46),
            frame=colors.HexColor("#86989b"),
            title="Hint",
            body=f"Restart service {note}.",
        )
    canvas.save()
    assert _figures(path) == []
    assert "Restartservice0." in _page_text(path).replace(" ", "")


def test_rounded_nodes_around_a_circle_are_a_figure(tmp_path: Path) -> None:
    """Rounded nodes of one colour, with a circle, still make a drawing."""
    path = tmp_path / "nodes.pdf"
    canvas = Canvas(str(path), pagesize=A4)
    canvas.setFillColor(colors.HexColor("#ccccff"))
    for left in (150, 330):
        canvas.roundRect(left, 600, 72, 30, 4, fill=1)
    canvas.circle(276, 560, 18, fill=1)
    canvas.line(222, 615, 330, 615)
    canvas.line(186, 600, 262, 572)
    canvas.line(366, 600, 290, 572)
    canvas.setFillColor(colors.black)
    canvas.drawString(160, 612, "Client")
    canvas.drawString(340, 612, "Server")
    canvas.save()
    assert [form for _, form in _figures(path)] == [False]
    assert "Client" not in _page_text(path)


def test_a_note_with_a_code_block_keeps_its_text(tmp_path: Path) -> None:
    """A rounded code block in a note box is one more band of the box."""
    path = tmp_path / "note.pdf"
    canvas = Canvas(str(path), pagesize=A4)
    _box(
        canvas,
        (72, 560, 451, 86),
        frame=colors.HexColor("#86989b"),
        title="Note",
        body="Run this first.",
    )
    canvas.setFillColor(colors.HexColor("#202020"))
    canvas.roundRect(76, 566, 443, 30, 3, stroke=0, fill=1)
    canvas.setFillColor(colors.HexColor("#f2f2f2"))
    canvas.roundRect(76.5, 566.5, 442, 29, 3, stroke=0, fill=1)
    canvas.setFillColor(colors.black)
    canvas.setFont("Courier", 9)
    canvas.drawString(82, 578, "make install")
    canvas.save()
    assert _figures(path) == []
    assert "makeinstall" in _page_text(path).replace(" ", "")


def test_bands_of_one_box_are_no_drawing() -> None:
    """A frame, its background and a block in it span one width: a box."""
    paths = [
        PathFacts((0, 0, 468, 78), bezier=False, fill=(134, 152, 155)),
        PathFacts((0.5, 0.5, 467.5, 77.5), bezier=False, fill=(247, 247, 247)),
        PathFacts((4, 6, 464, 36), bezier=False, fill=BLUE),
    ]
    assert is_drawing(paths) is False
    assert is_drawing(
        [*paths, PathFacts((10, 40, 90, 70), bezier=False, fill=RED)],
    )


def test_pages_shown_two_up_keep_their_text(tmp_path: Path) -> None:
    """A page of text shown in a form, as pdfpages does, is no figure."""
    path = tmp_path / "two-up.pdf"
    canvas = Canvas(str(path), pagesize=A4)
    lines = [
        f"Sheet line {n} with ordinary words that run along."
        for n in range(30)
    ]
    canvas.beginForm("sheet")
    canvas.rect(36, 36, 523, 770)
    canvas.setFont("Times-Roman", 11)
    for n, line in enumerate(lines):
        canvas.drawString(72, 760 - 22 * n, line)
    canvas.endForm()
    for left in (20, 300):
        canvas.saveState()
        canvas.translate(left, 230)
        canvas.scale(0.45, 0.45)
        canvas.doForm("sheet")
        canvas.restoreState()
    canvas.save()
    assert _figures(path) == []
    assert "Sheetline29withordinarywordsthatrunalong." in _page_text(
        path,
    ).replace(" ", "")


SHORT_LINES = ("Restart it now.", "Then log in.")


def _short_lines(canvas: Canvas, left: float, top: float) -> None:
    """Write the two short lines of a text box, from its first baseline."""
    canvas.setFillColor(colors.black)
    canvas.setFont("Helvetica", 10)
    for k, line in enumerate(SHORT_LINES):
        canvas.drawString(left, top - 14 * k, line)


def _info_icon(
    canvas: Canvas,
    left: float,
    bottom: float,
    colour: Color,
) -> None:
    """Draw an info icon, as an HTML page does: a circle and an i, one path."""
    canvas.setFillColor(colour)
    path = canvas.beginPath()
    path.circle(left + 6, bottom + 6, 6)
    path.rect(left + 5.4, bottom + 2.4, 1.2, 4.8)
    path.circle(left + 6, bottom + 9, 0.8)
    canvas.drawPath(path, stroke=0, fill=1, fillMode=1)


def _icon_box(canvas: Canvas) -> None:
    """A titled tcolorbox with a TikZ icon, a circle, in its title."""
    canvas.setFillColor(colors.HexColor("#006600"))
    canvas.roundRect(72, 600, 451, 62, 3, stroke=0, fill=1)
    canvas.setFillColor(colors.HexColor("#f2fff2"))
    canvas.roundRect(73.4, 601.4, 448.2, 42, 2, stroke=0, fill=1)
    canvas.setFillColor(colors.white)
    canvas.circle(84, 652, 4, stroke=0, fill=1)
    canvas.drawString(92, 648, "Tip")
    _short_lines(canvas, 80, 628)


def _title_tab_box(canvas: Canvas) -> None:
    """A tcolorbox whose boxed title is a narrow tab on its frame."""
    navy = colors.HexColor("#000099")
    canvas.setFillColor(navy)
    canvas.roundRect(72, 600, 451, 46, 3, stroke=0, fill=1)
    canvas.setFillColor(colors.HexColor("#f2f2ff"))
    canvas.roundRect(73.4, 601.4, 448.2, 43.2, 2, stroke=0, fill=1)
    canvas.setFillColor(navy)
    canvas.roundRect(84, 640, 34, 14, 2, stroke=0, fill=1)
    canvas.setFillColor(colors.white)
    canvas.drawString(88, 644, "Note")
    _short_lines(canvas, 80, 626)


def _bar_box(canvas: Canvas) -> None:
    """An mdframed box: a thick left line, a background, a title band."""
    canvas.setFillColor(colors.HexColor("#4d4dff"))
    canvas.rect(72, 596, 5, 46, stroke=0, fill=1)
    canvas.setFillColor(colors.HexColor("#f2f2ff"))
    canvas.rect(77, 596, 446, 46, stroke=0, fill=1)
    canvas.setFillColor(colors.HexColor("#d9d9ff"))
    canvas.rect(77, 625, 446, 17, stroke=0, fill=1)
    canvas.setFillColor(colors.black)
    canvas.drawString(82, 630, "Hint")
    _short_lines(canvas, 82, 613)


def _highlight_box(canvas: Canvas) -> None:
    """A tcolorbox whose first line has a word on a colorbox."""
    canvas.setFillColor(colors.HexColor("#b3b3b3"))
    canvas.roundRect(72, 600, 451, 41, 3, stroke=0, fill=1)
    canvas.setFillColor(colors.HexColor("#f9f9f9"))
    canvas.roundRect(73.4, 601.4, 448.2, 38.2, 2, stroke=0, fill=1)
    canvas.setFillColor(colors.HexColor("#ffe6e6"))
    canvas.rect(100, 620, 27, 12, stroke=0, fill=1)
    _short_lines(canvas, 80, 623)


def _note_table(canvas: Canvas) -> None:
    """A table of one row, as LibreOffice writes notes: a label, then text."""
    canvas.setFillColor(colors.HexColor("#3465a4"))
    canvas.rect(56.7, 723.8, 198.4, 47.6, stroke=0, fill=1)
    canvas.setFillColor(colors.HexColor("#eef3fb"))
    canvas.rect(255.1, 723.8, 198.4, 47.6, stroke=0, fill=1)
    canvas.setStrokeColor(colors.HexColor("#eef3fb"))
    for x in (56.7, 255.1, 453.5):
        canvas.line(x, 723.3, x, 771.9)
    for y in (723.5, 771.1):
        canvas.line(56.2, y, 454.0, y)
    canvas.setFillColor(colors.white)
    canvas.drawString(62, 757, "Note")
    _short_lines(canvas, 260, 757)


def _material(canvas: Canvas) -> None:
    """An MkDocs Material admonition: a ring, a title band, an icon."""
    blue = colors.HexColor("#448aff")
    ring = canvas.beginPath()
    ring.roundRect(72, 594, 451, 58, 3)
    ring.roundRect(72.9, 594.9, 449.2, 56.2, 2.2)
    canvas.setFillColor(blue)
    canvas.drawPath(ring, stroke=0, fill=1, fillMode=1)
    canvas.setFillColor(colors.HexColor("#ecf3ff"))
    canvas.rect(72.9, 630, 449.2, 21.1, stroke=0, fill=1)
    _info_icon(canvas, 80, 635, blue)
    canvas.setFillColor(colors.black)
    canvas.drawString(98, 637, "Note")
    _short_lines(canvas, 82, 616)


def _docusaurus(canvas: Canvas) -> None:
    """A Docusaurus admonition: a tinted box, a left border, an icon."""
    canvas.setFillColor(colors.HexColor("#eef9fd"))
    canvas.roundRect(72, 596, 451, 60, 4, stroke=0, fill=1)
    canvas.setFillColor(colors.HexColor("#4cb3d4"))
    canvas.rect(72, 596, 3.75, 60, stroke=0, fill=1)
    _info_icon(canvas, 84, 638, colors.HexColor("#193c47"))
    canvas.setFillColor(colors.black)
    canvas.drawString(100, 641, "INFO")
    _short_lines(canvas, 84, 620)


def _on_a_coloured_page(canvas: Canvas) -> None:
    """The box with an icon, on a page that pagecolor fills."""
    canvas.setFillColor(colors.HexColor("#fffdee"))
    canvas.rect(0, 0, *A4, stroke=0, fill=1)
    _icon_box(canvas)


@pytest.mark.parametrize(
    "draw",
    [
        _icon_box,
        _title_tab_box,
        _bar_box,
        _highlight_box,
        _note_table,
        _material,
        _docusaurus,
        _on_a_coloured_page,
    ],
    ids=[
        "icon",
        "title-tab",
        "bar",
        "colorbox",
        "note-table",
        "material",
        "docusaurus",
        "pagecolor",
    ],
)
def test_short_text_boxes_keep_their_text(
    tmp_path: Path,
    draw: Callable[[Canvas], None],
) -> None:
    """A box that holds its short lines in one large shape is no figure."""
    path = tmp_path / "box.pdf"
    canvas = Canvas(str(path), pagesize=A4)
    canvas.setFont("Helvetica", 10)
    canvas.drawString(72, 770, "The text above the box.")
    draw(canvas)
    canvas.save()
    assert _figures(path) == []
    text = _page_text(path).replace(" ", "")
    assert all(line.replace(" ", "") in text for line in SHORT_LINES)


def test_stacked_boxes_keep_their_text(tmp_path: Path) -> None:
    """Two boxes of two widths, a frame around a background each, are text."""
    path = tmp_path / "stacked.pdf"
    canvas = Canvas(str(path), pagesize=A4)
    boxes = (
        (615, 343.7, "#8080ff", "#f2f2ff", "First short box."),
        (582, 206.2, "#ff8080", "#fff2f2", "Second short."),
    )
    for bottom, width, frame, background, line in boxes:
        canvas.setFillColor(colors.HexColor(frame))
        canvas.roundRect(134, bottom, width, 27, 3, stroke=0, fill=1)
        canvas.setFillColor(colors.HexColor(background))
        canvas.roundRect(135.4, bottom + 1.4, width - 2.8, 24.2, 2, fill=1)
        canvas.setFillColor(colors.black)
        canvas.drawString(142, bottom + 10, line)
    canvas.save()
    assert _figures(path) == []
    assert "Secondshort." in _page_text(path).replace(" ", "")
