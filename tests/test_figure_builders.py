# Copyright 2026 Christophe Le Douarec
"""Tests for the figures written by the test builders."""

from typing import TYPE_CHECKING

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

from tests.builders import (
    FIGURES_EN,
    DocSpec,
    Figure,
    Section,
    build_docusaurus_like,
    build_mkdocs_like,
    build_pandoc_like,
    build_sphinx_like,
    figure_defect,
)
from tests.pdf_builders import build_fpdf2_pdf, build_reportlab_pdf

if TYPE_CHECKING:
    from pathlib import Path


def test_html_builders_write_their_figures(tmp_path: Path) -> None:
    """Each HTML builder writes its figures in its own way."""
    sphinx = build_sphinx_like(FIGURES_EN, tmp_path / "s", numbered=True)
    page = (sphinx / "index.html").read_text(encoding="utf-8")
    assert '<img alt="Installer steps" src="_images/setup.png" />' in page
    assert '<span class="caption-number">Fig. 1 </span>' in page
    assert (sphinx / "_images" / "pipeline.svg").is_file()
    mkdocs = build_mkdocs_like(FIGURES_EN, tmp_path / "m")
    reference = (mkdocs / "reference" / "index.html").read_text(
        encoding="utf-8",
    )
    assert "<figure><svg " in reference
    embedded = build_pandoc_like(FIGURES_EN, tmp_path / "p", embed=True)
    single = (embedded / "index.html").read_text(encoding="utf-8")
    assert 'src="data:image/png;base64,' in single
    assert not (embedded / "img").exists()
    docusaurus = build_docusaurus_like(FIGURES_EN, tmp_path / "d", base="/x/")
    names = sorted(p.name for p in (docusaurus / "assets/images").iterdir())
    assert [name.split("-")[0] for name in names] == [
        "dashboard",
        "pipeline",
        "quota",
        "setup",
    ]


def test_defects_change_the_written_figures(tmp_path: Path) -> None:
    """A defect changes the file, the alt or the presence of an image."""
    spec = figure_defect(FIGURES_EN, "setup", "jpeg10")
    spec = figure_defect(spec, "dashboard", "noalt")
    spec = figure_defect(spec, "quota", "dropped")
    site = build_sphinx_like(spec, tmp_path / "s")
    page = (site / "index.html").read_text(encoding="utf-8")
    assert 'src="_images/setup.jpg"' in page
    assert '<img src="_images/dashboard.jpg" />' in page
    assert "_images/quota" not in page
    assert "Requests per client" in page


def _objects(path: Path) -> list[dict[int, int]]:
    """Count the top level objects of each page by type."""
    counts = []
    with pdfium.PdfDocument(path) as pdf:
        for page in pdf:
            found: dict[int, int] = {}
            for obj in page.get_objects(max_depth=1):
                found[obj.type] = found.get(obj.type, 0) + 1
            counts.append(found)
    return counts


def test_pdf_builders_draw_images_and_figures(tmp_path: Path) -> None:
    """Raster figures are images, SVG figures are paths or forms."""
    image, path, form = (
        pdfium_c.FPDF_PAGEOBJ_IMAGE,
        pdfium_c.FPDF_PAGEOBJ_PATH,
        pdfium_c.FPDF_PAGEOBJ_FORM,
    )
    paths = _objects(build_reportlab_pdf(FIGURES_EN, tmp_path / "a.pdf"))
    assert sum(page.get(image, 0) for page in paths) == 3
    assert sum(page.get(path, 0) for page in paths) >= 5
    forms = _objects(
        build_reportlab_pdf(FIGURES_EN, tmp_path / "b.pdf", figure_forms=True),
    )
    assert sum(page.get(form, 0) for page in forms) == 1
    fpdf2 = _objects(build_fpdf2_pdf(FIGURES_EN, tmp_path / "c.pdf"))
    assert sum(page.get(image, 0) for page in fpdf2) == 3
    assert sum(page.get(path, 0) for page in fpdf2) >= 5


def _box(page: pdfium.PdfPage) -> tuple[float, float, float, float]:
    """Return the box that holds the paths of a page."""
    bounds = [
        obj.get_bounds()
        for obj in page.get_objects(max_depth=1)
        if obj.type == pdfium_c.FPDF_PAGEOBJ_PATH
    ]
    return (
        min(b[0] for b in bounds),
        min(b[1] for b in bounds),
        max(b[2] for b in bounds),
        max(b[3] for b in bounds),
    )


def _outside(path: Path, label: str) -> int:
    """Count the characters of a label outside the box of the paths."""
    with pdfium.PdfDocument(path) as pdf:
        page = pdf[0]
        left, bottom, right, top = _box(page)
        text = page.get_textpage()
        content = text.get_text_range()
        found = text.search(label).get_next()
        assert found is not None
        outside = 0
        for index in range(found[0], found[0] + found[1]):
            if content[index].strip():
                x0, y0, x1, y1 = text.get_charbox(index)
                x, y = (x0 + x1) / 2, (y0 + y1) / 2
                outside += not (left <= x <= right and bottom <= y <= top)
        return outside


def test_flipped_labels_stay_within_the_shapes(tmp_path: Path) -> None:
    """Every character of the flipped figure's labels is within its paths."""
    figure = Figure("pipeline", "Caption", "Alt", "svg", "other")
    section = Section("Cmd", "cmd", 1, ("Text.",), figures=(figure,))
    spec = DocSpec("T", "en", (section,))
    reportlab = build_reportlab_pdf(spec, tmp_path / "a.pdf")
    fpdf2 = build_fpdf2_pdf(spec, tmp_path / "b.pdf")
    for path in (reportlab, fpdf2):
        assert _outside(path, "pipeline-other in") == 0
        assert _outside(path, "pipeline-other out") == 0
