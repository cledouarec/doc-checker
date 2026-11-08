# Copyright 2026 Christophe Le Douarec
"""End to end comparison of documentations with figures."""

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

import pytest

from doc_checker.config import Config
from doc_checker.model.findings import Severity
from doc_checker.pipeline import run_compare
from tests.builders import (
    FIGURES_EN,
    Defect,
    DocSpec,
    build_docusaurus_like,
    build_mkdocs_like,
    build_pandoc_like,
    build_sphinx_like,
    figure_defect,
)
from tests.pdf_builders import build_fpdf2_pdf, build_reportlab_pdf

if TYPE_CHECKING:
    from pathlib import Path

    from doc_checker.model.findings import Finding

C, MA, MI, IN = (
    Severity.CRITICAL,
    Severity.MAJOR,
    Severity.MINOR,
    Severity.INFO,
)

type Builder = Callable[[DocSpec, Path], Path]

# Each entry builds the documentation with figures twice, by two generators
# or two layouts that write the same images.
FIGURE_PAIRS: dict[str, tuple[Builder, Builder]] = {
    "sphinx/mkdocs": (build_sphinx_like, build_mkdocs_like),
    "sphinx-chapters/mkdocs": (
        partial(build_sphinx_like, split="chapters"),
        build_mkdocs_like,
    ),
    "mkdocs/pandoc-embed": (
        build_mkdocs_like,
        partial(build_pandoc_like, embed=True),
    ),
    "sphinx-numbered/pandoc": (
        partial(build_sphinx_like, numbered=True),
        build_pandoc_like,
    ),
    "docusaurus-base/sphinx": (
        partial(build_docusaurus_like, base="/example/"),
        build_sphinx_like,
    ),
    "reportlab/fpdf2": (build_reportlab_pdf, build_fpdf2_pdf),
    "reportlab/reportlab-forms": (
        build_reportlab_pdf,
        partial(build_reportlab_pdf, figure_forms=True),
    ),
    "reportlab/reportlab-logo": (
        build_reportlab_pdf,
        partial(build_reportlab_pdf, logo=True),
    ),
    "reportlab/reportlab-reencoded": (
        build_reportlab_pdf,
        partial(build_reportlab_pdf, reencode=True),
    ),
}


@pytest.mark.parametrize("order", [(0, 1), (1, 0)], ids=["forward", "swapped"])
@pytest.mark.parametrize("pair", list(FIGURE_PAIRS))
def test_same_figures_produce_no_finding(
    tmp_path: Path,
    pair: str,
    order: tuple[int, int],
) -> None:
    """The same figures through two generators have no finding, both ways."""
    builders = FIGURE_PAIRS[pair]
    docs = (
        builders[0](FIGURES_EN, tmp_path / "first"),
        builders[1](FIGURES_EN, tmp_path / "second"),
    )
    result = run_compare(docs[order[0]], docs[order[1]], Config())
    stats = result.stats
    assert stats.matched > 0
    assert (stats.ref_images, stats.matched_images) == (4, 4)
    assert [
        (f.rule, f.severity) for f in result.findings if f.severity >= MI
    ] == []


# Defects planted in one figure of the new documentation, each with the one
# finding of at least minor severity, or of its rule, that it must give.
IMAGE_DEFECTS: list[tuple[str, Defect, str, Severity]] = [
    ("setup", "jpeg10", "image.degraded", MA),
    ("setup", "x0.4", "image.degraded", MA),
    ("dashboard", "blur", "image.degraded", MA),
    ("setup", "nearest", "image.degraded", MA),
    ("quota", "gray", "image.degraded", MA),
    ("setup", "jpeg95", "image.reencoded", IN),
    ("pipeline", "raster", "image.rasterized", MI),
    ("setup", "other", "image.replaced", MA),
    ("pipeline", "other", "image.replaced", MA),
    ("quota", "dropped", "image.missing", C),
    ("dashboard", "noalt", "image.alt_lost", MI),
]
PDF_DEFECTS = [case for case in IMAGE_DEFECTS if case[1] != "noalt"]


def _relevant(
    findings: list[Finding],
    rule: str,
) -> list[tuple[str, Severity]]:
    """Keep the findings of at least minor severity, and those of a rule."""
    return [
        (f.rule, f.severity)
        for f in findings
        if f.severity >= MI or f.rule == rule
    ]


@pytest.mark.parametrize(("name", "defect", "rule", "severity"), IMAGE_DEFECTS)
def test_image_defect_in_html_is_reported(
    tmp_path: Path,
    name: str,
    defect: Defect,
    rule: str,
    severity: Severity,
) -> None:
    """Each image defect of a site is reported once, with its severity."""
    ref = build_sphinx_like(FIGURES_EN, tmp_path / "ref")
    new = build_mkdocs_like(
        figure_defect(FIGURES_EN, name, defect),
        tmp_path / "new",
    )
    findings = run_compare(ref, new, Config()).findings
    assert _relevant(findings, rule) == [(rule, severity)]


@pytest.mark.parametrize(("name", "defect", "rule", "severity"), PDF_DEFECTS)
def test_image_defect_in_pdf_is_reported(
    tmp_path: Path,
    name: str,
    defect: Defect,
    rule: str,
    severity: Severity,
) -> None:
    """Each image defect of a PDF is reported once, with its severity."""
    ref = build_reportlab_pdf(FIGURES_EN, tmp_path / "ref.pdf")
    new = build_fpdf2_pdf(
        figure_defect(FIGURES_EN, name, defect),
        tmp_path / "new.pdf",
    )
    findings = run_compare(ref, new, Config()).findings
    assert _relevant(findings, rule) == [(rule, severity)]
