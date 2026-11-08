# Copyright 2026 Christophe Le Douarec
"""End to end comparison of PDF documents."""

from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

import pytest
from reportlab.lib.units import mm

from doc_checker.config import Config
from doc_checker.model.findings import Category, Severity
from doc_checker.pipeline import run_compare
from tests.builders import (
    FIGURES_EN,
    SAMPLE_EN,
    SAMPLE_FR,
    DocSpec,
    demote_heading,
    drop_section,
    move_section,
    number_titles,
    set_level,
)
from tests.pdf_builders import (
    ProseStyle,
    build_bullets_pdf,
    build_contents_pdf,
    build_fpdf2_pdf,
    build_lines_pdf,
    build_listing_pdf,
    build_numbered_pdf,
    build_prose_pdf,
    build_references_pdf,
    build_reportlab_pdf,
    build_table_pdf,
    build_turned_pdf,
    filler_sections,
    swap_pages,
)
from tests.test_e2e_compare import DEFECTS, Transform

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from doc_checker.compare.engine import CompareResult

C, MI = Severity.CRITICAL, Severity.MINOR
OPTIONS = {
    "plain": {},
    "toc": {"toc": True},
    "artifacts": {"toc": True, "artifacts": True},
    "no-outline": {"outline": False},
}


def _compare(
    tmp_path: Path,
    ref_spec: DocSpec,
    new_spec: DocSpec,
    **ref_options: bool,
) -> CompareResult:
    """Compare a reportlab PDF of a documentation with an fpdf2 PDF."""
    ref = build_reportlab_pdf(ref_spec, tmp_path / "ref.pdf", **ref_options)
    new = build_fpdf2_pdf(new_spec, tmp_path / "new.pdf")
    return run_compare(ref, new, Config())


@pytest.mark.parametrize("spec", [SAMPLE_EN, SAMPLE_FR], ids=["en", "fr"])
@pytest.mark.parametrize("options", list(OPTIONS.values()), ids=list(OPTIONS))
def test_generator_only_differences_produce_no_finding(
    tmp_path: Path,
    spec: DocSpec,
    options: dict[str, bool],
) -> None:
    """The same content made by two PDF generators has no finding."""
    findings = _compare(tmp_path, spec, spec, **options).findings
    assert [f.rule for f in findings if f.severity >= MI] == []


@pytest.mark.parametrize("spec", [SAMPLE_EN, SAMPLE_FR], ids=["en", "fr"])
def test_fixtures_exercise_the_layout_differences(
    tmp_path: Path,
    spec: DocSpec,
) -> None:
    """The two PDFs differ by chrome, contents, quotes, hyphens and breaks."""
    result = _compare(tmp_path, spec, spec, toc=True)
    ref, new = result.ref.normalization, result.new.normalization
    assert (
        ref["header_footer"] > 0,
        ref["toc"] > 0,
        ref["quote"] > 0,
    ) == (True, True, True)
    assert (
        new["hyphenation"] > 0,
        new["page_break"] > 0,
        new["header_footer"] > 0,
    ) == (True, True, True)
    assert result.stats.matched == result.stats.ref_blocks


@pytest.mark.parametrize(("spec", "transform", "rule", "severity"), DEFECTS)
def test_injected_defect_is_reported(
    tmp_path: Path,
    spec: DocSpec,
    transform: Transform,
    rule: str,
    severity: Severity,
) -> None:
    """Each injected defect is reported once, with its severity."""
    findings = _compare(tmp_path, spec, transform(spec)).findings
    relevant = [
        (f.rule, f.severity)
        for f in findings
        if f.severity >= MI or f.rule == rule
    ]
    assert relevant == [(rule, severity)]


def test_moved_section_is_info_only(tmp_path: Path) -> None:
    """A section moved elsewhere is only reported as moved."""
    moved = move_section(SAMPLE_EN, "limits", after="installation")
    findings = _compare(tmp_path, SAMPLE_EN, moved).findings
    assert findings
    assert {f.rule for f in findings} == {"block.moved"}


def test_missing_section_without_bookmarks(tmp_path: Path) -> None:
    """A section missing from a PDF without bookmarks is still reported."""
    dropped = drop_section(SAMPLE_EN, "exports")
    findings = _compare(tmp_path, SAMPLE_EN, dropped, outline=False).findings
    assert [(f.rule, f.severity) for f in findings if f.severity >= MI] == [
        ("section.missing", C),
    ]


def test_demoted_heading_is_reported(tmp_path: Path) -> None:
    """A heading of the outline written as a paragraph is a major finding."""
    demoted = demote_heading(SAMPLE_EN, "exports")
    findings = _compare(tmp_path, SAMPLE_EN, demoted).findings
    assert [(f.rule, f.severity) for f in findings if f.severity >= MI] == [
        ("structure.heading_lost", Severity.MAJOR),
    ]


def test_numbered_demoted_heading_is_reported(tmp_path: Path) -> None:
    """A numbered heading written as a paragraph only loses its kind."""
    numbered = number_titles(SAMPLE_EN)
    demoted = demote_heading(numbered, "exports")
    forward = _compare(tmp_path, numbered, demoted).findings
    swapped = run_compare(
        tmp_path / "new.pdf",
        tmp_path / "ref.pdf",
        Config(),
    ).findings
    assert [(f.rule, f.severity) for f in forward if f.severity >= MI] == [
        ("structure.heading_lost", Severity.MAJOR),
    ]
    assert [(f.rule, f.severity) for f in swapped if f.severity >= MI] == []


def test_swapped_pages_are_info_only(tmp_path: Path) -> None:
    """Two swapped pages are only reported as moved blocks."""
    ref = build_reportlab_pdf(SAMPLE_EN, tmp_path / "ref.pdf")
    new = swap_pages(ref, tmp_path / "new.pdf", 4, 5)
    findings = run_compare(ref, new, Config()).findings
    assert findings
    assert {f.rule for f in findings} == {"block.moved"}


type Builder = Callable[[Path], Path]

COMPOUND_WIDE = [
    [
        (
            "The client builds x86-64 binaries and signs them with SHA-256 "
            "digests."
        ),
        "It ships them over Wi-Fi links to IPv4-only gateways.",
    ],
    [
        "Each gateway logs ISO-8601 dates for COVID-19 reports.",
        "The payloads stay Base64-encoded until the end.",
    ],
]
COMPOUND_NARROW = [
    [
        "The client builds x86-",
        "64 binaries and signs them with SHA-",
        "256 digests. It ships them over Wi-",
        "Fi links to IPv4-",
        "only gateways.",
    ],
    [
        "Each gateway logs ISO-",
        "8601 dates for COVID-",
        "19 reports. The payloads stay Base64-",
        "encoded until the end.",
    ],
]
# Sections of paragraphs that wrap over several lines and fill a few pages:
# many paragraphs of four sentences, or a few of nine.
LONG_SECTIONS = filler_sections(4, paragraphs=8, sentences=4)
LONG_PARAGRAPHS = filler_sections(3, paragraphs=3, sentences=9)
# Fourteen chapters fill nine pages or more, two of them numbered i and ii.
ROMAN_SECTIONS = filler_sections(14, paragraphs=8, sentences=4)
# Paragraphs of one line, which end a sentence, among longer ones.
_SHORT = filler_sections(2, paragraphs=3, sentences=4)
ONE_LINERS = [
    [_SHORT[0][0], "A short paragraph.", "Another short one.", *_SHORT[0][1:]],
    [_SHORT[1][0], "One more short one.", "And the last.", *_SHORT[1][1:]],
]
# References whose first lines end a sentence or not, depending on the width.
REFERENCES = [
    (
        "Lamport, L. LaTeX: A Document Preparation System. Addison-Wesley, "
        "Reading, Massachusetts, second edition, 1994. Updated for LaTeX2e."
    ),
    (
        "Knuth, D. E. The TeXbook. Addison-Wesley, Reading, Massachusetts, "
        "1984. Volume A of Computers and Typesetting."
    ),
    (
        "Mittelbach, F. and Goossens, M. The LaTeX Companion. Addison-Wesley, "
        "Boston, second edition, 2004. Tools and Techniques for Computer "
        "Typesetting."
    ),
    (
        "Bringhurst, R. The Elements of Typographic Style. Hartley and Marks, "
        "Vancouver, fourth edition, 2012. Version 4.0: 20th anniversary "
        "edition."
    ),
    (
        "Tufte, E. R. The Visual Display of Quantitative Information. "
        "Graphics Press, Cheshire, Connecticut, second edition, 2001."
    ),
    (
        "Strunk, W. and White, E. B. The Elements of Style. Longman, New "
        "York, fourth edition, 1999. With a foreword by Roger Angell."
    ),
]
# A long table whose cells look like running headers and footers: their texts
# repeat from page to page once their numbers are left out, or as they are.
STEP_ROWS = [
    ("Step", "Duration", "Checked"),
    *((f"Step {i + 1}", f"{10 * i + 7} ms", "Yes") for i in range(160)),
]
# Numbered sections whose third title is broken into four lines when the
# titles are 120 pt wide, one line more than a heading has.
NUMBERED_SECTIONS = [
    ("1", "Install"),
    ("1.1", "Requirements"),
    ("1.2", "Download the archive from the release page of the project"),
    ("1.3", "Configure"),
    ("2", "Usage"),
]
WIDE_TITLES = 420.0
NARROW_TITLES = 120.0
# A listing with empty lines, which a numbered listing keeps as a number alone.
CODE_SAMPLE = [
    "import os",
    "",
    "def main():",
    "    path = os.getcwd()",
    "    print(path)",
    "",
    "main()",
]
# A listing whose closing braces have no more characters than their numbers
# have digits, and whose empty line keeps a number alone.
BRACES_SAMPLE = [
    "int main(void) {",
    "    int x = 0;",
    "    if (x) {",
    "        x = 1;",
    "    }",
    "    while (x) {",
    "        x = 0;",
    "    }",
    "}",
    "",
    "struct s {",
    "    int y;",
    "};",
    "x = 1",
    "}",
]


def _lines(text: str) -> list[str]:
    """Cut a text into lines of nine words."""
    words = text.split()
    return [" ".join(words[i : i + 9]) for i in range(0, len(words), 9)]


# Pages of one paragraph each, given line by line.
TURNED_PAGES = [
    _lines(texts[0]) for texts in filler_sections(3, paragraphs=1, sentences=9)
]
# Paragraphs given line by line, and the words of them that stand out.
EMPHASIS = [
    [
        "Install the package with the default configuration before you",
        "start the service, and never edit the files that it generates:",
        "the next upgrade replaces them without any warning.",
    ],
    [
        "Each worker retries a failed request three times, then reports",
        "it to the queue. Raise the limit only when the network is slow,",
        "as the retries delay every other request of the worker.",
    ],
]
EMPHASIZED = ("never", "three", "only")
# A page of prose whose lines a tab cuts in two, as two cells of a table, at
# 197 pt from their start: the first parts are 178 to 186 pt wide in
# Times-Roman 10. Each one ends with an m or a w, wider than half the size:
# on a turned page, PDFium moves the space after it off the line by that
# width.
_CELLS = [
    (
        "The service takes the requests that come from",
        "the clients of the cluster, and it logs each",
    ),
    (
        "of them, so that its operators can always know",
        "where a failed request came from and how",
    ),
    (
        "the worker handled it. A worker retries them",
        "three times, then it reports them to the",
    ),
    (
        "queue, which then keeps them until the review",
        "by the operators, who decide what to do",
    ),
    (
        "with the requests that the queue has for them",
        "and how to answer the clients that sent",
    ),
    (
        "them. The operators check the queues below",
        "the log every morning, before the start",
    ),
    (
        "of the next shift, and they clean up the system",
        "once a week, after the backup of the logs.",
    ),
]
CELL_LINES = ["\t".join(cells) for cells in _CELLS]
BULLET_ITEMS = [
    "Python 3.14 or newer is required.",
    "A working network connection is needed.",
    "The cache directory must be writable.",
]
# Each entry builds one text as two PDFs that differ only by their layout.
LAYOUTS: dict[str, tuple[Builder, Builder]] = {
    # LaTeX and Word number their headings, other generators do not.
    "numbered-headings": (
        partial(build_reportlab_pdf, SAMPLE_EN),
        partial(build_reportlab_pdf, SAMPLE_EN, numbered=True),
    ),
    # A long numbered title that a narrow column breaks into too many lines
    # for a heading: it becomes a paragraph that keeps its number.
    "numbered-title-wrap": (
        partial(
            build_numbered_pdf,
            NUMBERED_SECTIONS,
            title_width=WIDE_TITLES,
        ),
        partial(
            build_numbered_pdf,
            NUMBERED_SECTIONS,
            title_width=NARROW_TITLES,
        ),
    ),
    # The same, against a generator that does not number its titles.
    "plain-and-numbered-title-wrap": (
        partial(
            build_numbered_pdf,
            NUMBERED_SECTIONS,
            title_width=WIDE_TITLES,
            numbered=False,
        ),
        partial(
            build_numbered_pdf,
            NUMBERED_SECTIONS,
            title_width=NARROW_TITLES,
        ),
    ),
    # LaTeX books set "Chapter 1" on a line of its own above the title.
    "chapter-labels": (
        partial(build_reportlab_pdf, SAMPLE_EN),
        partial(build_reportlab_pdf, SAMPLE_EN, chapter_labels=True),
    ),
    # An outline that stops at the chapters, as hyperref's bookmarksdepth.
    "partial-outline": (
        partial(build_reportlab_pdf, SAMPLE_EN),
        partial(build_reportlab_pdf, SAMPLE_EN, outline_depth=1),
    ),
    "compound-breaks": (
        partial(build_lines_pdf, COMPOUND_WIDE),
        partial(build_lines_pdf, COMPOUND_NARROW),
    ),
    "contents-page": (
        partial(build_contents_pdf, contents=False),
        partial(build_contents_pdf, contents=True),
    ),
    "indent-paragraphs": (
        partial(build_prose_pdf, LONG_SECTIONS),
        partial(
            build_prose_pdf,
            LONG_SECTIONS,
            style=ProseStyle(space_after=0.0, first_indent=15.0),
        ),
    ),
    "wide-line-spacing": (
        partial(build_prose_pdf, LONG_PARAGRAPHS),
        partial(
            build_prose_pdf,
            LONG_PARAGRAPHS,
            style=ProseStyle(leading=17.5),
        ),
    ),
    "double-spacing": (
        partial(build_prose_pdf, LONG_PARAGRAPHS),
        partial(
            build_prose_pdf,
            LONG_PARAGRAPHS,
            style=ProseStyle(leading=23.0, space_after=0.0, first_indent=15.0),
        ),
    ),
    # Lines 17.5 pt apart and 3 pt more between the paragraphs, about a sixth
    # of the pitch: a small space after each paragraph, as LibreOffice and
    # LaTeX set it with wide lines.
    "small-paragraph-space": (
        partial(build_prose_pdf, LONG_SECTIONS),
        partial(
            build_prose_pdf,
            LONG_SECTIONS,
            style=ProseStyle(leading=17.5, space_after=3.0),
        ),
    ),
    "indented-one-liners": (
        partial(build_prose_pdf, ONE_LINERS),
        partial(
            build_prose_pdf,
            ONE_LINERS,
            style=ProseStyle(space_after=0.0, first_indent=15.0),
        ),
    ),
    "hanging-references": (
        partial(build_references_pdf, REFERENCES, right_margin=25 * mm),
        partial(build_references_pdf, REFERENCES, right_margin=70 * mm),
    ),
    # 92 pt is the baseline of the page number of a default LaTeX article,
    # 10.9 % of the height from the bottom, beyond the footer band.
    "page-number-low": (
        partial(build_prose_pdf, LONG_SECTIONS),
        partial(
            build_prose_pdf,
            LONG_SECTIONS,
            style=ProseStyle(bottom_margin=40 * mm, page_number_y=92.0),
        ),
    ),
    "narrow-table": (
        partial(build_table_pdf, margin=25 * mm),
        partial(build_table_pdf, margin=12 * mm),
    ),
    # At both margins, the first and last rows of each page lie beyond the
    # bands, on the outermost lines.
    "chrome-like-cells": (
        partial(build_table_pdf, margin=25 * mm, rows=STEP_ROWS),
        partial(build_table_pdf, margin=35 * mm, rows=STEP_ROWS),
    ),
    # A cell padding of 7 pt sets the rows 26 pt apart, more than two sizes:
    # the outermost rows of each page stand apart from the rest of the table,
    # and their checks repeat as they are.
    "padded-table-cells": (
        partial(build_table_pdf, margin=25 * mm, rows=STEP_ROWS, padding=7.0),
        partial(build_table_pdf, margin=35 * mm, rows=STEP_ROWS, padding=7.0),
    ),
    # The numbers of the lines are in a tiny proportional type.
    "numbered-listing": (
        partial(build_listing_pdf, CODE_SAMPLE, numbers=False),
        partial(build_listing_pdf, CODE_SAMPLE, numbers=True),
    ),
    # The numbers stand next to the code, in the segment of each line.
    "numbered-braces": (
        partial(build_listing_pdf, BRACES_SAMPLE, numbers=False),
        partial(build_listing_pdf, BRACES_SAMPLE, numbers=True),
    ),
    # The numbers stand 30 pt from the code, each one in a segment of its own.
    "numbered-braces-apart": (
        partial(build_listing_pdf, BRACES_SAMPLE, numbers=False),
        partial(
            build_listing_pdf,
            BRACES_SAMPLE,
            numbers=True,
            numbersep=30.0,
        ),
    ),
    # Five lines fit on a page: the sixth, an empty one, opens the second
    # page, where its number stands alone.
    "numbered-listing-page-break": (
        partial(build_listing_pdf, CODE_SAMPLE, numbers=False, page_lines=5),
        partial(build_listing_pdf, CODE_SAMPLE, numbers=True, page_lines=5),
    ),
    # The second page is turned, as a landscape page: its text runs up the
    # page, which a viewer turns a quarter clockwise.
    "turned-page": (
        partial(build_turned_pdf, TURNED_PAGES),
        partial(build_turned_pdf, TURNED_PAGES, turned={1}),
    ),
    # A DRAFT watermark crosses every page at 45 degrees.
    "watermark": (
        partial(build_prose_pdf, LONG_SECTIONS),
        partial(
            build_prose_pdf,
            LONG_SECTIONS,
            style=ProseStyle(watermark=True),
        ),
    ),
    # The words that stand out are set in synthetic italics: their glyphs
    # slant, but their baseline stays level.
    "synthetic-italics": (
        partial(build_lines_pdf, EMPHASIS),
        partial(build_lines_pdf, EMPHASIS, slanted=EMPHASIZED),
    ),
    # Each line changes text object at the same place, with no space glyph,
    # as LaTeX sets the cells of a table: the space that PDFium generates
    # between the two objects keeps them on one line, turned page or not.
    "turned-cells": (
        partial(build_turned_pdf, [CELL_LINES], tab=197.0),
        partial(build_turned_pdf, [CELL_LINES], tab=197.0, turned={0}),
    ),
    # Word sets its bullets in a symbol font, and PDFium returns them as
    # private use characters: they are bullets, not encoding damage.
    "word-bullets": (
        partial(build_bullets_pdf, BULLET_ITEMS, word=False),
        partial(build_bullets_pdf, BULLET_ITEMS, word=True),
    ),
    # Front matter numbered i, ii, with or without page labels.
    "roman-front-matter": (
        partial(
            build_prose_pdf,
            ROMAN_SECTIONS,
            style=ProseStyle(page_number_y=40.0, roman_pages=2),
        ),
        partial(
            build_prose_pdf,
            ROMAN_SECTIONS,
            style=ProseStyle(
                page_number_y=40.0,
                roman_pages=2,
                page_labels=True,
            ),
        ),
    ),
    "word-bullets-arrows": (
        partial(build_bullets_pdf, BULLET_ITEMS, word=False),
        partial(
            build_bullets_pdf,
            BULLET_ITEMS,
            word=True,
            bullet="\U0000f0d8",
        ),
    ),
    # The bullet and its text are one string, with no blank between them.
    "word-bullets-tight": (
        partial(build_bullets_pdf, BULLET_ITEMS, word=False),
        partial(build_bullets_pdf, BULLET_ITEMS, word=True, gap=False),
    ),
    # A figure drawn as paths, or the same figure in a form XObject.
    "figures-as-forms": (
        partial(build_reportlab_pdf, FIGURES_EN),
        partial(build_reportlab_pdf, FIGURES_EN, figure_forms=True),
    ),
    # A logo drawn on every page is chrome.
    "logo-on-pages": (
        partial(build_reportlab_pdf, FIGURES_EN),
        partial(build_reportlab_pdf, FIGURES_EN, logo=True),
    ),
}


@pytest.mark.parametrize("order", [(0, 1), (1, 0)], ids=["forward", "swapped"])
@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_layout_only_differences_produce_no_finding(
    tmp_path: Path,
    layout: str,
    order: tuple[int, int],
) -> None:
    """One text in two layouts has no finding, whichever is the reference."""
    first, second = LAYOUTS[layout]
    pdfs = (first(tmp_path / "first.pdf"), second(tmp_path / "second.pdf"))
    result = run_compare(pdfs[order[0]], pdfs[order[1]], Config())
    # Two PDFs that extract to nothing would have no finding either.
    assert result.stats.matched > 0
    assert [
        (f.rule, f.severity) for f in result.findings if f.severity >= MI
    ] == []


@pytest.mark.parametrize("order", [(0, 1), (1, 0)], ids=["forward", "swapped"])
def test_logo_chrome_on_one_side_only_gives_no_image_finding(
    tmp_path: Path,
    order: tuple[int, int],
) -> None:
    """A logo that only the longer document removes as chrome is no image."""
    two = replace(FIGURES_EN, sections=FIGURES_EN.sections[:2])
    pdfs = (
        build_reportlab_pdf(two, tmp_path / "two.pdf", logo=True),
        build_reportlab_pdf(two, tmp_path / "toc.pdf", logo=True, toc=True),
    )
    result = run_compare(pdfs[order[0]], pdfs[order[1]], Config())
    assert result.stats.matched_images > 0
    assert [
        f.rule for f in result.findings if f.category is Category.IMAGE
    ] == []


def test_wrapped_numbered_title_only_loses_its_kind(tmp_path: Path) -> None:
    """A numbered title broken into too many lines is an inferred change."""
    wide, narrow = (
        build_numbered_pdf(
            NUMBERED_SECTIONS,
            tmp_path / f"{width}.pdf",
            title_width=width,
        )
        for width in (WIDE_TITLES, NARROW_TITLES)
    )
    forward = run_compare(wide, narrow, Config()).findings
    swapped = run_compare(narrow, wide, Config()).findings
    assert [(f.rule, f.severity) for f in forward] == [
        ("structure.heading_lost", Severity.INFO),
    ]
    assert [(f.rule, f.severity) for f in swapped] == [
        ("structure.kind_changed", Severity.INFO),
    ]


def test_heading_level_change_is_reported(tmp_path: Path) -> None:
    """A heading one level deeper in the outline is a minor finding."""
    deeper = set_level(SAMPLE_EN, "exports", 3)
    findings = _compare(tmp_path, SAMPLE_EN, deeper).findings
    assert [(f.rule, f.severity) for f in findings if f.severity >= MI] == [
        ("structure.heading_level", Severity.MINOR),
    ]
