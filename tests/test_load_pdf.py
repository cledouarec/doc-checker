# Copyright 2026 Christophe Le Douarec
"""Tests for loading PDF documents."""

import logging
import re
import threading
import time
from typing import TYPE_CHECKING, override

import pytest
from pypdf import PdfWriter
from pypdf.generic import NameObject, TextStringObject

from doc_checker.compare.engine import compare_documents
from doc_checker.config import Config, PdfConfig
from doc_checker.errors import InputError
from doc_checker.extract import detect_format, load_document
from doc_checker.extract.pdf import load_pdf, reader
from doc_checker.images.pixels import TOO_LARGE
from doc_checker.model.document import HEADING_SOURCE, NUMBER_ATTR, BlockKind
from doc_checker.model.facts import OutlineEntry
from tests.builders import FIGURES_EN, SAMPLE_EN, SAMPLE_FR
from tests.pdf_builders import (
    build_figure_list_pdf,
    build_fpdf2_pdf,
    build_images_pdf,
    build_inline_code_pdf,
    build_listing_pdf,
    build_ornaments_pdf,
    build_outline_pdf,
    build_oversized_image_pdf,
    build_raw_pdf,
    build_reportlab_pdf,
    build_scanned_pdf,
    build_shared_resources_pdf,
    build_single_page_pdf,
    build_stacked_figures_pdf,
    build_turned_pdf,
)

if TYPE_CHECKING:
    from pathlib import Path

    from doc_checker.model.document import Document

NUMBERS = "text.number_changed"
SECTION_HEADINGS = [(s.title, s.level) for s in SAMPLE_EN.sections]


@pytest.fixture(scope="module")
def pdfs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Build the samples with both generators, and the special cases."""
    root = tmp_path_factory.mktemp("pdfs")
    return {
        "rl_en": build_reportlab_pdf(SAMPLE_EN, root / "en.pdf"),
        "rl_fr": build_reportlab_pdf(SAMPLE_FR, root / "fr.pdf"),
        "fp_en": build_fpdf2_pdf(SAMPLE_EN, root / "fp_en.pdf"),
        "fp_fr": build_fpdf2_pdf(SAMPLE_FR, root / "fp_fr.pdf"),
        "toc": build_reportlab_pdf(SAMPLE_EN, root / "toc.pdf", toc=True),
        "scan": build_reportlab_pdf(
            SAMPLE_EN,
            root / "scan.pdf",
            scanned_page=True,
            artifacts=True,
        ),
    }


def _load(path: Path) -> Document:
    """Load a PDF with the default settings."""
    return load_pdf(path, PdfConfig(), lang="auto")


def _dehyphen(key: str) -> str:
    """Return a key without the hyphens inside its words."""
    return re.sub(r"(?<=\w)-(?=\w)", "", key)


def _headings(doc: Document) -> list[tuple[str, int | None]]:
    """Return the text and the level of each heading of a document."""
    return [
        (b.text, b.level) for b in doc.blocks if b.kind is BlockKind.HEADING
    ]


def _with_outline(source: Path, out: Path, *titles: str) -> Path:
    """Copy a PDF, with top level bookmarks that point to its first page."""
    writer = PdfWriter(clone_from=source)
    for title in titles:
        writer.add_outline_item(title, 0)
    writer.write(out)
    return out


def test_reportlab_document(pdfs: dict[str, Path]) -> None:
    """A reportlab PDF gives its pages, metadata, blocks and chrome."""
    doc = _load(pdfs["rl_en"])
    assert (
        doc.format,
        doc.parts,
        doc.lang,
        doc.declared_lang,
        doc.tagged,
    ) == (
        "pdf",
        [str(n) for n in range(1, 10)],
        "en",
        "en",
        False,
    )
    assert doc.generator is not None
    assert doc.generator.startswith("ReportLab")
    assert [(b.kind.value, b.text, b.level) for b in doc.blocks[:3]] == [
        ("heading", "Getting started", 1),
        ("heading", "Installation", 2),
        (
            "paragraph",
            (
                "Install the package with the default configuration "
                "before you start."
            ),
            None,
        ),
    ]
    kinds = {b.text: b.kind.value for b in doc.blocks}
    assert kinds["Python 3.14 or newer is required."] == "list_item"
    assert kinds["pip install example\nexample --version"] == "code"
    assert (
        doc.normalization["code_line_number"],
        doc.normalization["bullet"],
    ) == (2, 2)
    assert {"Draft", "Page #"} <= set(doc.chrome)
    assert doc.chrome_keys == frozenset()
    block = next(
        b for b in doc.blocks if b.text.startswith("Install the package")
    )
    assert (block.source.href, block.section_path) == (
        "en.pdf#page=2",
        ("Getting started", "Installation"),
    )


def test_fpdf2_document(pdfs: dict[str, Path]) -> None:
    """An fpdf2 PDF is tagged, has no generator and joins its breaks."""
    doc = _load(pdfs["fp_en"])
    assert (doc.parts, doc.tagged, doc.generator) == (["1", "2"], True, None)
    assert doc.normalization["hyphenation"] > 0
    assert doc.normalization["page_break"] == 1


@pytest.mark.parametrize("lang", ["en", "fr"])
def test_both_builders_yield_same_blocks(
    pdfs: dict[str, Path],
    lang: str,
) -> None:
    """The reportlab and the fpdf2 builds of a text give the same blocks."""
    ref, new = _load(pdfs[f"rl_{lang}"]), _load(pdfs[f"fp_{lang}"])
    assert [_dehyphen(b.key_strict) for b in ref.blocks] == [
        _dehyphen(b.key_strict) for b in new.blocks
    ]
    assert [b.kind for b in ref.blocks] == [b.kind for b in new.blocks]


def test_toc_page(pdfs: dict[str, Path]) -> None:
    """The blocks of a contents page are typed and counted."""
    doc = _load(pdfs["toc"])
    toc = [b for b in doc.blocks if b.kind is BlockKind.TOC]
    assert toc
    assert {b.source.page for b in toc} == {1}
    assert (doc.parts[:2], doc.normalization["toc"]) == (["i", "1"], len(toc))


def test_values_beyond_the_page_count_are_no_contents(tmp_path: Path) -> None:
    """A table of values with dot leaders is no contents in a short PDF."""
    values = [
        "Max load . . . . . . 500",
        "Timeout . . . . . . 30",
        "Retries . . . . . . 3",
    ]
    doc = _load(build_turned_pdf([values], tmp_path / "values.pdf"))
    assert doc.blocks
    assert BlockKind.TOC not in {block.kind for block in doc.blocks}


def test_list_of_figures_is_contents(tmp_path: Path) -> None:
    """A list of figures, as LibreOffice writes it, is typed as contents."""
    doc = _load(build_figure_list_pdf(tmp_path / "figures.pdf"))
    kinds = [(b.kind, b.source.page) for b in doc.blocks]
    assert kinds.count((BlockKind.TOC, 1)) == 4
    assert [b.text for b in doc.blocks if b.kind is BlockKind.CAPTION] == [
        "Main screen of the application",
        "Flow of a request through the workers",
        "State of the scheduler",
        "Dashboard of a project",
    ]


def test_tiny_images_are_left_out_and_counted(tmp_path: Path) -> None:
    """Images of a pixel or three, used as rules, are no images."""
    doc = _load(
        build_images_pdf(tmp_path / "rules.pdf", pages=2, image_size=(3, 40)),
    )
    assert [b for b in doc.blocks if b.kind is BlockKind.IMAGE] == []
    assert doc.normalization["tiny_image"] == 2


@pytest.mark.parametrize("float_page", [False, True], ids=["stacked", "page"])
@pytest.mark.parametrize(
    ("captions", "edited", "rule"),
    [
        (("Sales in 2021", "Sales in 2022", "Sales in 2023"), "2024", NUMBERS),
        (("Release 3", "Release 4", "Release 5"), "6", NUMBERS),
        (
            ("Output of the CLI", "Schema of the XML", "Menu of the DVD"),
            "VM",
            None,
        ),
    ],
    ids=["years", "release", "acronyms"],
)
def test_stacked_captions_keep_their_changes(
    tmp_path: Path,
    captions: tuple[str, ...],
    edited: str,
    rule: str | None,
    *,
    float_page: bool,
) -> None:
    """Stacked captions that end alike stay captions, and are compared."""
    ref = _load(
        build_stacked_figures_pdf(
            tmp_path / "ref.pdf",
            captions,
            float_page=float_page,
        ),
    )
    assert [b.text for b in ref.blocks if b.kind is BlockKind.CAPTION] == [
        *captions,
    ]
    last = captions[-1].rsplit(" ", 1)[0]
    new = _load(
        build_stacked_figures_pdf(
            tmp_path / "new.pdf",
            [*captions[:-1], f"{last} {edited}"],
            float_page=float_page,
        ),
    )
    rules = [f.rule for f in compare_documents(ref, new, Config()).findings]
    assert len(rules) == 1
    assert rule is None or rules == [rule]


def test_scanned_page_and_artifacts(pdfs: dict[str, Path]) -> None:
    """A page of images is reported, and artifacts are no chrome."""
    doc = _load(pdfs["scan"])
    assert doc.warnings == ["1 page(s) without text, probably scanned: 10"]
    assert all(
        b.source.page != 10
        for b in doc.blocks
        if b.kind is not BlockKind.IMAGE
    )
    assert [b.kind for b in doc.blocks if b.source.page == 10] == [
        BlockKind.IMAGE,
    ]
    assert (doc.chrome, doc.normalization["artifact"] > 0) == ([], True)
    assert "Draft" not in {b.text for b in doc.blocks}


def test_fully_scanned_pdf(tmp_path: Path) -> None:
    """A PDF without text has no block and lists its first pages."""
    doc = _load(build_scanned_pdf(tmp_path / "scan.pdf", pages=12))
    assert (doc.blocks, doc.lang) == ([], None)
    # The same scan on every page is chrome.
    assert doc.normalization["image_chrome"] == 12
    assert doc.warnings == [
        (
            "12 page(s) without text, probably scanned: "
            "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, \U00002026"
        ),
    ]


def test_language_comes_from_the_text_without_lang(tmp_path: Path) -> None:
    """Without /Lang, the language is found in the text."""
    sentence = (
        "Le service accepte les demandes de tous les clients et il les "
        "traite dans le bon ordre."
    )
    path = build_raw_pdf(
        tmp_path / "fr.pdf",
        [f"BT /F1 10 Tf 72 700 Td ({sentence}) Tj ET"],
    )
    doc = _load(path)
    assert (doc.declared_lang, doc.lang) == (None, "fr")


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        ("fr-CA", "fr-CA"),
        ("en<script>alert(1)</script>", None),
        ("fr\x07\U0000202e", None),
        ("", None),
    ],
)
def test_pdf_language_must_be_a_language_tag(
    tmp_path: Path,
    declared: str,
    expected: str | None,
) -> None:
    """A /Lang that is not a language tag is ignored, as in HTML."""
    sentence = (
        "Le service accepte les demandes de tous les clients et il les "
        "traite dans le bon ordre."
    )
    source = build_raw_pdf(
        tmp_path / "fr.pdf",
        [f"BT /F1 10 Tf 72 700 Td ({sentence}) Tj ET"],
    )
    writer = PdfWriter(clone_from=source)
    writer.root_object[NameObject("/Lang")] = TextStringObject(declared)
    path = tmp_path / "lang.pdf"
    writer.write(path)
    doc = _load(path)
    assert (doc.declared_lang, doc.lang) == (expected, "fr")


def test_load_document_reads_pdf(pdfs: dict[str, Path]) -> None:
    """A PDF input is loaded as a PDF document."""
    assert load_document(pdfs["rl_fr"], Config()).format == "pdf"


def test_load_document_reads_the_page_facts_for_the_audit(
    pdfs: dict[str, Path],
) -> None:
    """The page facts of a PDF are read for the audit, and only then."""
    plain = load_document(pdfs["scan"], Config())
    audited = load_document(pdfs["scan"], Config(), audit=True)
    assert plain.pdf_pages == ()
    assert [page.scanned for page in audited.pdf_pages] == [False] * 9 + [
        True,
    ]
    assert audited.pdf_pages[0].chars > 0
    assert (audited.title, audited.outline) == (plain.title, plain.outline)
    assert audited.warnings == plain.warnings
    assert [b.text for b in audited.blocks] == [b.text for b in plain.blocks]


def test_outline_entries_are_printable_and_may_lack_a_page(
    tmp_path: Path,
) -> None:
    """A control in a title is a space; a bookmark may lead to no page."""
    path = build_outline_pdf(
        tmp_path / "o.pdf",
        [("(a\\007b)", 1), ("(Nowhere)", None)],
    )
    assert _load(path).outline == (
        OutlineEntry(0, "a b", 1),
        OutlineEntry(0, "Nowhere", None),
    )


def test_outline_beyond_its_limit_is_a_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bookmarks past the limit are not read, and a warning says so."""
    monkeypatch.setattr(reader, "MAX_BOOKMARKS", 2)
    path = build_outline_pdf(
        tmp_path / "o.pdf",
        [("(One)", 1), ("(Two)", 1), ("(Three)", 1)],
    )
    doc = _load(path)
    assert [entry.title for entry in doc.outline] == ["One", "Two"]
    assert doc.warnings == [
        "the outline has more than 2 bookmarks, the others were not read",
    ]


def test_page_labels_beyond_their_work_are_a_warning(
    pdfs: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once the labels asked spent their work, the next pages are numbered."""
    monkeypatch.setattr(reader, "PAGE_LABEL_WORK", 8)
    doc = load_pdf(pdfs["toc"], PdfConfig(), lang="auto", audit=True)
    assert doc.parts == ["i", "1", *(str(n) for n in range(3, 11))]
    assert [page.label for page in doc.pdf_pages] == doc.parts
    assert doc.warnings == [
        "the page labels exceed 8 bytes, the other pages are numbered",
    ]


def test_outline_titles_beyond_their_work_are_a_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once the titles asked spent their work, a warning ends the outline."""
    monkeypatch.setattr(reader, "OUTLINE_TITLE_WORK", 10)
    path = build_outline_pdf(
        tmp_path / "o.pdf",
        [("(One)", 1), ("(Two)", 1), ("(Three)", 1)],
    )
    doc = _load(path)
    assert [entry.title for entry in doc.outline] == ["One", "Two"]
    assert doc.warnings == [
        (
            "the outline titles exceed 10 bytes,"
            " the other bookmarks were not read"
        ),
    ]


class _Recorder(logging.Handler):
    """A logging handler that keeps the records it is given."""

    def __init__(self) -> None:
        """Start with no record."""
        super().__init__()
        self.records: list[logging.LogRecord] = []

    @override
    def emit(self, record: logging.LogRecord) -> None:
        """Keep a record."""
        self.records.append(record)


def test_outline_cycle_is_quiet(tmp_path: Path) -> None:
    """An outline that loops is read once around, and logs nothing."""
    path = build_outline_pdf(
        tmp_path / "o.pdf",
        [("(One)", 1), ("(Two)", 1)],
        cycle=True,
    )
    recorder = _Recorder()
    root = logging.getLogger()
    root.addHandler(recorder)
    try:
        doc = _load(path)
    finally:
        root.removeHandler(recorder)
    assert [entry.title for entry in doc.outline] == ["One", "Two"]
    assert recorder.records == []


@pytest.mark.parametrize("field", ["outline", "info", "label"])
def test_lone_surrogates_do_not_stop_the_loading(
    tmp_path: Path,
    field: str,
) -> None:
    """Half a UTF-16 character in a title or a label is replaced."""
    lone = "<FEFFD8000041>"  # A high surrogate without its low half, then A.
    path = build_outline_pdf(
        tmp_path / "s.pdf",
        [(lone if field == "outline" else "(Intro)", 1)],
        info_title=lone if field == "info" else "(Title)",
        label=lone if field == "label" else None,
    )
    doc = load_pdf(path, PdfConfig(), lang="auto", audit=True)
    read = {
        "outline": doc.outline[0].title,
        "info": doc.title,
        "label": doc.pdf_pages[0].label,
    }
    assert read[field] == "\U0000fffdA"


def test_pdf_header_may_follow_a_few_bytes(tmp_path: Path) -> None:
    """A PDF whose header comes after a few bytes is still a PDF."""
    data = build_raw_pdf(
        tmp_path / "raw.pdf",
        ["BT /F1 10 Tf 72 700 Td (Hello there.) Tj ET"],
    ).read_bytes()
    shifted = tmp_path / "shifted.pdf"
    shifted.write_bytes(b"\xef\xbb\xbf\n" + data)
    far = tmp_path / "far.bin"
    far.write_bytes(b"x" * 1024 + data)
    assert detect_format(shifted) == "pdf"
    assert load_document(shifted, Config()).format == "pdf"
    with pytest.raises(InputError, match="unsupported input"):
        detect_format(far)


def test_html_page_that_quotes_the_pdf_header_stays_html(
    tmp_path: Path,
) -> None:
    """An HTML file that quotes the PDF header near its top is still HTML."""
    page = tmp_path / "header.html"
    page.write_text(
        "<html><body><p>Every PDF starts with %PDF-1.7</p></body></html>",
        encoding="utf-8",
    )
    assert detect_format(page) == "html"


def test_pdf_header_at_the_start_wins_over_an_html_suffix(
    tmp_path: Path,
) -> None:
    """The PDF header at the start of a file wins over an HTML suffix."""
    named = build_raw_pdf(tmp_path / "named.html", [])
    assert detect_format(named) == "pdf"


def test_chrome_texts_follow_the_pages(pdfs: dict[str, Path]) -> None:
    """The chrome texts come once each, in page order, digits neutralized."""
    doc = _load(pdfs["rl_en"])
    assert doc.chrome == [
        "Example",
        "Draft",
        "Page #",
        "Getting started",
        "User guide",
        "Reference",
    ]
    # Four segments a page, but the first page has no running chapter title.
    assert doc.normalization["header_footer"] == 9 * 4 - 1


def test_damaged_page_names_the_file(tmp_path: Path) -> None:
    """A page that cannot be read is an input error naming the file."""
    good = build_raw_pdf(tmp_path / "good.pdf", [])
    # PDFium opens the file, but it has lost the page tree.
    damaged = tmp_path / "kids.pdf"
    damaged.write_bytes(good.read_bytes().replace(b"/Kids", b"/Kidz", 1))
    message = f"cannot read {damaged}: page 1: invalid or damaged PDF"
    with pytest.raises(InputError, match=re.escape(message)):
        _load(damaged)


def test_negative_font_size_does_not_stop_the_loading(tmp_path: Path) -> None:
    """A text set with a negative font size is loaded in a bounded time."""
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
    loaded: list[Document] = []
    worker = threading.Thread(
        target=lambda: loaded.append(_load(path)),
        daemon=True,
    )
    worker.start()
    worker.join(timeout=10.0)
    assert len(loaded) == 1
    assert any(
        "Normal text above the flipped one." in block.text
        for block in loaded[0].blocks
    )


def test_text_at_an_angle_is_reported(tmp_path: Path) -> None:
    """Text left out for its angle is counted and reported."""
    path = build_raw_pdf(
        tmp_path / "draft.pdf",
        [
            "BT /F1 10 Tf 72 700 Td (The service accepts requests.) Tj ET",
            (
                "q 0.7071 0.7071 -0.7071 0.7071 300 400 cm "
                "BT /F1 60 Tf 0 0 Td (DRAFT) Tj ET Q"
            ),
        ],
    )
    doc = _load(path)
    assert doc.warnings == ["1 page(s) with text at an angle, left out: 1"]
    assert doc.normalization["angled_text"] == 5


def test_scanned_pages_are_reported_before_angled_text(tmp_path: Path) -> None:
    """The warning about scanned pages comes before the one about angles."""
    # The page with text at an angle comes before the scanned page.
    draft = build_raw_pdf(
        tmp_path / "draft.pdf",
        [
            "BT /F1 10 Tf 72 700 Td (The service accepts requests.) Tj ET",
            (
                "q 0.7071 0.7071 -0.7071 0.7071 300 400 cm "
                "BT /F1 60 Tf 0 0 Td (DRAFT) Tj ET Q"
            ),
        ],
    )
    writer = PdfWriter()
    writer.append(draft)
    writer.append(build_scanned_pdf(tmp_path / "scan.pdf", pages=1))
    path = tmp_path / "both.pdf"
    writer.write(path)
    assert _load(path).warnings == [
        "1 page(s) without text, probably scanned: 2",
        "1 page(s) with text at an angle, left out: 1",
    ]


def test_unmatched_bookmarks_leave_headings_to_styles(tmp_path: Path) -> None:
    """Bookmarks that match no block let the styles tell the headings."""
    plain = build_reportlab_pdf(SAMPLE_EN, tmp_path / "p.pdf", outline=False)
    doc = _load(_with_outline(plain, tmp_path / "o.pdf", "Release notes"))
    assert _headings(doc) == SECTION_HEADINGS


def test_a_partial_outline_is_completed(tmp_path: Path) -> None:
    """One matching bookmark no longer hides the headings of the styles."""
    plain = build_reportlab_pdf(SAMPLE_EN, tmp_path / "p.pdf", outline=False)
    doc = _load(_with_outline(plain, tmp_path / "o.pdf", "Getting started"))
    assert _headings(doc) == SECTION_HEADINGS
    sources = [
        b.attrs.get(HEADING_SOURCE)
        for b in doc.blocks
        if b.kind is BlockKind.HEADING
    ]
    assert sources == ["outline"] + ["style"] * 8


def test_an_outline_of_chapters_keeps_the_sections(tmp_path: Path) -> None:
    """An outline that stops at the chapters keeps the section headings."""
    path = build_reportlab_pdf(SAMPLE_EN, tmp_path / "c.pdf", outline_depth=1)
    assert _headings(_load(path)) == SECTION_HEADINGS


def test_a_full_outline_gives_every_heading(pdfs: dict[str, Path]) -> None:
    """In a PDF with a full outline, every heading comes from it."""
    doc = _load(pdfs["rl_en"])
    sources = {
        b.attrs.get(HEADING_SOURCE)
        for b in doc.blocks
        if b.kind is BlockKind.HEADING
    }
    assert sources == {"outline"}


def test_styles_tell_headings_without_bookmarks(tmp_path: Path) -> None:
    """Without bookmarks, the styles of the blocks tell the headings."""
    plain = build_reportlab_pdf(SAMPLE_EN, tmp_path / "p.pdf", outline=False)
    assert _headings(_load(plain)) == SECTION_HEADINGS


def test_numbered_listing_in_tiny_type(tmp_path: Path) -> None:
    """A listing numbered in tiny proportional type is one code block."""
    code = [
        "import os",
        "",
        "def main():",
        "    print(os.getcwd())",
        "",
        "main()",
    ]
    pdf = build_listing_pdf(code, tmp_path / "listing.pdf", numbers=True)
    doc = _load(pdf)
    assert [b.kind.value for b in doc.blocks] == [
        "paragraph",
        "code",
        "paragraph",
    ]
    lines = doc.blocks[1].text.splitlines()
    assert (len(lines), lines[0], doc.normalization["code_line_number"]) == (
        4,
        "import os",
        6,
    )


def test_numbers_and_chapter_labels_leave_headings(tmp_path: Path) -> None:
    """Heading numbers and chapter labels leave the text of a PDF."""
    path = build_reportlab_pdf(
        SAMPLE_EN,
        tmp_path / "n.pdf",
        numbered=True,
        chapter_labels=True,
    )
    doc = _load(path)
    assert _headings(doc) == SECTION_HEADINGS
    assert {
        b.attrs[HEADING_SOURCE]
        for b in doc.blocks
        if b.kind is BlockKind.HEADING
    } == {"outline"}
    assert "Chapter 1" not in {b.text for b in doc.blocks}
    # Three chapter labels, then nine heading numbers.
    assert doc.normalization["heading_number"] == 12


def test_image_blocks_take_their_place(tmp_path: Path) -> None:
    """An image comes after the text above it and before its caption."""
    doc = load_pdf(
        build_reportlab_pdf(FIGURES_EN, tmp_path / "a.pdf"),
        PdfConfig(),
        lang="en",
    )
    kinds = [b.kind for b in doc.blocks]
    first = kinds.index(BlockKind.IMAGE)
    assert doc.blocks[first - 1].text.startswith("The installer downloads")
    assert (kinds[first + 1], doc.blocks[first + 1].text) == (
        BlockKind.CAPTION,
        "Steps of the installer",
    )
    assert doc.blocks[first].section_path == (
        "Getting started",
        "Installation",
    )


def test_logo_on_every_page_is_chrome(tmp_path: Path) -> None:
    """A logo drawn on each page is chrome, and counted."""
    doc = load_pdf(
        build_reportlab_pdf(FIGURES_EN, tmp_path / "a.pdf", logo=True),
        PdfConfig(),
        lang="en",
    )
    pages = len(doc.parts)
    assert (
        sum(b.image is not None and not b.image.vector for b in doc.blocks)
        == 3
    )
    assert doc.normalization["image_chrome"] == pages
    (logo,) = doc.chrome_images
    assert all(b.image is None or b.image.digest != logo for b in doc.blocks)


def test_pdf_with_hundreds_of_images_loads_quickly(tmp_path: Path) -> None:
    """Three hundred pages with an image each load in bounded time."""
    path = build_images_pdf(tmp_path / "many.pdf", pages=300)
    start = time.perf_counter()
    doc = load_pdf(path, PdfConfig(), lang="en")
    assert time.perf_counter() - start < 30.0
    assert sum(b.kind is BlockKind.IMAGE for b in doc.blocks) == 300


def test_no_images_still_finds_scanned_pages(pdfs: dict[str, Path]) -> None:
    """Without images, no block is an image, and scans are still reported."""
    doc = load_document(pdfs["scan"], Config(images=False))
    assert BlockKind.IMAGE not in {b.kind for b in doc.blocks}
    assert doc.warnings == ["1 page(s) without text, probably scanned: 10"]


def test_unreadable_image_is_a_warning(tmp_path: Path) -> None:
    """An image whose pixels cannot be read is kept, with a warning."""
    doc = _load(build_oversized_image_pdf(tmp_path / "large.pdf"))
    images = [b.image for b in doc.blocks if b.image is not None]
    assert [image.problem for image in images] == [TOO_LARGE]
    assert doc.warnings == ["1 image(s) could not be read: page 1 image 1"]


def _text(doc: Document) -> list[tuple[object, ...]]:
    """Return what is compared of the blocks of text of a document."""
    return [
        (b.kind, b.text, b.level, b.attrs, b.section_path, b.source)
        for b in doc.blocks
        if b.kind is not BlockKind.IMAGE
    ]


def _text_counts(doc: Document) -> dict[str, int]:
    """Return the normalization counts of a document, but of its images."""
    return {
        key: count
        for key, count in doc.normalization.items()
        if key not in {"image_chrome", "tiny_image"}
    }


@pytest.mark.parametrize("layout", ["ornaments", "figures"])
def test_text_does_not_depend_on_the_images(
    tmp_path: Path,
    layout: str,
) -> None:
    """Reading the images or not leaves the text and its counts the same."""
    if layout == "ornaments":
        path = build_ornaments_pdf(tmp_path / "o.pdf")
    else:
        path = build_reportlab_pdf(
            FIGURES_EN,
            tmp_path / "f.pdf",
            chapter_labels=True,
        )
    read = load_document(path, Config())
    skipped = load_document(path, Config(images=False))
    assert BlockKind.IMAGE in {b.kind for b in read.blocks}
    assert _text(read) == _text(skipped)
    assert _text_counts(read) == _text_counts(skipped)


def test_chapter_label_folds_past_an_ornament(tmp_path: Path) -> None:
    """An image between a chapter label and its title stays before it."""
    doc = _load(build_ornaments_pdf(tmp_path / "o.pdf"))
    headings = [
        (b.text, b.attrs.get(NUMBER_ATTR))
        for b in doc.blocks
        if b.kind is BlockKind.HEADING
    ]
    assert headings == [
        ("Introduction", "Chapter 1"),
        ("Usage", "Chapter 2"),
        ("Reference", "Chapter 3"),
        ("Appendix", "Chapter 4"),
    ]
    assert [b.kind for b in doc.blocks[:2]] == [
        BlockKind.IMAGE,
        BlockKind.HEADING,
    ]
    assert doc.normalization["heading_number"] == 4


def test_pages_sharing_many_xobjects_load_quickly(tmp_path: Path) -> None:
    """A thousand pages that share 5,000 XObjects load in bounded time."""
    path = build_shared_resources_pdf(
        tmp_path / "shared.pdf",
        pages=1000,
        xobjects=5000,
    )
    start = time.perf_counter()
    doc = load_pdf(path, PdfConfig(), lang="en")
    # About 3 s on a slow machine; a lookup of the XObjects of every page
    # took 32 s on a desktop.
    assert time.perf_counter() - start < 20.0
    assert (doc.normalization["image_chrome"], doc.warnings) == (1000, [])


def test_monospaced_words_are_read_for_the_audit(tmp_path: Path) -> None:
    """The words set in Courier are code words, for the audit only."""
    path = build_inline_code_pdf(
        tmp_path / "code.pdf",
        [("Run ", False), ("tlmgr update", True), (" now.", False)],
    )
    audited = load_pdf(path, PdfConfig(), lang="auto", audit=True)
    assert audited.code_words == frozenset({"tlmgr", "update"})
    assert _load(path).code_words == frozenset()
    assert [b.text for b in audited.blocks] == ["Run tlmgr update now."]


def test_code_fonts_give_code_words_for_the_audit(tmp_path: Path) -> None:
    """The typewriter font of cm-super sets code, for the audit only."""
    # pdfTeX does not set the fixed pitch flag of its fonts, and MONO_NAME
    # does not know that name: the other commands read no monospaced font.
    path = build_single_page_pdf(
        tmp_path / "sftt.pdf",
        "BT /F1 12 Tf 20 100 Td (tlmgr update) Tj ET",
        font="<< /Type /Font /Subtype /Type1 /BaseFont /SFTT1000 >>",
    )
    with reader.open_pdf(path) as pdf:
        chars = reader.read_page(pdf, 0, images=False).chars
    assert {(c.mono, c.code) for c in chars if not c.text.isspace()} == {
        (False, True),
    }
    audited = load_pdf(path, PdfConfig(), lang="auto", audit=True)
    plain = _load(path)
    assert audited.code_words == frozenset({"tlmgr", "update"})
    assert plain.code_words == frozenset()
    assert [(b.kind, b.text) for b in plain.blocks] == [
        (BlockKind.PARAGRAPH, "tlmgr update"),
    ]
