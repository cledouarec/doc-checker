# Copyright 2026 Christophe Le Douarec
"""Tests for the accessibility checks of the audit."""

import time
from dataclasses import replace
from typing import TYPE_CHECKING

from doc_checker.audit.accessibility import (
    FILE_NAME_ALT,
    GENERIC_LINK_TEXTS,
    UNMAPPED_PAGES_SHOWN,
    accessibility_drafts,
    prose_sample,
)
from doc_checker.audit.engine import audit_document
from doc_checker.compare.aggregate import PATTERN_MAX_CHARS
from doc_checker.config import AuditThresholds, Config
from doc_checker.extract.loader import load_document
from doc_checker.model.document import BlockKind, ImageRef, Link
from doc_checker.model.facts import (
    NO_STRUCTURE_TREE,
    HtmlPageFacts,
    PdfLink,
    PdfPageFacts,
    Spot,
    TaggedElement,
    TagTree,
)
from doc_checker.model.findings import Severity
from doc_checker.text.lang import LANG_SAMPLE_BLOCKS
from doc_checker.text.normalize import ELLIPSIS
from tests.builders import (
    SAMPLE_FR,
    link_text,
    make_audit_document,
    make_block,
    make_heading,
    make_image_block,
)
from tests.pdf_builders import build_tagged_pdf

if TYPE_CHECKING:
    from pathlib import Path

    from doc_checker.compare.drafts import Draft

PDF = "doc.pdf"
THRESHOLDS = AuditThresholds()
HOME = (HtmlPageFacts("index.html", "en", "Home"),)


def _found(drafts: list[Draft]) -> list[tuple[str, int | None]]:
    return [(d.rule, d.ref.source.page) for d in drafts if d.ref]


def test_untagged_pdf_without_language_nor_title() -> None:
    """One finding for each missing declaration."""
    doc = make_audit_document(
        make_block("Text.", file=PDF),
        fmt="pdf",
        tagged=False,
        declared_lang=None,
    )
    assert [d.rule for d in accessibility_drafts(doc, THRESHOLDS)] == [
        "a11y.pdf_untagged",
        "a11y.lang_missing",
        "a11y.title_missing",
    ]


def test_tagged_flag_without_tree_is_untagged() -> None:
    """A tagged flag without a structure tree is an untagged PDF."""
    doc = make_audit_document(
        make_block("Text.", file=PDF),
        fmt="pdf",
        tagged=True,
        title="Guide",
        tags=TagTree((), 0, NO_STRUCTURE_TREE),
    )
    rules = [d.rule for d in accessibility_drafts(doc, THRESHOLDS)]
    assert rules == ["a11y.pdf_untagged"]


def test_pdf_figures_and_tables_of_the_tree() -> None:
    """Missing or file name alt texts, tables without header cells."""
    tags = TagTree(
        (
            TaggedElement("figure", "Figure", 1),
            TaggedElement("figure", "Figure", 2, alt="setup.png"),
            TaggedElement("figure", "Figure", 3, alt=" ", actual="Flow"),
            TaggedElement("figure", "Figure", 3, alt="Installer steps"),
            TaggedElement("table", "Table", 4),
            TaggedElement("table", "Table", 5, headers=True),
        ),
        items=20,
    )
    doc = make_audit_document(
        make_block("Text.", file=PDF),
        fmt="pdf",
        tagged=True,
        title="Guide",
        tags=tags,
    )
    assert _found(accessibility_drafts(doc, THRESHOLDS)) == [
        ("a11y.alt_missing", 1),
        ("a11y.alt_filename", 2),
        ("a11y.table_headers", 4),
    ]


def test_html_images_without_alt_or_with_a_file_name() -> None:
    """An img without alt, an alt naming the file, an untitled SVG."""
    images = [
        ImageRef("_images/a.png", "png", "1" * 32),
        ImageRef("_images/b.png", "png", "2" * 32, alt="_images/b.png"),
        ImageRef("", "svg", "3" * 32, vector=True),
        ImageRef("_images/c.png", "png", "4" * 32, alt=""),
        ImageRef("_images/d.png", "png", "5" * 32, alt="Installer steps"),
    ]
    doc = make_audit_document(
        make_block("Text."),
        *map(make_image_block, images),
        html_pages=HOME,
    )
    assert [d.rule for d in accessibility_drafts(doc, THRESHOLDS)] == [
        "a11y.alt_missing",
        "a11y.alt_filename",
        "a11y.svg_title",
    ]


def test_pages_without_language_or_title() -> None:
    """Each page tells its own language and title."""
    pages = (
        *HOME,
        HtmlPageFacts("a.html", None, "A"),
        HtmlPageFacts("b.html", "<x>", None),
    )
    doc = make_audit_document(
        make_block("Text."),
        make_block("A text.", file="a.html"),
        make_block("B text.", file="b.html"),
        parts=["index.html", "a.html", "b.html"],
        html_pages=pages,
    )
    found = [
        (d.rule, d.ref.source.file, d.detail)
        for d in accessibility_drafts(doc, THRESHOLDS)
        if d.ref
    ]
    assert found == [
        ("a11y.lang_missing", "a.html", "the page declares no language"),
        (
            "a11y.lang_missing",
            "b.html",
            'the page declares "<x>", which is not a language tag',
        ),
        ("a11y.title_missing", "b.html", "the page has no title"),
    ]


def test_declared_language_differs_from_the_text() -> None:
    """French prose declared English is reported; code does not vote."""
    french = [
        make_block(link_text(text))
        for section in SAMPLE_FR.sections
        for text in section.paragraphs
    ]
    code = make_block("for x in data:\n    print(x)", BlockKind.CODE)
    doc = make_audit_document(*french, code, html_pages=HOME)
    rules = [d.rule for d in accessibility_drafts(doc, THRESHOLDS)]
    assert rules == ["a11y.lang_mismatch"]
    fine = replace(doc, declared_lang="fr", lang="fr")
    assert not accessibility_drafts(fine, THRESHOLDS)


def test_unmapped_characters() -> None:
    """Minor below one percent of the characters, major from it."""
    pages = (
        PdfPageFacts(1, "i", 1000, 0, scanned=False),
        PdfPageFacts(2, "1", 1000, 5, scanned=False),
        PdfPageFacts(3, "2", 1000, 3, scanned=False),
    )
    doc = make_audit_document(
        make_block("Text.", file=PDF),
        fmt="pdf",
        tagged=True,
        title="Guide",
        pdf_pages=pages,
    )
    (draft,) = accessibility_drafts(doc, THRESHOLDS)
    assert (draft.rule, draft.severity, draft.ref.source.page) == (
        "a11y.unmapped_chars",
        None,
        2,
    )
    assert draft.detail == (
        "8 of 3000 characters have no Unicode mapping, on page(s) 1, 2"
    )
    many = (PdfPageFacts(1, "1", 1000, 40, scanned=False),)
    (draft,) = accessibility_drafts(replace(doc, pdf_pages=many), THRESHOLDS)
    assert draft.severity is Severity.MAJOR


def test_scanned_pages_one_finding_each() -> None:
    """Each scanned page is a major finding."""
    pages = (
        PdfPageFacts(1, "1", 0, 0, scanned=True),
        PdfPageFacts(2, "2", 300, 0, scanned=False),
        PdfPageFacts(3, "3", 0, 0, scanned=True),
    )
    doc = make_audit_document(
        make_block("Text.", file=PDF),
        fmt="pdf",
        tagged=True,
        title="Guide",
        pdf_pages=pages,
    )
    assert _found(accessibility_drafts(doc, THRESHOLDS)) == [
        ("a11y.scanned_page", 1),
        ("a11y.scanned_page", 3),
    ]


def test_links_without_text_or_with_vague_text() -> None:
    """Empty links of a page, vague texts of HTML links and PDF links."""
    block = replace(
        make_block("For details click here or see the guide."),
        links=(
            Link("click here", "https://example.org", url="//example.org"),
            Link("the guide", "guide.html", page="guide.html", found=True),
        ),
    )
    pages = (replace(HOME[0], empty_links=(Spot("#install", 12),)),)
    doc = make_audit_document(block, html_pages=pages)
    drafts = accessibility_drafts(doc, THRESHOLDS)
    assert [(d.rule, d.pattern) for d in drafts] == [
        ("a11y.link_empty", "empty"),
        ("a11y.link_text", "click here"),
    ]
    pdf_pages = (
        PdfPageFacts(
            1,
            "1",
            100,
            0,
            scanned=False,
            links=(PdfLink("Click here.", "goto", "page 1", valid=True),),
        ),
    )
    pdf = make_audit_document(
        make_block("Text.", file=PDF),
        fmt="pdf",
        tagged=True,
        title="Guide",
        pdf_pages=pdf_pages,
    )
    assert [d.rule for d in accessibility_drafts(pdf, THRESHOLDS)] == [
        "a11y.link_text",
    ]


def _pdf(**facts: object) -> list[Draft]:
    """Return the drafts of a PDF, tagged with a title but for its facts."""
    doc = make_audit_document(
        make_block("Text.", file=PDF),
        fmt="pdf",
        **({"tagged": True, "title": "Guide"} | facts),
    )
    return accessibility_drafts(doc, THRESHOLDS)


def test_a_tree_read_in_part_is_not_untagged() -> None:
    """Only a missing tree is untagged; the elements read are checked."""
    for problem in ("encrypted", "too many items"):
        tags = TagTree((TaggedElement("figure", "Figure", 2),), 9, problem)
        assert _found(_pdf(tags=tags)) == [("a11y.alt_missing", 2)]
    assert _pdf(tags=TagTree((), 0, "refused")) == []
    assert _pdf(tags=None) == []


def test_document_findings_point_at_the_document() -> None:
    """The findings of a whole document show its name, without pattern."""
    pages = (
        PdfPageFacts(1, "1", 100, 0, scanned=False),
        PdfPageFacts(2, "ii", 100, 2, scanned=False),
    )
    drafts = _pdf(
        tagged=False,
        declared_lang=None,
        title=None,
        pdf_pages=pages,
    )
    sides = [
        (d.rule, d.ref.source.file, d.ref.source.page, d.ref.text, d.pattern)
        for d in drafts
        if d.ref
    ]
    assert sides == [
        ("a11y.pdf_untagged", PDF, None, PDF, None),
        ("a11y.lang_missing", PDF, None, PDF, None),
        ("a11y.title_missing", PDF, None, PDF, None),
        ("a11y.unmapped_chars", PDF, 2, PDF, None),
    ]
    assert [d.detail for d in drafts[:3]] == [
        (
            "the PDF has no structure tree: assistive technologies read it"
            " without headings, lists, tables or figures"
        ),
        "the PDF declares no language",
        "the PDF has no title in its metadata",
    ]
    assert drafts[3].detail == (
        "2 of 200 characters have no Unicode mapping, on page(s) ii"
    )


def test_language_mismatch_points_at_the_first_page() -> None:
    """The mismatch of a site is shown at its first page."""
    french = [
        make_block(link_text(text), file="b.html")
        for section in SAMPLE_FR.sections
        for text in section.paragraphs
    ]
    doc = make_audit_document(
        make_heading("Accueil", 1, file="a.html"),
        *french,
        parts=["a.html", "b.html"],
        declared_lang="en-US",
    )
    (draft,) = accessibility_drafts(doc, THRESHOLDS)
    assert (draft.rule, draft.ref.source.file, draft.ref.text) == (
        "a11y.lang_mismatch",
        "a.html",
        "a.html",
    )
    assert draft.pattern is None
    assert draft.detail == (
        'the document declares "en-US" but its text reads as French'
    )


def test_an_upper_case_declared_language_is_read() -> None:
    """The case of the declared language does not matter."""
    french = [
        make_block(link_text(text))
        for section in SAMPLE_FR.sections
        for text in section.paragraphs
    ]
    doc = make_audit_document(*french, declared_lang="EN-gb")
    (draft,) = accessibility_drafts(doc, THRESHOLDS)
    assert draft.detail == (
        'the document declares "EN-gb" but its text reads as French'
    )
    fine = replace(doc, declared_lang="FR-ca")
    assert accessibility_drafts(fine, THRESHOLDS) == []


def test_only_prose_tells_the_language() -> None:
    """Tables, code and an undeclared or other language are not judged."""
    english = " ".join(["the cat and the dog is on the mat of it"] * 4)
    table = make_block(english, BlockKind.TABLE)
    assert (
        accessibility_drafts(
            make_audit_document(table, declared_lang="fr", lang="fr"),
            THRESHOLDS,
        )
        == []
    )
    prose = make_block(english)
    (draft,) = accessibility_drafts(
        make_audit_document(prose, declared_lang="fr", lang="fr"),
        THRESHOLDS,
    )
    assert draft.detail == (
        'the document declares "fr" but its text reads as English'
    )
    for declared in (None, "de"):
        doc = make_audit_document(prose, declared_lang=declared, lang="fr")
        assert accessibility_drafts(doc, THRESHOLDS) == []


def test_only_the_first_prose_blocks_tell_the_language() -> None:
    """Code does not fill the sample; prose past it does not vote."""
    french = [
        make_block(link_text(text))
        for section in SAMPLE_FR.sections
        for text in section.paragraphs
    ]
    code = [
        make_block(f"x = {k}", BlockKind.CODE)
        for k in range(LANG_SAMPLE_BLOCKS)
    ]
    rules = [
        d.rule
        for d in accessibility_drafts(
            make_audit_document(*code, *french),
            THRESHOLDS,
        )
    ]
    assert rules == ["a11y.lang_mismatch"]
    notes = [make_block("Note.") for _ in range(LANG_SAMPLE_BLOCKS)]
    late = make_audit_document(*notes, *french)
    assert accessibility_drafts(late, THRESHOLDS) == []


def test_prose_sample_joins_the_first_prose_blocks() -> None:
    """Headings and paragraphs fill the sample, code and tables do not."""
    doc = make_audit_document(
        make_heading("Title", 1),
        make_block("x = 1", BlockKind.CODE),
        make_block("a | b", BlockKind.TABLE),
        *(make_block(f"Note {k}.") for k in range(LANG_SAMPLE_BLOCKS)),
    )
    sample = prose_sample(doc)
    assert sample.startswith("Title Note 0. Note 1.")
    assert sample.endswith(f" Note {LANG_SAMPLE_BLOCKS - 2}.")
    assert not prose_sample(make_audit_document())


def test_an_empty_href_is_quoted() -> None:
    """A link without a target is quoted as an empty string."""
    page = replace(HOME[0], empty_links=(Spot("", 4),))
    doc = make_audit_document(make_block("Text."), html_pages=(page,))
    (draft,) = accessibility_drafts(doc, THRESHOLDS)
    assert draft.detail == 'the link to "" has neither text nor label'


def test_a_blank_page_language_is_no_language() -> None:
    """A blank lang declares no language."""
    pages = (HtmlPageFacts("index.html", "", "Home"),)
    doc = make_audit_document(make_block("Text."), html_pages=pages)
    (draft,) = accessibility_drafts(doc, THRESHOLDS)
    assert (draft.rule, draft.detail, draft.pattern) == (
        "a11y.lang_missing",
        "the page declares no language",
        "lang",
    )


def test_page_findings_quote_the_page() -> None:
    """Page findings show the page, a table its first row, a link its href."""
    page = HtmlPageFacts(
        "a.html",
        "en",
        None,
        empty_links=(Spot("#install", 12),),
        bare_tables=(Spot("Option | Default", 30),),
    )
    doc = make_audit_document(
        make_block("Text.", file="a.html"),
        parts=["a.html"],
        html_pages=(page,),
    )
    found = [
        (
            d.rule,
            d.ref.source.file,
            d.ref.source.line,
            d.ref.text,
            d.ref.kind,
            d.detail,
            d.pattern,
        )
        for d in accessibility_drafts(doc, THRESHOLDS)
        if d.ref
    ]
    assert found == [
        (
            "a11y.title_missing",
            "a.html",
            None,
            "a.html",
            None,
            "the page has no title",
            "title",
        ),
        (
            "a11y.table_headers",
            "a.html",
            30,
            "Option | Default",
            BlockKind.TABLE,
            "the table has no header cell",
            "table",
        ),
        (
            "a11y.link_empty",
            "a.html",
            12,
            "#install",
            None,
            "the link to #install has neither text nor label",
            "empty",
        ),
    ]


def test_image_findings_show_the_image() -> None:
    """Image findings show the image, and quote an alt that names a file."""
    named = ImageRef("_images/b.png", "png", "2" * 32, alt="_images/b.png")
    svg = ImageRef("", "svg", "3" * 32, vector=True)
    titled = replace(svg, digest="4" * 32, alt="Pipeline")
    doc = make_audit_document(
        *map(make_image_block, (named, svg, titled)),
        html_pages=HOME,
    )
    drafts = accessibility_drafts(doc, THRESHOLDS)
    assert [(d.rule, d.ref.image, d.detail, d.pattern) for d in drafts] == [
        (
            "a11y.alt_filename",
            named,
            'the alternative text "_images/b.png" is a file name',
            "file name",
        ),
        (
            "a11y.svg_title",
            svg,
            "the drawing has no title nor label",
            "svg",
        ),
    ]


def test_alt_texts_of_an_img() -> None:
    """A missing alt is major; an alt with words is no file name."""
    missing = ImageRef("_images/a.svg", "svg", "1" * 32, vector=True)
    worded = ImageRef("a.png", "png", "2" * 32, alt="Screenshot of a.png")
    doc = make_audit_document(
        *map(make_image_block, (missing, worded)),
        html_pages=HOME,
    )
    (draft,) = accessibility_drafts(doc, THRESHOLDS)
    assert (draft.rule, draft.detail, draft.pattern) == (
        "a11y.alt_missing",
        "the image has no alternative text",
        "img",
    )


def test_file_name_alt_texts() -> None:
    """A path that ends with an image suffix, without blanks."""
    for alt in ("setup.png", "_images/x.JPEG", r"fig\a.tif", "a-b.svg"):
        assert FILE_NAME_ALT.match(alt), alt
    for alt in ("my setup.png", "setup", "setup.png now", "Figure 1."):
        assert not FILE_NAME_ALT.match(alt), alt


def test_pdf_images_are_judged_by_the_tree_only() -> None:
    """The image blocks of a PDF have no alt of their own to check."""
    image = make_image_block(ImageRef("", "png", "1" * 32), file=PDF)
    doc = make_audit_document(image, fmt="pdf", title="Guide")
    assert accessibility_drafts(doc, THRESHOLDS) == []


def test_tagged_findings_show_the_type_or_the_alt() -> None:
    """A figure or a table of the tree is shown by its type, at its page."""
    tags = TagTree(
        (
            TaggedElement("figure", "Figure", 1),
            TaggedElement("figure", "Figure", None, alt="setup.png"),
            TaggedElement("table", "Table", 4),
        ),
        items=5,
    )
    found = [
        (d.ref.source.page, d.ref.text, d.ref.kind, d.detail, d.pattern)
        for d in _pdf(tags=tags)
        if d.ref
    ]
    assert found == [
        (
            1,
            "Figure",
            BlockKind.IMAGE,
            "the figure has no alternative text",
            "Figure",
        ),
        (
            None,
            "setup.png",
            BlockKind.IMAGE,
            'the alternative text "setup.png" is a file name',
            "file name",
        ),
        (4, "Table", BlockKind.TABLE, "the table has no header cell", "table"),
    ]


def test_scanned_page_findings() -> None:
    """A scanned page is shown by its label."""
    pages = (PdfPageFacts(1, "iv", 0, 0, scanned=True),)
    (draft,) = _pdf(pdf_pages=pages)
    assert (
        draft.ref.source.page,
        draft.ref.text,
        draft.detail,
        draft.pattern,
    ) == (1, "page iv", "the page is an image without text", "scanned")


def test_unmapped_pages_are_listed_up_to_a_limit() -> None:
    """The first pages touched are listed, then an ellipsis."""
    pages = tuple(
        PdfPageFacts(n, str(n), 100, 1, scanned=False)
        for n in range(1, UNMAPPED_PAGES_SHOWN + 3)
    )
    (draft,) = _pdf(pdf_pages=pages)
    labels = ", ".join(str(n) for n in range(1, UNMAPPED_PAGES_SHOWN + 1))
    assert draft.detail == (
        f"12 of 1200 characters have no Unicode mapping, on page(s) {labels}"
        f", {ELLIPSIS}"
    )


def test_generic_link_texts_by_their_loose_key() -> None:
    """Case, accents and punctuation do not matter; an empty text is fine."""
    assert {
        "click here",
        "cliquez ici",
        "en savoir plus",
    } <= GENERIC_LINK_TEXTS
    links = (
        PdfLink("", "uri", "https://example.org", valid=None),
        PdfLink("Cliquez ICI !", "uri", "https://example.org", valid=None),
        PdfLink("the guide", "goto", "page 2", valid=True),
        PdfLink("Read more...", "goto", "page 2", valid=True),
    )
    pages = (PdfPageFacts(3, "3", 100, 0, scanned=False, links=links),)
    found = [
        (d.ref.source.page, d.ref.text, d.detail, d.pattern)
        for d in _pdf(pdf_pages=pages)
        if d.ref
    ]
    assert found == [
        (
            3,
            "Cliquez ICI !",
            'the text "Cliquez ICI !" does not tell where the link leads',
            "cliquez ici",
        ),
        (
            3,
            "Read more...",
            'the text "Read more..." does not tell where the link leads',
            "read more",
        ),
    ]


def test_html_generic_link_marks_its_text() -> None:
    """The side of a vague link marks its text in its block."""
    block = replace(
        make_block("To install it, read more below."),
        links=(Link("read more", "#install", page="index.html", found=True),),
    )
    doc = make_audit_document(block, html_pages=HOME)
    (draft,) = accessibility_drafts(doc, THRESHOLDS)
    assert [(s.op, s.text) for s in draft.ref.segments] == [
        ("equal", "To install it, "),
        ("mark", "read more"),
        ("equal", " below."),
    ]


def _marks(text: str, *links: str) -> list[list[tuple[str, str]]]:
    """Return the segments of the vague links of a block, one list each."""
    block = replace(
        make_block(text),
        links=tuple(
            Link(t, "a.html", page="a.html", found=True) for t in links
        ),
    )
    doc = make_audit_document(block, html_pages=HOME)
    return [
        [(s.op, s.text) for s in d.ref.segments]
        for d in accessibility_drafts(doc, THRESHOLDS)
        if d.ref
    ]


def test_a_vague_link_marks_whole_words() -> None:
    """The text of a link is marked as whole words, not inside a word."""
    assert _marks("There is a guide here.", "here") == [
        [("equal", "There is a guide "), ("mark", "here"), ("equal", ".")],
    ]
    assert _marks("Voici le guide, ici.", "ici") == [
        [("equal", "Voici le guide, "), ("mark", "ici"), ("equal", ".")],
    ]
    assert _marks("Please click\nhere now.", "click here") == [
        [("equal", "Please "), ("mark", "click\nhere"), ("equal", " now.")],
    ]


def test_links_of_one_text_mark_their_own_occurrence() -> None:
    """The k-th link of a text marks the k-th occurrence of that text."""
    assert _marks("Two links: here and here.", "here", "here") == [
        [("equal", "Two links: "), ("mark", "here"), ("equal", " and here.")],
        [("equal", "Two links: here and "), ("mark", "here"), ("equal", ".")],
    ]
    assert _marks("Only here.", "here", "here") == [
        [("equal", "Only "), ("mark", "here"), ("equal", ".")],
        [("equal", "Only "), ("mark", "here"), ("equal", ".")],
    ]
    assert _marks("Somewhere.", "here") == [[("equal", "Somewhere.")]]


def test_links_of_a_huge_block_are_judged_quickly() -> None:
    """Only a vague link is marked, and many of them in linear time."""
    entries = [f"func_{k:06d}() (in module mod)" for k in range(30_000)]
    index = make_block("\n".join(entries), BlockKind.TABLE)
    words = [f"entry{k:06d}" for k in range(80_000)]
    vague = ["here"] * 20_000
    blocks = [
        replace(index, links=tuple(Link(e, "api.html") for e in entries)),
        replace(
            make_block(", ".join(words), BlockKind.INDEX),
            links=tuple(Link(w, "x.html") for w in words),
        ),
        replace(
            make_block(", ".join(vague)),
            links=tuple(Link(v, "x.html") for v in vague),
        ),
    ]
    for block, count in zip(blocks, (0, 0, 20_000), strict=True):
        doc = make_audit_document(block, declared_lang=None)
        start = time.perf_counter()
        drafts = accessibility_drafts(doc, THRESHOLDS)
        # Under 0.8 s; a search of each link text through its block would
        # take minutes.
        assert time.perf_counter() - start < 3.0
        assert len(drafts) == count
    ends = [
        [(s.op, s.text) for s in d.ref.segments[1:]]
        for d in drafts[-2:]
        if d.ref
    ]
    assert ends == [
        [("mark", "here"), ("equal", ", here")],
        [("mark", "here")],
    ]


def test_quoted_texts_are_cut() -> None:
    """A long alt, href or language is cut in the details."""
    long = "x" * (PATTERN_MAX_CHARS + 20)
    cut = "x" * PATTERN_MAX_CHARS + ELLIPSIS
    page = HtmlPageFacts(
        "index.html",
        long,
        "Home",
        empty_links=(Spot(long, 3),),
    )
    image = ImageRef("a.png", "png", "1" * 32, alt=f"{long}.png")
    doc = make_audit_document(make_image_block(image), html_pages=(page,))
    details = [d.detail for d in accessibility_drafts(doc, THRESHOLDS)]
    assert details == [
        f'the page declares "{cut}", which is not a language tag',
        f"the link to {cut} has neither text nor label",
        f'the alternative text "{cut}" is a file name',
    ]


def test_audit_runs_accessibility_after_structure() -> None:
    """The family runs when asked, its findings after those of structure."""
    doc = make_audit_document(
        make_block("See Section ?? now."),
        html_pages=(HtmlPageFacts("index.html", None, "Home"),),
    )
    rules = [f.rule for f in audit_document(doc, Config()).findings]
    assert rules == ["structure.unresolved_reference", "a11y.lang_missing"]
    only = audit_document(doc, Config(checks=("a11y",))).findings
    assert [f.rule for f in only] == ["a11y.lang_missing"]
    none = audit_document(doc, Config(checks=("structure",))).findings
    assert [f.rule for f in none] == ["structure.unresolved_reference"]


def test_the_engine_passes_its_thresholds() -> None:
    """The thresholds of the configuration reach the checks."""
    pages = (PdfPageFacts(1, "1", 1000, 5, scanned=False),)
    doc = make_audit_document(
        make_block("Text.", file=PDF),
        fmt="pdf",
        title="Guide",
        pdf_pages=pages,
    )
    cfg = Config(checks=("a11y",))
    (finding,) = audit_document(doc, cfg).findings
    assert finding.severity is Severity.MINOR
    strict = replace(cfg, audit=replace(THRESHOLDS, unmapped_major=0.005))
    (finding,) = audit_document(doc, strict).findings
    assert finding.severity is Severity.MAJOR


def test_unmapped_severity_follows_its_threshold() -> None:
    """The share of unmapped characters that is major is configurable."""
    pages = (PdfPageFacts(1, "1", 1000, 5, scanned=False),)
    doc = make_audit_document(
        make_block("Text.", file=PDF),
        fmt="pdf",
        title="Guide",
        pdf_pages=pages,
    )
    strict = replace(THRESHOLDS, unmapped_major=0.005)
    (draft,) = accessibility_drafts(doc, strict)
    assert draft.severity is Severity.MAJOR


def test_tagged_pdf_files(tmp_path: Path) -> None:
    """A complete tagged PDF is clean; its faults are found in the file."""
    clean = build_tagged_pdf(tmp_path / "clean.pdf")
    doc = load_document(clean, Config(), audit=True)
    assert accessibility_drafts(doc, THRESHOLDS) == []
    faulty = build_tagged_pdf(
        tmp_path / "faulty.pdf",
        figure_alt=None,
        table_headers=False,
        lang=None,
        title=None,
    )
    doc = load_document(faulty, Config(), audit=True)
    assert _found(accessibility_drafts(doc, THRESHOLDS)) == [
        ("a11y.lang_missing", None),
        ("a11y.title_missing", None),
        ("a11y.alt_missing", 1),
        ("a11y.table_headers", 1),
    ]
    untagged = build_tagged_pdf(tmp_path / "untagged.pdf", tagged=False)
    doc = load_document(untagged, Config(), audit=True)
    assert [d.rule for d in accessibility_drafts(doc, THRESHOLDS)] == [
        "a11y.pdf_untagged",
    ]
