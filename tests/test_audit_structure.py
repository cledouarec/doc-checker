# Copyright 2026 Christophe Le Douarec
"""Tests for the structure checks of the audit."""

import time
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from doc_checker.audit.structure import structure_drafts
from doc_checker.config import Config
from doc_checker.extract.loader import load_document
from doc_checker.images.refs import raster_ref
from doc_checker.model.document import BlockKind, SourceRef
from doc_checker.model.facts import (
    HtmlPageFacts,
    OutlineEntry,
    Spot,
    TaggedElement,
    TagTree,
)
from tests.builders import (
    FIGURES_EN,
    SAMPLE_EN,
    build_docusaurus_like,
    build_sphinx_like,
    link_text,
    make_audit_document,
    make_block,
    make_heading,
    make_image_block,
    map_texts,
)
from tests.image_builders import drawing, encode
from tests.pdf_builders import build_reportlab_pdf

if TYPE_CHECKING:
    from pathlib import Path

    from doc_checker.compare.drafts import Draft
    from doc_checker.model.document import Block, Document

PDF = "doc.pdf"
ARROW = "\U00002192"


def _of(doc: Document, rule: str) -> list[Draft]:
    return [d for d in structure_drafts(doc) if d.rule == rule]


def test_skipped_heading_level_in_a_page() -> None:
    """An h4 after an h2 of its page skips a level; a page starts over."""
    doc = make_audit_document(
        make_heading("Guide", 1),
        make_block("Intro text of the guide."),
        make_heading("Install", 2),
        make_block("Install text."),
        make_heading("Deep", 4),
        make_block("Deep text."),
        make_heading("Other", 3, file="b.html"),
        make_block("Other text.", file="b.html"),
        parts=["index.html", "b.html"],
    )
    drafts = _of(doc, "structure.heading_skipped")
    assert [(d.ref.text, d.detail, d.pattern) for d in drafts] == [
        ("Deep", "h2 is followed by h4", f"h2{ARROW}h4"),
    ]


def test_empty_sections() -> None:
    """A heading followed by one of its level, or ending its page."""
    doc = make_audit_document(
        make_heading("Guide", 1),
        make_heading("Install", 2),
        make_block("Install text."),
        make_heading("Empty", 2),
        make_heading("Next", 2),
        make_block("Next text."),
        make_heading("Last", 2),
        make_heading("Title only", 1, file="b.html"),
        parts=["index.html", "b.html"],
    )
    drafts = _of(doc, "structure.empty_section")
    assert [(d.ref.text, d.detail) for d in drafts] == [
        ("Empty", "no content before the next heading of its level"),
        ("Last", "no content before the end of the page"),
    ]


def test_pdf_empty_sections_count_outline_headings_only() -> None:
    """A style heading is content; a title before its chapter is fine."""
    doc = make_audit_document(
        make_heading("Orbit Sync", 1, file=PDF, page=1),
        make_heading("Orbit Sync", 1, file=PDF, page=1, outline=True),
        make_block("Text of the chapter.", file=PDF),
        make_heading("Empty", 1, file=PDF, page=2, outline=True),
        make_heading("Next", 1, file=PDF, page=2, outline=True),
        make_block("More text.", file=PDF),
        fmt="pdf",
    )
    assert [d.ref.text for d in _of(doc, "structure.empty_section")] == [
        "Empty",
    ]


def test_identical_sibling_headings() -> None:
    """Two titles of one parent are reported once; other parents differ."""
    doc = make_audit_document(
        make_heading("Guide", 1),
        make_heading("Projects", 2),
        make_block("First text."),
        make_heading("projects", 2),
        make_block("Second text."),
        make_heading("Reference", 1),
        make_heading("Projects", 2),
        make_block("Third text."),
    )
    drafts = _of(doc, "structure.duplicate_heading")
    assert [(d.ref.text, d.detail) for d in drafts] == [
        ("projects", 'same title as an earlier heading under "Guide"'),
    ]


def test_pdf_duplicates_come_from_the_outline() -> None:
    """Bookmarks of one parent with one title; other parents differ."""
    outline = (
        OutlineEntry(0, "Guide", 1),
        OutlineEntry(1, "Overview", 1),
        OutlineEntry(1, "Overview", 2),
        OutlineEntry(0, "Reference", 3),
        OutlineEntry(1, "Overview", 3),
    )
    doc = make_audit_document(
        make_block("Text.", file=PDF),
        fmt="pdf",
        outline=outline,
    )
    drafts = _of(doc, "structure.duplicate_heading")
    assert [(d.ref.text, d.ref.source.page) for d in drafts] == [
        ("Overview", 2),
    ]


def test_pdf_skipped_levels_come_from_the_tags() -> None:
    """Tagged H2 then H4 skip a level, quoted by the heading of the page."""
    tags = TagTree(
        (
            TaggedElement("heading", "H1", 1, 1),
            TaggedElement("heading", "H2", 2, 2),
            TaggedElement("heading", "H4", 2, 4),
        ),
        items=9,
    )
    doc = make_audit_document(
        make_heading("Guide", 1, file=PDF, page=1, outline=True),
        make_block("Text.", file=PDF),
        make_heading("Install", 2, file=PDF, page=2, outline=True),
        make_block("Text.", file=PDF),
        make_heading("Deep", 3, file=PDF, page=2),
        make_block("Deep text.", file=PDF),
        fmt="pdf",
        tags=tags,
    )
    drafts = _of(doc, "structure.heading_skipped")
    assert [(d.ref.text, d.ref.source.page, d.detail) for d in drafts] == [
        ("Deep", 2, "H2 is followed by H4"),
    ]


def test_unresolved_references_in_texts() -> None:
    """LaTeX question marks, missing citations, literal Markdown links."""
    doc = make_audit_document(
        make_block("See Section ?? for the limits."),
        make_block("As shown in [?], the cache helps."),
        make_block("See [the details][missing] for more."),
        make_block("Really?? Use a[0][1] in the loop."),
        make_block("print(a ?? b)", BlockKind.CODE),
    )
    drafts = _of(doc, "structure.unresolved_reference")
    marked = [
        [s.text for s in d.ref.segments if s.op == "mark"] for d in drafts
    ]
    assert [d.pattern for d in drafts] == ["??", "[?]", "[text][label]"]
    assert marked == [["??"], ["[?]"], ["[the details][missing]"]]


def test_question_marks_in_inline_code_are_not_references() -> None:
    """A ``??`` that inline code holds is an operator, not a reference."""
    block = make_block("Use a ?? b to pick.")
    coded = replace(block, code_spans=("a ?? b",))
    rule = "structure.unresolved_reference"
    assert not _of(make_audit_document(coded), rule)
    assert len(_of(make_audit_document(block), rule)) == 1
    both = replace(
        make_block("Use a ?? b, see Section ?? and [x][y]."),
        code_spans=("a ?? b", "[x][y]"),
    )
    (draft,) = _of(make_audit_document(both), rule)
    assert [s.text for s in draft.ref.segments if s.op == "mark"] == ["??"]
    assert draft.detail == "1 unresolved reference(s) in this block"


@pytest.mark.parametrize(
    "text",
    [
        '"fd::<infd>[,<outfd>][/<anything>]" (as URL)',
        "objects/[0-9a-f][0-9a-f]",
        'Only valid setting is "--unified=??" to set it.',
        "The operator ??= assigns a default.",
    ],
)
def test_syntax_quoted_in_prose_is_no_reference(text: str) -> None:
    """Command syntax and character classes are not unresolved references."""
    doc = make_audit_document(make_block(text))
    assert not _of(doc, "structure.unresolved_reference")


@pytest.mark.parametrize(
    "text",
    [
        "See the [setup guide][setup] first.",
        "As shown in figure ??, the server.",
        "See [e-mail][x] for the address.",
        "See [a-b-c][d] for the range.",
    ],
)
def test_unresolved_references_are_still_found(text: str) -> None:
    """A Markdown reference without target and a LaTeX ?? are found."""
    doc = make_audit_document(make_block(text))
    assert [d.rule for d in structure_drafts(doc)] == [
        "structure.unresolved_reference",
    ]


def test_an_unclosed_bracket_is_read_in_linear_time() -> None:
    """A 200 kB text after an open bracket does not make the scan quadratic."""
    for text in ("[" + "a" * 200_000, "[a][" * 50_000):
        doc = make_audit_document(make_block(text))
        start = time.process_time()
        _of(doc, "structure.unresolved_reference")
        assert time.process_time() - start < 2.0


def test_markdown_references_are_not_sought_in_pdf() -> None:
    """Brackets of a PDF are text: a matrix, a quotation."""
    doc = make_audit_document(
        make_block("See [the details][missing] for more.", file=PDF),
        fmt="pdf",
    )
    assert not _of(doc, "structure.unresolved_reference")


def test_figures_without_caption_where_the_document_captions() -> None:
    """Only a document that captions a figure is checked, as information."""
    image = make_image_block(
        raster_ref(encode(drawing("setup")), name="a.png", source=None),
    )
    caption = make_block("The steps of the installer.", BlockKind.CAPTION)
    captioned = make_audit_document(image, caption, make_block("Text."), image)
    assert len(_of(captioned, "structure.figure_caption")) == 1
    bare = make_audit_document(make_block("Text."), image)
    assert not _of(bare, "structure.figure_caption")


def test_orphan_pages() -> None:
    """A page with content no other page links to; titles and entry not."""
    pages = (
        HtmlPageFacts(
            "index.html",
            "en",
            "Home",
            links_to=frozenset({"guide.html"}),
        ),
        HtmlPageFacts(
            "guide.html",
            "en",
            "Guide",
            links_to=frozenset({"index.html"}),
        ),
        HtmlPageFacts("old.html", "en", "Old"),
        HtmlPageFacts("tags.html", "en", "Tags"),
    )
    doc = make_audit_document(
        make_block("Home text."),
        make_block("Guide text.", file="guide.html"),
        make_block("Old text.", file="old.html"),
        make_heading("Tags", 1, file="tags.html"),
        parts=["index.html", "guide.html", "old.html", "tags.html"],
        html_pages=pages,
    )
    drafts = _of(doc, "structure.orphan_page")
    assert [d.ref.source.file for d in drafts] == ["old.html"]


def test_the_regular_expression_match_is_the_mark() -> None:
    """A question mark glued to a word comes first but is not marked."""
    doc = make_audit_document(make_block("Really?? See Section ?? now."))
    (draft,) = _of(doc, "structure.unresolved_reference")
    assert [(s.op, s.text) for s in draft.ref.segments] == [
        ("equal", "Really?? See Section "),
        ("mark", "??"),
        ("equal", " now."),
    ]
    assert draft.detail == "1 unresolved reference(s) in this block"


def test_one_draft_per_block_and_pattern() -> None:
    """Two marks of a block make one draft that counts them."""
    doc = make_audit_document(
        make_block("See Section ?? and Section ?? then [?]."),
    )
    drafts = _of(doc, "structure.unresolved_reference")
    assert [(d.pattern, d.detail) for d in drafts] == [
        ("??", "2 unresolved reference(s) in this block"),
        ("[?]", "1 unresolved reference(s) in this block"),
    ]


def test_references_left_unresolved_in_the_page() -> None:
    """A dead reference of the DOM is quoted in the block that shows it."""
    pages = (
        HtmlPageFacts(
            "index.html",
            "en",
            "Home",
            unresolved=(Spot("missing-label", 12),),
        ),
    )
    doc = make_audit_document(
        _at(make_block("Intro."), 10),
        _at(make_block("See missing-label for the details."), 11),
        html_pages=pages,
    )
    (draft,) = _of(doc, "structure.unresolved_reference")
    assert (draft.pattern, draft.detail) == (
        "xref",
        'reference "missing-label" not resolved',
    )
    marked = [s.text for s in draft.ref.segments if s.op == "mark"]
    assert (draft.ref.text, marked) == (
        "See missing-label for the details.",
        ["missing-label"],
    )


def test_a_reference_of_no_block_is_quoted_alone() -> None:
    """A dead reference that no block shows keeps its page and its line."""
    pages = (
        HtmlPageFacts(
            "index.html",
            "en",
            "Home",
            unresolved=(Spot("lost-label", 7),),
        ),
    )
    doc = make_audit_document(make_block("Intro."), html_pages=pages)
    (draft,) = _of(doc, "structure.unresolved_reference")
    assert (draft.ref.text, draft.ref.source.line) == ("lost-label", 7)


def test_pdf_skipped_levels_quote_the_type_when_counts_differ() -> None:
    """Without one extracted heading per tagged one, the type is quoted."""
    tags = TagTree(
        (
            TaggedElement("heading", "H1", 1, 1),
            TaggedElement("heading", "H3", 1, 3),
        ),
        items=4,
    )
    doc = make_audit_document(
        make_heading("Guide", 1, file=PDF, page=1, outline=True),
        make_block("Text.", file=PDF),
        fmt="pdf",
        tags=tags,
    )
    drafts = _of(doc, "structure.heading_skipped")
    assert [(d.ref.text, d.ref.source.page, d.pattern) for d in drafts] == [
        ("H3", 1, f"H1{ARROW}H3"),
    ]


def test_pdf_skipped_levels_quote_the_type_when_levels_differ() -> None:
    """Headings of a page whose levels do not follow its tags: the type."""
    tags = TagTree(
        (
            TaggedElement("heading", "H1", 1, 1),
            TaggedElement("heading", "H1", 2, 1),
            TaggedElement("heading", "H3", 2, 3),
        ),
        items=6,
    )
    stale = make_audit_document(
        make_heading("Guide", 1, file=PDF, page=1, outline=True),
        make_block("Text.", file=PDF),
        make_heading("Skipped level", 2, file=PDF, page=2, outline=True),
        make_block("Text.", file=PDF),
        make_heading("Empty part", 1, file=PDF, page=2, outline=True),
        make_block("Text.", file=PDF),
        fmt="pdf",
        tags=tags,
    )
    drafts = _of(stale, "structure.heading_skipped")
    assert [(d.ref.text, d.ref.source.page) for d in drafts] == [("H3", 2)]
    ordered = replace(
        stale,
        blocks=[stale.blocks[i] for i in (0, 1, 4, 5, 2, 3)],
    )
    drafts = _of(ordered, "structure.heading_skipped")
    assert [(d.ref.text, d.ref.source.page) for d in drafts] == [
        ("Skipped level", 2),
    ]


def test_pdf_headings_without_level_are_ignored() -> None:
    """A plain H is neither compared nor the level before the next one."""
    tags = TagTree(
        (
            TaggedElement("heading", "H1", 1, 1),
            TaggedElement("heading", "H", 1),
            TaggedElement("heading", "H2", 1, 2),
            TaggedElement("heading", "H", 2),
            TaggedElement("heading", "H4", 2, 4),
        ),
        items=9,
    )
    doc = make_audit_document(
        make_block("Text.", file=PDF),
        fmt="pdf",
        tags=tags,
    )
    drafts = _of(doc, "structure.heading_skipped")
    assert [d.detail for d in drafts] == ["H2 is followed by H4"]


def test_pdf_empty_section_at_the_end_of_the_document() -> None:
    """The last bookmark heading with nothing after it is empty."""
    doc = make_audit_document(
        make_heading("Guide", 1, file=PDF, page=1, outline=True),
        make_block("Text.", file=PDF),
        make_heading("Appendix", 1, file=PDF, page=2, outline=True),
        fmt="pdf",
    )
    drafts = _of(doc, "structure.empty_section")
    assert [(d.ref.text, d.detail) for d in drafts] == [
        ("Appendix", "no content before the end of the document"),
    ]


def test_headings_under_parents_of_one_title_are_not_siblings() -> None:
    """Only the second Examples is reported, not its own subsections."""
    doc = make_audit_document(
        make_heading("API", 1),
        make_heading("Examples", 2),
        make_heading("Basic", 3),
        make_block("First text."),
        make_heading("Examples", 2),
        make_heading("Basic", 3),
        make_block("Second text."),
    )
    drafts = _of(doc, "structure.duplicate_heading")
    assert [(d.ref.text, d.detail, d.pattern) for d in drafts] == [
        (
            "Examples",
            'same title as an earlier heading under "API"',
            "duplicate",
        ),
    ]


def test_top_level_duplicates_are_under_the_top() -> None:
    """Two first level titles of a page are siblings under the top."""
    doc = make_audit_document(
        make_heading("Notes", 1),
        make_block("Text."),
        make_heading("Notes", 1),
        make_block("More text."),
    )
    drafts = _of(doc, "structure.duplicate_heading")
    assert [d.detail for d in drafts] == [
        'same title as an earlier heading under "the top"',
    ]


def test_tables_without_caption_where_the_document_captions() -> None:
    """A table is captioned by the caption just before it, in its page."""
    caption = make_block("Table 1: limits of the API.", BlockKind.CAPTION)
    table = make_block("Name | Limit", BlockKind.TABLE)
    other = make_block("Key | Value", BlockKind.TABLE, file="b.html")
    doc = make_audit_document(
        caption,
        table,
        make_block("Text."),
        other,
        parts=["index.html", "b.html"],
    )
    drafts = _of(doc, "structure.table_caption")
    assert [(d.ref.text, d.pattern) for d in drafts] == [
        ("Key | Value", "table"),
    ]


def test_figure_drafts_describe_the_image() -> None:
    """The side of a figure without caption shows the image."""
    image = make_image_block(
        raster_ref(encode(drawing("setup")), name="a.png", source=None),
    )
    caption = make_block("The steps of the installer.", BlockKind.CAPTION)
    doc = make_audit_document(image, caption, make_block("Text."), image)
    (draft,) = _of(doc, "structure.figure_caption")
    assert draft.ref.image is image.image
    assert draft.ref.text.startswith("a.png, ")
    assert draft.pattern == "figure"


def test_chrome_images_are_no_figures() -> None:
    """A logo removed as page chrome needs no caption."""
    logo = make_image_block(
        raster_ref(encode(drawing("logo")), name="logo.png", source=None),
    )
    figure = make_image_block(
        raster_ref(encode(drawing("setup")), name="a.png", source=None),
    )
    caption = make_block("The steps of the installer.", BlockKind.CAPTION)
    assert logo.image is not None
    doc = make_audit_document(
        logo,
        make_block("Text."),
        figure,
        caption,
        chrome_images=frozenset({logo.image.digest}),
    )
    assert not _of(doc, "structure.figure_caption")


def test_orphan_page_side_and_detail() -> None:
    """An orphan page is quoted by its file."""
    pages = (
        HtmlPageFacts("index.html", "en", "Home"),
        HtmlPageFacts(
            "old.html",
            "en",
            "Old",
            links_to=frozenset({"old.html"}),
        ),
    )
    doc = make_audit_document(
        make_block("Home text."),
        make_block("Old text.", file="old.html"),
        parts=["index.html", "old.html"],
        html_pages=pages,
    )
    (draft,) = _of(doc, "structure.orphan_page")
    assert (draft.ref.text, draft.detail, draft.pattern) == (
        "old.html",
        "no other page links to this page",
        "orphan",
    )


def test_no_orphan_in_a_single_page_or_a_pdf() -> None:
    """A lone page and a PDF have no orphan."""
    lone = make_audit_document(
        make_block("Home text."),
        html_pages=(HtmlPageFacts("index.html", "en", "Home"),),
    )
    assert not _of(lone, "structure.orphan_page")
    pdf = make_audit_document(make_block("Text.", file=PDF), fmt="pdf")
    assert not _of(pdf, "structure.orphan_page")


def test_heading_skips_and_empty_sections_end_with_their_page() -> None:
    """An h1 of a page then an h3 of the next one is no skip."""
    doc = make_audit_document(
        make_heading("Guide", 1),
        make_block("Text."),
        make_heading("Deep", 3, file="b.html"),
        make_block("Text.", file="b.html"),
        parts=["index.html", "b.html"],
    )
    assert not _of(doc, "structure.heading_skipped")
    assert not _of(doc, "structure.empty_section")


def test_no_orphan_under_a_base_that_only_the_navigation_has(
    tmp_path: Path,
) -> None:
    """Pages linked by a navigation under the base of the site are linked."""
    spec = map_texts(SAMPLE_EN, link_text)
    site = build_docusaurus_like(spec, tmp_path / "site", base="/project/")
    old = site / "docs" / "old" / "index.html"
    old.parent.mkdir()
    old.write_text(
        '<!DOCTYPE html><html lang="en"><head><title>Old</title></head>'
        "<body><main><h1>Old notes</h1>"
        "<p>Notes kept from an older release of the tool.</p>"
        "</main></body></html>",
        encoding="utf-8",
    )
    doc = load_document(site, Config(), audit=True)
    drafts = _of(doc, "structure.orphan_page")
    assert [d.ref.source.file for d in drafts] == ["docs/old/index.html"]


def _at(block: Block, line: int) -> Block:
    """Return a block at a source line of its page."""
    return replace(block, source=replace(block.source, line=line))


def _xrefs(doc: Document) -> list[tuple[int | None, str, list[str]]]:
    """Return the line, text and marks of each unresolved reference."""
    return [
        (
            d.ref.source.line,
            d.ref.text,
            [s.text for s in d.ref.segments if s.op == "mark"],
        )
        for d in _of(doc, "structure.unresolved_reference")
    ]


def _with_spots(*spots: Spot) -> tuple[HtmlPageFacts, ...]:
    """Return the facts of the one page, with its dead references."""
    return (HtmlPageFacts("index.html", "en", "Home", unresolved=spots),)


def test_references_land_on_the_block_of_their_line() -> None:
    """One label in two paragraphs, and a label that a heading holds."""
    doc = make_audit_document(
        _at(make_heading("Installation", 1), 1),
        _at(make_block("First, see missing-label for the limits."), 2),
        _at(make_block("Some unrelated text."), 4),
        _at(make_block("Later, see missing-label again."), 5),
        _at(make_block("Then read Installation."), 6),
        html_pages=_with_spots(
            Spot("missing-label", 3),
            Spot("missing-label", 5),
            Spot("Installation", 7),
        ),
    )
    assert _xrefs(doc) == [
        (2, "First, see missing-label for the limits.", ["missing-label"]),
        (5, "Later, see missing-label again.", ["missing-label"]),
        (6, "Then read Installation.", ["Installation"]),
    ]
    drafts = _of(doc, "structure.unresolved_reference")
    assert drafts[0].detail == 'reference "missing-label" not resolved'


def test_references_are_placed_whatever_their_characters() -> None:
    """A no-break space or a ligature keeps a reference on its block."""
    doc = make_audit_document(
        _at(make_block("Intro."), 1),
        _at(make_block("A: see non\xa0breaking here."), 2),
        _at(make_block("D: see \U0000fb01le here."), 3),
        html_pages=_with_spots(
            Spot("non breaking", 2),
            Spot("\U0000fb01le", 3),
        ),
    )
    assert [(line, marks) for line, _, marks in _xrefs(doc)] == [
        (2, ["non\xa0breaking"]),
        (3, ["file"]),
    ]


def test_several_references_of_one_block() -> None:
    """Two references of a block make one draft that counts them."""
    doc = make_audit_document(
        _at(make_block("See first-label and second-label."), 2),
        html_pages=_with_spots(
            Spot("first-label", 2),
            Spot("second-label", 2),
        ),
    )
    (draft,) = _of(doc, "structure.unresolved_reference")
    assert (draft.pattern, draft.detail) == (
        "xref",
        "2 unresolved reference(s) in this block",
    )
    assert _xrefs(doc)[0][2] == ["first-label"]


def test_references_of_one_line_are_told_apart_by_their_text() -> None:
    """In a page written on one line, the text tells the block."""
    doc = make_audit_document(
        _at(make_block("Intro."), 1),
        _at(make_block("See one-label here."), 1),
        _at(make_block("Plain text."), 1),
        _at(make_block("See two-label here."), 1),
        html_pages=_with_spots(Spot("one-label", 1), Spot("two-label", 1)),
    )
    assert [text for _, text, _ in _xrefs(doc)] == [
        "See one-label here.",
        "See two-label here.",
    ]


def test_references_before_any_block_or_without_line_are_alone() -> None:
    """A reference above the first block, or without line, is quoted alone."""
    doc = make_audit_document(
        _at(make_block("Text."), 5),
        html_pages=_with_spots(Spot("early", 1), Spot("nowhere", None)),
    )
    assert _xrefs(doc) == [
        (1, "early", ["early"]),
        (None, "nowhere", ["nowhere"]),
    ]


def test_twenty_thousand_references_are_placed_quickly() -> None:
    """Placing many references, by line or along one line, is linear."""
    lined = make_audit_document(
        *(
            _at(make_block(f"Paragraph {k} about the tool."), k + 1)
            for k in range(20_000)
        ),
        html_pages=_with_spots(
            *(Spot(f"lost-{k}", k + 1) for k in range(20_000)),
        ),
    )
    flat = make_audit_document(
        *(
            _at(make_block(f"Paragraph {k} about the tool."), 1)
            for k in range(20_000)
        ),
        html_pages=_with_spots(*(Spot(f"lost-{k}", 1) for k in range(20_000))),
    )
    for doc, count in ((lined, 20_000), (flat, 1)):
        start = time.perf_counter()
        drafts = _of(doc, "structure.unresolved_reference")
        # Under 0.3 s; a search of each reference through every block would
        # take minutes.
        assert time.perf_counter() - start < 3.0
        assert len(drafts) == count


def test_a_dead_reference_of_a_sphinx_page_is_found(tmp_path: Path) -> None:
    """A planted reference of a real page is quoted in its paragraph."""
    site = build_sphinx_like(SAMPLE_EN, tmp_path / "site", split="chapters")
    page = site / "getting-started.html"
    first = '<section id="first-steps">'
    html = page.read_text(encoding="utf-8")
    assert first in html
    planted = (
        '<p>See <span class="xref std std-ref">missing-label</span>'
        " for the details.</p>"
    )
    page.write_text(html.replace(first, planted + first, 1), encoding="utf-8")
    doc = load_document(site, Config(), audit=True)
    drafts = _of(doc, "structure.unresolved_reference")
    assert [(d.ref.source.file, d.ref.text, d.detail) for d in drafts] == [
        (
            "getting-started.html",
            "See missing-label for the details.",
            'reference "missing-label" not resolved',
        ),
    ]
    assert _xrefs(doc)[0][2] == ["missing-label"]


def _on(block: Block, page: int) -> Block:
    """Return a block of a page of the PDF."""
    return replace(block, source=SourceRef(PDF, page=page))


def _picture(page: int) -> Block:
    """Return an image block of a page of the PDF."""
    ref = raster_ref(encode(drawing("setup")), name="", source=None)
    return _on(make_image_block(ref, file=PDF), page)


def test_pdf_figures_are_captioned_within_a_window() -> None:
    """A caption of the page, a few blocks away, captions a PDF figure."""

    def row(text: str, page: int) -> Block:
        return _on(make_block(text, file=PDF), page)

    def caption(text: str, page: int) -> Block:
        return _on(make_block(text, BlockKind.CAPTION, file=PDF), page)

    doc = make_audit_document(
        _picture(1),
        *(row(f"Row {k} of the table", 1) for k in range(3)),
        caption("Dashboard of a project", 1),
        _picture(2),
        _on(make_heading("Next", 2, file=PDF, page=2, outline=True), 2),
        caption("Caption after a heading", 2),
        _picture(3),
        _picture(3),
        caption("Two images, one figure", 3),
        caption("A caption of page four", 4),
        _picture(5),
        _picture(6),
        *(row(f"Line {k} of the text", 6) for k in range(9)),
        caption("Too far from its figure", 6),
        caption("Before its figure", 7),
        row("A line between them", 7),
        _picture(7),
        _picture(8),
        _on(make_block("a | b", BlockKind.TABLE, file=PDF), 8),
        caption("After a table", 8),
        fmt="pdf",
    )
    drafts = _of(doc, "structure.figure_caption")
    assert [d.ref.source.page for d in drafts] == [2, 5, 6, 8]


def test_html_figures_need_an_adjacent_caption() -> None:
    """A paragraph between an HTML figure and a caption leaves it bare."""
    image = make_image_block(
        raster_ref(encode(drawing("setup")), name="a.png", source=None),
    )
    caption = make_block("The steps of the installer.", BlockKind.CAPTION)
    doc = make_audit_document(
        image,
        caption,
        make_block("Text."),
        image,
        make_block("More text."),
        caption,
    )
    assert len(_of(doc, "structure.figure_caption")) == 1


def test_reportlab_figures_are_all_captioned(tmp_path: Path) -> None:
    """The figures of the reportlab build keep their captions."""
    path = build_reportlab_pdf(FIGURES_EN, tmp_path / "doc.pdf")
    doc = load_document(path, Config(), audit=True)
    assert not _of(doc, "structure.figure_caption")


def test_a_section_of_an_embedded_medium_is_not_empty() -> None:
    """A frame, a video or a sound under a heading is its content."""
    doc = make_audit_document(
        _at(make_heading("Guide", 1), 1),
        _at(make_block("Text."), 2),
        _at(make_heading("Video", 2), 3),
        _at(make_heading("Audio", 2), 5),
        _at(make_heading("Empty", 2), 6),
        _at(make_heading("Last", 2), 7),
        html_pages=(HtmlPageFacts("index.html", "en", "Home", media=(4, 5)),),
    )
    assert [d.ref.text for d in _of(doc, "structure.empty_section")] == [
        "Empty",
        "Last",
    ]


def test_a_section_of_content_without_block_is_not_empty() -> None:
    """A toctree, or a picture whose image is not read, is content."""
    doc = make_audit_document(
        _at(make_heading("Guide", 1), 1),
        _at(make_block("Text."), 2),
        _at(make_heading("User guide", 2), 3),
        _at(make_heading("Diagram", 2), 5),
        _at(make_heading("Empty", 2), 7),
        _at(make_heading("Last", 2), 8),
        html_pages=(
            HtmlPageFacts("index.html", "en", "Home", blockless=(4, 6)),
        ),
    )
    assert [d.ref.text for d in _of(doc, "structure.empty_section")] == [
        "Empty",
        "Last",
    ]


def test_a_video_section_of_a_real_page_is_not_empty(tmp_path: Path) -> None:
    """An iframe under its heading, read from a page, is content."""
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text(
        '<!DOCTYPE html>\n<html lang="en">\n<head><title>T</title></head>\n'
        "<body>\n<main>\n<h1>Guide</h1>\n<p>Intro of the guide.</p>\n"
        "<h2>Video</h2>\n"
        '<iframe src="https://www.youtube.com/embed/x" title="D"></iframe>\n'
        "<h2>Next</h2>\n<p>Text of the next part.</p>\n</main>\n</body>\n"
        "</html>\n",
        encoding="utf-8",
    )
    doc = load_document(site, Config(), audit=True)
    assert not _of(doc, "structure.empty_section")


def test_quoted_texts_are_cut() -> None:
    """A long reference or parent title is quoted cut."""
    long = "x" * 5_000
    doc = make_audit_document(
        make_heading(long, 1),
        make_heading("Notes", 2),
        make_block("Text."),
        make_heading("Notes", 2),
        _at(make_block(f"See {long} here."), 3),
        html_pages=_with_spots(Spot(long, 3)),
    )
    for draft in structure_drafts(doc):
        assert len(draft.detail or "") < 200
        assert len(draft.pattern or "") < 200


def test_rules_come_in_their_order_whatever_the_document_order() -> None:
    """Skips, empty sections, duplicates, references, captions, orphans."""
    image = make_image_block(
        raster_ref(encode(drawing("setup")), name="a.png", source=None),
    )
    caption = make_block("The steps of the installer.", BlockKind.CAPTION)
    table_caption = make_block("Table 1: limits.", BlockKind.CAPTION)
    pages = (
        HtmlPageFacts("index.html", "en", "Home"),
        HtmlPageFacts("old.html", "en", "Old"),
    )
    doc = make_audit_document(
        make_block("See Section ?? first."),
        image,
        make_block("Text."),
        image,
        caption,
        make_block("More text."),
        make_block("Name | Limit", BlockKind.TABLE),
        make_block("Even more text."),
        table_caption,
        make_block("Key | Value", BlockKind.TABLE),
        make_heading("Guide", 1),
        make_heading("Notes", 2),
        make_block("Text of the notes."),
        make_heading("Notes", 2),
        make_heading("Empty", 2),
        make_heading("Deep", 4),
        make_block("Deep text."),
        make_block("Old text.", file="old.html"),
        parts=["index.html", "old.html"],
        html_pages=pages,
    )
    assert [d.rule for d in structure_drafts(doc)] == [
        "structure.heading_skipped",
        "structure.empty_section",
        "structure.duplicate_heading",
        "structure.unresolved_reference",
        "structure.figure_caption",
        "structure.table_caption",
        "structure.orphan_page",
    ]


def test_blanks_of_a_reference_match_any_blanks() -> None:
    """A no-break space of the block or of the reference is a blank."""
    doc = make_audit_document(
        _at(make_block("A: see non\xa0breaking here."), 2),
        _at(make_block("B: see plain   words here."), 3),
        html_pages=_with_spots(
            Spot("non breaking", 2),
            Spot("plain\xa0words", 3),
        ),
    )
    assert [marks for _, _, marks in _xrefs(doc)] == [
        ["non\xa0breaking"],
        ["plain   words"],
    ]


def test_a_far_reference_of_a_one_line_page_is_found() -> None:
    """A reference far into a page written on one line lands on its block."""
    filler = (
        _at(make_block(f"Paragraph {k} of a long page, with a few words."), 1)
        for k in range(340)
    )
    doc = make_audit_document(
        _at(make_heading("Home", 1), 1),
        *filler,
        _at(make_block("See far-label for the details."), 1),
        html_pages=_with_spots(Spot("far-label", 1)),
    )
    assert sum(len(block.text) for block in doc.blocks) > 15_000
    assert _xrefs(doc) == [
        (1, "See far-label for the details.", ["far-label"]),
    ]
