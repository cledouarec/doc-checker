# Copyright 2026 Christophe Le Douarec
"""Tests for the document, finding and rule models."""

from collections import Counter
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from doc_checker.model.document import (
    BlockKind,
    Document,
    ImageRef,
    PixelSource,
    SourceRef,
    carry_anchors,
)
from doc_checker.model.findings import (
    Category,
    DiffSegment,
    Severity,
    Side,
    finding_id,
)
from doc_checker.report.context import CATEGORY_TITLES
from doc_checker.rules import RULES, get_rule
from tests.builders import make_block, make_document

if TYPE_CHECKING:
    from pathlib import Path


def _document(path: Path) -> Document:
    return Document(
        path=path,
        format="html",
        blocks=[],
        parts=[],
        lang=None,
        declared_lang=None,
        generator=None,
        normalization=Counter(),
        chrome=[],
        chrome_keys=frozenset(),
        warnings=[],
    )


def test_severity_parse_is_case_insensitive() -> None:
    """Severity names are parsed without regard to case."""
    assert Severity.parse("Major") is Severity.MAJOR


def test_severity_parse_rejects_unknown() -> None:
    """An unknown severity name raises a ``ValueError``."""
    with pytest.raises(ValueError, match="unknown severity"):
        Severity.parse("fatal")


def test_severity_order_and_label() -> None:
    """Severities are ordered and expose a lowercase label."""
    assert Severity.CRITICAL > Severity.MAJOR > Severity.MINOR > Severity.INFO
    assert Severity.MINOR.label == "minor"


def test_source_href() -> None:
    """A source reference renders its file and optional anchor."""
    assert SourceRef("guide/a.html", anchor="x").href == "guide/a.html#x"
    assert SourceRef("a.html").href == "a.html"


def test_source_fragment_and_pdf_href() -> None:
    """A page gives the fragment of a PDF link, an anchor takes its place."""
    assert SourceRef("doc.pdf", page=3).fragment == "page=3"
    assert SourceRef("doc.pdf", page=3).href == "doc.pdf#page=3"
    assert SourceRef("a.html", page=3, anchor="x").href == "a.html#x"
    assert SourceRef("a.html").fragment is None


def test_source_fragment_prefers_the_anchor() -> None:
    """The fragment is the anchor when there is both an anchor and a page."""
    assert SourceRef("a.html", page=3, anchor="x").fragment == "x"


def test_source_empty_anchor_falls_back_to_the_page() -> None:
    """An empty anchor is no anchor: the page tells where the block is."""
    assert SourceRef("doc.pdf", page=3, anchor="").href == "doc.pdf#page=3"


@pytest.mark.parametrize(
    "source",
    [
        SourceRef("a.html"),
        SourceRef("a.html", line=12),
        SourceRef("a.html", anchor=""),
    ],
)
def test_source_without_anchor_or_page_has_no_fragment(
    source: SourceRef,
) -> None:
    """A line number or an empty anchor does not make a fragment."""
    assert source.fragment is None
    assert source.href == "a.html"


def test_side_text_joins_segments() -> None:
    """The text of a side is the concatenation of its segments."""
    side = Side(
        SourceRef("a.html"),
        (DiffSegment("equal", "a "), DiffSegment("delete", "b")),
    )
    assert side.text == "a b"


def test_finding_id_is_stable_and_short() -> None:
    """Finding identifiers are stable, short and depend on the rule."""
    side = Side(SourceRef("a.html", anchor="s"), (DiffSegment("delete", "x"),))
    first = finding_id("block.missing", side, None)
    assert first == finding_id("block.missing", side, None)
    assert len(first) == 12
    assert first != finding_id("block.added", side, None)


def test_finding_id_ignores_the_page() -> None:
    """Finding identifiers do not change when the pagination changes."""
    segments = (DiffSegment("delete", "x"),)
    early = Side(SourceRef("doc.pdf", page=3), segments)
    late = Side(SourceRef("doc.pdf", page=30), segments)
    early_id = finding_id("block.missing", early, None)
    late_id = finding_id("block.missing", late, None)
    assert early_id == late_id


def test_finding_id_of_a_side_without_mark_is_unchanged() -> None:
    """The identifiers of compare, whose sides have no mark, stay the same."""
    source = SourceRef("guide.html", anchor="install")
    old = Side(
        source,
        (
            DiffSegment("equal", "Run the "),
            DiffSegment("delete", "old"),
            DiffSegment("insert", "new"),
            DiffSegment("equal", " installer."),
        ),
        BlockKind.PARAGRAPH,
    )
    new = Side(
        source,
        (DiffSegment("equal", "Run the new installer."),),
        BlockKind.PARAGRAPH,
    )
    assert finding_id("text.changed", old, new) == "21baec031731"
    assert finding_id("block.missing", old, None) == "7f665493b7bb"


def test_finding_id_tells_apart_the_marks_of_one_excerpt() -> None:
    """Two marks of one excerpt differ by their place or their text."""
    source = SourceRef("a.html")

    def marked(before: str, mark: str, after: str) -> Side:
        segments = (
            DiffSegment("equal", before),
            DiffSegment("mark", mark),
            DiffSegment("equal", after),
        )
        return Side(source, segments)

    first = marked("See ", "here", " or here.")
    second = marked("See here or ", "here", ".")
    assert first.text == second.text
    ids = {
        finding_id("a11y.link_text", side, None)
        for side in (first, second, marked("See ", "here or", " here."))
    }
    assert len(ids) == 3
    again = marked("See ", "here", " or here.")
    assert finding_id("a11y.link_text", again, None) in ids


def test_document_tagged_defaults_to_none(tmp_path: Path) -> None:
    """A document does not say whether it is tagged unless it is told so."""
    assert _document(tmp_path).tagged is None


def test_document_tagged_keeps_its_value(tmp_path: Path) -> None:
    """A PDF that is tagged, or is not, keeps the answer of its catalog."""
    pdf = replace(_document(tmp_path), format="pdf")
    assert replace(pdf, tagged=True).tagged is True
    assert replace(pdf, tagged=False).tagged is False


def test_document_resolve(tmp_path: Path) -> None:
    """Sources resolve inside a site directory or to the single file."""
    page = tmp_path / "index.html"
    page.write_text("<p>x</p>", encoding="utf-8")
    site = _document(tmp_path)
    single = _document(page)
    assert site.resolve(SourceRef("guide/a.html")) == tmp_path / "guide/a.html"
    assert single.resolve(SourceRef("index.html")) == page


def test_missing_content_rules_are_critical() -> None:
    """Missing blocks and sections are critical by default."""
    assert get_rule("block.missing").severity is Severity.CRITICAL
    assert get_rule("section.missing").severity is Severity.CRITICAL


def test_rule_registry_is_consistent() -> None:
    """Every rule is registered under its own identifier."""
    assert len(RULES) == 83
    assert all(spec.id == key for key, spec in RULES.items())


def test_carry_anchors_moves_dropped_anchors_within_a_page() -> None:
    """Anchors of dropped blocks go to the next kept block of their page."""
    blocks = [
        replace(make_block("Drop one."), anchors=("x",)),
        replace(make_block("Drop two."), anchors=("y", "x")),
        replace(make_block("Keep one."), anchors=("x", "z")),
        replace(make_block("Drop three."), anchors=("lost",)),
        make_block("Keep two.", file="other.html"),
    ]
    kept = carry_anchors(blocks, lambda block: block.text.startswith("Keep"))
    assert [(block.text, block.anchors) for block in kept] == [
        ("Keep one.", ("x", "y", "z")),
        ("Keep two.", ()),
    ]
    assert blocks[2].anchors == ("x", "z")


def test_image_without_pixels_is_not_readable() -> None:
    """An image without hashes has no pixels to compare."""
    svg = ImageRef(name="img/a.svg", format="svg", digest="ab" * 16)
    assert (svg.readable, svg.source, svg.alt) == (False, None, None)
    png = ImageRef("a.png", "png", "cd" * 16, phash=1, dhash=2)
    assert png.readable


def test_pixel_source_hides_its_bytes() -> None:
    """The bytes of a data URL stay out of the representation."""
    assert "PNG" not in repr(PixelSource(data=b"\x89PNG" * 100))


def test_text_blocks_and_sides_have_no_image() -> None:
    """Text blocks and their sides carry no image."""
    block = make_block("Some text here for the block.")
    assert block.image is None
    assert Side(block.source, ()).image is None


def test_documents_have_no_audit_facts_by_default() -> None:
    """A document starts without the facts that only the audit reads."""
    doc = make_document(make_block("Some text here for the block."))
    facts = (doc.title, doc.outline, doc.html_pages, doc.pdf_pages, doc.tags)
    assert facts == (None, (), (), (), None)


def test_a_side_can_mark_part_of_its_text() -> None:
    """A marked segment belongs to the text of its side."""
    side = Side(
        SourceRef("a.html"),
        (
            DiffSegment("equal", "See "),
            DiffSegment("mark", "??"),
            DiffSegment("equal", " now"),
        ),
    )
    assert side.text == "See ?? now"


def test_accessibility_is_the_last_category() -> None:
    """The new category comes last and has its report title."""
    assert list(Category)[-1] is Category.ACCESSIBILITY
    assert CATEGORY_TITLES[Category.ACCESSIBILITY] == "Accessibility"


def test_blocks_and_documents_have_no_code_by_default() -> None:
    """Inline code spans and monospaced words are empty by default."""
    assert make_block("x y").code_spans == ()
    assert make_document(make_block("x y")).code_words == frozenset()
