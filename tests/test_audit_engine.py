# Copyright 2026 Christophe Le Douarec
"""Tests for the engine of the audit."""

from dataclasses import replace
from typing import TYPE_CHECKING

from doc_checker.audit.engine import AuditStats, audit_document
from doc_checker.compare.drafts import EXCERPT_MAX_CHARS, marked_side
from doc_checker.config import Config, IgnoreConfig
from doc_checker.images.refs import raster_ref
from doc_checker.model.document import BlockKind, Link, SourceRef
from doc_checker.model.facts import HtmlPageFacts, PdfLink, PdfPageFacts
from doc_checker.model.findings import Severity
from doc_checker.text.normalize import ELLIPSIS
from tests.builders import (
    make_audit_document,
    make_block,
    make_heading,
    make_image_block,
)
from tests.image_builders import drawing, encode

if TYPE_CHECKING:
    from doc_checker.model.document import Document

STRUCTURE_ONLY = Config(checks=("structure",))


def test_audit_finalizes_its_drafts() -> None:
    """Minimum severity and ignored identifiers apply to the audit."""
    doc = make_audit_document(
        make_block("See Section ?? now."),
        make_heading("Empty", 2),
        make_heading("Next", 2),
        make_block("Text."),
    )
    result = audit_document(doc, STRUCTURE_ONLY)
    assert sorted(f.rule for f in result.findings) == [
        "structure.empty_section",
        "structure.unresolved_reference",
    ]
    assert all(f.new is None and f.ref is not None for f in result.findings)
    quiet = replace(STRUCTURE_ONLY, min_severity=Severity.MAJOR)
    assert [f.rule for f in audit_document(doc, quiet).findings] == [
        "structure.unresolved_reference",
    ]
    ignored = result.findings[0].id
    cfg = replace(STRUCTURE_ONLY, ignore=IgnoreConfig(findings=(ignored,)))
    assert ignored not in {f.id for f in audit_document(doc, cfg).findings}


def test_systematic_findings_are_grouped() -> None:
    """Six unresolved marks make one group of six occurrences."""
    doc = make_audit_document(
        *(make_block(f"See Section ?? for part {k}.") for k in range(6)),
    )
    findings = audit_document(doc, STRUCTURE_ONLY).findings
    assert [(f.rule, len(f.occurrences)) for f in findings] == [
        ("structure.unresolved_reference", 6),
    ]


def test_audit_counts_what_it_checked() -> None:
    """Pages, blocks, images and links of the document are counted."""
    doc = make_audit_document(make_heading("Guide", 1), make_block("Text."))
    stats = audit_document(doc, STRUCTURE_ONLY).stats
    assert (stats.pages, stats.blocks, stats.images, stats.links) == (
        1,
        2,
        0,
        0,
    )
    assert (stats.checks, stats.profile) == (("structure",), "screen")


def test_audit_runs_only_the_families_asked() -> None:
    """Without the structure family, nothing is checked."""
    doc = make_audit_document(make_block("See Section ?? now."))
    result = audit_document(doc, Config(checks=("a11y",)))
    assert result.findings == []
    assert (result.stats.links, result.stats.checks) == (0, ("a11y",))


def test_audit_stats_of_a_pdf() -> None:
    """Pages of a PDF, images apart, links, profile and images read."""
    image = make_image_block(
        raster_ref(encode(drawing("setup")), name="", source=None),
        file="doc.pdf",
    )
    pages = (
        PdfPageFacts(1, "1", 10, 0, scanned=False),
        PdfPageFacts(
            2,
            "2",
            10,
            0,
            scanned=False,
            links=(PdfLink("x", "uri", "https://example.org", valid=None),),
        ),
    )
    doc = make_audit_document(
        make_block("Text.", file="doc.pdf"),
        image,
        fmt="pdf",
        parts=["1", "2"],
        pdf_pages=pages,
    )
    cfg = replace(STRUCTURE_ONLY, profile="print", images=False)
    stats = audit_document(doc, cfg).stats
    assert stats == AuditStats(
        pages=2,
        blocks=1,
        images=1,
        links=1,
        checks=("structure",),
        profile="print",
        images_read=False,
    )


def test_marked_side_marks_the_first_occurrence() -> None:
    """Text before, the mark, then text after; empty pieces are left out."""
    source = SourceRef("index.html")
    side = marked_side(source, "See the guide now.", "the guide")
    assert [(s.op, s.text) for s in side.segments] == [
        ("equal", "See "),
        ("mark", "the guide"),
        ("equal", " now."),
    ]
    alone = marked_side(
        source,
        "the guide",
        "the guide",
        kind=BlockKind.HEADING,
    )
    assert [(s.op, s.text) for s in alone.segments] == [("mark", "the guide")]
    assert alone.kind is BlockKind.HEADING


def test_marked_side_without_its_mark() -> None:
    """Without a mark, or one not in the text, the text is shown whole."""
    source = SourceRef("index.html")
    long = "x" * (EXCERPT_MAX_CHARS + 10)
    for side in (
        marked_side(source, long),
        marked_side(source, long, "absent"),
    ):
        assert [s.op for s in side.segments] == ["equal"]
        assert side.text == long[:EXCERPT_MAX_CHARS] + ELLIPSIS


def test_marked_side_cuts_around_its_mark() -> None:
    """Half an excerpt is kept on each side of the mark."""
    half = EXCERPT_MAX_CHARS // 2
    before, after = "a" * (half + 5), "b" * (half + 5)
    side = marked_side(SourceRef("index.html"), f"{before}MARK{after}", "MARK")
    assert [(s.op, s.text) for s in side.segments] == [
        ("equal", ELLIPSIS + "a" * half),
        ("mark", "MARK"),
        ("equal", "b" * half + ELLIPSIS),
    ]


def _ids(doc: Document, rule: str) -> list[str]:
    """Return the identifiers of the findings of a rule of an audit."""
    return [
        f.id for f in audit_document(doc, Config()).findings if f.rule == rule
    ]


def test_findings_of_one_rule_in_one_block_have_their_own_ids() -> None:
    """Two broken links, two vague links, two kinds of reference differ."""
    links = replace(
        make_block(
            "Read the install guide and the upgrade guide, or here or here.",
        ),
        links=(
            Link("install guide", "install.html", page="install.html"),
            Link("upgrade guide", "upgrade.html", page="upgrade.html"),
            Link("here", "https://example.org", url="//example.org"),
            Link("here", "https://example.com", url="//example.com"),
        ),
    )
    doc = make_audit_document(
        links,
        make_block("See Section ?? and [?] for the limits."),
        html_pages=(HtmlPageFacts("index.html", "en", "Home"),),
    )
    for rule in (
        "link.page_missing",
        "a11y.link_text",
        "structure.unresolved_reference",
    ):
        ids = _ids(doc, rule)
        assert len(ids) == len(set(ids)) == 2


def test_same_findings_on_two_pdf_pages_share_their_id() -> None:
    """The page is not part of an identifier: a repeated text shares it."""
    links = (PdfLink("the manual", "file", "gone.pdf", valid=False),)
    pages = tuple(
        PdfPageFacts(number, str(number), 10, 0, scanned=False, links=links)
        for number in (1, 2)
    )
    blocks = [
        replace(
            make_block("See Section ?? below.", file="doc.pdf"),
            source=SourceRef("doc.pdf", page=number),
        )
        for number in (1, 2)
    ]
    doc = make_audit_document(
        *blocks,
        fmt="pdf",
        parts=["1", "2"],
        pdf_pages=pages,
    )
    for rule in ("link.file_missing", "structure.unresolved_reference"):
        first, second = _ids(doc, rule)
        assert first == second
