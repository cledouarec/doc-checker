# Copyright 2026 Christophe Le Douarec
"""Tests for the link checks of the audit."""

import time
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from doc_checker.audit.links import broken_link_drafts
from doc_checker.audit.structure import structure_drafts
from doc_checker.compare.aggregate import PATTERN_MAX_CHARS
from doc_checker.model.document import Document, Link
from doc_checker.model.facts import HtmlPageFacts, PdfLink, PdfPageFacts
from tests.builders import make_audit_document, make_block, make_heading

if TYPE_CHECKING:
    from pathlib import Path

    from doc_checker.compare.drafts import Draft


def _to_ref(text: str, fragment: str) -> Link:
    return Link(
        text,
        f"ref.html#{fragment}",
        page="ref.html",
        fragment=fragment,
        found=True,
    )


def test_internal_links_to_missing_targets() -> None:
    """A missing page, file and anchor are reported; working links not."""
    block = replace(
        make_block("See the guide, the tool and the limits."),
        links=(
            Link("the guide", "nothere.html", page="nothere.html"),
            Link("the tool", "files/tool.zip", page="files/tool.zip"),
            _to_ref("the limits", "nowhere"),
            _to_ref("the top", "top"),
            _to_ref("the text", ":~:text=limits"),
            Link(
                "the index",
                "genindex.html#a",
                page="genindex.html",
                fragment="a",
                found=True,
            ),
            Link("the site", "https://example.org", url="//example.org"),
        ),
    )
    pages = (
        HtmlPageFacts("index.html", "en", "Home"),
        HtmlPageFacts("ref.html", "en", "Ref", ids=frozenset({"limits"})),
    )
    doc = make_audit_document(
        block,
        make_block("Ref text.", file="ref.html"),
        parts=["index.html", "ref.html"],
        html_pages=pages,
    )
    drafts, checked = broken_link_drafts(doc)
    assert [(d.rule, d.pattern) for d in drafts] == [
        ("link.page_missing", "nothere.html"),
        ("link.file_missing", "files/tool.zip"),
        ("link.anchor_missing", "ref.html#nowhere"),
    ]
    assert checked == 6
    marked = [s.text for s in drafts[0].ref.segments if s.op == "mark"]
    assert marked == ["the guide"]
    assert drafts[2].detail == (
        '"the limits" links to ref.html, which has no anchor "nowhere"'
    )


def test_pdf_link_annotations() -> None:
    """A link to no page and a missing file; web links are not checked."""
    links = (
        PdfLink("see the limits", "goto", "page 2", valid=True),
        PdfLink("the lost one", "none", "", valid=False),
        PdfLink("the manual", "file", "gone.pdf", valid=False),
        PdfLink("the site", "uri", "https://example.org", valid=None),
    )
    pages = (PdfPageFacts(1, "1", 120, 0, scanned=False, links=links),)
    doc = make_audit_document(
        make_block("Text.", file="doc.pdf"),
        fmt="pdf",
        pdf_pages=pages,
    )
    drafts, checked = broken_link_drafts(doc)
    found = [(d.rule, d.ref.text, d.ref.source.page) for d in drafts]
    assert found == [
        ("link.destination_invalid", "the lost one", 1),
        ("link.file_missing", "the manual", 1),
    ]
    assert checked == 4


def test_structure_and_links_of_a_large_site_are_quick() -> None:
    """Fifty thousand blocks over four hundred pages, in seconds."""
    blocks = []
    pages = []
    for k in range(400):
        file = f"p{k}.html"
        pages.append(HtmlPageFacts(file, "en", f"Page {k}"))
        blocks.append(make_heading(f"Page {k}", 1, file=file))
        target = f"p{k + 1}.html"
        link = Link("next", target, page=target, found=True)
        blocks.extend(
            replace(make_block(f"Text {k} {n}.", file=file), links=(link,))
            for n in range(124)
        )
    doc = make_audit_document(
        *blocks,
        parts=[page.file for page in pages],
        html_pages=tuple(pages),
    )
    start = time.perf_counter()
    broken_link_drafts(doc)
    structure_drafts(doc)
    assert time.perf_counter() - start < 5.0


def _linked(*links: Link) -> Document:
    """Return a document whose first block holds the links, and ref.html."""
    block = replace(make_block("See the links of this page."), links=links)
    pages = (
        HtmlPageFacts("index.html", "en", "Home", ids=frozenset({"intro"})),
        HtmlPageFacts("ref.html", "en", "Ref", ids=frozenset({"limits"})),
        HtmlPageFacts("old.htm", "en", "Old"),
    )
    return make_audit_document(
        block,
        make_block("Ref text.", file="ref.html"),
        parts=["index.html", "ref.html", "old.htm"],
        html_pages=pages,
    )


def test_working_anchors_and_browser_fragments_pass() -> None:
    """Existing anchors, top in any case, and text directives pass."""
    doc = _linked(
        _to_ref("the limits", "limits"),
        _to_ref("the top", "Top"),
        _to_ref("the limits again", "limits:~:text=quota"),
        Link(
            "intro",
            "#intro",
            page="index.html",
            fragment="intro",
            found=True,
        ),
    )
    drafts, checked = broken_link_drafts(doc)
    assert not drafts
    assert checked == 4


def test_the_anchor_before_a_text_directive_is_checked() -> None:
    """An anchor followed by a text directive must exist."""
    doc = _linked(_to_ref("the limits", "nowhere:~:text=quota"))
    (draft,) = broken_link_drafts(doc)[0]
    assert (draft.rule, draft.pattern) == (
        "link.anchor_missing",
        "ref.html#nowhere",
    )


def test_missing_pages_by_their_suffix() -> None:
    """A missing .htm or .HTML target is a page, a folder leads to one."""
    doc = _linked(
        Link("old", "gone.htm", page="gone.htm"),
        Link("upper", "GONE.HTML", page="GONE.HTML"),
        Link("folder", "guide/", page="guide/index.html"),
        Link("data", "data.json", page="data.json"),
    )
    drafts, _ = broken_link_drafts(doc)
    assert [(d.rule, d.pattern) for d in drafts] == [
        ("link.page_missing", "gone.htm"),
        ("link.page_missing", "GONE.HTML"),
        ("link.page_missing", "guide/index.html"),
        ("link.file_missing", "data.json"),
    ]
    assert drafts[3].detail == (
        '"data" links to data.json, which does not exist'
    )


def test_links_out_of_the_site_are_not_checked() -> None:
    """A link above the root of the site is neither checked nor counted."""
    doc = _linked(
        Link("up", "../x.html"),
        Link("mail", "mailto:a@b.c", url="mailto:a@b.c"),
    )
    assert broken_link_drafts(doc) == ([], 0)


def test_pdf_link_details() -> None:
    """A PDF link is quoted by its text at its page."""
    links = (
        PdfLink("lost", "goto", "", valid=False),
        PdfLink("manual", "file", "gone.pdf", valid=False),
        PdfLink("other", "other", "", valid=None),
        PdfLink("unknown file", "file", "x.pdf", valid=None),
    )
    pages = (
        PdfPageFacts(1, "i", 10, 0, scanned=False),
        PdfPageFacts(2, "1", 10, 0, scanned=False, links=links),
    )
    doc = make_audit_document(
        make_block("Text.", file="doc.pdf"),
        fmt="pdf",
        pdf_pages=pages,
    )
    drafts, checked = broken_link_drafts(doc)
    assert [(d.detail, d.pattern, d.ref.source.page) for d in drafts] == [
        ('"lost" leads to no page of the document', "destination", 2),
        (
            '"manual" opens gone.pdf, which does not exist next to the PDF',
            "gone.pdf",
            2,
        ),
    ]
    assert checked == 4
    assert drafts[0].ref.source.file == "doc.pdf"


def test_structure_drafts_do_not_hold_links() -> None:
    """The links of a document give no structure draft."""
    doc = _linked(Link("old", "gone.htm", page="gone.htm"))
    assert all(not d.rule.startswith("link.") for d in structure_drafts(doc))


def test_an_existing_file_without_suffix_is_found(tmp_path: Path) -> None:
    """A link to a file of the site without suffix leads somewhere."""
    site = tmp_path / "site"
    (site / "data").mkdir(parents=True)
    (site / "LICENSE").write_text("MIT", encoding="utf-8")
    (site / "data" / "README").write_text("Data.", encoding="utf-8")
    links = (
        Link("license", "LICENSE", page="LICENSE/index.html"),
        Link("readme", "data/README#top", page="data/README/index.html"),
        Link("notice", "NOTICE", page="NOTICE/index.html"),
        Link("folder", "LICENSE/", page="LICENSE/index.html"),
    )
    block = replace(make_block("See the files of the site."), links=links)
    doc = make_audit_document(block, path=site)
    drafts, checked = broken_link_drafts(doc)
    assert [(d.rule, d.pattern) for d in drafts] == [
        ("link.page_missing", "NOTICE/index.html"),
        ("link.page_missing", "LICENSE/index.html"),
    ]
    assert checked == 4


def test_a_file_without_suffix_out_of_the_site_is_missing(
    tmp_path: Path,
) -> None:
    """A symbolic link of the site to a file outside it leads nowhere."""
    site = tmp_path / "site"
    site.mkdir()
    (tmp_path / "secret").write_text("Outside.", encoding="utf-8")
    try:
        (site / "escape").symlink_to(tmp_path / "secret")
    except OSError:
        pytest.skip("symbolic links are not allowed here")
    link = Link("escape", "escape", page="escape/index.html")
    block = replace(make_block("See the escape."), links=(link,))
    drafts, _ = broken_link_drafts(make_audit_document(block, path=site))
    assert [(d.rule, d.pattern) for d in drafts] == [
        ("link.page_missing", "escape/index.html"),
    ]


def test_quoted_link_texts_and_targets_are_cut() -> None:
    """A huge target or a link that wraps a whole block is quoted cut."""
    huge = "a" * 1_000_000 + ".html"
    card = "A card that links a whole block of text. " * 50
    block = replace(
        make_block(card),
        links=(
            Link("x", huge, page=huge),
            Link(card, "gone.html", page="gone.html"),
            _to_ref(card, "n" * 5_000),
        ),
    )
    pdf_links = (PdfLink(card, "file", "f" * 5_000 + ".pdf", valid=False),)
    doc = make_audit_document(
        block,
        make_block("Ref text.", file="ref.html"),
        parts=["index.html", "ref.html"],
        html_pages=(HtmlPageFacts("ref.html", "en", "Ref"),),
        pdf_pages=(
            PdfPageFacts(1, "1", 10, 0, scanned=False, links=pdf_links),
        ),
    )
    drafts, _ = broken_link_drafts(doc)
    assert len(drafts) == 4
    for draft in drafts:
        assert len(draft.detail or "") < 300
        assert len(draft.pattern or "") <= PATTERN_MAX_CHARS + 1


def _segments(draft: Draft) -> list[tuple[str, str]]:
    """Return the segments of the reference side of a draft."""
    assert draft.ref is not None
    return [(s.op, s.text) for s in draft.ref.segments]


def test_a_broken_link_marks_its_own_occurrence() -> None:
    """The text before a broken link, and a working link, are not marked."""
    block = replace(
        make_block(
            "The setup takes a minute: read the setup notes, then the setup"
            " checklist.",
        ),
        links=(
            Link("setup", "index.html", page="index.html", found=True),
            Link("setup", "gone.html", page="gone.html"),
        ),
    )
    (draft,) = broken_link_drafts(make_audit_document(block))[0]
    assert _segments(draft) == [
        ("equal", "The setup takes a minute: read the setup notes, then the "),
        ("mark", "setup"),
        ("equal", " checklist."),
    ]


def test_broken_links_of_one_text_mark_their_own_places() -> None:
    """A broken link before a working one of its text marks the first."""
    block = replace(
        make_block("Open setup first, then setup again, or here, or here."),
        links=(
            Link("setup", "gone.html", page="gone.html"),
            Link("setup", "index.html", page="index.html", found=True),
            Link("here", "a.html", page="a.html"),
            Link("here", "b.html", page="b.html"),
        ),
    )
    drafts, checked = broken_link_drafts(make_audit_document(block))
    assert checked == 4
    assert [_segments(d)[:2] for d in drafts] == [
        [("equal", "Open "), ("mark", "setup")],
        [
            ("equal", "Open setup first, then setup again, or "),
            ("mark", "here"),
        ],
        [
            ("equal", "Open setup first, then setup again, or here, or "),
            ("mark", "here"),
        ],
    ]


def test_many_broken_links_of_one_block_are_marked_quickly() -> None:
    """Eighty thousand broken links of one block, each at its place."""
    words = [f"w{k:07d}x" for k in range(80_000)]
    block = replace(
        make_block(" ".join(words)),
        links=tuple(
            Link(w, f"#m{k}", page="index.html", fragment=f"m{k}", found=True)
            for k, w in enumerate(words)
        ),
    )
    pages = (HtmlPageFacts("index.html", "en", "Home"),)
    doc = make_audit_document(block, html_pages=pages)
    start = time.perf_counter()
    drafts, checked = broken_link_drafts(doc)
    # About 0.5 s on a desktop; a search from each link took 13.7 s.
    assert time.perf_counter() - start < 5.0
    assert checked == len(drafts) == 80_000
    for k in (0, 1, 40_000, 79_999):
        marks = [t for op, t in _segments(drafts[k]) if op == "mark"]
        assert marks == [words[k]]
