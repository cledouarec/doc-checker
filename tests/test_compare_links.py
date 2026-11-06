# Copyright 2026 Christophe Le Douarec
"""Tests for the comparison of the links of matched blocks."""

from dataclasses import replace

import pytest

from doc_checker.compare.align import align
from doc_checker.compare.drafts import ARROW
from doc_checker.compare.engine import compare_documents
from doc_checker.compare.links import anchor_index, link_drafts
from doc_checker.config import Config, IgnoreConfig
from doc_checker.model.document import Block, BlockKind, Document, Link
from doc_checker.model.findings import DiffSegment, Finding, Severity
from tests.builders import SPLIT_PARTS, SPLIT_WHOLE, make_block, make_document

C, MA = Severity.CRITICAL, Severity.MAJOR
INSTALL = "Installation of the service"
USAGE = "Usage of the command line"
BODY = "Read the guide before you start the service."
OTHER = "Every option has a default value that suits most users."
REF_LINK = Link(
    "guide",
    "#usage",
    page="index.html",
    fragment="usage",
    found=True,
)


def _block(
    text: str,
    file: str,
    *anchors: str,
    heading: bool = False,
) -> Block:
    """Make a block of a page, with its anchors."""
    kind = BlockKind.HEADING if heading else BlockKind.PARAGRAPH
    block = make_block(text, kind, file=file, level=2 if heading else None)
    return replace(block, anchors=anchors)


def _link(text: str, fragment: str, page: str = "index.html") -> Link:
    """Make a link to an anchor of a page that exists."""
    return Link(text, f"#{fragment}", page=page, fragment=fragment, found=True)


def _sides(finding: Finding) -> list[tuple[str, tuple[DiffSegment, ...]]]:
    """Return the file and the segments of each side of a finding."""
    sides = (finding.ref, finding.new)
    return [(s.source.file, s.segments) for s in sides if s is not None]


def _doc(files: tuple[str, str], *links: Link) -> Document:
    """Make the guide in one page or two, its body holding the links."""
    first, second = files
    return make_document(
        _block(INSTALL, first, "install", heading=True),
        replace(make_block(BODY, file=first), links=links),
        _block(USAGE, second, "usage", heading=True),
        _block(OTHER, second, "defaults"),
    )


def _findings(
    ref_links: tuple[Link, ...],
    new_links: tuple[Link, ...],
) -> list[Finding]:
    """Compare the guide in one page with the guide in two pages."""
    ref = _doc(("index.html", "index.html"), *ref_links)
    new = _doc(("install/index.html", "usage/index.html"), *new_links)
    return compare_documents(ref, new, Config()).findings


def _rules(
    ref_link: Link,
    new_link: Link | None,
) -> list[tuple[str, Severity]]:
    """Compare the guide in one page with the guide in two pages."""
    links = (new_link,) if new_link else ()
    return [(f.rule, f.severity) for f in _findings((ref_link,), links)]


@pytest.mark.parametrize(
    ("new_link", "expected"),
    [
        (
            Link(
                "guide",
                "../usage/#usage",
                page="usage/index.html",
                fragment="usage",
                found=True,
            ),
            [],
        ),
        (Link("guide", "../usage/", page="usage/index.html", found=True), []),
        (
            Link(
                "guide",
                "../usage/#defaults",
                page="usage/index.html",
                fragment="defaults",
                found=True,
            ),
            [],
        ),
        (
            Link(
                "the guide",
                "../usage/#usage",
                page="usage/index.html",
                fragment="usage",
                found=True,
            ),
            [],
        ),
        (None, [("link.lost", MA)]),
        (
            Link(
                "guide",
                "../usage/#nope",
                page="usage/index.html",
                fragment="nope",
                found=True,
            ),
            [("link.broken", C)],
        ),
        (
            Link("guide", "../gone/", page="gone/index.html"),
            [("link.broken", C)],
        ),
        (
            Link(
                "guide",
                "#install",
                page="install/index.html",
                fragment="install",
                found=True,
            ),
            [("link.target_changed", MA)],
        ),
        (
            Link(
                "guide",
                "https://example.org/usage",
                url="//example.org/usage",
            ),
            [("link.target_changed", MA)],
        ),
    ],
)
def test_internal_links(
    new_link: Link | None,
    expected: list[tuple[str, Severity]],
) -> None:
    """Internal links compare by the blocks they point to."""
    assert _rules(REF_LINK, new_link) == expected


def test_external_links() -> None:
    """External links compare by their normalized URL."""
    site = Link("site", "https://example.org/", url="//example.org")
    same = Link("site", "http://Example.org", url="//example.org")
    other = Link("site", "https://example.com/", url="//example.com")
    assert _rules(site, same) == []
    assert _rules(site, other) == [("link.target_changed", MA)]


def test_links_already_broken_are_not_reported() -> None:
    """A link broken in the reference is no regression."""
    broken = Link(
        "guide",
        "#nope",
        page="index.html",
        fragment="nope",
        found=True,
    )
    gone = Link("guide", "../gone/", page="gone/index.html")
    assert _rules(broken, gone) == []


def test_files_compare_by_name() -> None:
    """A page that has no block is a file, compared by its name."""
    index = Link("index", "genindex.html", page="genindex.html", found=True)
    moved = Link("index", "../genindex.html", page="genindex.html", found=True)
    search = Link("index", "../search.html", page="search.html", found=True)
    assert _rules(index, moved) == []
    assert _rules(index, search) == [("link.target_changed", MA)]


def test_pages_compare_by_their_last_path_part() -> None:
    """A page without blocks is known by its folder or its file name."""
    foo = Link("api", "foo/", page="foo/index.html", found=True)
    bar = Link("api", "bar/", page="bar/index.html", found=True)
    flat = Link("api", "foo.html", page="foo.html", found=True)
    assert _rules(foo, bar) == [("link.target_changed", MA)]
    assert _rules(bar, foo) == [("link.target_changed", MA)]
    assert _rules(flat, foo) == []
    assert _rules(foo, flat) == []


def test_downloads_compare_by_their_file_name() -> None:
    """A download keeps its whole name, whatever folder holds it."""
    pdf = Link("manual", "a/manual.pdf", page="a/manual.pdf", found=True)
    moved = Link("manual", "b/manual.pdf", page="b/manual.pdf", found=True)
    html = Link("manual", "manual.html", page="manual.html", found=True)
    assert _rules(pdf, moved) == []
    assert _rules(pdf, html) == [("link.target_changed", MA)]


def test_anchor_index() -> None:
    """Pages lead to their first block, anchors to their own block."""
    blocks = [
        _block(INSTALL, "a.html", "install", heading=True),
        _block(BODY, "a.html", "body", "intro"),
        _block(USAGE, "b.html"),
    ]
    assert anchor_index(blocks) == {
        ("a.html", None): 0,
        ("a.html", "install"): 0,
        ("a.html", "body"): 1,
        ("a.html", "intro"): 1,
        ("b.html", None): 2,
    }


def test_anchors_of_left_out_blocks_move_to_the_next_block() -> None:
    """An anchor of page chrome leads to the next block of its page."""
    banner = _block("Docs for version 2.0.", "index.html", "top")
    top = Link("top", "#top", page="index.html", fragment="top", found=True)
    ref = _doc(("index.html", "index.html"), top)
    ref = replace(
        ref,
        blocks=[banner, *ref.blocks],
        chrome_keys=frozenset({banner.key_strict}),
    )
    install = Link(
        "top",
        "#install",
        page="install/index.html",
        fragment="install",
        found=True,
    )
    new = _doc(("install/index.html", "usage/index.html"), install)
    assert compare_documents(ref, new, Config()).findings == []


def test_patterns_name_the_targets() -> None:
    """A link finding names its targets, so that alike findings group."""
    setup = Link(
        "setup",
        "#install",
        page="index.html",
        fragment="install",
        found=True,
    )
    site = Link("site", "https://example.org/", url="//example.org")
    ref = _doc(("index.html", "index.html"), REF_LINK, setup, site)
    gone = Link("guide", "../gone/", page="gone/index.html")
    other = Link("site", "https://example.com/", url="//example.com")
    new = _doc(("install/index.html", "usage/index.html"), gone, other)
    pairs = align(ref.blocks, new.blocks, match=0.6, moved_match=0.8)
    found = link_drafts(pairs, ref.blocks, new.blocks)
    assert [(d.rule, d.pattern) for ds in found.values() for d in ds] == [
        ("link.broken", "gone/index.html"),
        ("link.lost", "index.html#install"),
        (
            "link.target_changed",
            f"https://example.org/{ARROW}https://example.com/",
        ),
    ]


def test_new_links_to_anchors_of_left_out_blocks_are_not_broken() -> None:
    """A new link to an anchor of page chrome leads to its next block."""
    install = Link(
        "top",
        "#install",
        page="index.html",
        fragment="install",
        found=True,
    )
    ref = _doc(("index.html", "index.html"), install)
    banner = _block("Docs for version 2.0.", "install/index.html", "top")
    top = Link(
        "top",
        "#top",
        page="install/index.html",
        fragment="top",
        found=True,
    )
    new = _doc(("install/index.html", "usage/index.html"), top)
    new = replace(
        new,
        blocks=[banner, *new.blocks],
        chrome_keys=frozenset({banner.key_strict}),
    )
    assert compare_documents(ref, new, Config()).findings == []


def test_duplicate_link_texts_pair_by_target_first() -> None:
    """Of two links with one text, the one whose target went is lost."""
    ref_links = (_link("here", "install"), _link("here", "usage"))
    new_links = (_link("here", "usage", "usage/index.html"),)
    assert [(f.rule, f.detail) for f in _findings(ref_links, new_links)] == [
        ("link.lost", '"here" no longer links to index.html#install'),
    ]


def test_lost_link_detail_and_sides() -> None:
    """A lost link shows its old target, and its text on the new side."""
    (finding,) = _findings((REF_LINK,), ())
    assert finding.detail == '"guide" no longer links to index.html#usage'
    assert _sides(finding) == [
        ("index.html", (DiffSegment("delete", "index.html#usage"),)),
        ("install/index.html", (DiffSegment("equal", "guide"),)),
    ]


def test_broken_link_detail_and_sides() -> None:
    """A broken link shows its old target and the target that is missing."""
    gone = Link("guide", "../gone/", page="gone/index.html")
    (finding,) = _findings((REF_LINK,), (gone,))
    assert finding.detail == (
        '"guide" links to ../gone/, which does not exist; '
        "it linked to index.html#usage before"
    )
    assert _sides(finding) == [
        ("index.html", (DiffSegment("delete", "index.html#usage"),)),
        ("install/index.html", (DiffSegment("insert", "gone/index.html"),)),
    ]


def test_changed_target_detail_and_sides() -> None:
    """A changed target shows the old target and the new one."""
    install = _link("guide", "install", "install/index.html")
    (finding,) = _findings((REF_LINK,), (install,))
    assert finding.detail == (
        '"guide" links to install/index.html#install instead of '
        "index.html#usage"
    )
    assert _sides(finding) == [
        ("index.html", (DiffSegment("delete", "index.html#usage"),)),
        (
            "install/index.html",
            (DiffSegment("insert", "install/index.html#install"),),
        ),
    ]


def test_external_link_that_leads_nowhere_changed_target() -> None:
    """An external link that became a missing page is not a broken link."""
    site = Link("site", "https://example.org/", url="//example.org")
    gone = Link("site", "../gone/", page="gone/index.html")
    assert _rules(site, gone) == [("link.target_changed", MA)]


def _between_headings(*blocks: Block) -> Document:
    """Make a page whose blocks sit between two headings."""
    return make_document(
        _block(INSTALL, "index.html", "install", heading=True),
        *blocks,
        _block(USAGE, "index.html", "usage", heading=True),
    )


@pytest.mark.parametrize("split", [True, False], ids=["split", "merged"])
def test_links_of_split_and_merged_blocks(*, split: bool) -> None:
    """The links of a split or merged block compare with those of its parts."""
    setup = _link("setup", "install")
    whole = make_block(SPLIT_WHOLE)
    first, second = (make_block(text) for text in SPLIT_PARTS)
    if split:
        ref = _between_headings(replace(whole, links=(setup, REF_LINK)))
        new = _between_headings(first, replace(second, links=(REF_LINK,)))
    else:
        ref = _between_headings(
            replace(first, links=(setup,)),
            replace(second, links=(REF_LINK,)),
        )
        new = _between_headings(replace(whole, links=(REF_LINK,)))
    findings = compare_documents(ref, new, Config()).findings
    assert [(f.rule, f.detail) for f in findings] == [
        ("block.split_merged", None),
        ("link.lost", '"setup" no longer links to index.html#install'),
    ]


def test_blocks_before_the_first_heading_are_their_own_sections() -> None:
    """A block that no heading comes before corresponds to its match only."""

    def page(link: Link) -> Document:
        return make_document(
            _block(BODY, "index.html", "intro"),
            _block(OTHER, "index.html", "defaults"),
            _block(USAGE, "index.html", "usage", heading=True),
            replace(make_block(INSTALL), links=(link,)),
        )

    ref = page(_link("notes", "intro"))
    same = compare_documents(ref, page(_link("notes", "intro")), Config())
    other = compare_documents(ref, page(_link("notes", "defaults")), Config())
    assert same.findings == []
    assert [f.rule for f in other.findings] == ["link.target_changed"]


def test_anchors_of_blocks_emptied_by_ignored_patterns_move_on() -> None:
    """A block that the ignored patterns empty gives its anchors on."""
    ref = _doc(("index.html", "index.html"), _link("top", "install"))
    version = _block("Version 2.0", "install/index.html", "top")
    top = _link("top", "top", "install/index.html")
    new = _doc(("install/index.html", "usage/index.html"), top)
    new = replace(new, blocks=[version, *new.blocks])
    cfg = Config(ignore=IgnoreConfig(patterns=(r"Version \d+\.\d+",)))
    assert compare_documents(ref, new, cfg).findings == []
