# Copyright 2026 Christophe Le Douarec
"""Tests for the links and anchors of HTML pages."""

from typing import TYPE_CHECKING

import pytest

from doc_checker.config import HtmlConfig
from doc_checker.extract.html import load_html
from doc_checker.extract.html.links import (
    normalize_url,
    resolve_links,
    site_base,
)
from doc_checker.model.document import Link
from tests.builders import SAMPLE_EN, build_docusaurus_like

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("href", "url"),
    [
        ("HTTPS://Example.org:443/Docs/", "//example.org/Docs"),
        ("http://example.org/docs/index.html", "//example.org/docs"),
        ("https://example.org:8443/a?b=1#c", "//example.org:8443/a?b=1#c"),
        ("mailto:Team@Example.org", "mailto:team@example.org"),
        ("javascript:alert(1)", None),
        ("DATA:text/html,x", None),
    ],
)
def test_normalize_url(href: str, url: str | None) -> None:
    """External URLs compare without scheme, host case or index page."""
    assert normalize_url(href) == url


def test_site_base_is_found_from_the_root_links() -> None:
    """The base of a site is the prefix that makes its root links exist."""
    pages = {"docs/a/index.html", "docs/b/index.html"}
    hrefs = ["/example/docs/a/", "/example/docs/b/#x", "/elsewhere/", "c.html"]
    assert site_base(hrefs, lambda page: page in pages) == "example/"
    assert not site_base(["/docs/a/"], lambda page: page in pages)


def _write(root: Path, rel: str, html: str) -> None:
    """Write one page of a site."""
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def test_links_resolve_to_pages_and_anchors(tmp_path: Path) -> None:
    """Links resolve through redirects to pages, anchors and files."""
    _write(
        tmp_path,
        "index.html",
        '<main><p>See <a href="guide/a.html#install">install</a>, '
        '<a href="https://example.org/">site</a> and '
        '<a href="old.html">old</a>.</p></main>',
    )
    _write(
        tmp_path,
        "guide/a.html",
        '<main><h2 id="install">Install</h2><p><a href="#install">here</a> '
        '<a href="../missing.html">gone</a> '
        '<a href="../genindex.html">index</a></p></main>',
    )
    _write(
        tmp_path,
        "old.html",
        '<html><head><meta http-equiv="refresh" '
        'content="0; url=guide/a.html"></head></html>',
    )
    _write(tmp_path, "genindex.html", "<main><p>Index</p></main>")
    doc = load_html(tmp_path, HtmlConfig(), lang="auto")
    links = {link.text: link for block in doc.blocks for link in block.links}
    assert links["install"] == Link(
        "install",
        "guide/a.html#install",
        page="guide/a.html",
        fragment="install",
        found=True,
    )
    assert links["site"].url == "//example.org"
    assert (links["old"].page, links["old"].found) == ("guide/a.html", True)
    assert (links["here"].page, links["here"].fragment) == (
        "guide/a.html",
        "install",
    )
    assert (links["gone"].page, links["gone"].found) == ("missing.html", False)
    assert (links["index"].page, links["index"].found) == (
        "genindex.html",
        True,
    )
    heading = next(b for b in doc.blocks if b.text == "Install")
    assert heading.anchors == ("install",)


def test_percent_encoded_targets(tmp_path: Path) -> None:
    """Non-ASCII page names and anchors resolve once decoded."""
    _write(
        tmp_path,
        "guide/index.html",
        '<main><p><a href="caf%C3%A9.html#%C3%A9t%C3%A9">summer</a>'
        "</p></main>",
    )
    _write(tmp_path, "guide/café.html", '<main><h2 id="été">Été</h2></main>')
    doc = load_html(tmp_path, HtmlConfig(), lang="auto")
    (link,) = [link for block in doc.blocks for link in block.links]
    assert (link.page, link.fragment, link.found) == (
        "guide/café.html",
        "été",
        True,
    )


def test_absolute_links_under_a_base(tmp_path: Path) -> None:
    """Absolute links under the base of a site resolve to its pages."""
    site = build_docusaurus_like(SAMPLE_EN, tmp_path, base="/example/")
    doc = load_html(site, HtmlConfig(), lang="auto")
    link = next(
        link
        for block in doc.blocks
        for link in block.links
        if link.text == "settings"
    )
    assert (link.page, link.fragment, link.found) == (
        "docs/reference/index.html",
        "command-line",
        True,
    )


def test_scheme_relative_links_are_external(tmp_path: Path) -> None:
    """A link that starts with two slashes is an external URL."""
    assert normalize_url("//Example.org:80/Docs/index.html") == (
        "//example.org/Docs"
    )
    links = resolve_links(
        [("cdn", "//CDN.example.org/lib/")],
        "index.html",
        base="",
        pages=frozenset({"index.html"}),
        redirects={},
        root=tmp_path,
    )
    assert links == (
        Link("cdn", "//CDN.example.org/lib/", url="//cdn.example.org/lib"),
    )
    assert links[0].target == "//CDN.example.org/lib/"


@pytest.mark.parametrize(
    "href",
    [
        "",
        "#",
        " \x01JavaScript:alert(1)",
        "java\x09script:alert(1)",
        "\x00DATA:text/html,x",
        "VBScript:msgbox(1)",
    ],
)
def test_empty_and_script_links_are_ignored(tmp_path: Path, href: str) -> None:
    """Empty, script and data targets give no link, whatever their case."""
    links = resolve_links(
        [("x", href)],
        "index.html",
        base="",
        pages=frozenset({"index.html"}),
        redirects={},
        root=tmp_path,
    )
    assert links == ()


def test_links_out_of_the_site_and_redirect_loops(tmp_path: Path) -> None:
    """A link above the root has no page, a redirect loop stops on a page."""
    refresh = '<meta http-equiv="refresh" content="0; url={}">'
    _write(tmp_path, "guide/a.html", refresh.format("b.html"))
    _write(tmp_path, "guide/b.html", refresh.format("a.html"))
    links = resolve_links(
        [("up", "../../x.html"), ("loop", "a.html#s"), ("query", "?v=2")],
        "guide/page.html",
        base="",
        pages=frozenset({"guide/page.html"}),
        redirects={"guide/a.html": "b.html", "guide/b.html": "a.html"},
        root=tmp_path,
    )
    assert links == (
        Link("up", "../../x.html"),
        Link(
            "loop",
            "a.html#s",
            page="guide/b.html",
            fragment="s",
            found=True,
        ),
        Link("query", "?v=2", page="guide/page.html", found=True),
    )
    assert [link.target for link in links] == [
        "../../x.html",
        "guide/b.html#s",
        "guide/page.html",
    ]


def test_redirects_under_a_base(tmp_path: Path) -> None:
    """A redirect written under the base lands on its page and anchor."""
    site = build_docusaurus_like(SAMPLE_EN, tmp_path, base="/example/")
    _write(
        site,
        "docs/old/index.html",
        '<html><head><meta http-equiv="refresh" '
        'content="0; url=/example/docs/reference/#limits"></head></html>',
    )
    _write(
        site,
        "docs/moved/index.html",
        '<main><p><a href="/example/docs/old/">old</a> and '
        '<a href="/example/docs/old/#command-line">own</a></p></main>',
    )
    doc = load_html(site, HtmlConfig(), lang="auto")
    links = {link.text: link for block in doc.blocks for link in block.links}
    assert (links["old"].page, links["old"].fragment, links["old"].found) == (
        "docs/reference/index.html",
        "limits",
        True,
    )
    assert links["own"].target == "docs/reference/index.html#command-line"


def test_links_to_the_base_without_its_slash(tmp_path: Path) -> None:
    """The base of a site is taken away even without its final slash."""
    links = resolve_links(
        [
            ("a", "/example"),
            ("b", "/example?x=1"),
            ("c", "/example#top"),
            ("d", "/examples/"),
        ],
        "docs/page.html",
        base="example/",
        pages=frozenset({"index.html"}),
        redirects={},
        root=tmp_path,
    )
    assert [(link.page, link.fragment, link.found) for link in links] == [
        ("index.html", None, True),
        ("index.html", None, True),
        ("index.html", "top", True),
        ("examples/index.html", None, False),
    ]


def test_site_base_prefers_the_shortest_then_the_first_candidate() -> None:
    """Among candidates as good, the shortest, then the first, is the base."""
    pages = {"a/index.html"}
    assert site_base(["/long/a/", "/ab/a/"], lambda page: page in pages) == (
        "ab/"
    )
    assert site_base(["/cd/a/", "/ab/a/"], lambda page: page in pages) == (
        "ab/"
    )


def test_link_texts_are_printable(tmp_path: Path) -> None:
    """The text of a link has no control or bidi character left."""
    links = resolve_links(
        [("a\U0000202eb\x07", "a.html")],
        "index.html",
        base="",
        pages=frozenset({"a.html"}),
        redirects={},
        root=tmp_path,
    )
    assert links == (Link("a b", "a.html", page="a.html", found=True),)


def test_identifiers_of_chrome_go_to_the_next_block(tmp_path: Path) -> None:
    """An identifier on page chrome moves to the next block of its page."""
    for name in ("a", "b", "c"):
        _write(
            tmp_path,
            f"{name}.html",
            f'<main><p id="top-{name}">Shared banner of the site.</p>'
            f'<h1 id="title">Page {name}</h1></main>',
        )
    doc = load_html(tmp_path, HtmlConfig(), lang="auto")
    assert doc.chrome == ["Shared banner of the site."]
    assert [(b.text, b.anchors) for b in doc.blocks] == [
        ("Page a", ("top-a", "title")),
        ("Page b", ("top-b", "title")),
        ("Page c", ("top-c", "title")),
    ]
