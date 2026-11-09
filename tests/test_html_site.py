# Copyright 2026 Christophe Le Douarec
"""Tests for site discovery, page parsing, ordering and chrome detection."""

from typing import TYPE_CHECKING

import lxml.html
import pytest

from doc_checker.config import HtmlConfig
from doc_checker.errors import InputError
from doc_checker.extract.html.chrome import find_chrome
from doc_checker.extract.html.ordering import (
    order_pages,
    path_sort_key,
    resolve_href,
)
from doc_checker.extract.html.site import HtmlPage, discover_pages, parse_page
from tests.builders import make_block

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, html: str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = html.encode() if isinstance(html, str) else html
    path.write_bytes(data)
    return path


def test_discover_pages_include_exclude(tmp_path: Path) -> None:
    """Site pages are found with the include and exclude patterns."""
    for rel in (
        "index.html",
        "guide/a.html",
        "_static/x.html",
        "genindex.html",
    ):
        _write(tmp_path / rel, "<p>x</p>")
    _write(tmp_path / "notes.txt", "x")
    found = [rel for _, rel in discover_pages(tmp_path, HtmlConfig())]
    assert found == ["index.html", "guide/a.html"]


def test_discover_single_file_and_non_ascii_names(tmp_path: Path) -> None:
    """A single file is one page and non ASCII names are kept."""
    page = _write(tmp_path / "guide" / "café.html", "<p>x</p>")
    assert discover_pages(page, HtmlConfig()) == [(page, "café.html")]
    found = [rel for _, rel in discover_pages(tmp_path, HtmlConfig())]
    assert found == ["guide/café.html"]


def test_parse_page_defaults_to_utf8(tmp_path: Path) -> None:
    """Pages without a declared charset are read as UTF-8."""
    path = _write(
        tmp_path / "a.html",
        "<html><body><p>Été déjà</p></body></html>",
    )
    assert "Été déjà" in parse_page(path, "a.html").tree.text_content()


def test_parse_page_honours_declared_charset(tmp_path: Path) -> None:
    """A declared charset is used to decode the page."""
    html = (
        '<html><head><meta charset="iso-8859-1"></head>'
        "<body><p>Été</p></body></html>"
    )
    path = _write(tmp_path / "a.html", html.encode("iso-8859-1"))
    assert "Été" in parse_page(path, "a.html").tree.text_content()


def test_parse_empty_page_raises(tmp_path: Path) -> None:
    """An empty page is an input error."""
    path = _write(tmp_path / "a.html", b"")
    with pytest.raises(InputError, match=r"cannot parse a\.html"):
        parse_page(path, "a.html")


def test_redirect_page(tmp_path: Path) -> None:
    """Meta refresh pages are recognized as redirections."""
    html = (
        '<html><head><meta http-equiv="Refresh" '
        'content="0; url=guide/a.html"></head></html>'
    )
    path = _write(tmp_path / "old.html", html)
    page = parse_page(path, "old.html")
    assert page.redirect_href == "guide/a.html"
    away = _write(tmp_path / "away.html", html.replace("guide", "https://x"))
    page = parse_page(away, "away.html")
    assert page.redirect_href is None


@pytest.mark.parametrize(
    ("href", "page", "expected"),
    [
        ("../intro/", "guide/index.html", "intro/index.html"),
        ("a.html#x", "guide/b.html", "guide/a.html"),
        ("https://x.org/a.html", "index.html", None),
        ("#top", "index.html", None),
        ("mailto:a@b.c", "index.html", None),
        ("../../out.html", "index.html", None),
        ("caf%C3%A9.html", "guide/index.html", "guide/café.html"),
    ],
)
def test_resolve_href(href: str, page: str, expected: str | None) -> None:
    """Links resolve to site relative pages, or to nothing."""
    assert resolve_href(href, page) == expected


def test_path_sort_key() -> None:
    """Index pages come first and numbers sort naturally."""
    paths = ["b/10.html", "b/9.html", "b/index.html", "a.html", "index.html"]
    assert sorted(paths, key=path_sort_key) == [
        "index.html",
        "a.html",
        "b/index.html",
        "b/9.html",
        "b/10.html",
    ]


def test_order_pages_nav_then_path() -> None:
    """Pages follow the navigation, then the path order."""
    pages = [
        HtmlPage(rel, lxml.html.fromstring("<p>x</p>"))
        for rel in ("c.html", "a.html", "index.html", "b.html")
    ]
    ordered = order_pages(pages, mode="auto", nav=["b.html", "a.html"])
    assert [p.rel_path for p in ordered] == [
        "index.html",
        "b.html",
        "a.html",
        "c.html",
    ]
    ordered = order_pages(pages, mode="path", nav=["b.html"])
    assert [p.rel_path for p in ordered] == [
        "index.html",
        "a.html",
        "b.html",
        "c.html",
    ]


def test_find_chrome() -> None:
    """Blocks repeated at the edges of most pages are chrome."""
    banner = make_block("Docs for version 2.0.")
    pages = [
        [banner, make_block(f"Unique content number {i}.")] for i in range(4)
    ]
    assert find_chrome(pages) == frozenset({banner.key_strict})
    assert find_chrome(pages[:2]) == frozenset()
    middle = [
        [make_block(f"Filler {i} {j}.") for j in range(12)] for i in range(4)
    ]
    for page in middle:
        page[6] = banner
    assert find_chrome(middle) == frozenset()


def test_page_over_the_size_limit_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A page larger than the limit is not read, and the error says why."""
    monkeypatch.setattr("doc_checker.extract.html.site.MAX_PAGE_BYTES", 1_000)

    def unread(_self: Path) -> bytes:
        msg = "the page was read"
        raise AssertionError(msg)

    monkeypatch.setattr("pathlib.Path.read_bytes", unread)
    page = tmp_path / "big.html"
    page.write_text("<p>" + "word " * 400 + "</p>", encoding="utf-8")
    with pytest.raises(InputError, match="more than the 1000 bytes"):
        parse_page(page, "big.html")
