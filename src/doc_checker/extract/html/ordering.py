# Copyright 2026 Christophe Le Douarec
"""Resolve internal links and order the pages of a site."""

import posixpath
import re
import urllib.parse
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Sequence

    from doc_checker.extract.html.site import HtmlPage

_SCHEME: Final = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")
_DIGITS: Final = re.compile(r"(\d+)")
_INDEX_NAMES: Final = frozenset({"index.html", "index.htm"})
_INDEX_PAGE: Final = "index.html"
_PARENT: Final = ".."

type SortKey = tuple[tuple[int, tuple[int | str, ...]], ...]


def resolve_href(href: str, page_rel: str) -> str | None:
    """Resolve a link to the site relative path of the target page.

    Args:
        href: Link target as written in the page.
        page_rel: Site relative path of the page that holds the link.

    Returns:
        The target page path, or ``None`` for external links, anchors in
        the same page and paths outside the site.
    """
    href = href.strip()
    if not href or href.startswith(("#", "//")) or _SCHEME.match(href):
        return None
    path = urllib.parse.unquote(urllib.parse.urlsplit(href).path)
    if not path:
        return None
    if path.startswith("/"):
        joined = posixpath.normpath(path.lstrip("/"))
    else:
        joined = posixpath.normpath(
            posixpath.join(posixpath.dirname(page_rel), path),
        )
    if joined == _PARENT or joined.startswith(f"{_PARENT}/"):
        return None
    if path.endswith("/") or not posixpath.splitext(joined)[1]:
        joined = posixpath.normpath(posixpath.join(joined, _INDEX_PAGE))
    return joined


def _natural(segment: str) -> tuple[int | str, ...]:
    """Return a key that sorts the numbers of a name numerically."""
    parts = _DIGITS.split(segment.casefold())
    return tuple(
        int(part) if index % 2 else part for index, part in enumerate(parts)
    )


def path_sort_key(rel_path: str) -> SortKey:
    """Return the key that sorts page paths, index pages first.

    Args:
        rel_path: Site relative page path.

    Returns:
        A key comparing path segments naturally, index pages first.
    """
    return tuple(
        (0 if segment.casefold() in _INDEX_NAMES else 1, _natural(segment))
        for segment in rel_path.split("/")
    )


def nav_links(page: HtmlPage, selectors: Sequence[str]) -> list[str]:
    """Return the internal pages linked from the navigation of a page.

    Args:
        page: Page holding the navigation, usually the entry page.
        selectors: Navigation selectors, the first that yields a link wins.

    Returns:
        Site relative page paths in navigation order, without duplicates.
    """
    for selector in selectors:
        links: dict[str, None] = {}
        for container in page.tree.cssselect(selector):
            for anchor in container.iter("a"):
                target = resolve_href(anchor.get("href") or "", page.rel_path)
                if target is not None:
                    links.setdefault(target)
        if links:
            return list(links)
    return []


def entry_page(pages: Sequence[HtmlPage]) -> HtmlPage:
    """Return the entry page of a site.

    Args:
        pages: Pages of the site, at least one.

    Returns:
        The root index page, or the first page in path order.
    """
    for page in pages:
        if page.rel_path.casefold() in _INDEX_NAMES:
            return page
    return min(pages, key=lambda page: path_sort_key(page.rel_path))


def order_pages(
    pages: Sequence[HtmlPage],
    *,
    mode: str,
    nav: Sequence[str],
) -> list[HtmlPage]:
    """Return the pages in reading order.

    Args:
        pages: Pages of the site.
        mode: ``"path"`` to sort by path, ``"auto"`` or ``"nav"`` to follow
            the navigation first.
        nav: Site relative paths in navigation order.

    Returns:
        The ordered pages.
    """
    by_path = sorted(pages, key=lambda page: path_sort_key(page.rel_path))
    if mode == "path" or not pages:
        return by_path
    known = {page.rel_path: page for page in pages}
    entry = entry_page(pages)
    ordered = {entry.rel_path: entry}
    for target in nav:
        if target in known:
            ordered.setdefault(target, known[target])
    for page in by_path:
        ordered.setdefault(page.rel_path, page)
    return list(ordered.values())
