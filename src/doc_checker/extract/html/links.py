# Copyright 2026 Christophe Le Douarec
"""Resolve the links of the pages of a site to URLs, pages and anchors.

A link with a scheme, or that starts with ``//``, is external: it keeps a
normalized URL. Any other link is internal: it leads to a page of the site,
through the redirection pages, and maybe to an anchor of that page. A site
served under a base, such as ``/project/``, writes its absolute links and
its redirections with that base, which is found from the links themselves.
"""

import re
import urllib.parse
from collections import Counter
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Final

from doc_checker.extract.html.ordering import INDEX_PAGE, SCHEME, resolve_href
from doc_checker.model.document import Link
from doc_checker.text.normalize import printable

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence
    from pathlib import Path

_UNSAFE_SCHEMES: Final = frozenset({"javascript", "data", "vbscript"})
_WEB_SCHEMES: Final = frozenset({"http", "https"})
_DEFAULT_PORTS: Final = frozenset({80, 443})
_MAILTO: Final = "mailto"
_SCHEME_RELATIVE: Final = "//"
_INDEX_SUFFIX: Final = f"/{INDEX_PAGE}"
_MAX_REDIRECTS: Final = 5
# What a browser takes away from a URL: the controls and spaces at either
# end, then the tabs and line breaks anywhere.
_URL_EDGES: Final = "".join(chr(code) for code in range(0x21))
_URL_BREAKS: Final = re.compile(r"[\t\n\r]")


def clean_href(href: str) -> str:
    """Return a link target as a browser reads it.

    Args:
        href: Target as written in the page, such as an ``href`` or the
            source of an image.

    Returns:
        The target without the controls and spaces at either end, then
        without its tabs and line breaks.
    """
    return _URL_BREAKS.sub("", href.strip(_URL_EDGES))


def _is_external(href: str) -> bool:
    """Return whether a cleaned link target has a scheme or a host."""
    return href.startswith(_SCHEME_RELATIVE) or bool(SCHEME.match(href))


def is_external(href: str) -> bool:
    """Return whether a target, as a browser reads it, has a scheme or a host.

    Args:
        href: Target as written in the page.

    Returns:
        Whether it has a scheme, such as ``https:`` or ``data:``, or starts
        with ``//``.
    """
    return _is_external(clean_href(href))


def _web_url(href: str) -> str:
    """Return the normalized form of an ``http``, ``https`` or ``//`` URL."""
    try:
        parts = urllib.parse.urlsplit(href)
        port = parts.port
    except ValueError:
        return href
    host = parts.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    if port is not None and port not in _DEFAULT_PORTS:
        host = f"{host}:{port}"
    path = parts.path.removesuffix(_INDEX_SUFFIX).rstrip("/")
    query = f"?{parts.query}" if parts.query else ""
    fragment = f"#{parts.fragment}" if parts.fragment else ""
    return f"//{host}{path}{query}{fragment}"


def normalize_url(href: str) -> str | None:
    """Return the form of an external link that two builds can compare.

    The controls and spaces at either end of the target, and its tabs and
    line breaks, are taken away first, as a browser does.

    Args:
        href: Target of the link, as written in the page.

    Returns:
        ``None`` for the ``javascript``, ``data`` and ``vbscript`` schemes,
        whatever their case. For ``http``, ``https`` and ``//``: ``//``,
        the host in lower case, its port unless 80 or 443, the path without
        ``/index.html`` nor a final ``/``, then the query and the fragment
        as written; a URL that cannot be parsed is kept. For ``mailto``,
        the address in lower case and the query as written. For another
        scheme, the scheme in lower case and the rest as written. A target
        without a scheme is kept.
    """
    href = clean_href(href)
    if href.startswith(_SCHEME_RELATIVE):
        return _web_url(href)
    match = SCHEME.match(href)
    if match is None:
        return href
    scheme, rest = href[: match.end() - 1].lower(), href[match.end() :]
    if scheme in _UNSAFE_SCHEMES:
        return None
    if scheme in _WEB_SCHEMES:
        return _web_url(href)
    if scheme == _MAILTO:
        address, mark, query = rest.partition("?")
        return f"{_MAILTO}:{address.lower()}{mark}{query}"
    return f"{scheme}:{rest}"


def _without_base(href: str, base: str) -> str:
    """Return an absolute target without the base of the site, if it has it.

    The base is taken away with or without its final ``/``: ``/project``,
    ``/project?q`` and ``/project#a`` lead to the root of the site, as
    ``/project/`` does.
    """
    if not base:
        return href
    prefix = f"/{base}"
    stem = prefix.removesuffix("/")
    if href.startswith(prefix):
        return "/" + href.removeprefix(prefix)
    if href == stem or href.startswith((f"{stem}?", f"{stem}#")):
        return "/" + href.removeprefix(stem)
    return href


def site_path(href: str, page: str, base: str) -> str | None:
    """Return the site relative path of an internal target, as a link's.

    The target is cleaned as a browser does, loses the base of the site if
    it is an absolute path that has it, then is resolved against its page,
    as `resolve_href` does.

    Args:
        href: Target as written in the page, such as the source of an image.
        page: Site relative path of the page that holds it.
        base: Base of the absolute links of the site, as `site_base` gives
            it.

    Returns:
        The POSIX path, relative to the site root; ``None`` for an external
        target, a target without a path, or one that goes above the root.
    """
    cleaned = clean_href(href)
    if _is_external(cleaned):
        return None
    return resolve_href(_without_base(cleaned, base), page)


def page_exists(pages: frozenset[str], root: Path) -> Callable[[str], bool]:
    """Return the check that a site relative page path exists.

    Args:
        pages: Site relative paths of the pages read.
        root: Directory of the site.

    Returns:
        A cached check: the page was read, or is a file under the root.
    """

    @cache
    def exists(page: str) -> bool:
        return page in pages or (root / page).is_file()

    return exists


def site_base(hrefs: Iterable[str], exists: Callable[[str], bool]) -> str:
    """Return the base under which a site writes its absolute links.

    The candidates are ``""`` and the first part, such as ``"project/"``,
    of the path of each link that starts with ``/`` but not ``//``. The
    base is the candidate whose removal from these links makes the most
    target pages exist; a link without the candidate stays as written.

    Args:
        hrefs: Targets of the links of the site, as written.
        exists: Tells whether a site relative page path exists.

    Returns:
        The best candidate: ``""`` on a tie, then the shortest one, then
        the first in alphabetical order.
    """

    def existing(path: str) -> str | None:
        target = resolve_href(path, INDEX_PAGE)
        return target if target is not None and exists(target) else None

    # Only the path of a link leads to a page: the links to the anchors of
    # one page count once.
    paths = dict.fromkeys(
        urllib.parse.urlsplit(href).path
        for href in map(clean_href, hrefs)
        if href.startswith("/") and not href.startswith(_SCHEME_RELATIVE)
    )
    # The targets of the paths as written, counted, are those of ``""``.
    # A candidate only changes the paths that start with it.
    plain = {path: existing(path) for path in paths}
    counts = Counter(target for target in plain.values() if target)
    groups: dict[str, list[str]] = {}
    candidates = dict.fromkeys([""])
    for path in paths:
        first, slash, _ = path[1:].partition("/")
        if first:
            groups.setdefault(f"{first}/", []).append(path)
        if first and slash:
            candidates.setdefault(f"{first}/")

    def found(base: str) -> int:
        if not base:
            return len(counts)
        group = groups[base]
        removed = Counter(plain[path] for path in group if plain[path])
        lost = sum(counts[target] == n for target, n in removed.items())
        added = dict.fromkeys(
            existing(_without_base(path, base)) for path in group
        )
        gained = sum(
            counts[target] == removed[target] for target in added if target
        )
        return len(counts) - lost + gained

    return min(
        candidates,
        key=lambda base: (-found(base), bool(base), len(base), base),
    )


def _internal(
    href: str,
    page: str,
    base: str,
) -> tuple[str | None, str | None]:
    """Return the page and the anchor of a cleaned internal target.

    The page is ``None`` when the target goes above the root of the site.
    """
    parts = urllib.parse.urlsplit(href)
    fragment = printable(urllib.parse.unquote(parts.fragment)) or None
    if not parts.path:
        return page, fragment
    return resolve_href(_without_base(href, base), page), fragment


@dataclass(frozen=True, slots=True)
class LinkResolver:
    """Resolve the links of the blocks of a site, as `resolve_links` does.

    One resolver serves every block of a site, so that its existence check
    keeps what it found.

    Attributes:
        base: Base of the absolute links of the site, as `site_base` gives
            it.
        redirects: URL of each redirection page, as written in it.
        exists: Tells whether a site relative page path exists, as
            `page_exists` does.
    """

    base: str
    redirects: Mapping[str, str]
    exists: Callable[[str], bool]

    def __call__(
        self,
        raw: Sequence[tuple[str, str]],
        page: str,
    ) -> tuple[Link, ...]:
        """Return the links of a block.

        Args:
            raw: Text and target, as written, of each link of the block.
            page: Site relative path of the page that holds the block.

        Returns:
            The links, in order.
        """
        return tuple(
            link
            for text, href in raw
            if (link := self.link(text, href, page)) is not None
        )

    def follow(
        self,
        page: str,
        fragment: str | None,
    ) -> tuple[str, str | None]:
        """Return where the redirections from a page lead, five at most.

        The anchor of a redirection serves a link that has none.
        """
        for _ in range(_MAX_REDIRECTS):
            href = self.redirects.get(page)
            if href is None:
                break
            target, anchor = _internal(clean_href(href), page, self.base)
            if target is None:
                break
            page, fragment = target, fragment or anchor
        return page, fragment

    def link(self, text: str, href: str, page: str) -> Link | None:
        """Return a resolved link, or ``None`` for a link to leave out."""
        cleaned = clean_href(href)
        if not cleaned or cleaned == "#":
            return None
        text, shown = printable(text), printable(href)
        if _is_external(cleaned):
            url = normalize_url(cleaned)
            return None if url is None else Link(text, shown, url=url)
        target, fragment = _internal(cleaned, page, self.base)
        if target is None:
            return Link(text, shown, fragment=fragment)
        target, fragment = self.follow(target, fragment)
        return Link(
            text,
            shown,
            page=target,
            fragment=fragment,
            found=self.exists(target),
        )


def resolve_links(  # ruff: ignore[too-many-arguments] - the links, their page, and the four facts of the site that resolve them
    raw: Sequence[tuple[str, str]],
    page: str,
    *,
    base: str,
    pages: frozenset[str],
    redirects: Mapping[str, str],
    root: Path,
) -> tuple[Link, ...]:
    """Resolve the links of a block.

    A target that is empty, ``#`` alone, or that `normalize_url` leaves out
    gives no link. A target with a scheme, or that starts with ``//``, is
    external. A target without a path, such as ``#anchor``, leads to the
    page that holds it. Any other target loses the base of the site, if it
    is an absolute path that has it, then leads to a page of the site, or
    to no page if it goes above the root of the site. The redirections
    are followed, five at most, their URLs resolved as the links are: the
    anchor of a redirection serves a link that has none.

    Args:
        raw: Text and target, as written, of each link of the block.
        page: Site relative path of the page that holds the block.
        base: Base of the absolute links of the site, as `site_base`
            gives it.
        pages: Site relative paths of the pages read.
        redirects: URL of each redirection page, as written in it.
        root: Directory of the site.

    Returns:
        The links, in order. The text of a link is made printable. The
        fragment of an internal link is decoded and made printable, so
        that it compares with the anchors.
    """
    return LinkResolver(base, redirects, page_exists(pages, root))(raw, page)
