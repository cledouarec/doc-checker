# Copyright 2026 Christophe Le Douarec
"""Find and parse the pages of an HTML file or site directory."""

import codecs
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Final

import lxml.etree
import lxml.html

from doc_checker.errors import InputError
from doc_checker.extract.html.ordering import path_sort_key, resolve_href

if TYPE_CHECKING:
    from lxml.html import HtmlElement

    from doc_checker.config import HtmlConfig

_BOMS: Final = (
    (codecs.BOM_UTF8, "utf-8"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)
_SNIFF_BYTES: Final = 2048
_CHARSET: Final = re.compile(
    rb"""charset\s*=\s*["']?([\w-]+)""",
    re.IGNORECASE,
)
_REFRESH_URL: Final = re.compile(
    r"""url\s*=\s*['"]?([^'"\s;]+)""",
    re.IGNORECASE,
)
MAX_PAGE_BYTES: Final = 100_000_000
_DEFAULT_CHARSET: Final = "utf-8"
_REFRESH: Final = "refresh"
# The marks of a CDATA section and of a comment, in the bytes of a page.
_CDATA_OPEN: Final = b"<![CDATA["
_CDATA_CLOSE: Final = b"]]>"
_COMMENT_OPEN: Final = b"<!--"
_COMMENT_CLOSE: Final = b"-->"


@dataclass(frozen=True, slots=True)
class HtmlPage:
    """A parsed page of a documentation site.

    Attributes:
        rel_path: POSIX path of the page, relative to the site root.
        tree: Parsed document.
        redirect_href: URL of the redirection as written in the page, when
            the page only redirects to a page of the site; the loader
            resolves it once the base of the site is known.
    """

    rel_path: str
    tree: HtmlElement
    redirect_href: str | None = None


def discover_pages(root: Path, cfg: HtmlConfig) -> list[tuple[Path, str]]:
    """Return the pages of an HTML file or site directory.

    Args:
        root: HTML file or site directory.
        cfg: HTML settings with the include and exclude patterns.

    Returns:
        Pairs of absolute path and site relative POSIX path, in path order.
    """
    if root.is_file():
        return [(root, root.name)]
    found = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = PurePosixPath(path.relative_to(root).as_posix())
        included = any(rel.full_match(pattern) for pattern in cfg.include)
        excluded = any(rel.full_match(pattern) for pattern in cfg.exclude)
        if included and not excluded:
            found.append((path, str(rel)))
    return sorted(found, key=lambda item: path_sort_key(item[1]))


def _charset(data: bytes) -> str:
    """Return the charset of a page: BOM, then declaration, then UTF-8."""
    for bom, name in _BOMS:
        if data.startswith(bom):
            return name
    match = _CHARSET.search(data[:_SNIFF_BYTES])
    if match:
        name = match.group(1).decode("ascii", "replace")
        try:
            codecs.lookup(name)
        except LookupError:
            return _DEFAULT_CHARSET
        return name
    return _DEFAULT_CHARSET


def _refresh_url(tree: HtmlElement) -> str | None:
    """Return the URL of a meta refresh redirection, as written, if any."""
    for meta in tree.iter("meta"):
        if (meta.get("http-equiv") or "").casefold() == _REFRESH:
            match = _REFRESH_URL.search(meta.get("content") or "")
            if match:
                return match.group(1)
    return None


@cache
def drops_cdata() -> bool:
    """Tell whether the HTML parser drops the CDATA sections of a page.

    libxml2 before 2.14, which the Windows wheels of lxml bundle, drops a
    CDATA section of the content, such as the text of an inline SVG, while
    libxml2 2.14 keeps it as a comment that holds ``[CDATA[``, its text and
    ``]]``. The parser is asked once, with a paragraph that holds a section.

    Returns:
        Whether the paragraph comes back without a comment.
    """
    probe = lxml.html.fragment_fromstring("<p><![CDATA[x]]></p>")
    return not any(node.tag is lxml.etree.Comment for node in probe.iter())


def keep_cdata(data: bytes) -> bytes:
    """Turn the CDATA sections of a page into the comments of libxml2 2.14.

    Each section, from ``<![CDATA[`` to the first ``]]>`` after it, becomes
    a comment that holds ``[CDATA[``, its text and ``]]``, which the SVG
    reader takes back, so that every parser reads it alike. A section inside
    a comment is left as it is, and so is the rest of the page from a
    section or a comment that is not closed. Each search starts where the
    last one ended: a hostile page costs a few reads of its bytes at most.

    Args:
        data: The bytes of the page.

    Returns:
        The bytes, with ``--`` added inside the ``<!`` and the ``>`` that
        bound each section.
    """
    parts: list[bytes] = []
    copied = start = 0
    begin = data.find(_CDATA_OPEN)
    while begin != -1:
        comment = data.find(_COMMENT_OPEN, start, begin)
        if comment != -1:
            close = data.find(_COMMENT_CLOSE, comment + len(_COMMENT_OPEN))
            if close == -1:
                break
            start = close + len(_COMMENT_CLOSE)
            if begin < start:
                begin = data.find(_CDATA_OPEN, start)
            continue
        end = data.find(_CDATA_CLOSE, begin + len(_CDATA_OPEN))
        if end == -1:
            break
        # "<![CDATA[x]]>" becomes "<!--" + "[CDATA[x]]" + "-->".
        parts.extend(
            (
                data[copied:begin],
                _COMMENT_OPEN,
                data[begin + 2 : end + 2],
                _COMMENT_CLOSE,
            ),
        )
        copied = start = end + len(_CDATA_CLOSE)
        begin = data.find(_CDATA_OPEN, start)
    parts.append(data[copied:])
    return b"".join(parts)


def parse_page(path: Path, rel_path: str) -> HtmlPage:
    """Read and parse one HTML page.

    A parser that drops the CDATA sections of a page, as `drops_cdata`
    tells, reads the page that `keep_cdata` makes, as libxml2 2.14 reads it.

    Args:
        path: File to read.
        rel_path: Site relative path of the page.

    Returns:
        The parsed page.

    Raises:
        InputError: If the file cannot be read, is larger than
            `MAX_PAGE_BYTES`, or holds no HTML.
    """
    try:
        size = path.stat().st_size
        if size > MAX_PAGE_BYTES:
            msg = (
                f"cannot read {rel_path}: {size} bytes, more than the "
                f"{MAX_PAGE_BYTES} bytes that a page may have"
            )
            raise InputError(msg)
        data = path.read_bytes()
    except OSError as exc:
        msg = f"cannot read {rel_path}: {exc.strerror}"
        raise InputError(msg) from None
    invalid = f"cannot parse {rel_path}: empty or invalid HTML"
    if not data.strip():
        raise InputError(invalid)
    if drops_cdata():
        data = keep_cdata(data)
    # Without huge_tree, libxml2 stops at a text or an attribute of ten
    # million characters, such as an embedded image, and drops the rest of
    # the page silently. The walk of the blocks bounds the nesting.
    parser = lxml.html.HTMLParser(encoding=_charset(data), huge_tree=True)
    try:
        tree = lxml.html.document_fromstring(data, parser=parser)
    except lxml.etree.ParserError, ValueError:
        raise InputError(invalid) from None
    url = _refresh_url(tree)
    if url is None or resolve_href(url, rel_path) is None:
        return HtmlPage(rel_path, tree)
    return HtmlPage(rel_path, tree, url)
