# Copyright 2026 Christophe Le Douarec
"""Find the images of HTML pages and read them safely.

An image is an ``img`` element, whose source is the largest candidate of its
``srcset``, else its ``src``, or an inline ``svg`` element, which is
identified by its source and its texts. Icons, the small ``svg`` of a theme
and the emoji images, are not images of the content.

A source is resolved as a link is, and its file is read only when it is
under the root of the site once the symbolic links are followed, and when
it is not too large. An external image is never fetched, a data URL is
decoded only below `DATA_URL_MAX_CHARS` characters, and an SVG is parsed
without entities, a DTD or the network, never rendered.
"""

import base64
import codecs
import posixpath
import re
import urllib.parse
from dataclasses import dataclass
from operator import itemgetter
from typing import TYPE_CHECKING, Final

from doc_checker.extract.html.links import (
    clean_href,
    is_external,
    normalize_url,
    site_path,
)
from doc_checker.images.pixels import (
    CANNOT_DECODE,
    EXTERNAL,
    MAX_FILE_BYTES,
    NOT_FOUND,
    OUTSIDE,
    TOO_LARGE,
    UnreadableImage,
    read_image_file,
)
from doc_checker.images.refs import content_digest, raster_ref, unreadable_ref
from doc_checker.images.svg import (
    ICON_CLASSES,
    is_icon,
    parse_svg,
    svg_identity,
)
from doc_checker.model.document import ImageRef, PixelSource
from doc_checker.text.normalize import printable

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from lxml.html import HtmlElement

    from doc_checker.images.svg import SvgIdentity

DATA_URL_MAX_CHARS: Final = 16_000_000
# The classes of emoji images, the same as those of emoji drawn in SVG.
ICON_IMG_CLASSES: Final = ICON_CLASSES

_SVG_TAG: Final = "svg"
_SVG_FORMAT: Final = "svg"
_SVG_MIME: Final = "image/svg+xml"
_SVG_STARTS: Final = (b"<?xml", b"<svg")
_SNIFF_BYTES: Final = 1024  # Head of a file in which an SVG start is sought.
_BASE64: Final = "base64"
_ALT_ATTRIBUTES: Final = ("alt", "aria-label")
# A data URL, once cleaned as a browser reads it.
_DATA_URL: Final = re.compile(r"data:", re.IGNORECASE)
# The URL of a srcset candidate: a run without ASCII whitespace, which
# starts after the blanks and commas that separate the candidates.
_CANDIDATE_URL: Final = re.compile(r"[^\t\n\f\r ,][^\t\n\f\r ]*")
_SPACE_BYTES: Final = re.compile(rb"[\t\n\f\r ]+")
_WIDTH: Final = re.compile(r"[0-9]+w")
_DENSITY: Final = re.compile(r"(?:[0-9]*\.)?[0-9]+(?:[eE][-+]?[0-9]+)?x")
_INTEGER: Final = re.compile(r"[0-9]+")


@dataclass(frozen=True, slots=True)
class RawImage:
    """An image as found in a page, before its source is read.

    Attributes:
        src: Source of an ``img``, as `best_source` chooses it.
        alt: Printable alternative text: the ``alt`` attribute, else the
            ``aria-label`` one; ``None`` without either.
        shown: Width and height of the ``img`` attributes, when both are
            integers, or the size of an inline SVG.
        svg: Identity of an inline SVG.
    """

    src: str | None
    alt: str | None
    shown: tuple[float, float] | None
    svg: SvgIdentity | None


def _candidates(srcset: str) -> Iterator[tuple[str, str]]:
    """Yield the URL and the descriptors of each candidate of a srcset.

    A URL is a run of characters other than blanks, so that it can hold
    commas, as a data URL does; the commas that end it end the candidate.
    Otherwise its descriptors run up to the next comma.
    """
    position = 0
    while (found := _CANDIDATE_URL.search(srcset, position)) is not None:
        url, position = found[0], found.end()
        if url.endswith(","):
            yield url.rstrip(","), ""
            continue
        end = srcset.find(",", position)
        end = len(srcset) if end < 0 else end
        yield url, srcset[position:end]
        position = end


def _descriptor(descriptors: str) -> tuple[bool, float] | None:
    """Return whether a candidate gives a width, and its width or density.

    Args:
        descriptors: Descriptors of the candidate, such as ``800w`` or
            ``2x``; none means ``1x``.

    Returns:
        ``(True, width)`` or ``(False, density)``; ``None`` for descriptors
        that tell neither.
    """
    tokens = descriptors.split()
    if not tokens:
        return False, 1.0
    for token in tokens:
        if _WIDTH.fullmatch(token):
            return True, float(token[:-1])
        if _DENSITY.fullmatch(token):
            return False, float(token[:-1])
    return None


def best_source(src: str | None, srcset: str | None) -> str | None:
    """Return the source of the largest version of an image.

    Args:
        src: The ``src`` attribute, if any.
        srcset: The ``srcset`` attribute, if any: candidates separated by
            commas, each a URL and maybe a descriptor.

    Returns:
        The candidate of the largest width, if any gives a width, else the
        one of the largest density, a candidate without a descriptor
        counting as ``1x``; the first one on a tie. Without a candidate,
        ``src`` without its blanks, or ``None`` if nothing is left.
    """
    widths: list[tuple[float, str]] = []
    densities: list[tuple[float, str]] = []
    for url, descriptors in _candidates(srcset or ""):
        found = _descriptor(descriptors)
        if found is not None:
            is_width, value = found
            (widths if is_width else densities).append((value, url))
    for candidates in (widths, densities):
        if candidates:
            return max(candidates, key=itemgetter(0))[1]
    return (src or "").strip() or None


def _is_emoji(element: HtmlElement) -> bool:
    """Return whether an image has, or is in, an emoji class."""
    for node in (element, *element.iterancestors()):
        classes = (node.get("class") or "").lower().split()
        if ICON_IMG_CLASSES.intersection(classes):
            return True
    return False


def _alt(element: HtmlElement) -> str | None:
    """Return the printable alternative text of an image, if it has one."""
    for name in _ALT_ATTRIBUTES:
        value = element.get(name)
        if value is not None:
            return printable(value)
    return None


def _shown(element: HtmlElement) -> tuple[float, float] | None:
    """Return the width and height attributes, when both are integers."""
    width = (element.get("width") or "").strip()
    height = (element.get("height") or "").strip()
    if _INTEGER.fullmatch(width) and _INTEGER.fullmatch(height):
        return float(width), float(height)
    return None


def raw_image(element: HtmlElement) -> RawImage | None:
    """Return the image of an ``img`` or ``svg`` element.

    Args:
        element: The element, from lxml's HTML parser.

    Returns:
        The image, or ``None`` for an icon: an ``svg`` that `is_icon`
        tells, or an ``img`` that has, or is in, a class of
        `ICON_IMG_CLASSES`.
    """
    if element.tag == _SVG_TAG:
        if is_icon(element):
            return None
        identity = svg_identity(element)
        shown = None
        if identity.width is not None and identity.height is not None:
            shown = (identity.width, identity.height)
        return RawImage(src=None, alt=None, shown=shown, svg=identity)
    if _is_emoji(element):
        return None
    return RawImage(
        src=best_source(element.get("src"), element.get("srcset")),
        alt=_alt(element),
        shown=_shown(element),
        svg=None,
    )


def _format(path: str) -> str:
    """Return the lower case suffix of a path, without its dot, or ""."""
    return posixpath.splitext(path)[1].removeprefix(".").lower()


def _url_path(url: str) -> str:
    """Return the path of a URL, or the URL if it cannot be parsed."""
    try:
        return urllib.parse.urlsplit(url).path
    except ValueError:
        return url


def _unreadable(  # ruff: ignore[too-many-arguments] - the facts of an image without pixels
    raw: RawImage,
    name: str,
    problem: str,
    *,
    data: bytes | None = None,
    svg: bool = False,
    size: int | None = None,
) -> ImageRef:
    """Return an image whose pixels are not read.

    Args:
        raw: The image as found in the page.
        name: Its name.
        problem: Why its pixels are not read.
        data: Its bytes, if they were read.
        svg: Whether it is an SVG.
        size: Its size in bytes, if known.

    Returns:
        The image, digested by its bytes, else by its name, else by its
        source as written, such as a data URL: two references to the same
        missing file are the same image, two broken data URLs are not.
    """
    if data is None:
        data = (name or raw.src or "").encode()
    return unreadable_ref(
        name=name,
        format=_SVG_FORMAT if svg else _format(_url_path(name)),
        digest=content_digest(data),
        problem=problem,
        alt=raw.alt,
        shown=raw.shown,
        size=size,
        vector=svg,
    )


def _svg_ref(raw: RawImage, name: str, data: bytes) -> ImageRef:
    """Return the image of the bytes of an SVG."""
    try:
        identity = parse_svg(data)
    except UnreadableImage as error:
        return _unreadable(
            raw,
            name,
            error.reason,
            data=data,
            svg=True,
            size=len(data),
        )
    return ImageRef(
        name=name,
        format=_SVG_FORMAT,
        digest=identity.digest,
        vector=True,
        texts=identity.texts,
        alt=raw.alt,
        shown=raw.shown,
        size=len(data),
    )


def _looks_like_svg(data: bytes) -> bool:
    """Return whether bytes start, after blanks, an XML or SVG document."""
    head = data[:_SNIFF_BYTES].removeprefix(codecs.BOM_UTF8).lstrip()
    return head.startswith(_SVG_STARTS)


def _data_url_ref(raw: RawImage, url: str, start: int) -> ImageRef:
    """Return the image of a data URL.

    Args:
        raw: The image as found in the page.
        url: The data URL, at most `DATA_URL_MAX_CHARS` characters long.
        start: Where its media type starts, after ``data:``.

    Returns:
        The image, with `CANNOT_DECODE` if its data cannot be decoded.
    """
    header, _, payload = url[start:].partition(",")
    params = [param.strip().lower() for param in header.split(";")]
    data = urllib.parse.unquote_to_bytes(payload)
    if len(params) > 1 and params[-1] == _BASE64:
        data = _SPACE_BYTES.sub(b"", data)
        try:
            data = base64.b64decode(
                data + b"=" * (-len(data) % 4),
                validate=True,
            )
        except ValueError:
            return _unreadable(raw, "", CANNOT_DECODE)
    if params[0] == _SVG_MIME:
        return _svg_ref(raw, "", data)
    return raster_ref(
        data,
        name="",
        source=PixelSource(data=data),
        alt=raw.alt,
        shown=raw.shown,
    )


@dataclass(frozen=True, slots=True)
class ImageResolver:
    """Read the images of the pages of a site.

    One resolver serves every page of a site. It reads the limits of this
    module when it is called.

    Attributes:
        root: Directory of the site; no file out of it is read.
        base: Base of the absolute links of the site, as `site_base` gives
            it.
    """

    root: Path
    base: str

    def __call__(self, raw: RawImage, page: str) -> ImageRef:
        """Return the image of a page, its pixels read if it can be.

        Args:
            raw: The image as found in the page.
            page: Site relative path of the page.

        Returns:
            The image. An inline SVG is identified by its source. A source
            longer than `DATA_URL_MAX_CHARS`, a data URL in practice, is
            `TOO_LARGE`, never cleaned nor decoded. The source is then read
            as a browser does, without its tabs and line breaks: an image
            without a source is `NOT_FOUND`; a data URL is decoded; an
            external image is `EXTERNAL`, named by its normalized URL.
            Another source is a file of the site: `OUTSIDE` if it goes
            above the root, or is not under it once the symbolic links are
            followed; `NOT_FOUND` if it has no path, such as ``#part``, or
            is not a file; `TOO_LARGE` beyond `MAX_FILE_BYTES`; else read
            as an SVG or a raster image.
        """
        if raw.svg is not None:
            return ImageRef(
                name="",
                format=_SVG_FORMAT,
                digest=raw.svg.digest,
                vector=True,
                texts=raw.svg.texts,
                alt=raw.alt,
                shown=raw.shown,
            )
        if len(raw.src or "") > DATA_URL_MAX_CHARS:
            return _unreadable(raw, "", TOO_LARGE)
        target = clean_href(raw.src or "")
        if not target:
            return _unreadable(raw, "", NOT_FOUND)
        data_url = _DATA_URL.match(target)
        if data_url is not None:
            return _data_url_ref(raw, target, data_url.end())
        if is_external(target):
            name = normalize_url(target) or printable(target)
            return _unreadable(raw, name, EXTERNAL)
        return self._file_ref(raw, target, page)

    def _site_file(self, path: str) -> Path:
        """Return the file of a site relative path, its links followed.

        Args:
            path: Site relative path.

        Returns:
            The resolved file.

        Raises:
            UnreadableImage: `OUTSIDE` if the file is not under the root
                once the symbolic links are followed, `NOT_FOUND` if the
                path cannot be resolved.
        """
        try:
            file = (self.root / path).resolve()
            inside = file.is_relative_to(self.root.resolve())
        except OSError, ValueError:
            raise UnreadableImage(NOT_FOUND) from None
        if not inside:
            raise UnreadableImage(OUTSIDE)
        return file

    def _file_ref(self, raw: RawImage, target: str, page: str) -> ImageRef:
        """Return the image of a file of the site.

        Args:
            raw: The image as found in the page.
            target: Its internal source, cleaned as `clean_href` does.
            page: Site relative path of the page.

        Returns:
            The image, named by its site relative path, or by its source
            when it has no path or leads out of the site.
        """
        path = site_path(target, page, self.base)
        if path is None:
            problem = OUTSIDE if _url_path(target) else NOT_FOUND
            return _unreadable(raw, printable(target), problem)
        try:
            file = self._site_file(path)
            data = read_image_file(file, MAX_FILE_BYTES)
        except UnreadableImage as error:
            return _unreadable(raw, path, error.reason)
        if _format(path) == _SVG_FORMAT or _looks_like_svg(data):
            return _svg_ref(raw, path, data)
        return raster_ref(
            data,
            name=path,
            source=PixelSource(file=file),
            alt=raw.alt,
            shown=raw.shown,
        )
