# Copyright 2026 Christophe Le Douarec
"""Read the facts of HTML pages that only the audit uses.

The identifiers, the links and the title are read from the whole page, its
navigation included. The links without a name, the references that lead
nowhere, the tables without a header and the content that gives no block
are read from the content of the page only, before its chrome is removed:
what would be removed, and what a drawing or a formula holds, does not
count, but for a toctree, which the chrome removes though it is the body of
its section.

The walks are iterative, as the document is not trusted.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from lxml import etree

from doc_checker.extract.html.dom import in_formula, in_svg, is_formula
from doc_checker.extract.html.images import first_role
from doc_checker.images.svg import local_name
from doc_checker.model.document import CELL_SEPARATOR
from doc_checker.model.facts import Spot
from doc_checker.text.normalize import printable

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from lxml.html import HtmlElement

LAYOUT_TABLE_CLASSES: Final = frozenset(
    {
        "highlighttable",
        "hlist",
        "footnote",
        "citation",
        "option-list",
        "field-list",
        "autosummary",
        "docinfo",
        "lntable",
        "rouge-table",
    },
)
LAYOUT_ROLES: Final = frozenset({"presentation", "none"})

_NAME_ATTRIBUTES: Final = ("aria-label", "aria-labelledby", "title")
_SVG: Final = "svg"
_IMAGE_TAGS: Final = ("img", _SVG)
_IMG_NAMES: Final = ("alt", "aria-label")
_UNRESOLVED_REFERENCE: Final = frozenset({"xref", "std"})
# The elements that embed a medium, which no block shows.
_MEDIA_TAGS: Final = frozenset({"iframe", "video", "audio", "object", "embed"})
# The elements that show a picture, which no block shows when the images are
# not read.
_PICTURE_TAGS: Final = frozenset({"img", _SVG, "picture"})
_HEADING_TAGS: Final = frozenset(f"h{level}" for level in range(1, 7))
# The class of the list of the pages of a Sphinx toctree.
_TOCTREE: Final = "toctree-wrapper"
_PROBLEMATIC: Final = "problematic"
# The elements that may hold an identifier: an id, or the name of an anchor.
_IDENTIFIED: Final = etree.XPath("//*[@id] | //a[@name]")


@dataclass(frozen=True, slots=True)
class ContentSpots:
    """The places of the content that the audit points at.

    Attributes:
        empty_links: Links without an accessible name, by their href.
        unresolved: References that lead nowhere, by their text.
        bare_tables: Tables without a header cell, by their first row.
        media: Source lines of the elements that embed a medium: iframe,
            video, audio, object and embed.
        blockless: Source lines of the other content that gives no block:
            a toctree, and a picture when the images are not read.
    """

    empty_links: tuple[Spot, ...]
    unresolved: tuple[Spot, ...]
    bare_tables: tuple[Spot, ...]
    media: tuple[int, ...] = ()
    blockless: tuple[int, ...] = ()


def _elements(tree: HtmlElement) -> Iterator[HtmlElement]:
    """Yield the elements of a tree, but comments and instructions."""
    for node in tree.iter():
        if isinstance(node.tag, str):
            yield node


def page_ids(tree: HtmlElement) -> frozenset[str]:
    """Return the identifiers of a whole page.

    Args:
        tree: Parsed page.

    Returns:
        The printable ``id`` of each element and ``name`` of each ``a``,
        but the empty ones.
    """
    found: set[str] = set()
    for node in _IDENTIFIED(tree):
        names = ("id", "name") if node.tag == "a" else ("id",)
        found.update(
            identifier
            for name in names
            if (identifier := printable(node.get(name) or ""))
        )
    return frozenset(found)


def page_hrefs(tree: HtmlElement) -> tuple[str, ...]:
    """Return the ``href`` of each link of a whole page, as written.

    Args:
        tree: Parsed page.

    Returns:
        The targets, in order; a link without ``href`` gives none.
    """
    return tuple(
        href
        for anchor in tree.iter("a")
        if (href := anchor.get("href")) is not None
    )


def page_title(tree: HtmlElement) -> str | None:
    """Return the title of the head of a page.

    Args:
        tree: Parsed page.

    Returns:
        The text of the first ``title`` of the head, printable, its blanks
        collapsed; ``None`` if there is none or it is blank.
    """
    head = tree.find("head")
    title = head.find("title") if head is not None else None
    if title is None:
        return None
    return printable("".join(title.itertext())) or None


def _left_out(
    root: HtmlElement,
    removed: frozenset[HtmlElement],
) -> set[HtmlElement]:
    """Return the elements of a root that go, or lie in a drawing or formula.

    One walk from the root down tells each element: it is left out when its
    parent is, when it is removed, when its parent is an ``svg`` element or
    when it is a formula. The root itself is never removed; it is left out
    when a drawing or a formula holds it.

    Args:
        root: Root of the content of a page.
        removed: The elements that the selectors match.

    Returns:
        The elements left out.
    """
    left: set[HtmlElement] = set()
    if in_svg(root) or in_formula(root):
        left.add(root)
    for node in root.iterdescendants():
        if not isinstance(node.tag, str):
            continue
        parent = node.getparent()
        if (
            parent in left
            or node in removed
            or parent.tag == _SVG
            or is_formula(node)
        ):
            left.add(node)
    return left


def _hidden(element: HtmlElement) -> bool:
    """Return whether an element is hidden from assistive technology."""
    return (
        element.get("aria-hidden") or ""
    ).strip().lower() == "true" or element.get("hidden") is not None


def _names_link(node: HtmlElement) -> bool:
    """Return whether an ``img`` or ``svg`` gives its link a name."""
    if node.tag == "img":
        if _hidden(node) or first_role(node) in LAYOUT_ROLES:
            return False
        return any(printable(node.get(n) or "") for n in _IMG_NAMES)
    if _hidden(node):
        return False
    if printable(node.get("aria-label") or ""):
        return True
    return any(
        isinstance(child.tag, str) and local_name(child.tag) == "title"
        for child in node
    )


def _has_name(anchor: HtmlElement) -> bool:
    """Return whether a link has an accessible name.

    The name is an attribute of the link, or its text, but the text that is
    hidden from assistive technology, or a picture with an alternative text.
    """
    if any(printable(anchor.get(name) or "") for name in _NAME_ATTRIBUTES):
        return True
    stack = [anchor]
    while stack:
        node = stack.pop()
        if printable(node.text or "") and isinstance(node.tag, str):
            return True
        for child in node:
            if printable(child.tail or ""):
                return True
            if not isinstance(child.tag, str):
                continue
            if child.tag in _IMAGE_TAGS:
                if _names_link(child):
                    return True
            elif not _hidden(child):
                stack.append(child)
    return False


def _is_unresolved(element: HtmlElement) -> bool:
    """Return whether an element is a reference that leads nowhere."""
    classes = (element.get("class") or "").split()
    if _PROBLEMATIC in classes:
        return True
    return _UNRESOLVED_REFERENCE.issubset(classes) and (
        next(element.iterancestors("a"), None) is None
    )


def _is_data_table(table: HtmlElement) -> bool:
    """Return whether a table has no header cell and is not for layout."""
    if next(table.iter("th"), None) is not None:
        return False
    if first_role(table) in LAYOUT_ROLES:
        return False
    classes = (table.get("class") or "").split()
    return not LAYOUT_TABLE_CLASSES.intersection(classes)


def _first_row(table: HtmlElement) -> str:
    """Return the cells of the first row that has text, joined."""
    for row in table.iter("tr"):
        cells = [
            printable("".join(cell.itertext()))
            for cell in row
            if cell.tag in {"th", "td"}
        ]
        if any(cells):
            return printable(CELL_SEPARATOR.join(cells))
    return ""


def _blockless(
    root: HtmlElement,
    left: set[HtmlElement],
    *,
    images: bool,
) -> list[int]:
    """Return the source lines of the content that gives no block.

    Args:
        root: Root of the content of a page.
        left: The elements left out, as `_left_out` tells.
        images: Whether the images are read, and give blocks.

    Returns:
        In document order, the line of each toctree that holds a link, is
        not hidden and lies in no element left out; then, when the images
        are not read, that of each ``img``, ``svg`` and ``picture`` element
        that is not left out. What a heading holds, which its heading block
        shows, gives no line.
    """
    lines: list[int] = []
    headed: set[HtmlElement] = set()
    for node in _elements(root):
        parent = node.getparent()
        if node.tag in _HEADING_TAGS or parent in headed:
            headed.add(node)
            continue
        line = node.sourceline
        if line is None:
            continue
        if _TOCTREE in (node.get("class") or "").split():
            if parent not in left and not _hidden(node) and _has_link(node):
                lines.append(line)
        elif not images and node.tag in _PICTURE_TAGS and node not in left:
            lines.append(line)
    return lines


def _has_link(element: HtmlElement) -> bool:
    """Return whether an element holds a link with a target."""
    return any(anchor.get("href") is not None for anchor in element.iter("a"))


def content_spots(
    root: HtmlElement,
    selectors: Sequence[str],
    *,
    images: bool = True,
) -> ContentSpots:
    """Find the links without a name, dead references, bare tables, media.

    Call it before the elements of the selectors are removed from the root.

    Args:
        root: Root of the content of a page.
        selectors: CSS selectors of the elements that are removed.
        images: Whether the images are read, and give blocks.

    Returns:
        The places found, in document order. An element that a selector
        matches, or that one of its ancestors under the root does, and one
        in a drawing or a formula, are left out; the root itself is never
        removed. A table without text gives no place, nor does a medium
        without a source line. The content that gives no block is found by
        `_blockless`.
    """
    removed = frozenset(
        element
        for selector in selectors
        for element in root.cssselect(selector)
    )
    left = _left_out(root, removed)
    links: list[Spot] = []
    unresolved: list[Spot] = []
    tables: list[Spot] = []
    media: list[int] = []
    for node in _elements(root):
        if node.tag in _MEDIA_TAGS:
            line = node.sourceline
            if line is not None and node not in left:
                media.append(line)
            continue
        if node.tag == "a":
            href = node.get("href")
            if href is None or _has_name(node):
                continue
            found, spot = links, Spot(printable(href), node.sourceline)
        elif _is_unresolved(node):
            text = printable("".join(node.itertext()))
            found, spot = unresolved, Spot(text, node.sourceline)
        elif node.tag == "table" and _is_data_table(node):
            text = _first_row(node)
            if not text:
                continue
            found, spot = tables, Spot(text, node.sourceline)
        else:
            continue
        if node not in left:
            found.append(spot)
    return ContentSpots(
        tuple(links),
        tuple(unresolved),
        tuple(tables),
        tuple(media),
        tuple(_blockless(root, left, images=images)),
    )
