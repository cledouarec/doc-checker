# Copyright 2026 Christophe Le Douarec
"""Read the facts of HTML pages that only the audit uses.

The identifiers, the links and the title are read from the whole page, its
navigation included. The links without a name, the references that lead
nowhere and the tables without a header are read from the content of the
page only, before its chrome is removed: what would be removed, and what a
drawing or a formula holds, does not count.

The walks are iterative, as the document is not trusted.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from doc_checker.extract.html.dom import in_formula, in_svg
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
_IMAGE_TAGS: Final = ("img", "svg")
_IMG_NAMES: Final = ("alt", "aria-label")
_UNRESOLVED_REFERENCE: Final = frozenset({"xref", "std"})
_PROBLEMATIC: Final = "problematic"


@dataclass(frozen=True, slots=True)
class ContentSpots:
    """The places of the content that the audit points at.

    Attributes:
        empty_links: Links without an accessible name, by their href.
        unresolved: References that lead nowhere, by their text.
        bare_tables: Tables without a header cell, by their first row.
    """

    empty_links: tuple[Spot, ...]
    unresolved: tuple[Spot, ...]
    bare_tables: tuple[Spot, ...]


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
    for node in _elements(tree):
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


def _is_removed(
    element: HtmlElement,
    root: HtmlElement,
    removed: frozenset[HtmlElement],
) -> bool:
    """Return whether an element, or an ancestor under the root, goes."""
    for node in (element, *element.iterancestors()):
        if node is root:
            return False
        if node in removed:
            return True
    return False


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


def content_spots(
    root: HtmlElement,
    selectors: Sequence[str],
) -> ContentSpots:
    """Find the links without a name, the dead references and bare tables.

    Call it before the elements of the selectors are removed from the root.

    Args:
        root: Root of the content of a page.
        selectors: CSS selectors of the elements that are removed.

    Returns:
        The places found, in document order. An element that a selector
        matches, or that one of its ancestors under the root does, and one
        in a drawing or a formula, are left out; the root itself is never
        removed. A table without text gives no place.
    """
    removed = frozenset(
        element
        for selector in selectors
        for element in root.cssselect(selector)
    )
    links: list[Spot] = []
    unresolved: list[Spot] = []
    tables: list[Spot] = []
    for node in _elements(root):
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
        if not (
            _is_removed(node, root, removed)
            or in_svg(node)
            or in_formula(node)
        ):
            found.append(spot)
    return ContentSpots(tuple(links), tuple(unresolved), tuple(tables))
