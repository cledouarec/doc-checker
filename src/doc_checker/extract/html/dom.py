# Copyright 2026 Christophe Le Douarec
"""Turn the content of an HTML page into raw blocks."""

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from doc_checker.extract.common import DEFAULT_ADMONITION_TITLES
from doc_checker.model.document import BlockKind
from doc_checker.text.normalize import printable

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from lxml.html import HtmlElement


@dataclass(frozen=True, slots=True)
class RawBlock:
    """A block as found in the page, before normalization.

    Attributes:
        kind: Kind of the block.
        text: Text with ASCII whitespace collapsed, except for code.
        line: Source line of the element.
        anchor: Printable identifier of the nearest anchor.
        level: Heading level, for headings.
        role: Special role, such as ``admonition-title``.
    """

    kind: BlockKind
    text: str
    line: int | None
    anchor: str | None
    level: int | None = None
    role: str | None = None


INLINE_TAGS: Final[frozenset[str]] = frozenset(
    {
        "a",
        "abbr",
        "b",
        "bdi",
        "bdo",
        "br",
        "cite",
        "code",
        "data",
        "dfn",
        "em",
        "i",
        "kbd",
        "mark",
        "q",
        "s",
        "samp",
        "small",
        "span",
        "strong",
        "sub",
        "sup",
        "time",
        "u",
        "var",
        "img",
        "wbr",
        "del",
        "ins",
        "tt",
        "font",
        "label",
    },
)

_HEADING_LEVELS: Final = {f"h{level}": level for level in range(1, 7)}
_ITEM_TEXT_TAGS: Final = frozenset({"p"})
_FORMULA_CLASSES: Final = frozenset({"math", "arithmatex"})
_ADMONITION_TITLE: Final = "admonition-title"
_WHITESPACE: Final = re.compile(r"[ \t\r\n\f\v]+")


def _classes(element: HtmlElement) -> frozenset[str]:
    """Return the classes of an element."""
    return frozenset((element.get("class") or "").split())


def _collapse(text: str) -> str:
    """Return the text with ASCII whitespace collapsed, nbsp kept."""
    return _WHITESPACE.sub(" ", text).strip(" ")


def _raw_text(element: HtmlElement, skip: frozenset[str] = frozenset()) -> str:
    """Return the text of an element, ``br`` giving a newline.

    Args:
        element: Element to read.
        skip: Tags whose content is left out, their tail being kept.

    Returns:
        The concatenated text.
    """
    parts: list[str] = []

    def visit(node: HtmlElement) -> None:
        if node.tag == "br":
            parts.append("\n")
        elif node.text:
            parts.append(node.text)
        for child in node:
            if isinstance(child.tag, str) and child.tag not in skip:
                visit(child)
            if child.tail:
                parts.append(child.tail)

    visit(element)
    return "".join(parts)


def strip_elements(root: HtmlElement, selectors: Iterable[str]) -> int:
    """Remove the elements matched by CSS selectors, keeping their tail.

    Args:
        root: Element to clean.
        selectors: CSS selectors of the elements to remove.

    Returns:
        The number of removed elements.
    """
    removed = 0
    for selector in selectors:
        for element in root.cssselect(selector):
            if element.getparent() is not None:
                element.drop_tree()
                removed += 1
    return removed


class _Walker:
    """Walk the DOM in document order and collect raw blocks."""

    def __init__(self) -> None:
        self.blocks: list[RawBlock] = []
        self.anchor: str | None = None
        self.run: list[str] = []
        self.run_line: int | None = None
        self.handlers: dict[str, Callable[[HtmlElement], None]] = {
            **dict.fromkeys(_HEADING_LEVELS, self.heading),
            **dict.fromkeys(("p", "dt", "summary"), self.paragraph),
            "li": self.list_item,
            "pre": self.code,
            "table": self.table,
            "figcaption": self.caption,
        }

    def emit(
        self,
        kind: BlockKind,
        text: str,
        element: HtmlElement,
        **extra: int | str | None,
    ) -> None:
        """Record a block unless its text is empty."""
        if text.strip():
            self.blocks.append(
                RawBlock(
                    kind,
                    text,
                    element.sourceline,
                    self.anchor,
                    **extra,  # type: ignore[arg-type]
                ),
            )

    def add_inline(self, text: str | None, element: HtmlElement) -> None:
        """Append inline text to the current loose text run."""
        if text:
            if not self.run:
                self.run_line = element.sourceline
            self.run.append(text)

    def flush(self) -> None:
        """Turn the current loose text run into a paragraph."""
        text = _collapse("".join(self.run))
        if text:
            self.blocks.append(
                RawBlock(
                    BlockKind.PARAGRAPH,
                    text,
                    self.run_line,
                    self.anchor,
                ),
            )
        self.run = []
        self.run_line = None

    def update_anchor(self, element: HtmlElement) -> None:
        """Make the element identifier the current anchor if it is one.

        The identifier is made printable, and an identifier that has
        nothing left is not an anchor.
        """
        identifier = printable(element.get("id") or "")
        if not identifier:
            return
        tag = element.tag
        if (
            tag in _HEADING_LEVELS
            or tag == "section"
            or (tag == "div" and "section" in _classes(element))
        ):
            self.anchor = identifier

    def walk(self, element: HtmlElement) -> None:
        """Walk the children of a container element."""
        self.add_inline(element.text, element)
        self.walk_children(element, list(element))

    def walk_children(
        self,
        element: HtmlElement,
        children: list[HtmlElement],
    ) -> None:
        """Walk some children of a container element, in order."""
        for child in children:
            if not isinstance(child.tag, str):
                pass
            elif child.tag in INLINE_TAGS:
                self.add_inline(_raw_text(child), child)
            else:
                self.flush()
                self.visit(child)
            self.add_inline(child.tail, element)
        self.flush()

    def visit(self, element: HtmlElement) -> None:
        """Visit a block level element."""
        self.update_anchor(element)
        handler = self.handlers.get(element.tag)
        if handler is not None:
            handler(element)
        elif element.tag == "div" and _classes(element) & _FORMULA_CLASSES:
            text = _collapse(_raw_text(element))
            self.emit(BlockKind.FORMULA, text, element)
        else:
            self.walk(element)

    def heading(self, element: HtmlElement) -> None:
        """Record a heading."""
        text = _collapse(_raw_text(element))
        level = _HEADING_LEVELS[element.tag]
        self.emit(BlockKind.HEADING, text, element, level=level)

    def paragraph(self, element: HtmlElement) -> None:
        """Record a paragraph, or an admonition title that is not default."""
        text = _collapse(_raw_text(element))
        is_title = element.tag == "summary" or (
            _ADMONITION_TITLE in _classes(element)
        )
        if not is_title:
            self.emit(BlockKind.PARAGRAPH, text, element)
        elif text.casefold().rstrip(":") not in DEFAULT_ADMONITION_TITLES:
            self.emit(
                BlockKind.PARAGRAPH,
                text,
                element,
                role=_ADMONITION_TITLE,
            )

    def list_item(self, element: HtmlElement) -> None:
        """Record a list item, then the blocks nested in it.

        The item text is its inline content and its leading paragraphs; the
        first other block (nested list, code, table, admonition...) and
        everything after it are visited as blocks of their own.
        """
        parts = [element.text or ""]
        children = list(element)
        rest = len(children)
        for position, child in enumerate(children):
            if isinstance(child.tag, str) and child.tag not in INLINE_TAGS:
                if child.tag not in _ITEM_TEXT_TAGS:
                    rest = position
                    break
                parts.append(f" {_raw_text(child)} ")
            elif isinstance(child.tag, str):
                parts.append(_raw_text(child))
            parts.append(child.tail or "")
        self.emit(BlockKind.LIST_ITEM, _collapse("".join(parts)), element)
        self.walk_children(element, children[rest:])

    def code(self, element: HtmlElement) -> None:
        """Record a code block with its raw text."""
        self.emit(BlockKind.CODE, _raw_text(element), element)

    def table(self, element: HtmlElement) -> None:
        """Record a table, or walk a code table with line numbers."""
        if "highlighttable" in _classes(element):
            self.walk(element)
            return
        caption = element.find("caption")
        if caption is not None:
            self.caption(caption)
        rows = []
        for row in element.iter("tr"):
            cells = [
                _collapse(_raw_text(cell))
                for cell in row
                if cell.tag in {"th", "td"}
            ]
            if any(cells):
                rows.append(" | ".join(cells))
        self.emit(BlockKind.TABLE, "\n".join(rows), element)

    def caption(self, element: HtmlElement) -> None:
        """Record a caption."""
        text = _collapse(_raw_text(element))
        self.emit(BlockKind.CAPTION, text, element)


def blocks_from_root(root: HtmlElement) -> list[RawBlock]:
    """Return the blocks of a content root, in document order.

    Args:
        root: Content element of a page, already stripped of chrome.

    Returns:
        The raw blocks.
    """
    walker = _Walker()
    walker.visit(root)
    walker.flush()
    return walker.blocks
