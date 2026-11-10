# Copyright 2026 Christophe Le Douarec
"""Turn the content of an HTML page into raw blocks.

A block keeps the links, the ``a`` elements with an ``href`` and some
printable text, of the elements whose text makes it: the whole element of
a heading, a paragraph or a table, the inline content and the leading
paragraphs of a list item, the inline elements of a loose text run. A code
block has no link.

A block also keeps the identifiers that designate it: those, ``id`` and the
``name`` of an ``a``, of the elements walked since the previous block
without giving one, such as an empty ``section`` or ``div``, then those of
the elements whose text makes it. They are printable, in document order and
without duplicates.

A block of text keeps the text of its inline code, the elements of
`INLINE_CODE_TAGS` of the elements whose text makes it that no other such
element holds. A code block and an image have none.

An ``svg`` element is an image, never text: neither its texts nor its links
belong to a block of text. An image block follows the block of text whose
elements hold it, or comes in its place when that text is empty; an inline
``svg`` that is walked, as a block level element, is an image block of its
own. Icons give no block, and are counted.
"""

import re
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from lxml import etree

from doc_checker.extract.html.css import select
from doc_checker.extract.html.images import raw_image
from doc_checker.model.document import CELL_SEPARATOR, BlockKind
from doc_checker.text.lexicon import is_admonition_title
from doc_checker.text.normalize import printable

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence

    from lxml.html import HtmlElement

    from doc_checker.extract.html.images import RawImage


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
        links: Printable text and ``href`` of each link of the block, in
            order.
        ids: Printable identifiers that designate the block, in order.
        image: The image of an `IMAGE` block.
        code: Text of each inline code of the block, its ASCII whitespace
            collapsed, in order; empty ones are left out.
    """

    kind: BlockKind
    text: str
    line: int | None
    anchor: str | None
    level: int | None = None
    role: str | None = None
    links: tuple[tuple[str, str], ...] = ()
    ids: tuple[str, ...] = ()
    image: RawImage | None = None
    code: tuple[str, ...] = ()


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

# The inline elements whose text is code: source code, keys, outputs and
# variables.
INLINE_CODE_TAGS: Final = frozenset({"code", "kbd", "samp", "tt", "var"})
_HEADING_LEVELS: Final = {f"h{level}": level for level in range(1, 7)}
_ITEM_TEXT_TAGS: Final = frozenset({"p"})
_FORMULA_CLASSES: Final = frozenset({"math", "arithmatex"})
_ADMONITION_TITLE: Final = "admonition-title"
# The first children of a title of blocks that may hold its own title.
_TITLE_BLOCKS: Final = frozenset({"div", "p"})
_WHITESPACE: Final = re.compile(r"[ \t\r\n\f\v]+")
_SVG: Final = "svg"
_IMAGE_TAGS: Final = ("img", _SVG)
_ICON: Final = "icon"  # Normalization counter of the icons left out.
# The kinds of blocks without links, and those whose images are not read:
# the images of code and of formulas, such as rendered math, are not
# figures.
_LINKLESS: Final = frozenset({BlockKind.CODE, BlockKind.IMAGE})
_IMAGELESS: Final = frozenset({BlockKind.CODE, BlockKind.FORMULA})
# What marks a formula, whose drawings and pictures are not figures: the
# elements of MathML and MathJax, and the classes, in lower case, of the
# formulas and of KaTeX and MathJax; pandoc --webtex gives `img.math`.
_FORMULA_TAGS: Final = frozenset({"math", "mjx-container"})
_FORMULA_MARKS: Final = frozenset({*_FORMULA_CLASSES, "katex", "mathjax"})
# Most levels of elements under the root of a page that are walked; the
# walk recurses, and libxml2 stopped at 255 before huge trees were allowed.
MAX_DEPTH: Final = 250
_TOO_DEEP: Final = etree.XPath("/".join(["*"] * (MAX_DEPTH + 1)))


def _classes(element: HtmlElement) -> frozenset[str]:
    """Return the classes of an element."""
    return frozenset((element.get("class") or "").split())


def _collapse(text: str) -> str:
    """Return the text with ASCII whitespace collapsed, nbsp kept."""
    return _WHITESPACE.sub(" ", text).strip(" ")


def in_svg(element: HtmlElement) -> bool:
    """Return whether an ``svg`` element holds an element."""
    return next(element.iterancestors(_SVG), None) is not None


def _raw_text(element: HtmlElement) -> str:
    """Return the text of an element, ``br`` giving a newline.

    The content of an ``svg`` element is left out, its tail being kept.

    Args:
        element: Element to read.

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
            if isinstance(child.tag, str) and child.tag != _SVG:
                visit(child)
            if child.tail:
                parts.append(child.tail)

    visit(element)
    return "".join(parts)


def is_formula(element: HtmlElement) -> bool:
    """Return whether an element is a formula, by its tag or its classes."""
    classes = (element.get("class") or "").lower().split()
    return element.tag in _FORMULA_TAGS or bool(
        _FORMULA_MARKS.intersection(classes),
    )


def in_formula(element: HtmlElement) -> bool:
    """Return whether an element is, or is in, a formula."""
    return any(
        is_formula(node) for node in (element, *element.iterancestors())
    )


def _images_in(parts: Iterable[HtmlElement]) -> Iterator[HtmlElement]:
    """Yield the ``img`` and ``svg`` elements of parts that are figures.

    An element that an ``svg`` element holds is part of that drawing, and
    one of a formula is part of the formula.
    """
    for part in parts:
        for node in part.iter(*_IMAGE_TAGS):
            if not in_svg(node) and not in_formula(node):
                yield node


def prune_deep(root: HtmlElement) -> bool:
    """Remove the elements nested more than `MAX_DEPTH` levels under a root.

    The text around the elements removed is kept.

    Args:
        root: Root of a page, changed in place.

    Returns:
        Whether some element was removed.
    """
    deep = _TOO_DEEP(root)
    for element in deep:
        element.drop_tree()
    return bool(deep)


def _identifiers(element: HtmlElement) -> Iterator[str]:
    """Yield the printable identifiers of an element, but empty ones."""
    names = ("id", "name") if element.tag == "a" else ("id",)
    for name in names:
        identifier = printable(element.get(name) or "")
        if identifier:
            yield identifier


def _all_identifiers(parts: Iterable[HtmlElement]) -> Iterator[str]:
    """Yield the identifiers of elements and of their descendants."""
    for part in parts:
        for node in part.iter():
            if isinstance(node.tag, str):
                yield from _identifiers(node)


def _links(parts: Iterable[HtmlElement]) -> tuple[tuple[str, str], ...]:
    """Return the printable text and the ``href`` of the links with text.

    The links of an ``svg`` element are left out, as its texts are.
    """
    found: list[tuple[str, str]] = []
    for part in parts:
        for anchor in part.iter("a"):
            href = anchor.get("href")
            if href is None or in_svg(anchor):
                continue
            text = printable(_raw_text(anchor))
            if text:
                found.append((text, href))
    return tuple(found)


def _inline_code(parts: Iterable[HtmlElement]) -> tuple[str, ...]:
    """Return the text of the inline code of elements, in document order.

    Each part is walked once, in document order, without entering an
    element of `INLINE_CODE_TAGS`, whose text holds the codes in it, nor
    an ``svg`` element, whose text is part of the drawing; a part that
    holds no inline code is not walked. The text has its ASCII whitespace
    collapsed; an empty one is left out.
    """
    found: list[str] = []
    for part in parts:
        if next(part.iter(*INLINE_CODE_TAGS), None) is None:
            continue
        stack = [part]
        while stack:
            node = stack.pop()
            if node.tag in INLINE_CODE_TAGS:
                text = _collapse(_raw_text(node))
                if text:
                    found.append(text)
            elif node.tag != _SVG:
                stack.extend(
                    child
                    for child in reversed(node)
                    if isinstance(child.tag, str)
                )
    return tuple(found)


def _found(root: HtmlElement, selectors: Iterable[str]) -> list[HtmlElement]:
    """Return the elements CSS selectors match, once, in document order."""
    matched: set[HtmlElement] = set()
    for selector in selectors:
        matched.update(select(root, selector))
    if not matched:
        return []
    return [node for node in root.iter() if node in matched]


def _inline_only(element: HtmlElement) -> bool:
    """Return whether an element holds only inline elements and drawings.

    The descendants are read in document order, one at a time, until the
    first block level one; those of an ``svg`` element are not read: a
    drawing, such as an icon, is no text.
    """
    stack = [iter(element)]
    while stack:
        node = next(stack[-1], None)
        if node is None:
            stack.pop()
        elif isinstance(node.tag, str) and node.tag != _SVG:
            if node.tag not in INLINE_TAGS:
                return False
            stack.append(iter(node))
    return True


def _titled(element: HtmlElement) -> None:
    """Give an element the class ``admonition-title``, its others kept."""
    classes = (*element.get("class", "").split(), _ADMONITION_TITLE)
    element.set("class", " ".join(dict.fromkeys(classes)))


def mark_titles(root: HtmlElement, selectors: Sequence[str]) -> None:
    """Mark the admonition titles that CSS selectors find.

    Each element found is visited once, in document order. One whose text
    is empty leaves, its tail kept, if it is inline. A block level element
    that holds only inline elements and drawings becomes a ``p`` of class
    ``admonition-title``, empty or not, its other classes and its
    identifier kept, and the walk decides whether the title is a default
    one. A block level element that holds blocks keeps its tag and takes
    the class, so that the walk reads its blocks apart, and its first
    child, if it is a ``div`` or a ``p``, is marked in the same way: its
    title, such as the ``div.title`` of an Asciidoctor icon cell. The
    marking stops at an element already marked, so that nested titles are
    read once. An inline
    element, such as a bold label, leaves, its tail kept, when its text is
    empty or a default title.

    Args:
        root: Content root, changed in place.
        selectors: CSS selectors of the titles.
    """
    marked: set[HtmlElement] = set()
    for element in _found(root, selectors):
        if element.tag in INLINE_TAGS:
            text = _collapse(_raw_text(element))
            if element.getparent() is not None and (
                not text or is_admonition_title(text)
            ):
                element.drop_tree()
            continue
        title: HtmlElement | None = element
        while (
            title is not None
            and title not in marked
            and not _inline_only(title)
        ):
            _titled(title)
            marked.add(title)
            title = next(
                (child for child in title if isinstance(child.tag, str)),
                None,
            )
            if title is not None and title.tag not in _TITLE_BLOCKS:
                title = None
        if title is not None and title not in marked:
            title.tag = "p"
            _titled(title)
            marked.add(title)


def mark_code(root: HtmlElement, selectors: Sequence[str]) -> None:
    """Make inline code of the elements that CSS selectors find.

    Args:
        root: Content root, changed in place.
        selectors: CSS selectors of the elements whose text is code.
    """
    for element in _found(root, selectors):
        element.tag = "code"


def in_admonition_block(element: HtmlElement) -> bool:
    """Return whether the parent of an element is an Asciidoctor admonition."""
    parent = element.getparent()
    return parent is not None and "admonitionblock" in _classes(parent)


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
        for element in select(root, selector):
            if element.getparent() is not None:
                element.drop_tree()
                removed += 1
    return removed


class _Walker:
    """Walk the DOM in document order and collect raw blocks."""

    def __init__(self, *, images: bool, counter: Counter[str]) -> None:
        self.images = images
        self.counter = counter
        self.blocks: list[RawBlock] = []
        self.anchor: str | None = None
        self.waiting: list[str] = []
        self.run: list[str] = []
        self.run_parts: list[HtmlElement] = []
        self.run_images: list[HtmlElement] = []
        self.run_line: int | None = None
        self.handlers: dict[str, Callable[[HtmlElement], None]] = {
            **dict.fromkeys(_HEADING_LEVELS, self.heading),
            **dict.fromkeys(("p", "dt", "summary"), self.paragraph),
            "li": self.list_item,
            "pre": self.code,
            "table": self.table,
            "figcaption": self.caption,
            _SVG: self.drawing,
        }

    def hold(self, parts: Iterable[HtmlElement]) -> None:
        """Keep the identifiers of elements without a block for the next."""
        self.waiting.extend(_all_identifiers(parts))

    def record(
        self,
        kind: BlockKind,
        text: str,
        line: int | None,
        parts: Sequence[HtmlElement],
        **extra: int | str | RawImage | None,
    ) -> None:
        """Record a block, with the links, code and identifiers of its parts.

        A code block and an image have neither links nor inline code.
        """
        ids = tuple(dict.fromkeys((*self.waiting, *_all_identifiers(parts))))
        self.waiting = []
        textless = kind in _LINKLESS
        self.blocks.append(
            RawBlock(
                kind,
                text,
                line,
                self.anchor,
                links=() if textless else _links(parts),
                ids=ids,
                code=() if textless else _inline_code(parts),
                **extra,  # type: ignore[arg-type]
            ),
        )

    def add_image(
        self,
        element: HtmlElement,
        parts: Sequence[HtmlElement] = (),
    ) -> None:
        """Record the block of an image, or count an icon.

        Args:
            element: The ``img`` or ``svg`` element.
            parts: Elements whose identifiers the image block takes, or
                that an icon keeps for the next block; none for an image
                whose identifiers went to its block of text.
        """
        raw = raw_image(element)
        if raw is None:
            self.counter[_ICON] += 1
            self.hold(parts)
        else:
            self.record(
                BlockKind.IMAGE,
                "",
                element.sourceline,
                parts,
                image=raw,
            )

    def follow(self, parts: Iterable[HtmlElement]) -> None:
        """Record the images of the elements of a block of text, in order.

        The block of text, recorded or not, took their identifiers.
        """
        if self.images:
            for element in _images_in(parts):
                self.add_image(element)

    def emit(
        self,
        kind: BlockKind,
        text: str,
        element: HtmlElement,
        *,
        parts: Sequence[HtmlElement] | None = None,
        **extra: int | str | None,
    ) -> None:
        """Record a block unless its text is empty, then its images.

        Args:
            kind: Kind of the block.
            text: Text of the block.
            element: Element of the block.
            parts: Elements whose text makes the block, the element itself
                by default.
            **extra: Other fields of the block.
        """
        parts = (element,) if parts is None else parts
        if text.strip():
            self.record(kind, text, element.sourceline, parts, **extra)
        else:
            self.hold(parts)
        if kind not in _IMAGELESS:
            self.follow(parts)

    def add_inline(self, text: str | None, element: HtmlElement) -> None:
        """Append inline text to the current loose text run."""
        if text:
            if not self.run:
                self.run_line = element.sourceline
            self.run.append(text)

    def add_element(self, element: HtmlElement) -> None:
        """Append an inline element, and its images, to the text run."""
        self.run_parts.append(element)
        if self.images:
            self.run_images.extend(_images_in((element,)))
        self.add_inline(_raw_text(element), element)

    def flush(self) -> None:
        """Turn the current loose text run into a paragraph, then images."""
        text = _collapse("".join(self.run))
        if text:
            self.record(
                BlockKind.PARAGRAPH,
                text,
                self.run_line,
                self.run_parts,
            )
        else:
            self.hold(self.run_parts)
        images = self.run_images
        self.run = []
        self.run_parts = []
        self.run_images = []
        self.run_line = None
        for element in images:
            self.add_image(element)

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
                self.add_element(child)
            else:
                self.flush()
                self.visit(child)
            self.add_inline(child.tail, element)
        self.flush()

    def visit(self, element: HtmlElement) -> None:
        """Visit a block level element."""
        self.update_anchor(element)
        self.waiting.extend(_identifiers(element))
        if _ADMONITION_TITLE in _classes(element) and all(
            node.tag in INLINE_TAGS
            for node in element.iterdescendants()
            if isinstance(node.tag, str)
        ):
            self.paragraph(element)
            return
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
        elif not is_admonition_title(text):
            self.emit(
                BlockKind.PARAGRAPH,
                text,
                element,
                role=_ADMONITION_TITLE,
            )
        else:
            self.hold((element,))
            self.follow((element,))

    def drawing(self, element: HtmlElement) -> None:
        """Record an ``svg`` element as an image, never reading its text.

        The drawing of a formula is no image.
        """
        if self.images and not in_formula(element):
            self.add_image(element, (element,))
        else:
            self.hold((element,))

    def list_item(self, element: HtmlElement) -> None:
        """Record a list item, then the blocks nested in it.

        The item text is its inline content and its leading paragraphs; the
        first other block (nested list, code, table, admonition...) and
        everything after it are visited as blocks of their own.
        """
        texts = [element.text or ""]
        children = list(element)
        rest = len(children)
        for position, child in enumerate(children):
            if isinstance(child.tag, str) and child.tag not in INLINE_TAGS:
                if child.tag not in _ITEM_TEXT_TAGS:
                    rest = position
                    break
                texts.append(f" {_raw_text(child)} ")
            elif isinstance(child.tag, str):
                texts.append(_raw_text(child))
            texts.append(child.tail or "")
        self.emit(
            BlockKind.LIST_ITEM,
            _collapse("".join(texts)),
            element,
            parts=[c for c in children[:rest] if isinstance(c.tag, str)],
        )
        self.walk_children(element, children[rest:])

    def code(self, element: HtmlElement) -> None:
        """Record a code block with its raw text."""
        self.emit(BlockKind.CODE, _raw_text(element), element)

    def table(self, element: HtmlElement) -> None:
        """Record a table, or walk a code table or an admonition table."""
        if "highlighttable" in _classes(element) or in_admonition_block(
            element,
        ):
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
                rows.append(CELL_SEPARATOR.join(cells))
        self.emit(
            BlockKind.TABLE,
            "\n".join(rows),
            element,
            parts=[
                child
                for child in element
                if isinstance(child.tag, str) and child is not caption
            ],
        )

    def caption(self, element: HtmlElement) -> None:
        """Record a caption."""
        text = _collapse(_raw_text(element))
        self.emit(BlockKind.CAPTION, text, element)


def blocks_from_root(
    root: HtmlElement,
    *,
    images: bool = True,
    counter: Counter[str] | None = None,
) -> list[RawBlock]:
    """Return the blocks of a content root, in document order.

    An ``img`` or ``svg`` element that an inline element of a loose text
    run holds, and that neither an ``svg`` nor a formula holds, follows the
    paragraph of the run, or comes in its place when the run has no text;
    one of the elements of a heading, a paragraph, the text of a list item,
    a table or a caption follows the block of that element. An ``svg``
    element that is walked is an image block of its own, unless a formula
    holds it. An image block takes the identifiers waiting for a block, as
    a block of text does.

    Args:
        root: Content element of a page, already stripped of chrome and
            pruned by `prune_deep`, as the walk recurses.
        images: Whether the images give blocks; without them, the
            identifiers of an ``svg`` element wait for the next block.
        counter: Normalization counter, counting the icons left out
            (``icon``), if images give blocks.

    Returns:
        The raw blocks.
    """
    walker = _Walker(
        images=images,
        counter=Counter() if counter is None else counter,
    )
    walker.visit(root)
    walker.flush()
    return walker.blocks
