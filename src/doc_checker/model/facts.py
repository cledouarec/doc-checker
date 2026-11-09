# Copyright 2026 Christophe Le Douarec
"""Facts about a document that only the audit reads."""

from dataclasses import dataclass
from typing import Final, Literal

# Why the audit has no structure tree to read in a PDF.
NO_STRUCTURE_TREE: Final = "no structure tree"

type LinkAction = Literal["goto", "uri", "file", "none", "other"]
type TagRole = Literal["heading", "figure", "table"]


@dataclass(frozen=True, slots=True)
class Spot:
    """A place in a page that an audit finding points at.

    Attributes:
        text: Printable text of the spot: an href, a reference, a table row.
        line: Source line of its element, if known.
    """

    text: str
    line: int | None = None


@dataclass(frozen=True, slots=True)
class HtmlPageFacts:
    """What the audit reads of one HTML page.

    Attributes:
        file: Site relative path of the page.
        lang: Printable lang of its html element, if any.
        title: Printable, collapsed head title; None if blank.
        ids: Identifiers of the whole page: id and a[name].
        links_to: Pages its a[href] lead to.
        empty_links: Links without an accessible name.
        unresolved: References that lead nowhere.
        bare_tables: Tables without a header cell.
        media: Source lines of the embedded media of the content: iframe,
            video, audio, object and embed, which give no block.
        blockless: Source lines of the other content that gives no block:
            a toctree, and a picture when the images are not read.
    """

    file: str
    lang: str | None
    title: str | None
    ids: frozenset[str] = frozenset()
    links_to: frozenset[str] = frozenset()
    empty_links: tuple[Spot, ...] = ()
    unresolved: tuple[Spot, ...] = ()
    bare_tables: tuple[Spot, ...] = ()
    media: tuple[int, ...] = ()
    blockless: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class PdfLink:
    """A link annotation of a PDF page.

    Attributes:
        text: Printable text under the annotation.
        action: Kind of action of the annotation.
        target: "page N", a URI, a file path, or "".
        valid: Whether the destination is valid; None if not checked.
    """

    text: str
    action: LinkAction
    target: str
    valid: bool | None


@dataclass(frozen=True, slots=True)
class PdfPageFacts:
    """What the audit reads of one PDF page.

    Attributes:
        number: Page number, from 1.
        label: Label of the page.
        chars: Characters kept on the page.
        unmapped: Characters without Unicode mapping.
        scanned: Whether the page is a scan.
        links: Link annotations of the page.
    """

    number: int
    label: str
    chars: int
    unmapped: int
    scanned: bool
    links: tuple[PdfLink, ...] = ()


@dataclass(frozen=True, slots=True)
class OutlineEntry:
    """A bookmark of the outline of a PDF.

    Attributes:
        level: Depth in the outline, from 0.
        title: Printable title.
        page: Page it leads to, from 1; None if unknown.
    """

    level: int
    title: str
    page: int | None


@dataclass(frozen=True, slots=True)
class TaggedElement:
    """A heading, figure or table of the structure tree of a PDF.

    Attributes:
        role: Role of the element.
        type: Standard type after the role map, such as H2 or Figure.
        page: Page of the element, from 1; None if unknown.
        level: Number n of a heading Hn.
        alt: Printable /Alt.
        actual: Printable /ActualText.
        headers: Whether a table holds a TH.
    """

    role: TagRole
    type: str
    page: int | None
    level: int | None = None
    alt: str | None = None
    actual: str | None = None
    headers: bool = False


@dataclass(frozen=True, slots=True)
class TagTree:
    """The structure tree of a tagged PDF, reduced to what the audit reads.

    Attributes:
        elements: Headings, figures and tables.
        items: Items walked.
        problem: Why the tree was not read in full, if so.
    """

    elements: tuple[TaggedElement, ...]
    items: int
    problem: str | None = None
