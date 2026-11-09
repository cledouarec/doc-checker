# Copyright 2026 Christophe Le Douarec
"""Check the accessibility of a document.

A PDF without a structure tree, one that does not declare a tag structure
or whose catalog has no ``/StructTreeRoot``, is untagged; a tree that was
read in part, or not at all for another reason, is not. A PDF declares its
language in its catalog and its title in its metadata; an HTML page, its
language in the ``lang`` of its ``html`` element, which has to be a
language tag, and its title in its head. A document that declares French
or English, but whose prose, its first `LANG_SAMPLE_BLOCKS` blocks of
`PROSE_KINDS`, plainly reads as the other language, as `strong_language`
tells, has a language mismatch. The characters of a PDF without a Unicode
mapping give one finding, at the first page that has some, which is major
from a share of the characters of the document; each scanned page gives
one.

An image of an HTML page needs an alternative text, an inline SVG a title
or a label, and an alternative text that is a file name, `FILE_NAME_ALT`,
says nothing; an empty alternative text marks a decorative image. Each
``Figure`` of the structure tree of a PDF needs a ``/Alt`` or an
``/ActualText``, and its ``/Alt`` is no file name; each ``Table`` of the
tree, like each table of the content of an HTML page, needs a header cell.
The image blocks of a PDF, whose alternative texts the tree holds, are not
judged by themselves.

A link of the content of an HTML page needs a name, and the text of a link
of a block, or under a link annotation of a PDF, has to tell where the link
leads: a text of `GENERIC_LINK_TEXTS`, compared by its loose key, does not.

The texts of the document that a detail quotes are cut.
"""

import re
from collections import Counter
from itertools import islice
from typing import TYPE_CHECKING, Final

from doc_checker.audit.sides import LinkMarks, quote
from doc_checker.compare.drafts import Draft, marked_side
from doc_checker.images.refs import describe
from doc_checker.model.document import BlockKind, SourceRef
from doc_checker.model.facts import NO_STRUCTURE_TREE
from doc_checker.model.findings import Severity
from doc_checker.text.lang import (
    LANG_SAMPLE_BLOCKS,
    LANGUAGE_NAMES,
    declared_language,
    strong_language,
)
from doc_checker.text.normalize import (
    ELLIPSIS,
    loose_key,
    normalize_display,
    printable,
    strict_key,
)

if TYPE_CHECKING:
    from doc_checker.config import AuditThresholds
    from doc_checker.model.document import Block, Document
    from doc_checker.model.facts import HtmlPageFacts, TaggedElement
    from doc_checker.model.findings import Side

FILE_NAME_ALT: Final = re.compile(
    r"(?i)^[\w\-./\\]+\.(?:png|jpe?g|gif|svg|webp|bmp|tiff?|pdf|eps|avif)$",
)
GENERIC_LINK_TEXTS: Final = frozenset(
    {
        "click here",
        "here",
        "click",
        "read more",
        "learn more",
        "more",
        "this link",
        "link",
        "this page",
        "cliquez ici",
        "ici",
        "cliquer ici",
        "lire la suite",
        "en savoir plus",
        "ce lien",
        "lien",
        "cette page",
    },
)
# The blocks whose text tells the language of a document: headings,
# paragraphs, items, captions, quotes and admonitions.
PROSE_KINDS: Final = frozenset(
    {
        BlockKind.HEADING,
        BlockKind.PARAGRAPH,
        BlockKind.LIST_ITEM,
        BlockKind.CAPTION,
        BlockKind.QUOTE,
        BlockKind.ADMONITION,
    },
)
UNMAPPED_PAGES_SHOWN: Final = 10

_UNTAGGED: Final = "a11y.pdf_untagged"
_LANG_MISSING: Final = "a11y.lang_missing"
_LANG_MISMATCH: Final = "a11y.lang_mismatch"
_TITLE_MISSING: Final = "a11y.title_missing"
_UNMAPPED: Final = "a11y.unmapped_chars"
_SCANNED: Final = "a11y.scanned_page"
_ALT_MISSING: Final = "a11y.alt_missing"
_ALT_FILENAME: Final = "a11y.alt_filename"
_SVG_TITLE: Final = "a11y.svg_title"
_TABLE_HEADERS: Final = "a11y.table_headers"
_LINK_EMPTY: Final = "a11y.link_empty"
_LINK_TEXT: Final = "a11y.link_text"

_NO_TREE: Final = (
    "the PDF has no structure tree: assistive technologies read it without"
    " headings, lists, tables or figures"
)
_FILE_NAME: Final = "file name"
_SUBTAG: Final = "-"
_MORE_PAGES: Final = f", {ELLIPSIS}"


def _link_key(text: str) -> str:
    """Return the loose key of the text of a link."""
    counter: Counter[str] = Counter()
    return loose_key(strict_key(normalize_display(text, counter), counter))


def _document_side(doc: Document) -> Side:
    """Return the side of a finding about the whole document.

    It shows the name of a PDF, or the first page of a site.
    """
    name = doc.parts[0] if doc.format == "html" and doc.parts else ""
    name = name or doc.path.name
    return marked_side(SourceRef(name), printable(name))


def _is_untagged(doc: Document) -> bool:
    """Return whether a PDF has no structure tree.

    A tree that was not read, or read in part, for another reason, such as
    encryption or a budget, is no proof that the PDF is untagged.
    """
    tags = doc.tags
    return doc.tagged is False or (
        tags is not None and tags.problem == NO_STRUCTURE_TREE
    )


def prose_sample(doc: Document) -> str:
    """Return the prose whose text tells the language of a document.

    Args:
        doc: The document.

    Returns:
        The texts of its first `LANG_SAMPLE_BLOCKS` blocks of `PROSE_KINDS`,
        joined by spaces.
    """
    return " ".join(
        islice(
            (block.text for block in doc.blocks if block.kind in PROSE_KINDS),
            LANG_SAMPLE_BLOCKS,
        ),
    )


def _language_mismatch(doc: Document, side: Side) -> Draft | None:
    """Return the draft of a document whose prose reads as another language.

    Args:
        doc: The audited document.
        side: The side of a finding about the whole document.

    Returns:
        The draft when the document declares French or English and its
        `prose_sample` plainly reads as the other language; else ``None``.
    """
    declared = doc.declared_lang
    if declared is None:
        return None
    primary = declared.partition(_SUBTAG)[0].casefold()
    if primary not in LANGUAGE_NAMES:
        return None
    found = strong_language(prose_sample(doc))
    if found is None or found == primary:
        return None
    detail = (
        f'the document declares "{quote(declared)}" but its text reads as'
        f" {LANGUAGE_NAMES[found]}"
    )
    return Draft(_LANG_MISMATCH, None, side, None, detail)


def _unmapped(doc: Document, thresholds: AuditThresholds) -> Draft | None:
    """Return the draft of the characters of a PDF without Unicode mapping.

    Args:
        doc: The PDF.
        thresholds: Thresholds of the audit.

    Returns:
        One draft at the first page with such characters, which counts them
        and lists the labels of the first `UNMAPPED_PAGES_SHOWN` pages that
        have some; major when they reach ``unmapped_major`` of the
        characters of the document. ``None`` without such characters.
    """
    touched = [page for page in doc.pdf_pages if page.unmapped]
    if not touched:
        return None
    total = sum(page.unmapped for page in touched)
    chars = sum(page.chars for page in doc.pdf_pages)
    labels = ", ".join(
        quote(page.label) for page in touched[:UNMAPPED_PAGES_SHOWN]
    )
    if len(touched) > UNMAPPED_PAGES_SHOWN:
        labels += _MORE_PAGES
    severity = (
        Severity.MAJOR if total >= thresholds.unmapped_major * chars else None
    )
    source = SourceRef(doc.path.name, page=touched[0].number)
    detail = (
        f"{total} of {chars} characters have no Unicode mapping,"
        f" on page(s) {labels}"
    )
    side = marked_side(source, printable(doc.path.name))
    return Draft(_UNMAPPED, severity, side, None, detail)


def _document_drafts(
    doc: Document,
    thresholds: AuditThresholds,
) -> list[Draft]:
    """Return the drafts about the whole document.

    Returns:
        For a PDF, its missing tree, its missing language, a language
        mismatch, its missing title and its unmapped characters; for a site,
        a language mismatch.
    """
    side = _document_side(doc)
    mismatch = _language_mismatch(doc, side)
    if doc.format != "pdf":
        return [] if mismatch is None else [mismatch]
    drafts: list[Draft] = []
    if _is_untagged(doc):
        drafts.append(Draft(_UNTAGGED, None, side, None, _NO_TREE))
    if doc.declared_lang is None:
        detail = "the PDF declares no language"
        drafts.append(Draft(_LANG_MISSING, None, side, None, detail))
    if mismatch is not None:
        drafts.append(mismatch)
    if not doc.title:
        detail = "the PDF has no title in its metadata"
        drafts.append(Draft(_TITLE_MISSING, None, side, None, detail))
    unmapped = _unmapped(doc, thresholds)
    if unmapped is not None:
        drafts.append(unmapped)
    return drafts


def _bare_table(side: Side) -> Draft:
    """Return the draft of a table without a header cell."""
    detail = "the table has no header cell"
    return Draft(_TABLE_HEADERS, None, side, None, detail, "table")


def _html_page_drafts(page: HtmlPageFacts) -> list[Draft]:
    """Return the drafts of an HTML page.

    Returns:
        Its missing or wrong language, its missing title, then its tables
        without header cell and its links without a name, in order.
    """
    side = marked_side(SourceRef(page.file), printable(page.file))
    drafts: list[Draft] = []
    lang = page.lang or ""
    if not lang.strip():
        detail = "the page declares no language"
        drafts.append(Draft(_LANG_MISSING, None, side, None, detail, "lang"))
    elif declared_language(lang) is None:
        detail = (
            f'the page declares "{quote(lang)}", which is not a language tag'
        )
        drafts.append(Draft(_LANG_MISSING, None, side, None, detail, "lang"))
    if not page.title:
        detail = "the page has no title"
        drafts.append(Draft(_TITLE_MISSING, None, side, None, detail, "title"))
    for spot in page.bare_tables:
        source = SourceRef(page.file, line=spot.line)
        table = marked_side(source, spot.text, kind=BlockKind.TABLE)
        drafts.append(_bare_table(table))
    for spot in page.empty_links:
        link = marked_side(SourceRef(page.file, line=spot.line), spot.text)
        target = quote(spot.text) or '""'
        detail = f"the link to {target} has neither text nor label"
        drafts.append(Draft(_LINK_EMPTY, None, link, None, detail, "empty"))
    return drafts


def _scanned_drafts(doc: Document) -> list[Draft]:
    """Return one draft for each scanned page of a PDF, shown by its label."""
    return [
        Draft(
            _SCANNED,
            None,
            marked_side(
                SourceRef(doc.path.name, page=page.number),
                f"page {quote(page.label)}",
            ),
            None,
            "the page is an image without text",
            "scanned",
        )
        for page in doc.pdf_pages
        if page.scanned
    ]


def _image_draft(block: Block) -> Draft | None:
    """Return the draft of the alternative text of an HTML image, if wrong.

    Returns:
        For an image without an alternative text, an untitled inline SVG,
        whose format is SVG and which has no name, or else a missing
        alternative text; for one that is a file name, its draft; else
        ``None``. An empty alternative text marks a decorative image.
    """
    image = block.image
    if image is None:
        return None
    side = marked_side(
        block.source,
        describe(image),
        kind=block.kind,
        image=image,
    )
    alt = image.alt
    if alt is None:
        if image.format == "svg" and not image.name:
            detail = "the drawing has no title nor label"
            return Draft(_SVG_TITLE, None, side, None, detail, "svg")
        detail = "the image has no alternative text"
        return Draft(_ALT_MISSING, None, side, None, detail, "img")
    if FILE_NAME_ALT.match(alt):
        detail = f'the alternative text "{quote(alt)}" is a file name'
        return Draft(_ALT_FILENAME, None, side, None, detail, _FILE_NAME)
    return None


def _link_text_draft(side: Side, text: str, key: str) -> Draft:
    """Return the draft of a link whose text tells nothing.

    Args:
        side: The side of the link.
        text: The text of the link.
        key: Its loose key, a text of `GENERIC_LINK_TEXTS`, which is the
            pattern of the draft.

    Returns:
        The draft.
    """
    detail = f'the text "{quote(text)}" does not tell where the link leads'
    return Draft(_LINK_TEXT, None, side, None, detail, key)


def _block_drafts(doc: Document) -> list[Draft]:
    """Return the drafts of the blocks, in order.

    Returns:
        For each block, that of its image, on an HTML page, then that of
        each of its links whose text tells nothing. The text of a link is
        judged first; only a block with a link whose text tells nothing has
        its links placed, as `LinkMarks` does.
    """
    html = doc.format == "html"
    drafts: list[Draft] = []
    for block in doc.blocks:
        if html and block.kind is BlockKind.IMAGE:
            image = _image_draft(block)
            if image is not None:
                drafts.append(image)
        marks: LinkMarks | None = None
        for index, link in enumerate(block.links):
            key = _link_key(link.text)
            if key not in GENERIC_LINK_TEXTS:
                continue
            marks = marks or LinkMarks(block, [o.text for o in block.links])
            drafts.append(_link_text_draft(marks.side(index), link.text, key))
    return drafts


def _tagged_draft(doc: Document, element: TaggedElement) -> Draft | None:
    """Return the draft of a figure or a table of the structure tree.

    Returns:
        For a figure without a ``/Alt`` nor an ``/ActualText`` that is not
        blank, a missing alternative text; for one whose ``/Alt`` is a file
        name, that draft; for a table without a ``TH``, a missing header.
        The side shows the type of the element, or the file name, at its
        page.
    """
    source = SourceRef(doc.path.name, page=element.page)
    if element.role == "table" and not element.headers:
        return _bare_table(
            marked_side(source, element.type, kind=BlockKind.TABLE),
        )
    if element.role != "figure":
        return None
    alt = (element.alt or "").strip()
    if not alt and not (element.actual or "").strip():
        side = marked_side(source, element.type, kind=BlockKind.IMAGE)
        detail = "the figure has no alternative text"
        return Draft(_ALT_MISSING, None, side, None, detail, element.type)
    if alt and FILE_NAME_ALT.match(alt):
        side = marked_side(source, alt, kind=BlockKind.IMAGE)
        detail = f'the alternative text "{quote(alt)}" is a file name'
        return Draft(_ALT_FILENAME, None, side, None, detail, _FILE_NAME)
    return None


def _tag_drafts(doc: Document) -> list[Draft]:
    """Return the drafts of the figures and tables of a structure tree.

    Only the elements that were read are judged: a tree read in part gives
    the drafts of its part; a missing tree gives none.
    """
    tags = doc.tags
    if tags is None or tags.problem == NO_STRUCTURE_TREE:
        return []
    return [
        draft
        for element in tags.elements
        if (draft := _tagged_draft(doc, element)) is not None
    ]


def _annotation_drafts(doc: Document) -> list[Draft]:
    """Return the drafts of the link annotations of a PDF that tell nothing.

    The text under an annotation is judged; an empty one, as when the text
    of its page was not all read, is not.
    """
    drafts: list[Draft] = []
    for page in doc.pdf_pages:
        for link in page.links:
            key = _link_key(link.text)
            if key not in GENERIC_LINK_TEXTS:
                continue
            source = SourceRef(doc.path.name, page=page.number)
            side = marked_side(source, link.text)
            drafts.append(_link_text_draft(side, link.text, key))
    return drafts


def accessibility_drafts(
    doc: Document,
    thresholds: AuditThresholds,
) -> list[Draft]:
    """Return the drafts of the accessibility of a document.

    Args:
        doc: The audited document.
        thresholds: Thresholds of the audit.

    Returns:
        The drafts about the whole document; then those of each HTML page,
        or of each scanned PDF page; then those of the blocks, in order;
        then those of the figures and tables of the structure tree of a
        PDF; then those of its link annotations.
    """
    drafts = _document_drafts(doc, thresholds)
    for page in doc.html_pages:
        drafts += _html_page_drafts(page)
    drafts += _scanned_drafts(doc)
    drafts += _block_drafts(doc)
    drafts += _tag_drafts(doc)
    drafts += _annotation_drafts(doc)
    return drafts
