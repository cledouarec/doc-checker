# Copyright 2026 Christophe Le Douarec
"""Check the internal links of an HTML site and the links of a PDF.

Each internal link of the blocks of a site is checked: one to a page that
does not exist is ``link.page_missing`` if the target is a page, by its
suffix, else ``link.file_missing``, but a link without final slash to a
file of the site without suffix, such as ``LICENSE``, which the target
names as a folder, leads somewhere; one to an anchor that the target page
does not have, among the identifiers of the whole page, is
``link.anchor_missing``. The ``top`` fragment, which browsers serve, and a
text directive, ``:~:``, are not anchors to check, and nor are the anchors
of a page that exists but was not read. An external link, or one that goes
out of the site, is never checked.

Each link annotation of a PDF is checked: a destination that leads to no
page of the document, or a link without destination nor action, is
``link.destination_invalid``; a link that opens a local file that is not
there is ``link.file_missing``. A URI is never checked.

The texts of the document that a detail or a pattern quotes are cut.
"""

import urllib.parse
from typing import TYPE_CHECKING, Final

from doc_checker.audit.sides import LinkMarks, quote
from doc_checker.compare.drafts import Draft, marked_side
from doc_checker.extract.html.ordering import INDEX_PAGE
from doc_checker.model.document import SourceRef
from doc_checker.text.normalize import printable

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from doc_checker.model.document import Block, Document, Link
    from doc_checker.model.facts import PdfLink

UNCHECKED_FRAGMENTS: Final = frozenset({"top"})
TEXT_FRAGMENT: Final = ":~:"
PAGE_SUFFIXES: Final = (".html", ".htm")

_PAGE_MISSING: Final = "link.page_missing"
_FILE_MISSING: Final = "link.file_missing"
_ANCHOR_MISSING: Final = "link.anchor_missing"
_DESTINATION_INVALID: Final = "link.destination_invalid"
# What ends the target of a link to a folder, or to a path without suffix.
_FOLDER_PAGE: Final = f"/{INDEX_PAGE}"
# The actions of an annotation that lead to a page of the document.
_PAGE_ACTIONS: Final = frozenset({"goto", "none"})


def _anchor(fragment: str | None) -> str | None:
    """Return the anchor of a fragment, without its text directive."""
    if not fragment:
        return None
    return fragment.partition(TEXT_FRAGMENT)[0] or None


def _is_site_file(link: Link, root: Path) -> bool:
    """Return whether a link without suffix leads to a file of the site.

    A path without suffix is resolved as a folder, to its index page; a
    link to it without a final slash may lead to a file of that name, such
    as ``LICENSE``.

    Args:
        link: An internal link, which has a page.
        root: Directory of the site.

    Returns:
        Whether the target is the index page of a folder, the link has no
        final slash, and a file of the name of the folder exists in the
        site, a link to it resolved inside the root.
    """
    target = link.page or ""
    if not target.endswith(_FOLDER_PAGE):
        return False
    if urllib.parse.urlsplit(link.href).path.endswith("/"):
        return False
    file = root / target.removesuffix(_FOLDER_PAGE)
    try:
        return file.is_file() and file.resolve().is_relative_to(root.resolve())
    except OSError:
        return False


def _html_problem(
    link: Link,
    ids: Mapping[str, frozenset[str]],
    root: Path,
) -> tuple[str, str, str] | None:
    """Return what is wrong with an internal link of a block, if anything.

    Args:
        link: An internal link, which has a page.
        ids: The identifiers of each page that was read.
        root: Directory of the site.

    Returns:
        The rule, the detail and the pattern of a missing page or file, or
        of a missing anchor of a page that was read; else ``None``. A link
        to a file of the site without suffix, as `_is_site_file` tells,
        leads somewhere.
    """
    target = link.page or ""
    page, text = quote(printable(target)), quote(link.text)
    if not (link.found or _is_site_file(link, root)):
        rule = (
            _PAGE_MISSING
            if target.lower().endswith(PAGE_SUFFIXES)
            else _FILE_MISSING
        )
        return rule, f'"{text}" links to {page}, which does not exist', page
    anchor = _anchor(link.fragment)
    known = ids.get(target)
    if (
        anchor is None
        or known is None
        or anchor in known
        or anchor.lower() in UNCHECKED_FRAGMENTS
    ):
        return None
    name = quote(anchor)
    detail = f'"{text}" links to {page}, which has no anchor "{name}"'
    return _ANCHOR_MISSING, detail, quote(f"{printable(target)}#{anchor}")


def _block_drafts(
    block: Block,
    ids: Mapping[str, frozenset[str]],
    root: Path,
) -> tuple[list[Draft], int]:
    """Return the drafts of the broken internal links of a block.

    Args:
        block: A block of a site.
        ids: The identifiers of each page that was read.
        root: Directory of the site.

    Returns:
        The drafts, in order, each side marking the text of its link at
        its place, as `LinkMarks` places the links of a block that has a
        broken one; and the number of internal links checked.
    """
    drafts: list[Draft] = []
    checked = 0
    marks: LinkMarks | None = None
    for index, link in enumerate(block.links):
        if link.url is not None or link.page is None:
            continue
        checked += 1
        problem = _html_problem(link, ids, root)
        if problem is None:
            continue
        marks = marks or LinkMarks(block, [o.text for o in block.links])
        rule, detail, pattern = problem
        side = marks.side(index)
        drafts.append(Draft(rule, None, side, None, detail, pattern))
    return drafts, checked


def _pdf_draft(doc: Document, number: int, link: PdfLink) -> Draft | None:
    """Return the draft of a link annotation of a PDF page, if it is broken.

    Args:
        doc: The PDF.
        number: The page of the annotation, from 1.
        link: The annotation.

    Returns:
        The draft of a destination that leads to no page, or of a file
        that is not there; else ``None``.
    """
    if link.valid is not False:
        return None
    side = marked_side(SourceRef(doc.path.name, page=number), link.text)
    text, target = quote(link.text), quote(link.target)
    if link.action == "file":
        detail = (
            f'"{text}" opens {target}, which does not exist next to the PDF'
        )
        return Draft(_FILE_MISSING, None, side, None, detail, target)
    if link.action in _PAGE_ACTIONS:
        detail = f'"{text}" leads to no page of the document'
        return Draft(
            _DESTINATION_INVALID,
            None,
            side,
            None,
            detail,
            "destination",
        )
    return None


def broken_link_drafts(doc: Document) -> tuple[list[Draft], int]:
    """Return the drafts of the broken links of a document.

    Args:
        doc: The audited document.

    Returns:
        The drafts, those of the links of the blocks of a site, in order,
        then those of the link annotations of a PDF, page by page; and the
        number of links checked: the internal links of the blocks, those
        that are not looked into included, and every annotation.
    """
    ids = {page.file: page.ids for page in doc.html_pages}
    root = doc.path.parent if doc.path.is_file() else doc.path
    drafts: list[Draft] = []
    checked = 0
    for block in doc.blocks:
        found, count = _block_drafts(block, ids, root)
        drafts += found
        checked += count
    for page in doc.pdf_pages:
        checked += len(page.links)
        drafts += [
            draft
            for link in page.links
            if (draft := _pdf_draft(doc, page.number, link)) is not None
        ]
    return drafts, checked
