# Copyright 2026 Christophe Le Douarec
"""Read the structure tree of a tagged PDF, in one bounded pass of pypdf.

The audit checks the headings, the figures and the tables that the tags of a
PDF declare. pypdf, a second parser of the same untrusted file, walks its
structure tree once, depth first from its root, with an explicit stack. The
walk visits `MAX_TAG_ITEMS` items at most, reads `MAX_TAG_DEPTH` levels of
elements at most, and visits each indirect object once; a type follows the
role map for `MAX_ROLE_STEPS` steps at most.

pypdf reads in strict mode only. A lenient pypdf repairs a damaged file at a
cost that grows with the file for each object it repairs, and logs some of
these repairs only; a strict one raises instead. Even strict, it would scan
the whole file for each object that the cross-reference table misplaces: such
an object is refused before pypdf looks for it.

pypdf inflates whole, and keeps, each cross-reference stream of the file as
it opens it, and each object stream that the walk reaches. The reader of the
module inflates each of them before pypdf reads it, and counts the bytes:
each cross-reference stream to `MAX_XREF_STREAM_BYTES` at most, each object
stream to `MAX_TAG_STREAM_BYTES`, and all of them to `TAG_INFLATE_WORK`, with
`MAX_XREF_STREAMS` cross-reference streams at most. No filter runs an outer
decoder.

pypdf decrypts what it can with the empty password, which is RC4 without a
cryptography package; an encrypted file that it cannot read only says so.
pypdf says nothing while it reads. An error leaves a tree without elements,
which tells the reason.
"""

import codecs
import logging
import re
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import islice
from typing import TYPE_CHECKING, Any, Final, override

import pypdf
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from pypdf import apply_configuration, get_configuration
from pypdf.errors import PdfReadError
from pypdf.generic import (
    ArrayObject,
    ByteStringObject,
    DictionaryObject,
    IndirectObject,
    NameObject,
    NullObject,
    StreamObject,
    TextStringObject,
)

from doc_checker.model.facts import NO_STRUCTURE_TREE, TaggedElement, TagTree
from doc_checker.text.normalize import printable

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator
    from pathlib import Path

    from pypdf.generic import PdfObject

    from doc_checker.model.facts import TagRole

MAX_TAG_ITEMS: Final = 500_000  # Most items of a tree that are visited.
MAX_TAG_DEPTH: Final = 100  # Most levels of elements that are read.
MAX_ROLE_STEPS: Final = 8  # Most steps through the role map for a type.
TAG_TEXT_LIMIT: Final = 200  # Most characters of an /Alt or /ActualText.
PROBLEM_LIMIT: Final = 80  # Most characters of an error of pypdf told.
# Most children of an element in which the walk looks for the page of its
# marked content: the references to marked content come first in practice.
MARKED_LOOKAHEAD: Final = 16
# Most bytes that an object stream that the walk reaches inflates to: the
# largest one seen in real PDFs inflates to 201 kB.
MAX_TAG_STREAM_BYTES: Final = 1_000_000
# Most bytes that a cross-reference stream inflates to while pypdf opens the
# file: the 2,000,000 entries that pypdf reads at most take 14 MB.
MAX_XREF_STREAM_BYTES: Final = 16_000_000
# Most bytes that the streams that pypdf keeps inflate to, all together: the
# cross-reference streams, then the object streams that the walk reaches,
# each counted `_OBJECT_STREAM_WEIGHT` times; past them, the walk stops and
# the tree is read in part. Real files count from 30 to 230 bytes of these
# streams for each item of their tree, so that densely tagged LaTeX is read
# in full up to about 550 pages; a hostile file takes about 9 s and 0.9 GB.
TAG_INFLATE_WORK: Final = 32_000_000
# Most cross-reference streams that pypdf follows through /Prev: each update
# of a file adds one.
MAX_XREF_STREAMS: Final = 1_000

_NOT_READ: Final = "structure tree not read: "
_ENCRYPTED: Final = f"{_NOT_READ}encrypted"
# Most characters of a text string that are read: the rest is cut before
# `printable`, so that a long string that many elements share costs little
# for each.
_TEXT_READ: Final = 512
# The standard types that the audit reads, as names; and the numbered
# headings, from H1, whose digits are bounded so that their level stays a
# small int.
_STANDARD: Final = frozenset({"/Figure", "/Table", "/TH", "/H"})
_NUMBERED_HEADING: Final = re.compile(r"/H[1-9][0-9]{0,8}")
# The types of the dictionaries that refer to content rather than hold it.
_REFERENCES: Final = frozenset({"/MCR", "/OBJR"})
_END: Final = object()  # Marks the end of the children of a frame.
# The limits of pypdf on the bytes that a filter inflates a stream to.
_INFLATE_LIMITS: Final = (
    "zlib_maximum_output_length",
    "lzw_maximum_output_length",
    "run_length_maximum_output_length",
    "brotli_maximum_output_length",
)
# How many times the bytes that a stream inflates to its filters may handle
# in all: pypdf counts what each filter takes and gives, so that a last
# filter that pypdf does not limit, such as ASCII85, is bounded too.
_DECODING_WORK: Final = 2
# How many times its bytes an object stream counts in `TAG_INFLATE_WORK`:
# pypdf parses tiny objects at up to four times the cost of a byte of a
# cross-reference stream, so that the object streams of a tree inflate to
# half the budget at most.
_OBJECT_STREAM_WEIGHT: Final = 2

# The children of an item that are left to visit, their depth, the page that
# they inherit and the nearest open table.
type _Frame = tuple[Iterator[object], int, int | None, _Found | None]


class _BoundError(BaseException):
    """Stops a reading at one of its bounds.

    It derives from `BaseException`, so that no handler of errors in pypdf
    catches it: pypdf drops the older cross-reference sections of a file,
    without a word, when one of them raises an error.
    """


class _BudgetError(_BoundError):
    """Stops a reading whose streams inflate past `TAG_INFLATE_WORK`."""


def _limits(cap: int) -> dict[str, Any]:
    """Return the settings of pypdf that bound what a stream inflates to.

    Args:
        cap: The most bytes that a stream may inflate to.

    Returns:
        The output of each filter that pypdf limits, at `cap` at most; the
        bytes that all the filters of a stream handle, at `_DECODING_WORK`
        times `cap` at most; each one lowered only, from the current
        settings; and no outer decoder for JBIG2.
    """
    configuration = get_configuration()
    limits: dict[str, Any] = {
        name: min(cap, getattr(configuration, name))
        for name in _INFLATE_LIMITS
    }
    limits["stream_decoding_work_maximum_length"] = min(
        _DECODING_WORK * cap,
        configuration.stream_decoding_work_maximum_length,
    )
    limits["jbig2dec_binary"] = None
    return limits


@contextmanager
def _quiet() -> Generator[None]:
    """Silence pypdf while it reads, and set its limits for the reading.

    The warnings of Python are ignored, and the logger of pypdf drops all but
    its critical messages, until the reading ends. The arrays that pypdf
    reads may hold `MAX_TAG_ITEMS` items, if its own limit is lower, and its
    streams inflate as `_limits` tells for `MAX_XREF_STREAM_BYTES`, each
    limit as the module holds it when called. The level of the logger and
    the limits come back after the reading, however it ends.

    Yields:
        Nothing.
    """
    logger = logging.getLogger("pypdf")
    level = logger.level
    items = max(MAX_TAG_ITEMS, get_configuration().array_maximum_element_count)
    limits = _limits(MAX_XREF_STREAM_BYTES)
    try:
        logger.setLevel(logging.CRITICAL)
        with (
            warnings.catch_warnings(action="ignore"),
            apply_configuration(array_maximum_element_count=items, **limits),
        ):
            yield
    finally:
        logger.setLevel(level)


def _resolved(value: object) -> object:
    """Return the object that a reference leads to, or the value itself."""
    return value.get_object() if isinstance(value, IndirectObject) else value


def _decoded(raw: bytes) -> str:
    """Decode the start of a text string that pypdf left undecoded.

    Args:
        raw: The bytes of the string.

    Returns:
        Its first `_TEXT_READ` characters at least, if it has them: UTF-16
        or UTF-8 after their byte order marks, else Latin-1, with each byte
        that does not decode replaced.
    """
    head = bytes(raw[: 4 * _TEXT_READ + 3])
    if head.startswith(codecs.BOM_UTF16_BE):
        return head[2:].decode("utf-16-be", "replace")
    if head.startswith(codecs.BOM_UTF16_LE):
        return head[2:].decode("utf-16-le", "replace")
    if head.startswith(codecs.BOM_UTF8):
        return head[3:].decode("utf-8", "replace")
    return head.decode("latin-1")


def _text(value: object) -> str | None:
    """Return the printable text of a text string, such as an /Alt.

    Args:
        value: The entry of the string, or ``None`` without the entry.

    Returns:
        The text of its first `_TEXT_READ` characters, printable and cut at
        `TAG_TEXT_LIMIT`; ``None`` if it is not a string. pypdf decodes a
        string in PDFDocEncoding or in UTF-16; a string of UTF-8, which
        PDF 2.0 allows, or one that pypdf could not decode is decoded here.
    """
    value = _resolved(value)
    if isinstance(value, TextStringObject):
        raw = value.original_bytes
        utf8 = raw.startswith(codecs.BOM_UTF8)
        text = _decoded(raw) if utf8 else value[:_TEXT_READ]
    elif isinstance(value, ByteStringObject):
        text = _decoded(value)
    else:
        return None
    return printable(text[:_TEXT_READ], limit=TAG_TEXT_LIMIT)


def _is_standard(name: str) -> bool:
    """Return whether a name is a standard type that the audit reads."""
    return name in _STANDARD or _NUMBERED_HEADING.fullmatch(name) is not None


def _role(name: NameObject, role_map: DictionaryObject) -> str | None:
    """Follow the role map from a type to a standard type.

    Args:
        name: The type of an element.
        role_map: The role map of the tree.

    Returns:
        The first standard type that the audit reads, as `_is_standard`
        tells, met in `MAX_ROLE_STEPS` steps at most and without coming back
        to a type, without its slash; or ``None``.
    """
    seen = {name}
    for _ in range(MAX_ROLE_STEPS):
        if _is_standard(name):
            return name[1:]
        following = _resolved(role_map.get(name))
        if not isinstance(following, NameObject) or following in seen:
            return None
        seen.add(following)
        name = following
    return name[1:] if _is_standard(name) else None


@dataclass(slots=True)
class _Found:
    """A heading, a figure or a table, which the walk can still change."""

    role: TagRole
    type: str
    page: int | None
    level: int | None = None
    alt: str | None = None
    actual: str | None = None
    headers: bool = False

    def frozen(self) -> TaggedElement:
        """Return the element, as the walk leaves it."""
        return TaggedElement(
            self.role,
            self.type,
            self.page,
            level=self.level,
            alt=self.alt,
            actual=self.actual,
            headers=self.headers,
        )


def _found(
    kind: str | None,
    page: int | None,
    element: DictionaryObject,
) -> _Found | None:
    """Make the record of an element that the audit reads.

    Args:
        kind: The standard type of the element, as `_role` gives it.
        page: Its page, from 1, if known.
        element: The element.

    Returns:
        A heading for ``H`` or ``Hn``, with the level n; a figure for
        ``Figure``, with its texts; a table for ``Table``; ``None`` for
        another type.
    """
    match kind:
        case None | "TH":
            return None
        case "Figure":
            return _Found(
                "figure",
                kind,
                page,
                alt=_text(element.get("/Alt")),
                actual=_text(element.get("/ActualText")),
            )
        case "Table":
            return _Found("table", kind, page)
        case "H":
            return _Found("heading", kind, page)
        case _:
            return _Found("heading", kind, page, level=int(kind[1:]))


def _is_reference(item: DictionaryObject) -> bool:
    """Return whether a dictionary refers to content, not holds it."""
    kind = _resolved(item.get("/Type"))
    return isinstance(kind, NameObject) and kind in _REFERENCES


@dataclass(slots=True)
class _Walk:
    """The walk of a structure tree, depth first, with an explicit stack.

    Each integer, reference, array or dictionary of the tree counts as an
    item, a reference already visited too; a dictionary with a type is an
    element, unless it refers to content.

    Attributes:
        limit: The most items to visit.
        pages: The number from 1 of each page, by its object number.
        role_map: The role map of the tree.
        items: The items visited.
        problem: Why the tree is read in part, if it is.
        seen: The indirect objects visited, by number and generation.
        roles: The standard type of each type met, or ``None``.
        found: The headings, figures and tables, in the order of the tree.
        stack: The frames of the items whose children are left to visit.
    """

    limit: int
    pages: dict[int, int] = field(default_factory=dict)
    role_map: DictionaryObject = field(default_factory=DictionaryObject)
    items: int = 0
    problem: str | None = None
    seen: set[tuple[int, int]] = field(default_factory=set)
    roles: dict[NameObject, str | None] = field(default_factory=dict)
    found: list[_Found] = field(default_factory=list)
    stack: list[_Frame] = field(default_factory=list)

    def run(self, kids: object) -> TagTree:
        """Walk the tree from the children of its root.

        Args:
            kids: The ``/K`` entry of the root.

        Returns:
            The headings, figures and tables of the tree, the items visited,
            and why the tree is read in part, if it is: more than `limit`
            items, or object streams that inflate past their budget, after
            which the walk stops; or more than `MAX_TAG_DEPTH` levels of
            elements, whose deeper elements are left; the first reason met.
        """
        self.stack.append((iter((kids,)), 1, None, None))
        try:
            self._walk()
        except _BudgetError as exc:
            self._cut(f"structure tree with {exc}, read in part")
        return TagTree(
            tuple(found.frozen() for found in self.found),
            self.items,
            self.problem,
        )

    def _walk(self) -> None:
        """Visit the items of the stack, until it is empty or too many."""
        while self.stack:
            children, depth, page, table = self.stack[-1]
            child = next(children, _END)
            if child is _END:
                self.stack.pop()
                continue
            if self.items >= self.limit:
                self._cut(
                    f"structure tree of more than {self.limit} items,"
                    " read in part",
                )
                break
            self.items += 1
            self._visit(child, depth, page, table)

    def _cut(self, problem: str) -> None:
        """Keep the first reason why the tree is read in part."""
        if self.problem is None:
            self.problem = problem

    def _visit(
        self,
        item: object,
        depth: int,
        page: int | None,
        table: _Found | None,
    ) -> None:
        """Visit an item, unless it is an indirect object already visited.

        Args:
            item: The item.
            depth: The level of the elements that it holds.
            page: The page that it inherits, if any.
            table: The nearest open table, if any.
        """
        if isinstance(item, IndirectObject):
            key = (item.idnum, item.generation)
            if key in self.seen:
                return
            self.seen.add(key)
            item = item.get_object()
        if isinstance(item, ArrayObject):
            self.stack.append((iter(item), depth, page, table))
        elif isinstance(item, IndirectObject):
            self.stack.append((iter((item,)), depth, page, table))
        elif (
            isinstance(item, DictionaryObject)
            and "/S" in item
            and not _is_reference(item)
        ):
            self._element(item, depth, page, table)

    def _element(
        self,
        element: DictionaryObject,
        depth: int,
        page: int | None,
        table: _Found | None,
    ) -> None:
        """Read an element, then put its children on the stack.

        An element deeper than `MAX_TAG_DEPTH` is left, with its children.
        Its page is that of its ``/Pg``, else that of its first child that
        refers to marked content with a ``/Pg``, else the inherited one; it
        gives its page to the nearest open table if that has none. A ``TH``
        gives that table its headers, and a ``Table`` opens a table.

        Args:
            element: The element.
            depth: Its level.
            page: The page that it inherits, if any.
            table: The nearest open table, if any.
        """
        if depth > MAX_TAG_DEPTH:
            self._cut(
                f"structure tree deeper than {MAX_TAG_DEPTH} levels,"
                " read in part",
            )
            return
        kids = element.get("/K")
        own = self._page(element.get("/Pg"))
        if own is None:
            own = self._marked_page(kids)
        page = page if own is None else own
        if table is not None and table.page is None:
            table.page = page
        kind = self._standard(element.get("/S"))
        found = _found(kind, page, element)
        if found is not None:
            self.found.append(found)
        if kind == "Table":
            table = found
        elif kind == "TH" and table is not None:
            table.headers = True
        if kids is not None:
            self.stack.append((iter((kids,)), depth + 1, page, table))

    def _page(self, value: object) -> int | None:
        """Return the number of the page that a ``/Pg`` refers to, if any."""
        if isinstance(value, IndirectObject):
            return self.pages.get(value.idnum)
        return None

    def _marked_page(self, kids: object) -> int | None:
        """Return the page of the first reference to marked content.

        Args:
            kids: The ``/K`` entry of an element. It is not looked at if it
                is an indirect object already visited, whose children are
                not the element's. Of its children, the first
                `MARKED_LOOKAHEAD` are looked at, and only those written in
                it: a child that is an indirect object is not read ahead of
                the walk, which may never reach it.

        Returns:
            The page of the ``/Pg`` of the first of these children of the
            type ``MCR`` that has one, if any.
        """
        if isinstance(kids, IndirectObject):
            if (kids.idnum, kids.generation) in self.seen:
                return None
            kids = kids.get_object()
        children = kids if isinstance(kids, ArrayObject) else (kids,)
        for child in islice(children, MARKED_LOOKAHEAD):
            if (
                isinstance(child, DictionaryObject)
                and _resolved(child.get("/Type")) == "/MCR"
            ):
                page = self._page(child.get("/Pg"))
                if page is not None:
                    return page
        return None

    def _standard(self, name: object) -> str | None:
        """Return the standard type of a type, as `_role` tells, once each."""
        name = _resolved(name)
        if not isinstance(name, NameObject):
            return None
        if name not in self.roles:
            self.roles[name] = _role(name, self.role_map)
        return self.roles[name]


def _page_numbers(reader: PdfReader) -> dict[int, int]:
    """Number the pages of a document by their object numbers.

    Args:
        reader: The document.

    Returns:
        The number from 1 of each page that is an indirect object, the first
        one for a page that the page tree holds twice.
    """
    numbers: dict[int, int] = {}
    for number, page in enumerate(reader.pages, 1):
        reference = page.indirect_reference
        if reference is not None:
            numbers.setdefault(reference.idnum, number)
    return numbers


def _in_place(
    reader: pypdf.PdfReader,
    reference: IndirectObject,
    offset: int,
) -> bool:
    """Return whether an object starts where the cross-reference table says.

    Args:
        reader: The document.
        reference: The object.
        offset: Where the table puts it.

    Returns:
        Whether the header that pypdf reads there names the object.
    """
    stream = reader.stream
    position = stream.tell()
    try:
        stream.seek(offset)
        header = reader.read_object_header(stream)
    except ValueError, PdfReadError:
        return False
    finally:
        stream.seek(position)
    return header == (reference.idnum, reference.generation)


@dataclass(slots=True)
class _Inflation:
    """The bytes that the streams of a reading inflate to.

    Attributes:
        limit: The most bytes that all of them may inflate to.
        cap: The most bytes that the next one may inflate to.
        room: The bytes left.
        sections: The cross-reference streams counted.
        streams: The object numbers of the object streams counted.
    """

    limit: int
    cap: int
    room: int = field(init=False)
    sections: int = 0
    streams: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        """Leave the whole limit."""
        self.room = self.limit

    def charge(self, stream: StreamObject, *, weight: int = 1) -> None:
        """Inflate a stream, which pypdf keeps inflated, and count its bytes.

        Args:
            stream: The stream, before pypdf reads it.
            weight: How many times its bytes count.

        Raises:
            _BudgetError: If the streams counted pass `limit`.
            _BoundError: If the stream inflates past `cap`.
        """
        size = len(stream.get_data())
        self.room -= weight * size
        if self.room < 0:
            msg = f"streams inflated past a budget of {self.limit} bytes"
            raise _BudgetError(msg)
        if size > self.cap:
            msg = f"stream inflated to more than {self.cap} bytes"
            raise _BoundError(msg)

    def charge_section(self, stream: StreamObject) -> None:
        """Count a cross-reference stream, and the bytes it inflates to.

        Args:
            stream: The stream, before pypdf reads it.

        Raises:
            _BoundError: If there are more than `MAX_XREF_STREAMS`, as the
                module holds it when called, or if the stream inflates past
                `cap`.
        """
        self.sections += 1
        if self.sections > MAX_XREF_STREAMS:
            msg = f"more than {MAX_XREF_STREAMS} cross-reference streams"
            raise _BoundError(msg)
        self.charge(stream)


class PdfReader(pypdf.PdfReader):
    """The reader of pypdf, bounded for the walk of a structure tree.

    It keeps the name of the reader of pypdf, which the tests replace. While
    it opens a file, each cross-reference stream that pypdf keeps is counted
    and inflated before pypdf reads it, as `_Inflation.charge_section`
    tells. Once open, a fetch of an object that pypdf has not cached checks
    the header that the cross-reference table points at, in constant time,
    as pypdf does: even strict, pypdf would scan the whole file for an
    object that the table misplaces, and for one that it does not hold, once
    for each such object. An object stream is inflated and counted before
    pypdf parses its objects, as `_Inflation.charge` tells, its bytes
    counted `_OBJECT_STREAM_WEIGHT` times. The fetches of pypdf itself,
    through the page tree or an object stream, are checked too.
    """

    def __init__(
        self,
        stream: Path,
        *,
        strict: bool,
        inflation: _Inflation,
    ) -> None:
        """Open a file, counting its cross-reference streams.

        Args:
            stream: The PDF file.
            strict: Whether pypdf raises rather than repairs.
            inflation: The bytes that its streams may inflate to.
        """
        self._inflation = inflation
        self._open = False
        super().__init__(stream, strict=strict)
        self._open = True

    @override
    def cache_indirect_object(
        self,
        generation: int,
        idnum: int,
        obj: PdfObject | None,
    ) -> PdfObject | None:
        """Keep an object that pypdf has read, counting a cross-reference one.

        Returns:
            The object, as pypdf keeps it.
        """
        if (
            isinstance(obj, StreamObject)
            and _resolved(obj.get("/Type")) == "/XRef"
        ):
            self._inflation.charge_section(obj)
        return super().cache_indirect_object(generation, idnum, obj)

    @override
    def get_object(
        self,
        indirect_reference: int | IndirectObject,
    ) -> PdfObject | None:
        """Fetch an object, unless the table misplaces it.

        Returns:
            The object, or a null object if the table does not hold it.

        Raises:
            PdfReadError: If the table misplaces it.
        """
        if not self._open:
            return super().get_object(indirect_reference)
        reference = indirect_reference
        if not isinstance(reference, IndirectObject):
            reference = IndirectObject(reference, 0, self)
        number, generation = reference.idnum, reference.generation
        if self.cache_get_indirect_object(generation, number) is None:
            if generation == 0 and number in self.xref_objStm:
                self._inflate(self.xref_objStm[number][0])
            else:
                offset = self.xref.get(generation, {}).get(number)
                if offset is None:
                    return NullObject()
                free = self.xref_free_entry.get(generation, {}).get(number)
                if not free and not _in_place(self, reference, offset):
                    msg = f"object {number} {generation} is misplaced"
                    raise PdfReadError(msg)
        return super().get_object(reference)

    def _inflate(self, number: int) -> None:
        """Inflate and count an object stream, once, before pypdf parses it.

        Args:
            number: The object number of the stream.
        """
        if number in self._inflation.streams:
            return
        self._inflation.streams.add(number)
        stream = self.get_object(IndirectObject(number, 0, self))
        if isinstance(stream, StreamObject):
            self._inflation.charge(stream, weight=_OBJECT_STREAM_WEIGHT)


def _read(reader: PdfReader, walk: _Walk, inflation: _Inflation) -> TagTree:
    """Walk the structure tree of a document that pypdf has opened.

    Args:
        reader: The document, opened in strict mode, and decrypted if it is
            encrypted.
        walk: The walk, without its pages and its role map yet.
        inflation: The bytes that its streams may inflate to, those of its
            cross-reference streams counted.

    Returns:
        The tree, as `_Walk.run` reads it, each object stream inflating to
        `MAX_TAG_STREAM_BYTES` at most, as the module holds it when called,
        and within the limits that `_limits` tells for it; or none if the
        catalog has no structure tree root that is a dictionary.
    """
    inflation.cap = MAX_TAG_STREAM_BYTES
    with apply_configuration(**_limits(MAX_TAG_STREAM_BYTES)):
        root = _resolved(reader.root_object.get("/StructTreeRoot"))
        if not isinstance(root, DictionaryObject):
            return TagTree((), 0, NO_STRUCTURE_TREE)
        role_map = _resolved(root.get("/RoleMap"))
        if isinstance(role_map, DictionaryObject):
            walk.role_map = role_map
        walk.pages = _page_numbers(reader)
        return walk.run(root.get("/K"))


def _encrypted(path: Path) -> bool:
    """Return whether PDFium finds a file encrypted.

    pypdf may fail on an encrypted file before it tells, such as when it
    needs a cryptography package for AES-256. PDFium opens the file as the
    loader does, without inflating its object streams.

    Args:
        path: The PDF file.

    Returns:
        Whether the file has a security handler, or a password that is not
        empty; ``False`` if PDFium cannot open it otherwise.
    """
    try:
        with pdfium.PdfDocument(path) as pdf:
            return pdfium_c.FPDF_GetSecurityHandlerRevision(pdf) != -1
    except pdfium.PdfiumError as exc:
        return exc.err_code == pdfium_c.FPDF_ERR_PASSWORD
    except OSError:
        return False


def read_tags(path: Path) -> TagTree:
    """Read the headings, the figures and the tables of a tagged PDF.

    pypdf reads the file in strict mode, through the `PdfReader` of the
    module, decrypted with the empty password if it is encrypted. The
    structure tree is walked as `_Walk` tells, with `MAX_TAG_ITEMS` as the
    module holds it when called. A type is the standard type that the role
    map of the tree leads it to, as `_role` tells. pypdf says nothing while
    it reads, and inflates its streams within the limits that `_quiet` and
    `_read` tell, `TAG_INFLATE_WORK` bytes in all.

    Args:
        path: The PDF file.

    Returns:
        The tree. It is empty, and tells why, if the file has no structure
        tree, or if pypdf fails or passes a bound before the walk: the
        reason is then ``encrypted`` for a file that pypdf or PDFium finds
        encrypted, else the error, cut at `PROBLEM_LIMIT` characters.
    """
    walk = _Walk(MAX_TAG_ITEMS)
    inflation = _Inflation(TAG_INFLATE_WORK, MAX_XREF_STREAM_BYTES)
    encrypted = False
    with _quiet():
        try:
            reader = PdfReader(path, strict=True, inflation=inflation)
            encrypted = bool(reader.is_encrypted)
            if encrypted and not reader.decrypt(""):
                return TagTree((), 0, _ENCRYPTED)
            tree = _read(reader, walk, inflation)
        except (Exception, _BoundError) as exc:  # ruff: ignore[blind-except] - pypdf raises many kinds of errors on hostile files
            if encrypted or _encrypted(path):
                return TagTree((), walk.items, _ENCRYPTED)
            reason = printable(
                str(exc) or type(exc).__name__,
                limit=PROBLEM_LIMIT,
            )
            return TagTree((), walk.items, f"{_NOT_READ}{reason}")
    return tree
