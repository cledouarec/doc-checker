# Copyright 2026 Christophe Le Douarec
"""Count the work of the reading of a page by pdfminer, before it reads it.

pdfminer interprets the content streams of a page and of the forms that they
draw, in a time that grows with their bytes. `PageWork` counts that work from
the streams themselves, against a budget, before pdfminer reads the page: a
page beyond `MAX_TABLE_PAGE_WORK` is not read. The count reads the content as
pdfminer would, with a lexer of its own that leaves out the strings, the
comments and the data of the inline images, in a time linear in the content:
`_hidden`. The search of `tables` uses it as a fast first filter.
"""

import re
import zlib
from bisect import bisect_right
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pdfminer.pdfpage import PDFPage
    from pdfminer.pdftypes import PDFStream

# Most work of the reading of a page by pdfminer, as `PageWork` counts it:
# about the bytes that pdfminer interprets, at 1.4 us a byte of operators that
# draw nothing and up to 4 us a byte of text. The pages of the PGF manual that
# the other budgets let through need 151,148 at most, those of the ruled
# samples 35,759; at the limit, a page takes 0.7 to 2 s to read.
MAX_TABLE_PAGE_WORK: Final = 500_000
# Most forms within forms that `PageWork` follows: a page that draws deeper
# forms is beyond `MAX_TABLE_PAGE_WORK`.
MAX_FORM_DEPTH: Final = 8
# A form or an image drawn by its name: the name, blanks, then the operator Do,
# which a blank or a delimiter ends. The name runs to the first blank or
# delimiter, so that the match is linear.
_DRAW: Final = re.compile(rb"/([^/%\[\]()<>{}\s]+)\s+Do(?![^/%\[\]()<>{}\s])")
# An escape in a name: # and up to two hexadecimal digits, as pdfminer reads
# it; # alone is dropped.
_NAME_ESCAPE: Final = re.compile(rb"#([0-9A-Fa-f]{0,2})")
# A Do operator: Do between blanks or delimiters, or the ends of the content.
_DO: Final = re.compile(rb"(?<![^/%\[\]()<>{}\s])Do(?![^/%\[\]()<>{}\s])")
# What opens a span of content that hides its operators from pdfminer: a
# string, a comment, or the keywords that begin an inline image and its data.
_HIDING: Final = re.compile(
    rb"[(%]|(?<![^/%\[\]()<>{}\s])(?:BI|ID)(?![^/%\[\]()<>{}\s])",
)
# What a string turns on: its parentheses, and the escapes of the next byte.
_STRING_MARK: Final = re.compile(rb"[()\\]")
_LINE_END: Final = re.compile(rb"[\r\n]")
_OPEN: Final = ord("(")
_BACKSLASH: Final = ord("\\")
# A token of the dictionary of an inline image, as far as its filter goes: a
# name, its escapes unread, an opening bracket, or a run of other bytes up to a
# blank, a name or a bracket.
_INLINE_TOKEN: Final = re.compile(rb"/([^/%\[\]()<>{}\s]*)|\[|[^/\[\s]+")
# The names of the ASCII85 filter, which ends the data of an inline image at ~>
# rather than at EI, under the only key that pdfminer reads for it, /F.
_ASCII85: Final = frozenset(("A85", "ASCII85Decode"))
_BLANKS: Final = frozenset(b" \t\n\r\x0b\x0c")  # The blanks of bytes.isspace.
# The value of the resources and the dictionaries that a document lacks: one
# mapping, never changed, whose address no other object takes.
_EMPTY: Final[Mapping[Any, Any]] = MappingProxyType({})
# Most references that `PageWork` follows from a font given in place.
_FONT_DEPTH: Final = 4
_INFLATE_CHUNK: Final = 65_536  # Bytes of a compressed stream fed at a time.
# The resources that pdfminer reads again each time that a form is drawn.
_CALL_RESOURCES: Final = ("Font", "ColorSpace", "XObject")


def _unescaped(name: bytes) -> str:
    """Read a name of a content stream as pdfminer reads it.

    Args:
        name: The name, without its slash.

    Returns:
        The name with its escapes read, in UTF-8, or as the text of its bytes
        if it is not UTF-8.
    """
    raw = _NAME_ESCAPE.sub(
        lambda match: bytes((int(match[1], 16),)) if match[1] else b"",
        name,
    )
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return str(raw)


def _string_end(data: bytes, start: int) -> int:
    """Find where a string of content ends.

    Args:
        data: The decoded content.
        start: Where the string starts, after its opening parenthesis.

    Returns:
        Where the string ends, after its closing parenthesis, its nested
        parentheses and escapes read as pdfminer reads them; the end of the
        content if the string is open.
    """
    depth = 1
    at = start
    while (mark := _STRING_MARK.search(data, at)) is not None:
        at = mark.end()
        if data[mark.start()] == _BACKSLASH:
            at += 1
        elif data[mark.start()] == _OPEN:
            depth += 1
        else:
            depth -= 1
            if not depth:
                return at
    return len(data)


def _ascii85(data: bytes, start: int, end: int) -> bool:
    """Tell whether the dictionary of an inline image names ASCII85 first.

    Args:
        data: The decoded content.
        start: Where the dictionary starts, after ``BI``.
        end: Where it ends, at ``ID``.

    Returns:
        Whether the name ``F`` comes, after blanks and an opening bracket at
        most, before the name of the ASCII85 filter, each name with its
        escapes read as `_unescaped` reads them.
    """
    after = 0  # 1 after the name F, 2 after it and an opening bracket.
    for token in _INLINE_TOKEN.finditer(data, start, end):
        if token[1] is None:
            after = 2 if after == 1 and token[0] == b"[" else 0
            continue
        name = _unescaped(token[1])
        if after and name in _ASCII85:
            return True
        after = 1 if name == "F" else 0
    return False


def _inline_end(data: bytes, start: int, target: bytes) -> int:
    """Find where the data of an inline image ends, as pdfminer finds it.

    pdfminer looks for the first byte of the end mark, then reads on, byte
    after byte, the rest of the mark and a blank; a byte that does not fit
    starts the search afresh after it.

    Args:
        data: The decoded content.
        start: Where the data starts, after ``ID`` and one byte.
        target: The end mark, ``EI`` or ``~>``.

    Returns:
        Where the data ends, after the mark and its blank, or the end of the
        content.
    """
    at = start
    while (found := data.find(target[:1], at)) >= 0:
        at = found + 1
        for expected in (*target[1:], None):
            if at >= len(data):
                return len(data)
            byte = data[at]
            at += 1
            if byte != expected and (
                expected is not None or byte not in _BLANKS
            ):
                break
        else:
            return at
    return len(data)


def _hidden(data: bytes) -> list[tuple[int, int]]:
    """Find the spans of some content that hide their operators from pdfminer.

    Args:
        data: The decoded content.

    Returns:
        In order, the span of each string, as `_string_end` reads it, of each
        comment, to the end of its line, and of the data of each inline image
        that ``BI`` begins, from its keyword ``ID`` to its end, as
        `_inline_end` finds it.
    """
    spans: list[tuple[int, int]] = []
    image = -1  # Where the dictionary of an inline image starts, once open.
    at = 0
    while (mark := _HIDING.search(data, at)) is not None:
        start = mark.start()
        at = mark.end()
        match mark[0]:
            case b"(":
                end = _string_end(data, at)
            case b"%":
                line = _LINE_END.search(data, at)
                end = len(data) if line is None else line.start()
            case b"BI":
                image = at
                continue
            case b"ID" if image >= 0:
                a85 = _ascii85(data, image, start)
                end = _inline_end(data, at + 1, b"~>" if a85 else b"EI")
                image = -1
            case _:
                continue
        spans.append((start, end))
        at = end
    return spans


def _operators(data: bytes, drawn: set[int]) -> int:
    """Count the Do operators of some content that `_DRAW` did not find.

    Args:
        data: The decoded content.
        drawn: Where the operators that `_DRAW` found start.

    Returns:
        The number of the other Do keywords that stand outside the strings
        and the comments of the content, as `_hidden` finds them.
    """
    others = [m.start() for m in _DO.finditer(data) if m.start() not in drawn]
    if not others:
        return 0
    spans = _hidden(data)
    starts = [start for start, _ in spans]
    count = 0
    for at in others:
        index = bisect_right(starts, at) - 1
        count += index < 0 or at >= spans[index][1]
    return count


class OverWorkError(Exception):
    """A page needs more work than `MAX_TABLE_PAGE_WORK` to be read."""


class PageWork:
    """Count the work that pdfminer would do to read a page, before it does.

    pdfminer interprets each byte of the content streams of a page, then of
    each form that they draw, each time that they draw it, after it has read
    the resources of the form again: a name or a procedure set each, and each
    font given in place rather than referred to, which it builds again. So a
    page costs the bytes of its content streams, decoded, and each drawing of
    a form costs the bytes of the form, the names of its resources, all that
    its fonts given in place reach, and the drawings of its own forms. A form
    that is drawn again in the same resources costs what it cost the first
    time. The drawings are found by `_DRAW`; each other Do operator, outside
    the strings and the comments, costs a drawing of the dearest form of its
    resources. A page that draws forms more than `MAX_FORM_DEPTH` deep needs
    too much work.

    Each stage of Flate of a stream is inflated in pieces, never past what is
    left of the budget, and leniently, as pdfminer does: the bytes before an
    error count. The other filters are decoded whole by pdfminer, as its
    reading would: the reading process of `read_tables` bounds them, as it
    bounds what this count leaves out, such as the fonts that the resources
    refer to.

    Attributes:
        room: The work left before `MAX_TABLE_PAGE_WORK`.
    """

    def __init__(self, room: int) -> None:
        """Start the count of a page.

        Args:
            room: The most work that the page may need.
        """
        from pdfminer import pdftypes, psparser  # ruff: ignore[import-outside-top-level] - optional extra

        self.room = room
        self._types = pdftypes
        self._form_type = psparser.LIT("Form")
        # The work of a drawing of a form, by form and resources, with the
        # form and the resources themselves, which stay alive with the
        # count so that their addresses never go to other objects.
        self._drawings: dict[tuple[int, int], tuple[object, object, int]] = {}
        # The work of the reading of resources, by resources, likewise.
        self._resources: dict[int, tuple[object, int]] = {}

    def spend(self, work: int) -> None:
        """Count some work.

        Args:
            work: The work.

        Raises:
            OverWorkError: If the page needs more than its budget.
        """
        self.room -= work
        if self.room < 0:
            raise OverWorkError

    def _check(self, work: int) -> None:
        """Stop a count that passes the work left.

        Args:
            work: The work counted so far, not spent yet.

        Raises:
            OverWorkError: If it passes the work left.
        """
        if work > self.room:
            raise OverWorkError

    def read(self, page: PDFPage) -> None:
        """Count the work of the reading of a page.

        Args:
            page: The page, as pdfminer reads it.
        """
        parts = []
        for content in page.contents:
            part = self._data(self._types.stream_value(content))
            self.spend(len(part))
            parts.append(part)
        resources = self._dict(page.resources)
        self.spend(self._resource_work(resources))
        self.spend(self._drawings_work(b"".join(parts), resources, 0))

    def _dict(self, value: object) -> Mapping[Any, Any]:
        """Return a value as a dictionary, or the shared empty one."""
        value = self._types.resolve1(value)
        return value if isinstance(value, dict) else _EMPTY

    def _data(self, stream: PDFStream) -> bytes:
        """Decode a stream, up to the work left and one byte more.

        Args:
            stream: The stream.

        Returns:
            Its data, decoded and deciphered as pdfminer decodes it; for a
            stream of Flate stages alone, its first bytes only if a stage
            would pass the work left.
        """
        if stream.data is not None:
            return stream.data
        filters = stream.get_filters()
        if not all(
            name in self._types.LITERALS_FLATE_DECODE for name, _ in filters
        ) or any(
            "Predictor" in self._dict(params) for _, params in filters[:-1]
        ):
            return stream.get_data()
        data = stream.rawdata or b""
        if stream.decipher:
            data = stream.decipher(
                stream.objid,
                stream.genno,
                data,
                stream.attrs,
            )
        for _ in filters:
            data = self._inflated(data)
            if len(data) > self.room:
                return data
        if filters and "Predictor" in self._dict(filters[-1][1]):
            # A predictor never lengthens the data that fits.
            return stream.get_data()
        return data

    def _inflated(self, raw: bytes) -> bytes:
        """Inflate data, up to the work left and one byte more.

        Args:
            raw: The data, compressed with Flate.

        Returns:
            The inflated data, or its first bytes if the whole would pass the
            work left; up to an error, if the data is damaged.
        """
        inflater = zlib.decompressobj()
        parts: list[bytes] = []
        size = 0
        try:
            for start in range(0, len(raw), _INFLATE_CHUNK):
                part = inflater.decompress(
                    raw[start : start + _INFLATE_CHUNK],
                    self.room + 1 - size,
                )
                parts.append(part)
                size += len(part)
                if size > self.room or inflater.eof:
                    break
        except zlib.error:
            pass  # pdfminer reads what comes before the error, or nothing.
        return b"".join(parts)

    def _resource_work(self, resources: Mapping[Any, Any]) -> int:
        """Count the work of the reading of resources by pdfminer.

        Args:
            resources: The resources.

        Returns:
            The number of their fonts, colour spaces, forms and procedure
            sets, plus all that their fonts given in place reach, as `_reach`
            counts it.

        Raises:
            OverWorkError: If the count passes the work left.
        """
        key = id(resources)
        if key not in self._resources:
            work = sum(
                len(self._dict(resources.get(name)))
                for name in _CALL_RESOURCES
            )
            procedures = self._types.resolve1(resources.get("ProcSet"))
            if isinstance(procedures, list):
                work += len(procedures)
            for font in self._dict(resources.get("Font")).values():
                if not isinstance(font, self._types.PDFObjRef):
                    work += self._reach(font, _FONT_DEPTH, set())
                if work > self.room:
                    raise OverWorkError
            self._resources[key] = (resources, work)
        return self._resources[key][1]

    def _reach(self, value: object, depth: int, seen: set[int]) -> int:
        """Count all that a value reaches, which pdfminer reads for a font.

        Args:
            value: The value.
            depth: How many references are followed from it, at most.
            seen: The objects already counted, which are not counted again.

        Returns:
            The number of the items of its arrays and dictionaries, and of
            their streams, plus the decoded bytes of its streams.

        Raises:
            OverWorkError: If the count passes the work left.
        """
        if isinstance(value, self._types.PDFObjRef):
            if not depth or value.objid in seen:
                return 0
            seen.add(value.objid)
            value = self._types.resolve1(value)
            depth -= 1
        if isinstance(value, self._types.PDFStream):
            work = len(self._data(value)) + len(value.attrs)
            items = list(value.attrs.values())
        elif isinstance(value, dict):
            work, items = len(value), list(value.values())
        elif isinstance(value, list):
            work, items = len(value), value
        else:
            return 0
        for item in items:
            if work > self.room:
                raise OverWorkError
            work += self._reach(item, depth, seen)
        return work

    def _form(self, forms: Mapping[Any, Any], name: str) -> PDFStream | None:
        """Return the form of some resources that a name draws, if it is one.

        Args:
            forms: The XObjects of the resources.
            name: The name.

        Returns:
            The XObject of the name if it is a form, as pdfminer draws it:
            with a bounding box; else None.
        """
        xobject = forms.get(name)
        if xobject is None:
            return None
        form = self._types.stream_value(xobject)
        if form.get("Subtype") is self._form_type and "BBox" in form:
            return form
        return None

    def _drawings_work(
        self,
        data: bytes,
        resources: Mapping[Any, Any],
        depth: int,
    ) -> int:
        """Count the drawings of the forms of some content.

        Args:
            data: The decoded content.
            resources: The resources in which its names are read.
            depth: How deep the content is among forms, the page being 0.

        Returns:
            The work of each drawing of a form that `_DRAW` finds, plus the
            work of a drawing of the dearest form of the resources for each
            other Do operator, as `_operators` counts them. The count stops
            with `OverWorkError` once it passes the work left.
        """
        forms = self._dict(resources.get("XObject"))
        if not forms:
            return 0
        work = 0
        drawn = set()
        for match in _DRAW.finditer(data):
            drawn.add(match.end() - len(b"Do"))
            form = self._form(forms, _unescaped(match[1]))
            if form is not None:
                work += self._drawing_work(form, resources, depth + 1)
                self._check(work)
        others = _operators(data, drawn)
        if others:
            dearest = 0
            for name in list(forms):
                form = self._form(forms, name)
                if form is not None:
                    cost = self._drawing_work(form, resources, depth + 1)
                    dearest = max(dearest, cost)
            work += others * dearest
            self._check(work)
        return work

    def _drawing_work(
        self,
        form: PDFStream,
        resources: Mapping[Any, Any],
        depth: int,
    ) -> int:
        """Count a drawing of a form, and those of its own forms.

        Args:
            form: The form.
            resources: The resources of the content that draws it, which a
                form without resources of its own uses.
            depth: How deep the form is among forms, from 1.

        Returns:
            The work of the drawing.

        Raises:
            OverWorkError: If the count passes the work left, or the form is
                deeper than `MAX_FORM_DEPTH`.
        """
        own = form.get("Resources")
        used = self._dict(own) if own else resources
        key = (id(form), id(used))
        if key in self._drawings:
            return self._drawings[key][2]
        if depth > MAX_FORM_DEPTH:
            raise OverWorkError
        work = self._resource_work(used)
        self._check(work)
        data = self._data(form)
        work += len(data)
        self._check(work)
        work += self._drawings_work(data, used, depth)
        self._check(work)
        self._drawings[key] = (form, used, work)
        return work
