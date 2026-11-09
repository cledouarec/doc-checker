# Copyright 2026 Christophe Le Douarec
"""Tests for the structure tree of tagged PDF documents."""

import base64
import logging
import re
import struct
import subprocess  # ruff: ignore[suspicious-subprocess-import] - runs this interpreter on a fixed command
import sys
import time
import tracemalloc
import warnings
import zlib
from typing import TYPE_CHECKING, NoReturn

from pypdf import (
    Configuration,
    PdfWriter,
    apply_configuration,
    get_configuration,
)
from pypdf.errors import DependencyError, PdfReadError
from pypdf.generic import DictionaryObject

from doc_checker.config import PdfConfig
from doc_checker.extract.pdf import tags
from doc_checker.extract.pdf.loader import load_pdf
from doc_checker.extract.pdf.tags import (
    MAX_ROLE_STEPS,
    MAX_TAG_DEPTH,
    MAX_TAG_ITEMS,
    TAG_TEXT_LIMIT,
    read_tags,
)
from doc_checker.model.facts import TagTree
from doc_checker.text.normalize import ELLIPSIS
from tests.builders import SAMPLE_FR
from tests.pdf_builders import build_fpdf2_pdf, build_tagged_pdf

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    import pytest

# An entry of the cross-reference table of the PDFs that the tests build.
_XREF_ENTRY = re.compile(rb"\d{10} 00000 n")
# The header of the PDFs that `_tree_pdf` writes, with its binary comment.
_HEADER = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n"
# The content of the page of the PDFs that `_tree_pdf` writes.
_PAGE_TEXT = b"BT /F1 12 Tf 72 700 Td (Hello tagged world) Tj ET"
# The owner password of the encrypted PDFs that the tests write.
_OWNER = "owner"
# What pypdf says when it needs a cryptography package for AES.
_NO_AES = "cryptography>=3.1 is required for AES algorithm"
# The most time that a hostile file may cost, in seconds: each of these files
# took more than 10 s before the reading was bounded.
_QUICK = 5.0


def _misplace_objects(path: Path) -> Path:
    """Point every entry of the cross-reference table at the first object.

    pypdf then finds each other object by a scan of the whole file.
    """
    data = path.read_bytes()
    first = f"{data.index(b'1 0 obj'):010d} 00000 n".encode("ascii")
    path.write_bytes(_XREF_ENTRY.sub(first, data))
    return path


def _stream(entries: bytes, data: bytes) -> bytes:
    """Return a stream object: its dictionary entries, then its data."""
    head = b"<< %s /Length %d >>\nstream\n" % (entries, len(data))
    return head + data + b"\nendstream"


def _tree_pdf(
    out: Path,
    root: bytes,
    objects: Mapping[int, bytes],
    *,
    kids: bytes = b"3 0 R",
    offsets: Mapping[int, int] | None = None,
) -> Path:
    """Write a one page tagged PDF with a cross-reference table.

    Args:
        out: The PDF file to write.
        root: The ``/K`` entry of the root of the structure tree.
        objects: More objects, by number from 7: the page is 3.
        kids: The ``/Kids`` entry of the page tree.
        offsets: Entries of the cross-reference table to write at these
            offsets, by object number, whether or not the object exists.

    Returns:
        The PDF file.
    """
    every = {
        1: (
            b"<< /Type /Catalog /Pages 2 0 R /MarkInfo << /Marked true >>"
            b" /StructTreeRoot 4 0 R >>"
        ),
        2: b"<< /Type /Pages /Kids [%s] /Count 1 >>" % kids,
        3: (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]"
            b" /Resources << /Font << /F1 5 0 R >> >> /Contents 6 0 R >>"
        ),
        4: b"<< /Type /StructTreeRoot /K %s >>" % root,
        5: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        6: _stream(b"", _PAGE_TEXT),
        **objects,
    }
    data = bytearray(_HEADER)
    where: dict[int, int] = {}
    for number in sorted(every):
        where[number] = len(data)
        data += b"%d 0 obj\n%s\nendobj\n" % (number, every[number])
    where.update(offsets or {})
    size = max(where) + 1
    table = len(data)
    data += b"xref\n0 %d\n0000000000 65535 f \n" % size
    for number in range(1, size):
        if number in where:
            data += b"%010d 00000 n \n" % where[number]
        else:
            data += b"0000000000 65535 f \n"
    data += b"trailer\n<< /Size %d /Root 1 0 R >>\n" % size
    data += b"startxref\n%d\n%%%%EOF\n" % table
    out.write_bytes(bytes(data))
    return out


def _object_stream_pdf(out: Path, *, listed: int, missing: int) -> Path:
    """Write a tagged PDF whose tree holds objects that no stream holds.

    The cross-reference stream puts each of `missing` objects in an object
    stream of `listed` objects that does not list it.

    Args:
        out: The PDF file to write.
        listed: How many objects the object stream lists.
        missing: How many objects the root of the tree holds.

    Returns:
        The PDF file.
    """
    holder, empty, table = 7, 8, 9
    first, lost = 1_000, 1_000 + listed
    refs = b" ".join(b"%d 0 R" % (lost + i) for i in range(missing))
    head = b" ".join(b"%d %d" % (first + i, 9 * i) for i in range(listed))
    content = head + b"\n" + b" ".join([b"<</S/P>>"] * listed)
    objects = {
        1: (
            b"<< /Type /Catalog /Pages 2 0 R /MarkInfo << /Marked true >>"
            b" /StructTreeRoot 4 0 R >>"
        ),
        2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        3: (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]"
            b" /Resources << /Font << /F1 5 0 R >> >> /Contents 6 0 R >>"
        ),
        4: b"<< /Type /StructTreeRoot /K [%s] >>" % refs,
        5: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        6: _stream(b"", _PAGE_TEXT),
        holder: _stream(
            b"/Type /ObjStm /N %d /First %d" % (listed, len(head) + 1),
            content,
        ),
        empty: _stream(b"/Type /ObjStm /N 0 /First 1", b"\n"),
    }
    data = bytearray(b"%PDF-1.7\n")
    where: dict[int, int] = {}
    for number in sorted(objects):
        where[number] = len(data)
        data += b"%d 0 obj\n%s\nendobj\n" % (number, objects[number])
    where[table] = len(data)
    size = lost + missing
    rows = []
    for number in range(size):
        if number in where:
            rows.append(struct.pack(">BIH", 1, where[number], 0))
        elif first <= number < lost:
            rows.append(struct.pack(">BIH", 2, empty, 0))
        elif number >= lost:
            rows.append(struct.pack(">BIH", 2, holder, 0))
        else:
            rows.append(struct.pack(">BIH", 0, 0, 0))
    rows_data = b"".join(rows)
    data += b"%d 0 obj\n" % table + _stream(
        b"/Type /XRef /Size %d /W [1 4 2] /Root 1 0 R" % size,
        rows_data,
    )
    data += b"\nendobj\nstartxref\n%d\n%%%%EOF\n" % where[table]
    out.write_bytes(bytes(data))
    return out


def _alts_in_streams_pdf(
    out: Path,
    *,
    figures: int,
    alt: bytes,
    last: bytes | None = None,
    ascii85: bool = False,
) -> Path:
    """Write a tagged PDF whose figures take their /Alt in object streams.

    The /Alt of each figure is an object alone in an object stream of its
    own, compressed with Flate.

    Args:
        out: The PDF file to write.
        figures: How many figures the root of the tree holds.
        alt: The object of each /Alt, such as a string.
        last: The object of the /Alt of the last figure, if another one.
        ascii85: Whether each object stream is also encoded in ASCII85,
            after Flate, so that it inflates to the ASCII85 text first.

    Returns:
        The PDF file.
    """
    alts = range(100, 100 + figures)
    elements = b" ".join(
        b"<< /S /Figure /Pg 3 0 R /Alt %d 0 R >>" % alt for alt in alts
    )
    objects = {
        1: (
            b"<< /Type /Catalog /Pages 2 0 R /MarkInfo << /Marked true >>"
            b" /StructTreeRoot 4 0 R >>"
        ),
        2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        3: (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]"
            b" /Resources << /Font << /F1 5 0 R >> >> /Contents 6 0 R >>"
        ),
        4: b"<< /Type /StructTreeRoot /K [%s] >>" % elements,
        5: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        6: _stream(b"", _PAGE_TEXT),
    }
    holders = {}
    for holder, number in enumerate(alts, 10):
        head = b"%d 0 " % number
        content = head + (alt if last is None or number < alts[-1] else last)
        filters = b"/FlateDecode"
        if ascii85:
            content = base64.a85encode(content, adobe=True)
            filters = b"[/FlateDecode /ASCII85Decode]"
        objects[holder] = _stream(
            b"/Type /ObjStm /N 1 /First %d /Filter %s" % (len(head), filters),
            zlib.compress(content, 9),
        )
        holders[number] = holder
    data = bytearray(_HEADER)
    where: dict[int, int] = {}
    for number in sorted(objects):
        where[number] = len(data)
        data += b"%d 0 obj\n%s\nendobj\n" % (number, objects[number])
    table = max(alts) + 1
    where[table] = len(data)
    rows = bytearray()
    for number in range(table + 1):
        if number in where:
            rows += struct.pack(">BIH", 1, where[number], 0)
        elif number in holders:
            rows += struct.pack(">BIH", 2, holders[number], 0)
        else:
            rows += struct.pack(">BIH", 0, 0, 0)
    data += b"%d 0 obj\n" % table + _stream(
        b"/Type /XRef /Size %d /W [1 4 2] /Root 1 0 R" % (table + 1),
        bytes(rows),
    )
    data += b"\nendobj\nstartxref\n%d\n%%%%EOF\n" % where[table]
    out.write_bytes(bytes(data))
    return out


def _xref_chain_pdf(out: Path, *, sections: int, padding: int) -> Path:
    """Write a tagged PDF whose cross-reference streams follow a /Prev chain.

    The first stream holds the objects of the file; each of the others holds
    only itself, then `padding` zero bytes, compressed with Flate.

    Args:
        out: The PDF file to write.
        sections: How many cross-reference streams the chain holds.
        padding: The zero bytes after the entry of each stream but the
            first.

    Returns:
        The PDF file.
    """
    objects = {
        1: (
            b"<< /Type /Catalog /Pages 2 0 R /MarkInfo << /Marked true >>"
            b" /StructTreeRoot 4 0 R >>"
        ),
        2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        3: (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]"
            b" /Resources << /Font << /F1 5 0 R >> >> /Contents 6 0 R >>"
        ),
        4: b"<< /Type /StructTreeRoot /K [8 0 R] >>",
        5: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        6: _stream(b"", _PAGE_TEXT),
        8: b"<< /S /Figure /Pg 3 0 R >>",
    }
    data = bytearray(_HEADER)
    where: dict[int, int] = {}
    for number in sorted(objects):
        where[number] = len(data)
        data += b"%d 0 obj\n%s\nendobj\n" % (number, objects[number])
    size = 9 + sections
    rows = bytearray(struct.pack(">BIH", 0, 0, 65535))
    for number in range(1, 9):
        used = int(number in where)
        rows += struct.pack(">BIH", used, where.get(number, 0), 0)
    previous = b""
    for number in range(9, 9 + sections):
        where[number] = len(data)
        entry = struct.pack(">BIH", 1, where[number], 0)
        if number == 9:
            body, index = bytes(rows) + entry, b"0 10"
        else:
            body, index = entry + b"\0" * padding, b"%d 1" % number
        data += b"%d 0 obj\n" % number + _stream(
            b"/Type /XRef /Size %d /Index [%s] /W [1 4 2] /Root 1 0 R"
            b" /Filter /FlateDecode%s" % (size, index, previous),
            zlib.compress(body, 9),
        )
        data += b"\nendobj\n"
        previous = b" /Prev %d" % where[number]
    data += b"startxref\n%d\n%%%%EOF\n" % where[9 + sections - 1]
    out.write_bytes(bytes(data))
    return out


def _rc4(path: Path) -> Path:
    """Write a copy of a PDF encrypted with RC4 and an empty user password."""
    writer = PdfWriter(clone_from=path)
    writer.encrypt(
        user_password="",
        owner_password=_OWNER,
        algorithm="RC4-128",
    )
    out = path.with_name(f"rc4-{path.name}")
    writer.write(out)
    return out


def _types(path: Path) -> list[str | None]:
    """Return the types of the elements of the tree of a PDF, in order."""
    return [element.type for element in read_tags(path).elements]


class _Locked:
    """A reader of an encrypted PDF that the empty password does not open."""

    is_encrypted = True

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        """Take the arguments of a reader."""

    @staticmethod
    def decrypt(_password: str) -> int:
        """Fail to decrypt."""
        return 0


def test_tagged_pdf_gives_its_headings_figures_and_tables(
    tmp_path: Path,
) -> None:
    """The elements that the audit checks come in the order of the tree."""
    path = build_tagged_pdf(
        tmp_path / "t.pdf",
        headings=((1, "Guide"), (2, "Install"), (4, "Deep")),
    )
    tree = read_tags(path)
    assert tree.problem is None
    found = [(e.role, e.type, e.page, e.level) for e in tree.elements]
    assert found == [
        ("heading", "H1", 1, 1),
        ("heading", "H2", 1, 2),
        ("heading", "H4", 1, 4),
        ("figure", "Figure", 1, None),
        ("table", "Table", 1, None),
    ]
    figure, table = tree.elements[3], tree.elements[4]
    assert (figure.alt, table.headers) == ("A diagram", True)


def test_missing_alt_and_header_cells_are_told(tmp_path: Path) -> None:
    """A figure without alt and a table of data cells only."""
    path = build_tagged_pdf(
        tmp_path / "t.pdf",
        figure_alt=None,
        table_headers=False,
    )
    roles = {e.role: e for e in read_tags(path).elements}
    assert (roles["figure"].alt, roles["figure"].actual) == (None, None)
    assert roles["table"].headers is False


def test_role_map_leads_to_standard_types(tmp_path: Path) -> None:
    """Custom types reach a standard one through the role map, in steps."""
    path = build_tagged_pdf(
        tmp_path / "t.pdf",
        headings=((2, "Install"),),
        types={"H2": "Title2"},
        role_map={"Title2": "Heading", "Heading": "H2"},
    )
    headings = [e for e in read_tags(path).elements if e.role == "heading"]
    assert [(e.type, e.level) for e in headings] == [("H2", 2)]


def test_role_map_cycle_stops(tmp_path: Path) -> None:
    """A role map that loops gives no standard type, and no hang."""
    path = build_tagged_pdf(
        tmp_path / "t.pdf",
        headings=((2, "Install"),),
        types={"H2": "A"},
        role_map={"A": "B", "B": "A"},
    )
    assert [e.role for e in read_tags(path).elements] == ["figure", "table"]


def test_cyclic_tree_is_walked_once(tmp_path: Path) -> None:
    """An element that holds its own ancestor is not walked again."""
    tree = read_tags(build_tagged_pdf(tmp_path / "t.pdf", cycle=True))
    assert tree.problem is None
    assert [e.type for e in tree.elements].count("H1") == 1


def test_deep_tree_is_read_in_part(tmp_path: Path) -> None:
    """A tree deeper than the limit is cut, and says why."""
    tree = read_tags(build_tagged_pdf(tmp_path / "t.pdf", nesting=150))
    assert tree.problem == (
        f"structure tree deeper than {MAX_TAG_DEPTH} levels, read in part"
    )


def test_wide_tree_is_read_in_part(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tree of more items than the limit is cut, and says why."""
    monkeypatch.setattr(tags, "MAX_TAG_ITEMS", 50)
    tree = read_tags(build_tagged_pdf(tmp_path / "t.pdf", paragraphs=200))
    assert tree.problem == "structure tree of more than 50 items, read in part"
    assert tree.items <= 51


def test_unreadable_tree_is_a_problem(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An error of pypdf gives no element and the reason."""
    message = "bad xref"

    def broken(*_args: object, **_kwargs: object) -> NoReturn:
        raise PdfReadError(message)

    monkeypatch.setattr(tags, "PdfReader", broken)
    tree = read_tags(build_tagged_pdf(tmp_path / "t.pdf"))
    assert (tree.elements, tree.problem) == (
        (),
        "structure tree not read: bad xref",
    )


def test_encrypted_tree_is_a_problem(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A PDF that the empty password does not open has no tree to read."""
    monkeypatch.setattr(tags, "PdfReader", _Locked)
    tree = read_tags(build_tagged_pdf(tmp_path / "t.pdf"))
    assert tree == TagTree((), 0, "structure tree not read: encrypted")


def test_untagged_pdf_has_no_tree(tmp_path: Path) -> None:
    """Without a structure tree, the tree says so."""
    tree = read_tags(build_tagged_pdf(tmp_path / "t.pdf", tagged=False))
    assert (tree.elements, tree.problem) == ((), "no structure tree")


def test_fpdf2_tags_its_section_titles_as_figures(tmp_path: Path) -> None:
    """fpdf2 2.9 writes a figure without alt for each section title."""
    path = build_fpdf2_pdf(SAMPLE_FR, tmp_path / "f.pdf")
    figures = [e for e in read_tags(path).elements if e.role == "figure"]
    assert len(figures) == 5
    assert {(e.alt, e.actual) for e in figures} == {(None, None)}


def test_audit_reads_the_tags_and_warns_of_problems(tmp_path: Path) -> None:
    """The audit gets the tree; compare does not; a cut tree is a warning."""
    path = build_tagged_pdf(tmp_path / "t.pdf")
    assert load_pdf(path, PdfConfig(), lang="auto").tags is None
    doc = load_pdf(path, PdfConfig(), lang="auto", audit=True)
    assert doc.tags is not None
    assert len(doc.tags.elements) == 4
    deep = build_tagged_pdf(tmp_path / "d.pdf", nesting=150)
    warnings = load_pdf(deep, PdfConfig(), lang="auto", audit=True).warnings
    assert warnings[-1] == (
        f"structure tree deeper than {MAX_TAG_DEPTH} levels, read in part;"
        " the checks of the tags may be incomplete"
    )


def test_audit_of_an_untagged_pdf_reads_no_tree(tmp_path: Path) -> None:
    """A PDF that does not declare its tags has no tree and no warning."""
    path = build_tagged_pdf(tmp_path / "t.pdf", tagged=False)
    doc = load_pdf(path, PdfConfig(), lang="auto", audit=True)
    assert (doc.tags, doc.warnings) == (None, [])


def test_large_tree_is_read_quickly(tmp_path: Path) -> None:
    """Twenty thousand elements are read in seconds."""
    path = build_tagged_pdf(tmp_path / "t.pdf", paragraphs=20_000)
    start = time.perf_counter()
    tree = read_tags(path)
    assert tree.problem is None
    assert tree.items > 20_000
    assert time.perf_counter() - start < 10.0


def test_alt_is_printable_and_cut(tmp_path: Path) -> None:
    """A long alt with controls keeps its printable start, then an ellipsis."""
    alt = r"a\007b " + "x" * 50_000
    path = build_tagged_pdf(tmp_path / "t.pdf", figure_alt=alt)
    (figure,) = [e for e in read_tags(path).elements if e.role == "figure"]
    assert figure.alt == ("a b " + "x" * 50_000)[:TAG_TEXT_LIMIT] + ELLIPSIS


def test_lone_surrogate_in_alt_is_replaced(tmp_path: Path) -> None:
    """A UTF-16 alt with half of a pair keeps its other characters."""
    path = build_tagged_pdf(tmp_path / "t.pdf", figure_alt="<FEFFD8000041>")
    (figure,) = [e for e in read_tags(path).elements if e.role == "figure"]
    assert figure.alt == "\U0000fffdA"


def test_utf8_alt_is_decoded(tmp_path: Path) -> None:
    """A PDF 2.0 text string in UTF-8, after its byte order mark."""
    path = build_tagged_pdf(tmp_path / "t.pdf", figure_alt="<EFBBBF43C3A9>")
    (figure,) = [e for e in read_tags(path).elements if e.role == "figure"]
    assert figure.alt == "Cé"


def test_arrays_longer_than_pypdf_allows_are_read(tmp_path: Path) -> None:
    """The size of an array is bounded by the items, not by pypdf alone."""
    path = build_tagged_pdf(tmp_path / "t.pdf", paragraphs=200)
    with apply_configuration(array_maximum_element_count=100):
        tree = read_tags(path)
    assert tree.problem is None
    assert tree.items > 200


def test_misplaced_objects_are_refused(tmp_path: Path) -> None:
    """Strict pypdf does not repair what the table misplaces."""
    path = _misplace_objects(build_tagged_pdf(tmp_path / "t.pdf"))
    tree = read_tags(path)
    assert (tree.elements, tree.problem) == (
        (),
        "structure tree not read: object 9 0 is misplaced",
    )


def test_reading_says_nothing_and_restores_pypdf_logging(
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """A damaged file is refused silently, and pypdf logs as before."""
    logger = logging.getLogger("pypdf")
    before = (logger.level, logger.propagate, list(logger.handlers))
    read_tags(_misplace_objects(build_tagged_pdf(tmp_path / "t.pdf")))
    assert capfd.readouterr() == ("", "")
    assert (logger.level, logger.propagate, list(logger.handlers)) == before


def test_pypdf_settings_come_back_after_an_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The logging, the warnings and the limits of pypdf come back."""
    message = "bad xref"

    def broken(*_args: object, **_kwargs: object) -> NoReturn:
        raise PdfReadError(message)

    monkeypatch.setattr(tags, "PdfReader", broken)
    logger = logging.getLogger("pypdf")

    def settings() -> tuple[object, ...]:
        """Return what the reading changes while it runs."""
        return (
            logger.level,
            logger.propagate,
            list(logger.handlers),
            get_configuration(),
            list(warnings.filters),
        )

    before = settings()
    read_tags(build_tagged_pdf(tmp_path / "t.pdf"))
    assert settings() == before


def test_many_misplaced_objects_are_refused_quickly(tmp_path: Path) -> None:
    """A table that misplaces every object is refused at the first one."""
    path = build_tagged_pdf(tmp_path / "t.pdf", paragraphs=10_000)
    _misplace_objects(path)
    start = time.perf_counter()
    tree = read_tags(path)
    assert time.perf_counter() - start < _QUICK
    assert (tree.elements, tree.problem) == (
        (),
        "structure tree not read: object 9 0 is misplaced",
    )


def test_misplaced_objects_that_are_nowhere_are_refused_quickly(
    tmp_path: Path,
) -> None:
    """References to objects that the table puts at the catalog."""
    count = 3_000
    refs = b" ".join(b"%d 0 R" % (100 + i) for i in range(count))
    path = _tree_pdf(
        tmp_path / "t.pdf",
        b"[%s]" % refs,
        {7: _stream(b"", b"x" * 2_000_000)},
        offsets=dict.fromkeys(range(100, 100 + count), len(_HEADER)),
    )
    start = time.perf_counter()
    tree = read_tags(path)
    assert time.perf_counter() - start < _QUICK
    assert (tree.elements, tree.problem) == (
        (),
        "structure tree not read: object 100 0 is misplaced",
    )


def test_streams_of_wrong_length_are_refused_quickly(tmp_path: Path) -> None:
    """Elements written as streams whose length is wrong are refused."""
    count = 32_000
    refs = b" ".join(b"%d 0 R" % (100 + i) for i in range(count))
    element = b"<< /S /H1 /Pg 3 0 R /Length 0 >>\nstream\nXYZ\nendstream"
    path = _tree_pdf(
        tmp_path / "t.pdf",
        b"[%s]" % refs,
        dict.fromkeys(range(100, 100 + count), element),
    )
    start = time.perf_counter()
    tree = read_tags(path)
    assert time.perf_counter() - start < _QUICK
    assert tree.elements == ()
    assert tree.problem is not None
    assert tree.problem.startswith(
        "structure tree not read: Unable to find 'endstream' marker",
    )


def test_page_tree_of_streams_is_refused_quickly(tmp_path: Path) -> None:
    """A page tree whose kids are streams of a wrong length is refused."""
    count = 24_000
    numbers = range(100, 100 + count)
    kids = b" ".join([b"3 0 R", *(b"%d 0 R" % number for number in numbers)])
    kid = b"<< /Length 0 >>\nstream\nXYZ\nendstream"
    path = _tree_pdf(
        tmp_path / "t.pdf",
        b"<< /S /Figure /Pg 3 0 R >>",
        dict.fromkeys(numbers, kid),
        kids=kids,
    )
    start = time.perf_counter()
    tree = read_tags(path)
    assert time.perf_counter() - start < _QUICK
    assert tree.elements == ()
    assert tree.problem is not None
    assert tree.problem.startswith(
        "structure tree not read: Unable to find 'endstream' marker",
    )


def test_objects_missing_from_their_stream_are_refused_quickly(
    tmp_path: Path,
) -> None:
    """Objects that their object stream does not list are refused."""
    path = _object_stream_pdf(tmp_path / "t.pdf", listed=10_000, missing=250)
    start = time.perf_counter()
    tree = read_tags(path)
    assert time.perf_counter() - start < _QUICK
    assert tree.elements == ()
    assert tree.problem is not None
    assert tree.problem.startswith("structure tree not read: ")


def test_large_file_with_unused_entries_is_read_in_full(
    tmp_path: Path,
) -> None:
    """Unused entries of the table at offset 0 do not stop a sound file."""
    path = _tree_pdf(
        tmp_path / "t.pdf",
        b"[8 0 R]",
        {
            7: _stream(b"", b"x" * 25_000_000),
            8: b"<< /S /Figure /Pg 3 0 R /K 0 >>",
        },
        offsets=dict.fromkeys(range(50, 62), 0),
    )
    tree = read_tags(path)
    assert ([e.type for e in tree.elements], tree.problem) == (
        ["Figure"],
        None,
    )


def test_rc4_encrypted_tree_is_read_in_full(tmp_path: Path) -> None:
    """An encrypted PDF that the empty password opens is read in full."""
    path = _rc4(build_tagged_pdf(tmp_path / "t.pdf"))
    tree = read_tags(path)
    assert tree.problem is None
    assert [e.type for e in tree.elements] == ["H1", "H2", "Figure", "Table"]


def test_file_that_pypdf_cannot_decrypt_is_a_problem(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When pypdf needs a cryptography package to open a file, as for AES."""

    class NoAes:
        """A reader that lacks AES to check the empty password."""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            """Fail to open the file."""
            raise DependencyError(_NO_AES)

    monkeypatch.setattr(tags, "PdfReader", NoAes)
    tree = read_tags(_rc4(build_tagged_pdf(tmp_path / "t.pdf")))
    assert tree == TagTree((), 0, "structure tree not read: encrypted")


def test_encrypted_strings_that_pypdf_cannot_decrypt_are_a_problem(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When pypdf needs a cryptography package for a string, as for AES."""

    class AesStrings:
        """A reader that opens an AES file, but lacks AES for its strings."""

        is_encrypted = True

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            """Open the file."""

        @staticmethod
        def decrypt(_password: str) -> int:
            """Accept the empty password."""
            return 1

        @staticmethod
        def get_object(_reference: object) -> NoReturn:
            """Fail to decrypt an object."""
            raise DependencyError(_NO_AES)

        @property
        def root_object(self) -> NoReturn:
            """Fail to decrypt the catalog."""
            raise DependencyError(_NO_AES)

    monkeypatch.setattr(tags, "PdfReader", AesStrings)
    tree = read_tags(build_tagged_pdf(tmp_path / "t.pdf"))
    assert tree == TagTree((), 0, "structure tree not read: encrypted")


def test_damaged_encrypted_file_is_a_problem(tmp_path: Path) -> None:
    """Any failure on an encrypted file says only that it is encrypted."""
    path = _misplace_objects(_rc4(build_tagged_pdf(tmp_path / "t.pdf")))
    tree = read_tags(path)
    assert tree == TagTree((), 0, "structure tree not read: encrypted")


def test_huge_object_stream_is_not_inflated(tmp_path: Path) -> None:
    """An object stream that inflates past the limit of a stream is refused."""
    path = _alts_in_streams_pdf(
        tmp_path / "t.pdf",
        figures=10,
        alt=b"(" + b"a" * 2_000_000 + b")",
    )
    start = time.perf_counter()
    tree = read_tags(path)
    assert time.perf_counter() - start < _QUICK
    assert (tree.elements, tree.problem) == (
        (),
        (
            "structure tree not read: Limit reached while decompressing."
            " 975 bytes remaining."
        ),
    )


def test_ascii85_stage_is_bounded(tmp_path: Path) -> None:
    """A last ASCII85 filter does not inflate past the limit of a stream."""
    expanding, longer = (
        _alts_in_streams_pdf(
            tmp_path / f"{size}.pdf",
            figures=1,
            alt=b"(" + b"\0" * size + b")",
            ascii85=True,
        )
        for size in (1_200_000, 2_400_000)
    )
    assert read_tags(expanding) == TagTree(
        (),
        2,
        "structure tree not read: stream inflated to more than 1000000 bytes",
    )
    assert read_tags(longer) == TagTree(
        (),
        2,
        "structure tree not read: Stream decoding exceeded the maximum"
        " accumulated decoding work of 2000000 bytes.",
    )


def test_object_streams_inflate_within_a_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Object streams under the limit of a stream are cut by the budget."""
    monkeypatch.setattr(tags, "TAG_INFLATE_WORK", 5_000_000)
    path = _alts_in_streams_pdf(
        tmp_path / "t.pdf",
        figures=20,
        alt=b"(" + b"a" * 900_000 + b")",
    )
    tree = read_tags(path)
    assert tree.problem == (
        "structure tree with streams inflated past a budget of 5000000"
        " bytes, read in part"
    )
    assert [e.alt for e in tree.elements] == [
        "a" * TAG_TEXT_LIMIT + ELLIPSIS,
    ] * 2


def test_dense_object_streams_are_cut_before_they_are_parsed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stream is counted before pypdf parses its objects, the costliest."""
    monkeypatch.setattr(tags, "TAG_INFLATE_WORK", 2_200_000)
    path = _alts_in_streams_pdf(
        tmp_path / "t.pdf",
        figures=3,
        alt=b"[" + b"0 " * 250_000 + b"]",
        last=b"(" + b"a" * 500_000,
    )
    start = time.perf_counter()
    tree = read_tags(path)
    assert time.perf_counter() - start < _QUICK
    assert tree.problem == (
        "structure tree with streams inflated past a budget of 2200000"
        " bytes, read in part"
    )
    assert len(tree.elements) == 2


def test_chain_of_cross_reference_streams_is_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cross-reference streams that pypdf keeps count in the budget."""
    monkeypatch.setattr(tags, "MAX_XREF_STREAM_BYTES", 1_000_000)
    monkeypatch.setattr(tags, "TAG_INFLATE_WORK", 5_000_000)
    path = _xref_chain_pdf(tmp_path / "t.pdf", sections=40, padding=900_000)
    tracemalloc.start()
    start = time.perf_counter()
    try:
        tree = read_tags(path)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert time.perf_counter() - start < _QUICK
    assert peak < 20_000_000
    assert tree == TagTree(
        (),
        0,
        (
            "structure tree not read: streams inflated past a budget of"
            " 5000000 bytes"
        ),
    )


def test_cross_reference_streams_are_counted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A chain of more cross-reference streams than allowed is refused."""
    monkeypatch.setattr(tags, "MAX_XREF_STREAMS", 5)
    path = _xref_chain_pdf(tmp_path / "t.pdf", sections=10, padding=0)
    assert read_tags(path) == TagTree(
        (),
        0,
        "structure tree not read: more than 5 cross-reference streams",
    )
    sound = _xref_chain_pdf(tmp_path / "s.pdf", sections=5, padding=0)
    assert [e.type for e in read_tags(sound).elements] == ["Figure"]


def test_pypdf_limits_while_it_opens_and_walks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """While pypdf reads, it inflates little and runs no outer decoder."""
    seen: list[Configuration] = []

    class Recorder:
        """A reader that keeps the configuration of pypdf as it is used."""

        is_encrypted = False

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            """Keep the configuration while the file opens."""
            seen.append(get_configuration())

        @property
        def root_object(self) -> DictionaryObject:
            """Keep the configuration while the tree is walked."""
            seen.append(get_configuration())
            return DictionaryObject()

    monkeypatch.setattr(tags, "PdfReader", Recorder)
    tree = read_tags(build_tagged_pdf(tmp_path / "t.pdf"))
    assert tree.problem == "no structure tree"
    opening, walking = seen
    for configuration, cap in ((opening, 16_000_000), (walking, 1_000_000)):
        assert configuration.zlib_maximum_output_length == cap
        assert configuration.lzw_maximum_output_length == cap
        assert configuration.run_length_maximum_output_length == cap
        assert configuration.brotli_maximum_output_length == cap
        assert configuration.stream_decoding_work_maximum_length == 2 * cap
        assert configuration.jbig2dec_binary is None
        assert configuration.array_maximum_element_count >= MAX_TAG_ITEMS


def test_odd_types_of_elements_are_ignored(tmp_path: Path) -> None:
    """A /Type that is an array or a dictionary leaves the element read."""
    path = _tree_pdf(
        tmp_path / "t.pdf",
        b"[7 0 R 8 0 R]",
        {
            7: b"<< /Type [/StructElem] /S /H1 /Pg 3 0 R /K 0 >>",
            8: b"<< /Type << >> /S /H2 /Pg 3 0 R /K 1 >>",
        },
    )
    tree = read_tags(path)
    assert ([e.type for e in tree.elements], tree.problem) == (
        ["H1", "H2"],
        None,
    )


def test_children_are_not_read_ahead_for_a_page(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The page of an element is not sought in children the walk leaves."""
    monkeypatch.setattr(tags, "MAX_TAG_ITEMS", 100)
    big = b"[" + b"[]" * 50_000 + b"]"
    refs = b" ".join(b"%d 0 R" % (10 + i) for i in range(10))
    path = _tree_pdf(
        tmp_path / "t.pdf",
        b"<< /S /Div /K [%s] >>" % refs,
        dict.fromkeys(range(10, 20), big),
    )
    tracemalloc.start()
    try:
        tree = read_tags(path)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert tree.items == 100
    assert peak < 20_000_000


def test_heading_numbers_start_at_one_without_leading_zeros(
    tmp_path: Path,
) -> None:
    """H0 and H007 are not standard headings."""
    path = build_tagged_pdf(
        tmp_path / "t.pdf",
        headings=((0, "Zero"), (7, "Seven")),
        types={"H7": "H007"},
    )
    assert [e for e in read_tags(path).elements if e.role == "heading"] == []


def test_loading_a_pdf_does_not_import_pypdf() -> None:
    """Only the audit needs pypdf: the loaders of compare do not import it."""
    code = (
        "import sys, doc_checker.extract.loader,"
        " doc_checker.extract.pdf.loader; print('pypdf' in sys.modules)"
    )
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] - a fixed command of this interpreter
        [sys.executable, "-c", code],
        capture_output=True,
        check=True,
        encoding="utf-8",
    )
    assert result.stdout.strip() == "False"


def test_page_comes_from_the_first_marked_content_child(
    tmp_path: Path,
) -> None:
    """Without /Pg, a reference to marked content gives the page."""
    path = _tree_pdf(
        tmp_path / "t.pdf",
        b"7 0 R",
        {
            7: b"<< /S /Document /K [8 0 R 9 0 R] >>",
            8: b"<< /S /H1 /K [<< /Type /MCR /Pg 3 0 R /MCID 0 >>] >>",
            9: b"<< /S /H2 /K 1 >>",
        },
    )
    found = [(e.type, e.page) for e in read_tags(path).elements]
    assert found == [("H1", 1), ("H2", None)]


def test_table_takes_its_page_from_a_cell(tmp_path: Path) -> None:
    """A table without a page takes the page of an element in it."""
    path = _tree_pdf(
        tmp_path / "t.pdf",
        b"7 0 R",
        {
            7: b"<< /S /Table /K [8 0 R] >>",
            8: b"<< /S /TR /K [9 0 R] >>",
            9: b"<< /S /TH /Pg 3 0 R /K 0 >>",
        },
    )
    (table,) = read_tags(path).elements
    assert (table.type, table.page, table.headers) == ("Table", 1, True)


def test_plain_heading_and_actual_text(tmp_path: Path) -> None:
    """A heading H has no level; a figure may have an /ActualText only."""
    path = _tree_pdf(
        tmp_path / "t.pdf",
        b"[7 0 R 8 0 R]",
        {
            7: b"<< /S /H /Pg 3 0 R /K 0 >>",
            8: b"<< /S /Figure /Pg 3 0 R /ActualText (Chart of sales) /K 1 >>",
        },
    )
    heading, figure = read_tags(path).elements
    assert (heading.role, heading.type, heading.level) == (
        "heading",
        "H",
        None,
    )
    assert (figure.alt, figure.actual) == (None, "Chart of sales")


def test_role_map_is_followed_for_a_bounded_number_of_steps(
    tmp_path: Path,
) -> None:
    """A type reaches a standard type in MAX_ROLE_STEPS steps, not more."""

    def chain(steps: int) -> dict[str, str]:
        """Map R0 to R1 and so on, the last one to H2, in so many steps."""
        mapped = {f"R{i}": f"R{i + 1}" for i in range(steps - 1)}
        return mapped | {f"R{steps - 1}": "H2"}

    near, far = (
        build_tagged_pdf(
            tmp_path / f"{steps}.pdf",
            headings=((2, "Install"),),
            types={"H2": "R0"},
            role_map=chain(steps),
        )
        for steps in (MAX_ROLE_STEPS, MAX_ROLE_STEPS + 1)
    )
    assert "H2" in _types(near)
    assert "H2" not in _types(far)
