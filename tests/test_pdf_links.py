# Copyright 2026 Christophe Le Douarec
"""Tests for the link annotations of PDF pages."""

import sys
import time
from contextlib import closing
from typing import TYPE_CHECKING

import pypdfium2.raw as pdfium_c
import pytest

from doc_checker.config import PdfConfig
from doc_checker.extract.pdf import loader, reader
from doc_checker.extract.pdf.loader import load_pdf
from doc_checker.extract.pdf.reader import (
    file_link_target,
    open_pdf,
    pdf_file_path,
    read_page,
)
from doc_checker.model.facts import OutlineEntry
from doc_checker.text.normalize import ELLIPSIS
from tests.builders import SAMPLE_EN
from tests.pdf_builders import (
    build_linked_pdf,
    build_many_links_pdf,
    build_reportlab_pdf,
    build_scanned_pdf,
    build_shared_link_pdf,
    build_single_page_pdf,
)

if TYPE_CHECKING:
    from pathlib import Path

# PDFium gives the path of a file with colons for slashes on macOS, where a
# path that holds a colon is therefore never checked.
_NOT_ON_MACOS = pytest.mark.skipif(
    sys.platform == "darwin",
    reason="PDFium turns the slashes of a path into colons on macOS",
)


def test_link_annotations_tell_their_targets(tmp_path: Path) -> None:
    """Each kind of link is read with its text, its target and its check."""
    (tmp_path / "manual.pdf").write_bytes(b"%PDF-1.4\n")
    path = build_linked_pdf(
        tmp_path / "links.pdf",
        [
            ("the limits", "page:2"),
            ("the target", "name:target"),
            ("the lost one", "name:missing"),
            ("the site", "uri:https://example.org/cli"),
            ("the manual", "launch:manual.pdf"),
            ("the gone one", "launch:gone.pdf"),
            ("the remote", "gotor:gone.pdf"),
            ("the share", "launch://host/share/a.pdf"),
            ("nothing", "none"),
        ],
    )
    with open_pdf(path) as pdf:
        page = read_page(pdf, 0, link_folder=tmp_path)
    found = [(k.text, k.action, k.target, k.valid) for k in page.links]
    assert found == [
        ("the limits", "goto", "page 2", True),
        ("the target", "goto", "page 2", True),
        ("the lost one", "none", "", False),
        ("the site", "uri", "https://example.org/cli", None),
        ("the manual", "file", "manual.pdf", True),
        ("the gone one", "file", "gone.pdf", False),
        ("the remote", "file", "gone.pdf", False),
        ("the share", "file", "//host/share/a.pdf", None),
        ("nothing", "none", "", False),
    ]
    assert not page.more_links


@pytest.mark.parametrize(
    ("path", "valid"),
    [
        pytest.param("file:manual.pdf", None, marks=_NOT_ON_MACOS),
        pytest.param("C:/manual.pdf", None, marks=_NOT_ON_MACOS),
        pytest.param("C:manual.pdf", None, marks=_NOT_ON_MACOS),
        ("/manual.pdf", None),
        pytest.param("mailto:someone@example.org", None, marks=_NOT_ON_MACOS),
        ("../manual.pdf", None),
        ("sub/../../manual.pdf", None),
        pytest.param("sub/../manual.pdf", True, marks=_NOT_ON_MACOS),
    ],
)
def test_only_paths_inside_the_folder_are_checked(
    tmp_path: Path,
    path: str,
    valid: object,
) -> None:
    """A path with a scheme, a drive, a root, or out of the folder, is not."""
    folder = tmp_path / "docs"
    (folder / "sub").mkdir(parents=True)
    for place in (tmp_path, folder):
        (place / "manual.pdf").write_bytes(b"%PDF-1.4\n")
    linked = build_linked_pdf(folder / "a.pdf", [("one", f"launch:{path}")])
    with open_pdf(linked) as pdf:
        page = read_page(pdf, 0, link_folder=folder)
    assert [(k.action, k.target, k.valid) for k in page.links] == [
        ("file", path, valid),
    ]


@pytest.mark.parametrize(
    ("raw", "platform", "path"),
    [
        ("\\host\\share\\a.pdf", "win32", "//host/share/a.pdf"),
        ("C:\\manual.pdf", "win32", "C:/manual.pdf"),
        ("\\\\manual.pdf", "win32", "/manual.pdf"),
        ("docs\\manual.pdf", "win32", "docs/manual.pdf"),
        ("C:manual.pdf", "win32", "C:manual.pdf"),
        ("::host:share:a.pdf", "darwin", "//host/share/a.pdf"),
        ("C::manual.pdf", "darwin", "C//manual.pdf"),
        (":manual.pdf", "darwin", "/manual.pdf"),
        ("docs:manual.pdf", "darwin", "docs/manual.pdf"),
        ("//host/share/a.pdf", "linux", "//host/share/a.pdf"),
        ("C:\\x:y.pdf", "linux", "C:\\x:y.pdf"),
        ("docs/manual.pdf", "linux", "docs/manual.pdf"),
    ],
)
def test_file_paths_lose_the_form_of_the_platform(
    raw: str,
    platform: str,
    path: str,
) -> None:
    """The slashes that PDFium turns into its platform's form come back."""
    assert pdf_file_path(raw, platform) == path


@pytest.mark.parametrize(
    ("raw", "platform", "expected"),
    [
        ("docs\\manual.pdf", "win32", ("docs/manual.pdf", True)),
        ("docs:manual.pdf", "darwin", ("docs/manual.pdf", None)),
        ("manual.pdf", "darwin", ("manual.pdf", True)),
        ("docs/gone.pdf", "linux", ("docs/gone.pdf", False)),
        ("docs/a\x1bb.pdf", "linux", ("docs/a b.pdf", False)),
    ],
)
def test_file_targets_are_checked_in_the_form_of_their_platform(
    tmp_path: Path,
    raw: str,
    platform: str,
    expected: tuple[str, bool | None],
) -> None:
    """On macOS a path with a colon is never checked, as a URL may hold one."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "manual.pdf").write_bytes(b"%PDF-1.4\n")
    (tmp_path / "manual.pdf").write_bytes(b"%PDF-1.4\n")
    assert file_link_target(raw, tmp_path, platform) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("..foo.pdf", ("..foo.pdf", True)),
        ("docs/..b.pdf", ("docs/..b.pdf", True)),
        ("docs/..gone.pdf", ("docs/..gone.pdf", False)),
        ("..", ("..", None)),
        ("../x.pdf", ("../x.pdf", None)),
        ("..\\x.pdf", ("..\\x.pdf", None)),
        ("docs/../../x.pdf", ("docs/../../x.pdf", None)),
    ],
)
def test_a_name_that_starts_with_two_dots_stays_in_the_folder(
    tmp_path: Path,
    raw: str,
    expected: tuple[str, bool | None],
) -> None:
    """Only a component that is ``..`` itself may lead out of the folder."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "..b.pdf").write_bytes(b"%PDF-1.4\n")
    (tmp_path / "..foo.pdf").write_bytes(b"%PDF-1.4\n")
    assert file_link_target(raw, tmp_path, "linux") == expected


def test_link_targets_are_printable(tmp_path: Path) -> None:
    """A control character of a URI or a path comes out as a space."""
    path = build_linked_pdf(
        tmp_path / "a.pdf",
        [
            ("the site", "uri:https://a\\033b.example"),
            ("the file", "launch:a\\033b.pdf"),
        ],
    )
    with open_pdf(path) as pdf:
        page = read_page(pdf, 0, link_folder=tmp_path)
    assert [(k.action, k.target, k.valid) for k in page.links] == [
        ("uri", "https://a b.example", None),
        ("file", "a b.pdf", False),
    ]


def test_link_texts_stop_at_the_work_of_their_page(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Past the work of a page, links keep their target but not their text."""
    path = build_many_links_pdf(tmp_path / "m.pdf", pages=1, per_page=10)
    with (
        open_pdf(path) as pdf,
        closing(pdf[0]) as page,
        closing(page.get_textpage()) as textpage,
    ):
        chars = textpage.count_chars()
    monkeypatch.setattr(reader, "LINK_TEXT_WORK", 4 * chars)
    with open_pdf(path) as pdf:
        read = read_page(pdf, 0, link_folder=tmp_path)
    assert read.untexted_links == 6
    doc = load_pdf(path, PdfConfig(), lang="auto", audit=True)
    links = doc.pdf_pages[0].links
    assert [link.text for link in links] == [
        *(f"Link {n}" for n in range(1, 5)),
        *[""] * 6,
    ]
    assert [link.target for link in links] == [
        f"https://example.org/{n}" for n in range(1, 11)
    ]
    assert "1 page(s) where the text of 6 links was not read: 1" in (
        doc.warnings
    )


def test_a_shared_annotation_is_resolved_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every reference to one annotation shares one target, read once."""
    calls: list[object] = []
    get_action = pdfium_c.FPDFLink_GetAction

    def counted(link: object) -> object:
        """Count a call, then make it."""
        calls.append(link)
        return get_action(link)

    monkeypatch.setattr(pdfium_c, "FPDFLink_GetAction", counted)
    path = build_shared_link_pdf(
        tmp_path / "s.pdf",
        pages=2,
        references=50,
        action="uri:https://example.org/shared",
    )
    doc = load_pdf(path, PdfConfig(), lang="auto", audit=True)
    links = [link for page in doc.pdf_pages for link in page.links]
    assert [link.text for link in links] == ["Shared link"] * 100
    assert {link.target for link in links} == {"https://example.org/shared"}
    assert len(calls) == 1
    assert len({id(link.target) for link in links}) == 1


def test_other_actions_are_neither_checked_nor_pages(tmp_path: Path) -> None:
    """A GoToE or a Named action is another action, not a page."""
    path = build_linked_pdf(
        tmp_path / "a.pdf",
        [("the attachment", "gotoe:a.pdf"), ("the next", "named:NextPage")],
    )
    with open_pdf(path) as pdf:
        page = read_page(pdf, 0, link_folder=tmp_path)
    assert [(k.text, k.action, k.target, k.valid) for k in page.links] == [
        ("the attachment", "other", "", None),
        ("the next", "other", "", None),
    ]


def test_page_out_of_the_document_is_invalid(tmp_path: Path) -> None:
    """A destination to a page number the document lacks is not valid."""
    path = build_linked_pdf(tmp_path / "a.pdf", [("far", "page:9")])
    with open_pdf(path) as pdf:
        page = read_page(pdf, 0, link_folder=tmp_path)
    assert [(k.action, k.target, k.valid) for k in page.links] == [
        ("goto", "page 9", False),
    ]


def test_too_long_a_target_is_not_read(tmp_path: Path) -> None:
    """A URI longer than the limit is left unread, its target empty."""
    uri = "https://example.org/" + "a" * reader.LINK_TARGET_LIMIT
    path = build_linked_pdf(tmp_path / "a.pdf", [("long", f"uri:{uri}")])
    with open_pdf(path) as pdf:
        page = read_page(pdf, 0, link_folder=tmp_path)
    assert [(k.text, k.action, k.target, k.valid) for k in page.links] == [
        ("long", "uri", "", None),
    ]


def test_links_are_not_read_without_a_folder(tmp_path: Path) -> None:
    """A page read without a link folder has no links."""
    path = build_linked_pdf(tmp_path / "a.pdf", [("the limits", "page:2")])
    with open_pdf(path) as pdf:
        page = read_page(pdf, 0)
    assert (page.links, page.more_links) == ((), False)


def test_links_are_read_only_for_the_audit(tmp_path: Path) -> None:
    """Page facts come with the audit; the title and outline always."""
    path = build_linked_pdf(tmp_path / "a.pdf", [("the limits", "page:2")])
    plain = load_pdf(path, PdfConfig(), lang="auto")
    assert plain.pdf_pages == ()
    audited = load_pdf(path, PdfConfig(), lang="auto", audit=True)
    assert [page.number for page in audited.pdf_pages] == [1, 2]
    assert [len(page.links) for page in audited.pdf_pages] == [1, 0]


def test_title_and_outline_are_kept(tmp_path: Path) -> None:
    """The metadata title and each bookmark with its page, from 1."""
    path = build_reportlab_pdf(SAMPLE_EN, tmp_path / "a.pdf")
    doc = load_pdf(path, PdfConfig(), lang="auto")
    assert doc.title == "Example"
    assert doc.outline[0] == OutlineEntry(0, "Getting started", 1)
    assert doc.outline[1] == OutlineEntry(1, "Installation", 2)


def test_page_facts_count_unmapped_and_scanned_pages(tmp_path: Path) -> None:
    """Characters without mapping and scanned pages are told per page."""
    unmapped = build_single_page_pdf(
        tmp_path / "u.pdf",
        "BT /F1 12 Tf 20 100 Td (ABC) Tj ET",
        font=(
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
            "/Encoding << /Differences [66 /nosuchglyph] >> >>"
        ),
    )
    doc = load_pdf(unmapped, PdfConfig(), lang="auto", audit=True)
    assert (doc.pdf_pages[0].chars, doc.pdf_pages[0].unmapped) == (3, 1)
    assert not doc.pdf_pages[0].scanned
    scanned = build_scanned_pdf(tmp_path / "s.pdf")
    pages = load_pdf(scanned, PdfConfig(), lang="auto", audit=True).pdf_pages
    assert [page.scanned for page in pages] == [True, True]


def test_links_beyond_the_page_limit_are_a_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A page of more links than the limit keeps the first ones, and says."""
    monkeypatch.setattr(reader, "MAX_PAGE_LINKS", 10)
    path = build_many_links_pdf(tmp_path / "m.pdf", pages=1, per_page=40)
    doc = load_pdf(path, PdfConfig(), lang="auto", audit=True)
    assert len(doc.pdf_pages[0].links) == 10
    assert (
        "1 page(s) with more than 10 links, the others were not read: 1"
    ) in doc.warnings


def test_link_limits_of_many_pages_are_one_warning_each(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pages past a link limit are listed in one warning per limit."""
    monkeypatch.setattr(reader, "MAX_PAGE_LINKS", 10)
    path = build_many_links_pdf(tmp_path / "m.pdf", pages=12, per_page=20)
    with (
        open_pdf(path) as pdf,
        closing(pdf[0]) as page,
        closing(page.get_textpage()) as textpage,
    ):
        chars = textpage.count_chars()
    monkeypatch.setattr(reader, "LINK_TEXT_WORK", 4 * chars)
    doc = load_pdf(path, PdfConfig(), lang="auto", audit=True)
    untexted = sum(
        not link.text for page in doc.pdf_pages for link in page.links
    )
    listed = f"1, 2, 3, 4, 5, 6, 7, 8, 9, 10, {ELLIPSIS}"
    cut = "12 page(s) with more than 10 links, the others were not read"
    unread = f"12 page(s) where the text of {untexted} links was not read"
    assert [w for w in doc.warnings if "links" in w] == [
        f"{cut}: {listed}",
        f"{unread}: {listed}",
    ]


def test_links_beyond_the_document_limit_are_one_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Past the document limit no link is read, and one warning says so."""
    monkeypatch.setattr(loader, "MAX_DOCUMENT_LINKS", 25)
    path = build_many_links_pdf(tmp_path / "m.pdf", pages=4, per_page=10)
    doc = load_pdf(path, PdfConfig(), lang="auto", audit=True)
    assert [len(page.links) for page in doc.pdf_pages] == [10, 10, 5, 0]
    assert [link.target for link in doc.pdf_pages[2].links] == [
        f"https://example.org/{n}" for n in range(21, 26)
    ]
    assert [w for w in doc.warnings if "links" in w] == [
        "more than 25 links in the document, the others were not read",
    ]


def test_document_limit_met_at_a_page_end_warns_if_links_follow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A limit met at the end of a page warns only if more links follow."""
    monkeypatch.setattr(loader, "MAX_DOCUMENT_LINKS", 20)
    exact = build_many_links_pdf(tmp_path / "e.pdf", pages=2, per_page=10)
    doc = load_pdf(exact, PdfConfig(), lang="auto", audit=True)
    assert [len(page.links) for page in doc.pdf_pages] == [10, 10]
    assert not [w for w in doc.warnings if "links" in w]
    beyond = build_many_links_pdf(tmp_path / "b.pdf", pages=3, per_page=10)
    doc = load_pdf(beyond, PdfConfig(), lang="auto", audit=True)
    assert [len(page.links) for page in doc.pdf_pages] == [10, 10, 0]
    assert [w for w in doc.warnings if "links" in w] == [
        "more than 20 links in the document, the others were not read",
    ]


def test_many_links_are_read_quickly(tmp_path: Path) -> None:
    """Twelve thousand link annotations add little to the extraction."""
    path = build_many_links_pdf(tmp_path / "m.pdf", pages=300, per_page=40)
    start = time.perf_counter()
    doc = load_pdf(path, PdfConfig(), lang="auto", audit=True)
    elapsed = time.perf_counter() - start
    assert sum(len(page.links) for page in doc.pdf_pages) == 12_000
    assert elapsed < 30.0
