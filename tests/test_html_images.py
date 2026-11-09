# Copyright 2026 Christophe Le Douarec
"""Tests for reading the image sources of HTML pages safely."""

import base64
import socket
import sys
import urllib.parse
from typing import TYPE_CHECKING

import pytest

from doc_checker.extract.html import images as html_images
from doc_checker.extract.html.images import ImageResolver, RawImage
from doc_checker.images.pixels import CANNOT_DECODE
from doc_checker.images.refs import content_digest
from tests.image_builders import drawing, encode

if TYPE_CHECKING:
    from pathlib import Path

    from doc_checker.model.document import ImageRef

PNG = encode(drawing("source"))
PNG_DATA = base64.b64encode(PNG).decode()
SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="300" height="200">'
    "<text>Data drawing</text></svg>"
)


@pytest.fixture
def site(tmp_path: Path) -> Path:
    """Write a site with an image, a directory and a file outside it."""
    root = tmp_path / "site"
    (root / "img" / "folder.png").mkdir(parents=True)
    (root / "img" / "ok.png").write_bytes(PNG)
    (tmp_path / "outside.png").write_bytes(PNG)
    return root


def _read(root: Path, src: str | None) -> ImageRef:
    """Read an ``img`` source of the index page of a site."""
    raw = RawImage(src=src, alt=None, shown=None, svg=None)
    return ImageResolver(root, "")(raw, "index.html")


@pytest.mark.parametrize(
    ("src", "problem"),
    [
        ("img/ok.png", None),
        ("../outside.png", "outside the site"),
        ("/../outside.png", "outside the site"),
        ("%2e%2e/outside.png", "outside the site"),
        ("img/folder.png", "not found"),
        ("img/gone.png", "not found"),
        ("img/a%00b.png", "not found"),
        ("#figure", "not found"),
        ("?version=2", "not found"),
        ("", "not found"),
        (None, "not found"),
        ("https://example.org/img/ok.png", "external"),
        ("//example.org/img/ok.png", "external"),
        ("data:image/png;base64,!!!", "cannot decode"),
        (f"data:image/png;base64,{PNG_DATA}", None),
        (f" da\tta:image/png;base64,{PNG_DATA[:50]}\n{PNG_DATA[50:]}", None),
        (f"data:image/svg+xml,{urllib.parse.quote(SVG)}", None),
    ],
)
def test_sources_are_read_only_in_the_site(
    site: Path,
    src: str | None,
    problem: str | None,
) -> None:
    """Each kind of source is read, or refused with its reason."""
    assert _read(site, src).problem == problem


def test_symbolic_link_out_of_the_site_is_not_followed(site: Path) -> None:
    """A link to a file outside the site is refused once resolved."""
    try:
        (site / "link.png").symlink_to(site.parent / "outside.png")
    except OSError:
        pytest.skip("symbolic links are not allowed here")
    image = _read(site, "link.png")
    assert (image.name, image.problem) == ("link.png", "outside the site")


def test_file_over_the_size_limit_is_not_read(
    site: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A file larger than the limit is too large, its pixels unread."""
    monkeypatch.setattr(html_images, "MAX_FILE_BYTES", len(PNG) - 1)
    image = _read(site, "img/ok.png")
    assert (image.problem, image.readable) == ("too large", False)


def test_file_over_the_size_limit_keeps_its_size(
    site: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A file too large to be read keeps the size of its file."""
    monkeypatch.setattr(html_images, "MAX_FILE_BYTES", len(PNG) - 1)
    image = _read(site, "img/ok.png")
    assert (image.problem, image.size) == ("too large", len(PNG))
    assert _read(site, "img/gone.png").size is None


def test_external_image_is_never_fetched(
    site: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An external image keeps its URL and touches neither network nor site."""

    def refuse(*_: object, **__: object) -> None:
        msg = "network used"
        raise AssertionError(msg)

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    image = _read(site, "HTTPS://Example.org/img/ok.png")
    assert (image.name, image.problem, image.readable) == (
        "//example.org/img/ok.png",
        "external",
        False,
    )
    assert image.digest == content_digest(image.name.encode())


def test_data_url_over_the_limit_is_not_decoded(
    site: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A long data URL is too large and digested by its own text."""
    monkeypatch.setattr(html_images, "DATA_URL_MAX_CHARS", 100)
    src = f"data:image/png;base64,{PNG_DATA}"
    image = _read(site, src)
    assert (image.problem, image.readable) == ("too large", False)
    assert image.digest == content_digest(src.encode())


@pytest.mark.parametrize(
    ("src", "size"),
    [
        (f"data:image/png;base64,{PNG_DATA}", len(PNG_DATA) * 3 // 4),
        (f"  DATA:image/png;BASE64,{PNG_DATA}", len(PNG_DATA) * 3 // 4),
        ("data:image/svg+xml," + "x" * 200, 200),
        ("img/" + "x" * 200 + ".png", None),
    ],
)
def test_data_url_over_the_limit_has_an_estimated_size(
    site: Path,
    monkeypatch: pytest.MonkeyPatch,
    src: str,
    size: int | None,
) -> None:
    """A long data URL is as heavy as its payload, decoded or not."""
    monkeypatch.setattr(html_images, "DATA_URL_MAX_CHARS", 100)
    image = _read(site, src)
    assert (image.problem, image.size) == ("too large", size)


# PostScript that Pillow would hand to Ghostscript, which prints a line.
EPS = (
    b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 10 10\n%%EndComments\n"
    b"newpath 0 0 moveto 10 10 lineto stroke\n"
    b"(run by ghostscript) print flush\n%%EOF\n"
)
# The audit events of a program started by the process.
SPAWN_EVENTS = ("subprocess.", "os.exec", "os.posix_spawn", "os.spawn")


def test_postscript_is_never_run(site: Path) -> None:
    """PostScript behind a raster name or a data URL runs no program."""
    (site / "img" / "figure.png").write_bytes(EPS)
    data_url = f"data:image/png;base64,{base64.b64encode(EPS).decode()}"
    spawned: list[str] = []
    watching = [True]

    def watch(event: str, _args: tuple[object, ...]) -> None:
        if watching[0] and event.startswith(SPAWN_EVENTS):
            spawned.append(event)

    sys.addaudithook(watch)
    try:
        images = [_read(site, "img/figure.png"), _read(site, data_url)]
    finally:
        watching[0] = False
    assert [image.problem for image in images] == [CANNOT_DECODE] * 2
    assert spawned == []


def test_unreadable_images_without_a_name_keep_apart(site: Path) -> None:
    """Two broken data URLs are two images, not one of an empty name."""
    first = _read(site, "data:image/png;base64,!!1")
    second = _read(site, "data:image/png;base64,!!2")
    assert (first.name, second.name) == ("", "")
    assert len({first.digest, second.digest, content_digest(b"")}) == 3
