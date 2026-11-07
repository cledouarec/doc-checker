# Copyright 2026 Christophe Le Douarec
"""Tests for the identity of SVG images."""

import time
import tracemalloc
from typing import TYPE_CHECKING

import lxml.html
import pytest

from doc_checker.images.pixels import UnreadableImage
from doc_checker.images.svg import (
    SVG_MAX_BYTES,
    is_icon,
    parse_svg,
    svg_identity,
)

if TYPE_CHECKING:
    from pathlib import Path

SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 200">'
    '<rect x="10" y="10" width="80" height="40" fill="#cde"/>'
    '<text x="20" y="35">Client</text>'
    '<path d="M90 30 L150 30" stroke="black"/>'
    '<text x="160" y="35">Server <tspan>side</tspan></text></svg>'
)
SAME_SVG = (
    '<?xml version="1.0" encoding="UTF-8"?>\n<!-- drawn by hand -->\n'
    '<svg   viewBox="0 0 300 200" xmlns="http://www.w3.org/2000/svg">\n'
    '  <rect fill="#cde" height="40" width="80" y="10" x="10"/>\n'
    '  <text y="35" x="20">Client</text>\n'
    '  <path stroke="black" d="M90 30 L150 30"/>\n'
    '  <text x="160" y="35">Server   <tspan>side</tspan></text>\n</svg>\n'
)


def test_inline_and_file_svg_are_one_image() -> None:
    """An SVG file and the same SVG written in a page are one image."""
    page = lxml.html.document_fromstring(f"<html><body>{SVG}</body></html>")
    inline = svg_identity(page.find(".//svg"))
    assert parse_svg(SVG.encode()) == inline
    assert inline.texts == ("Client", "Server side")
    assert (inline.width, inline.height) == (300.0, 200.0)


def test_identity_ignores_the_layout_of_the_source() -> None:
    """Declaration, comments, blanks and attribute order do not count."""
    assert parse_svg(SAME_SVG.encode()) == parse_svg(SVG.encode())


def test_another_drawing_has_another_identity() -> None:
    """A changed shape changes the identity, not the texts."""
    other = parse_svg(SVG.replace('width="80"', 'width="90"').encode())
    assert other.digest != parse_svg(SVG.encode()).digest
    assert other.texts == ("Client", "Server side")


def test_scripts_and_handlers_are_left_out() -> None:
    """A script or an event handler is not part of the drawing."""
    hostile = SVG.replace(
        "<rect",
        '<script>alert(1)</script><rect onclick="alert(2)"',
        1,
    )
    assert parse_svg(hostile.encode()) == parse_svg(SVG.encode())


def test_external_entities_are_not_expanded(tmp_path: Path) -> None:
    """An external entity is never read."""
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP-SECRET", encoding="utf-8")
    data = (
        '<?xml version="1.0"?><!DOCTYPE svg ['
        f'<!ENTITY x SYSTEM "{secret.as_uri()}">]>'
        '<svg xmlns="http://www.w3.org/2000/svg"><text>&x;</text></svg>'
    ).encode()
    try:
        identity = parse_svg(data)
    except UnreadableImage:
        return
    assert "TOP-SECRET" not in " ".join(identity.texts)


def _entity_bomb(depth: int = 9) -> bytes:
    """Return an SVG whose nested entities would expand ten times a level."""
    entities = ['<!ENTITY e0 "abcdefghij">']
    for level in range(1, depth):
        body = f"&e{level - 1};" * 10
        entities.append(f'<!ENTITY e{level} "{body}">')
    return (
        f'<?xml version="1.0"?><!DOCTYPE svg [{"".join(entities)}]>'
        '<svg xmlns="http://www.w3.org/2000/svg">'
        f"<text>&e{depth - 1};</text></svg>"
    ).encode()


def test_entity_bomb_is_not_expanded() -> None:
    """Nested entities are neither expanded nor slow."""
    start = time.perf_counter()
    try:
        texts = parse_svg(_entity_bomb()).texts
    except UnreadableImage:
        texts = ()
    assert time.perf_counter() - start < 1.0
    assert len(" ".join(texts)) < 1000


def test_large_or_broken_svg_is_unreadable() -> None:
    """Too large, broken or not an SVG: the image cannot be read."""
    with pytest.raises(UnreadableImage, match="too large"):
        parse_svg(b"<svg>" + b" " * SVG_MAX_BYTES + b"</svg>")
    with pytest.raises(UnreadableImage, match="cannot decode"):
        parse_svg(b"<svg><rect></svg")
    with pytest.raises(UnreadableImage, match="cannot decode"):
        parse_svg(b"<html/>")


@pytest.mark.parametrize(
    ("markup", "icon"),
    [
        ('<svg viewBox="0 0 24 24"><path d="M0 0"/></svg>', True),
        ('<svg width="16" height="16px"><path d="M0 0"/></svg>', True),
        ('<span class="twemoji"><svg viewBox="0 0 300 300"/></span>', True),
        ('<svg viewBox="0 0 300 200"><path d="M0 0"/></svg>', False),
        ('<svg width="400" height="30"><path d="M0 0"/></svg>', False),
    ],
)
def test_icons_are_told_apart(markup: str, *, icon: bool) -> None:
    """Small or emoji SVG are icons, figures are not."""
    root = lxml.html.fragment_fromstring(f"<div>{markup}</div>")
    assert is_icon(root.find(".//svg")) is icon


def test_tail_of_the_root_is_not_part_of_the_drawing() -> None:
    """The text after an inline SVG belongs to its paragraph."""
    page = lxml.html.fragment_fromstring(f"<p>See {SVG} here.</p>")
    inline = svg_identity(page.find(".//svg"))
    assert inline == parse_svg(SVG.encode())
    assert "here." not in inline.texts


def test_comment_inside_a_text_does_not_split_it() -> None:
    """A comment inside a label leaves the label in one piece."""
    inline = lxml.html.fragment_fromstring(
        "<div><svg><text>Cli<!-- x -->ent</text></svg></div>",
    ).find(".//svg")
    file = parse_svg(b"<svg><text>Cli<!-- x -->ent</text></svg>")
    assert svg_identity(inline) == file
    assert file.texts == ("Client",)


CDATA_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 30 20">'
    "<style><![CDATA[.a{fill:red}]]></style>"
    '<text class="a"><![CDATA[Label]]></text></svg>'
)


def test_cdata_is_the_same_inline_and_in_a_file() -> None:
    """A CDATA section reads the same with either parser."""
    page = lxml.html.fragment_fromstring(f"<div>{CDATA_SVG}</div>")
    inline = svg_identity(page.find(".//svg"))
    assert parse_svg(CDATA_SVG.encode()) == inline
    assert inline.texts == ("Label",)


def test_another_style_is_another_drawing() -> None:
    """Two SVG that differ only by their style are not the same."""
    other = parse_svg(CDATA_SVG.replace("red", "blue").encode())
    assert other.digest != parse_svg(CDATA_SVG.encode()).digest


def test_size_falls_back_to_the_view_box() -> None:
    """A percentage or a missing side is read from the viewBox."""
    percent = parse_svg(
        b'<svg width="100%" height="50" viewBox="0 0 300 200"/>',
    )
    assert (percent.width, percent.height) == (300.0, 50.0)
    both = parse_svg(b'<svg width="100%" viewBox="0 0 300 200"/>')
    assert (both.width, both.height) == (300.0, 200.0)


def test_nested_texts_are_read_once() -> None:
    """Nested text elements cost linear time and memory, read once."""
    depth = 250
    body = "word " * 960_000
    data = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300">'
        f"{'<text>' * depth}{body}{'</text>' * depth}</svg>"
    ).encode()
    assert 4_500_000 < len(data) < SVG_MAX_BYTES
    start = time.perf_counter()
    identity = parse_svg(data)
    assert time.perf_counter() - start < 3.0
    assert identity.texts == (body.strip(),)
    tracemalloc.start()
    try:
        parse_svg(data)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 100_000_000
