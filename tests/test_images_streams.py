# Copyright 2026 Christophe Le Douarec
"""Tests for the filter chains of PDF image streams and their codec headers."""

import os
import struct
import time
import zlib

import pytest

from doc_checker.images.streams import (
    full_filters,
    inflated_length,
    jpeg_header,
    jpx_header,
)
from tests.image_builders import (
    cmyk_encoded,
    drawing,
    encode,
    jpeg_of_components,
    jpx_announcing,
    jpx_of_components,
)

_JPEG = encode(drawing("setup"), "JPEG", quality=90)


def test_filters_get_their_full_names() -> None:
    """The abbreviations of the filters become their full names."""
    names = ["AHx", "A85", "LZW", "Fl", "RL", "CCF", "DCT", "JPX", "Foo"]
    assert full_filters(names) == (
        "ASCIIHexDecode",
        "ASCII85Decode",
        "LZWDecode",
        "FlateDecode",
        "RunLengthDecode",
        "CCITTFaxDecode",
        "DCTDecode",
        "JPXDecode",
        "Foo",
    )


def test_a_flate_bomb_is_counted_within_bounds() -> None:
    """Only the bytes asked for are inflated, whatever the stream holds."""
    bomb = zlib.compress(bytes(64 << 20), 9)
    assert inflated_length([bomb], 1000) == 1000
    assert inflated_length([bomb[:1], bomb[1:]], 5 << 20) == 5 << 20


def test_a_stream_that_ends_after_a_large_piece_is_counted() -> None:
    """A stream that ends past a megabyte gives its length, then stops."""
    assert inflated_length([zlib.compress(bytes(3 << 20))], 10 << 20) == (
        3 << 20
    )


def test_inflated_length_keeps_what_comes_before_an_error() -> None:
    """A wrong checksum or a cut stream still counts its bytes, as PDFium."""
    stream = zlib.compress(os.urandom(6000))
    wrong_check = stream[:-4] + b"\x00\x00\x00\x00"
    assert inflated_length([wrong_check], 10**6) == 6000
    assert 0 < inflated_length([stream[: len(stream) // 2]], 10**6) < 6000
    assert inflated_length([b"\x00\x01junk"], 10) == 0
    assert inflated_length([], 10) == 0


def test_jpeg_header_follows_the_segments() -> None:
    """The frame header is read past other bytes and other segments."""
    # An APP1 segment that holds a frame header of 16 by 16 pixels.
    fake = b"\xff\xc0\x00\x11\x08\x00\x10\x00\x10\x03" + bytes(9)
    app1 = b"\xff\xe1" + struct.pack(">H", len(fake) + 2) + fake
    hidden = _JPEG[:2] + app1 + _JPEG[2:]
    assert jpeg_header(_JPEG) == (400, 250, 3)
    assert jpeg_header(b"junk before" + _JPEG) == (400, 250, 3)
    assert jpeg_header(hidden) == (400, 250, 3)


def test_codec_headers_give_their_components() -> None:
    """The components that a frame header or a SIZ marker announce."""
    assert jpeg_header(cmyk_encoded("JPEG")) == (400, 250, 4)
    assert jpeg_header(jpeg_of_components(5)) == (400, 250, 5)
    assert jpx_header(cmyk_encoded("JPEG2000")) == (400, 250, 4)
    assert jpx_header(jpx_of_components(16)) == (400, 250, 16)


def test_header_walks_are_linear_and_bounded() -> None:
    """Runs of 0xFF, and floods of markers or boxes, take little time."""
    start = time.perf_counter()
    assert jpeg_header(b"\xff\xd8" + b"\xff" * 1_000_000 + b"\x00") is None
    assert jpeg_header(b"\xff\xd8" + b"\xff\xd0" * 1_000_000) is None
    assert jpeg_header(b"\xff\xd8" + b"\xff\xe0\x00\x02" * 100_000) is None
    jp2 = encode(drawing("setup"), "JPEG2000")
    boxes = jp2[:12] + b"\x00\x00\x00\x08free" * 100_000 + jp2[12:]
    assert jpx_header(boxes) is None
    assert time.perf_counter() - start < 2.0
    padded = _JPEG[:2] + b"\xff" * 100_000 + _JPEG[2:]
    assert jpeg_header(padded) == (400, 250, 3)


@pytest.mark.parametrize(
    "data",
    [
        b"no start of image",
        b"\xff\xd8\xff\xda\x00\x02",  # A scan before any frame.
        b"\xff\xd8\xff\xd0" + _JPEG[2:],  # A restart before the frame.
        b"\xff\xd8" + _JPEG,  # Two starts of image.
        b"\xff\xd8\xff\xc0\x00\x11\x08\x00",  # A frame header cut short.
        b"\xff\xd8\xff\xc0\x00\x11\x08\x00\x00\x01\x90\x03",  # No height.
        b"\xff\xd8\xff\xc0\x00\x08\x08\x00\xfa\x01\x90\x00",  # No component.
    ],
)
def test_jpeg_without_a_header(data: bytes) -> None:
    """A JPEG whose frame header cannot be read has no header."""
    assert jpeg_header(data) is None


def test_jpx_header_of_a_codestream_and_of_a_jp2_file() -> None:
    """The SIZ marker is read in a bare codestream and in a JP2 file."""
    jp2 = encode(drawing("setup"), "JPEG2000")
    assert jpx_header(jpx_announcing(400, 250)) == (400, 250, 3)
    assert jpx_header(jp2) == (400, 250, 3)
    assert jpx_header(b"junk" + jpx_announcing(400, 250)) is None
    assert jpx_header(jp2[:40]) is None
    no_component = jpx_announcing(400, 250)
    assert jpx_header(no_component[:40] + bytes(2) + no_component[42:]) is None
