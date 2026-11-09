# Copyright 2026 Christophe Le Douarec
"""The filter chains of PDF image streams, and the headers of their codecs.

PDFium decodes a JPEG or a JPEG 2000 at the size that its own header
announces, whatever the dictionary of the image says, and inflates the
simple filters of a stream whole, a thousandfold if a stream is built so. So
only the filter chains that the generators of documents write are accepted,
and the size that a codec announces is read from its header, as its decoder
reads it, before PDFium decodes it. Each reading here is bounded, and linear
in the bytes it reads.
"""

import re
import zlib
from typing import TYPE_CHECKING, Final, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

ASCII_HEX: Final = "ASCIIHexDecode"
ASCII_85: Final = "ASCII85Decode"
LZW: Final = "LZWDecode"
FLATE: Final = "FlateDecode"
RUN_LENGTH: Final = "RunLengthDecode"
CCITT: Final = "CCITTFaxDecode"
JBIG2: Final = "JBIG2Decode"
DCT: Final = "DCTDecode"
JPX: Final = "JPXDecode"
# The filter chains of the image streams that generators write: pixels, with
# no codec; a codec alone; a JPEG or a JPEG 2000 wrapped in one filter.
PIXEL_CHAINS: Final = frozenset(
    {(), (FLATE,), (RUN_LENGTH,), (ASCII_85, FLATE), (ASCII_HEX, FLATE)},
)
CODEC_CHAINS: Final = frozenset({(DCT,), (JPX,), (JBIG2,), (CCITT,)})
WRAPPED_CHAINS: Final = frozenset(
    {
        (ASCII_85, DCT),
        (ASCII_HEX, DCT),
        (FLATE, DCT),
        (ASCII_85, JPX),
        (FLATE, JPX),
    },
)

# The abbreviations of the filter names, which PDFium accepts as the names.
_ABBREVIATIONS: Final = {
    "AHx": ASCII_HEX,
    "A85": ASCII_85,
    "LZW": LZW,
    "Fl": FLATE,
    "RL": RUN_LENGTH,
    "CCF": CCITT,
    "DCT": DCT,
    "JPX": JPX,
}
_ZLIB_METHOD: Final = 8  # Deflate, the only method of a zlib stream.
_ZLIB_WINDOW: Final = 7  # The largest window that a zlib header announces.
_ZLIB_DICTIONARY: Final = 0x20  # The flag of a preset dictionary.
_ZLIB_CHECK: Final = 31  # The two bytes of a zlib header are its multiple.
_ZLIB_HEADER: Final = 2
_INFLATE_CHUNK: Final = 1 << 14  # Bytes of a zlib stream inflated at a time.
_INFLATE_OUT: Final = 1 << 20  # Most bytes inflated, and dropped, at a time.
_SOI: Final = b"\xff\xd8"  # The marker that starts a JPEG.
# A marker of a JPEG: the last of one or more 0xFF, then a byte that is
# neither 0 nor 0xFF. One 0xFF in the pattern keeps the search linear.
_MARKER: Final = re.compile(rb"\xff([\x01-\xfe])")
# A marker that is not part of the data of a scan: neither a stuffed 0xFF00
# nor a restart marker RSTn, 0xD0 to 0xD7. Linear, like `_MARKER`.
_SCAN_MARKER: Final = re.compile(rb"\xff([\x01-\xcf\xd8-\xfe])")
_SOS: Final = 0xDA  # The marker that starts a scan.
_EOI: Final = 0xD9  # The marker that ends a JPEG.
# The markers without a segment, which a scan walk passes over.
_STANDALONE: Final = frozenset({0x01, 0xD8})
# The most scans that a JPEG may have: the progressive ones of the encoders
# have ten or so, and the decoder works through each scan in turn.
MAX_JPEG_SCANS: Final = 64
# The markers that end the header: SOS and EOI, and the markers without a
# segment, TEM, RSTn and SOI, which have no place before a frame header.
_HEADER_END: Final = frozenset({0x01, *range(0xD0, 0xDB)})
# The JPEG segments, or the JP2 boxes, read at most before the frame header
# or the codestream: sound files hold a few dozen.
_MOST_SEGMENTS: Final = 4096
# The frame headers: SOF0 to SOF15, but DHT, JPG and DAC.
_FRAMES: Final = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}
_FRAME_BYTES: Final = 8  # Length, precision, height, width and components.
_JP2_SIGNATURE: Final = b"\x00\x00\x00\x0cjP  \r\n\x87\n"
_JP2_CODESTREAM: Final = b"jp2c"  # The box that holds the codestream.
_BOX_HEADER: Final = 8  # Length and type of a box.
_LONG_BOX_HEADER: Final = 16  # With an extended length.
_LONG_BOX: Final = 1  # The length of a box whose extended length follows.
_LAST_BOX: Final = 0  # The length of a box that runs to the end of the file.
_CODESTREAM: Final = b"\xff\x4f\xff\x51"  # SOC, then SIZ.
_SIZE_AT: Final = 8  # Offset of Xsiz in the codestream, after SOC and Lsiz.
_SIZE_BYTES: Final = 16  # Xsiz, Ysiz, XOsiz and YOsiz.
_COMPONENTS_AT: Final = 40  # Offset of Csiz, after the sizes of the tiles.


class CodecHeader(NamedTuple):
    """What the header of a JPEG or a JPEG 2000 announces.

    Attributes:
        width: The width of the image, in pixels.
        height: The height of the image, in pixels.
        components: The number of its components, such as 3 for RGB.
    """

    width: int
    height: int
    components: int


def full_filters(names: Sequence[str]) -> tuple[str, ...]:
    """Return the full names of the filters of a stream.

    Args:
        names: The filters as the stream names them, maybe abbreviated, such
            as ``DCT``.

    Returns:
        The full names, such as ``DCTDecode``, in the same order.
    """
    return tuple(_ABBREVIATIONS.get(name, name) for name in names)


def _zlib_header(head: bytes) -> bool:
    """Tell whether two bytes are a zlib header that zlib accepts.

    Args:
        head: The first two bytes of a stream.

    Returns:
        Whether they announce deflate, a window zlib allows, no preset
        dictionary, and pass their check.
    """
    return not (
        head[0] & 0x0F != _ZLIB_METHOD
        or head[0] >> 4 > _ZLIB_WINDOW
        or head[1] & _ZLIB_DICTIONARY
        or int.from_bytes(head) % _ZLIB_CHECK
    )


class _Inflater:
    """Inflate a raw deflate stream piece by piece, counting its bytes."""

    def __init__(self) -> None:
        """Start the inflation."""
        self._inflater = zlib.decompressobj(-zlib.MAX_WBITS)

    def count(self, data: bytes, limit: int) -> tuple[int, bool]:
        """Inflate one piece, and drop what it gives.

        Args:
            data: The piece.
            limit: Most bytes to inflate.

        Returns:
            The bytes inflated, and whether the inflation is over: at its
            end, at an error or at the limit.
        """
        count = 0
        pending = data
        # A call that gives less than asked has used all its input; one that
        # gives all may have more to give. Past the end, the bytes left stay
        # in the unconsumed tail, so the end stops the loop.
        while count < limit and not self._inflater.eof:
            asked = min(limit - count, _INFLATE_OUT)
            try:
                out = self._inflater.decompress(pending, asked)
            except zlib.error:
                return count, True  # PDFium keeps what came before.
            count += len(out)
            if len(out) < asked:
                break
            pending = self._inflater.unconsumed_tail
        return count, count >= limit or self._inflater.eof


def inflated_length(chunks: Iterable[bytes], limit: int) -> int:
    """Count the bytes that a zlib stream inflates to, holding none of them.

    The header is checked as zlib checks it, but not the checksum at the
    end, and the bytes inflated before an error count, as PDFium keeps them.

    Args:
        chunks: The zlib stream, a piece at a time.
        limit: Most bytes to count.

    Returns:
        The bytes that the stream inflates to, at most `limit`.
    """
    head = b""
    inflater = None
    count = 0
    for chunk in chunks:
        data = chunk
        if inflater is None:
            head += chunk
            if len(head) < _ZLIB_HEADER:
                continue
            if not _zlib_header(head[:_ZLIB_HEADER]):
                return 0
            inflater = _Inflater()
            data = head[_ZLIB_HEADER:]
        for start in range(0, len(data), _INFLATE_CHUNK):
            piece = data[start : start + _INFLATE_CHUNK]
            inflated, over = inflater.count(piece, limit - count)
            count += inflated
            if over:
                return count
    return count


def _jpeg_segments(data: bytes) -> Iterator[tuple[int, int]]:
    """Yield the marker and the start of each segment of a JPEG header.

    As PDFium, the JPEG starts at its first SOI marker. As libjpeg, the
    markers are read in order, each segment skipped by its length, and the
    bytes between two segments are passed over. The walk is linear in the
    data, and reads at most `_MOST_SEGMENTS` segments.

    Args:
        data: The JPEG, maybe after other bytes.

    Yields:
        The marker of each segment before the first scan, and the position
        of its length, which is in the data. The walk stops at a marker
        without a segment, which has no place in a header.
    """
    start = data.find(_SOI)
    if start < 0:
        return
    position = start + len(_SOI)
    for _ in range(_MOST_SEGMENTS):
        match = _MARKER.search(data, position)
        if match is None:
            return
        marker, position = match[1][0], match.end()
        if marker in _HEADER_END or position + 2 > len(data):
            return
        yield marker, position
        length = int.from_bytes(data[position : position + 2])
        if length < len(b"xx"):
            return
        position += length


def jpeg_scans(data: bytes) -> int:
    """Count the scans of a JPEG, without decoding any.

    The markers are read in order, each segment skipped by its length, and
    the data of a scan up to the next marker that is neither a stuffed byte,
    a restart marker nor fill. The walk is linear in the data, never reads
    past its end, and stops at `MAX_JPEG_SCANS` plus one scans, or after
    `_MOST_SEGMENTS` markers, which counts as too many scans.

    Args:
        data: The JPEG, maybe after other bytes, maybe truncated or not a
            JPEG at all.

    Returns:
        The number of scans, at most `MAX_JPEG_SCANS` plus one; 0 if there is
        no SOI marker. It never raises.
    """
    start = data.find(_SOI)
    if start < 0:
        return 0
    position, scans = start + len(_SOI), 0
    for _ in range(_MOST_SEGMENTS):
        match = _SCAN_MARKER.search(data, position)
        if match is None:
            return scans
        marker, position = match[1][0], match.end()
        if marker == _EOI:
            return scans
        if marker in _STANDALONE:
            continue
        if marker == _SOS:
            scans += 1
            if scans > MAX_JPEG_SCANS:
                return scans
        if position + 2 > len(data):
            return scans
        length = int.from_bytes(data[position : position + 2])
        position += max(length, len(b"xx"))
    return MAX_JPEG_SCANS + 1  # Past the bound on markers: not decoded.


def jpeg_header(data: bytes) -> CodecHeader | None:
    """Read the size and the components of a JPEG from its frame header.

    Args:
        data: The JPEG, maybe after other bytes, as PDFium allows.

    Returns:
        What the first frame announces, or ``None`` if no frame header comes
        before the first scan, or if it announces no width, no height or no
        component.
    """
    for marker, position in _jpeg_segments(data):
        if marker in _FRAMES:
            frame = data[position : position + _FRAME_BYTES]
            if len(frame) < _FRAME_BYTES:
                return None
            header = CodecHeader(
                width=int.from_bytes(frame[5:7]),
                height=int.from_bytes(frame[3:5]),
                components=frame[7],
            )
            return header if all(header) else None
    return None


def _jp2_codestream(data: bytes) -> bytes:
    """Return the codestream of a JP2 file, or nothing if it has none.

    At most `_MOST_SEGMENTS` boxes are read before the codestream.

    Args:
        data: The JP2 file.

    Returns:
        The bytes of its ``jp2c`` box, or nothing.
    """
    position = 0
    for _ in range(_MOST_SEGMENTS):
        if position + _BOX_HEADER > len(data):
            return b""
        length = int.from_bytes(data[position : position + 4])
        kind = data[position + 4 : position + _BOX_HEADER]
        header = _BOX_HEADER
        if length == _LONG_BOX:
            header = _LONG_BOX_HEADER
            length = int.from_bytes(
                data[position + _BOX_HEADER : position + header],
            )
        elif length == _LAST_BOX:
            length = len(data) - position
        if kind == _JP2_CODESTREAM:
            return data[position + header : position + length]
        if length < header:
            return b""
        position += length
    return b""


def jpx_header(data: bytes) -> CodecHeader | None:
    """Read the size and the components of a JPEG 2000 from its SIZ marker.

    Args:
        data: A JP2 file, whose ``jp2c`` box holds the codestream, or a bare
            codestream, as PDFium tells them apart.

    Returns:
        The width and the height of the image area, ``Xsiz - XOsiz`` and
        ``Ysiz - YOsiz``, and the components, ``Csiz``; or ``None`` if they
        cannot be read, or if one of them is not positive.
    """
    stream = _jp2_codestream(data) if data.startswith(_JP2_SIGNATURE) else data
    if not stream.startswith(_CODESTREAM) or len(stream) < _COMPONENTS_AT + 2:
        return None
    xsiz, ysiz, xosiz, yosiz = (
        int.from_bytes(stream[at : at + 4])
        for at in range(_SIZE_AT, _SIZE_AT + _SIZE_BYTES, 4)
    )
    header = CodecHeader(
        width=xsiz - xosiz,
        height=ysiz - yosiz,
        components=int.from_bytes(stream[_COMPONENTS_AT : _COMPONENTS_AT + 2]),
    )
    return header if min(header) > 0 else None
