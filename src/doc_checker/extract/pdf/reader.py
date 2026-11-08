# Copyright 2026 Christophe Le Douarec
"""Read the characters, images, bookmarks and metadata of PDF documents.

PDFium does the reading. Each character keeps its position and its style, so
that the next steps can rebuild lines, blocks and headings. A page is read
upright, turned so that the baselines of most of its characters run from left
to right, and its characters whose baseline runs at another angle are left
out. Its raster images keep their box on the upright page and their
signature, their pixels being dropped once read. Its vector figures are found
whether images are read or not, and the text drawn in them is left out and
counted; when images are read, each figure is rendered for its signature, and
the render dropped. Pages are read one at a time, so that the caller never
holds the characters of a whole document.
"""

import ctypes
import math
import re
from collections import Counter
from contextlib import closing
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

from doc_checker.errors import InputError
from doc_checker.extract.pdf.figures import vector_figures
from doc_checker.images.load import (
    RenderGuard,
    decode_object,
    image_objects,
    object_stream,
    render_figure,
)
from doc_checker.images.pixels import (
    NOT_RENDERED,
    PAGE_LIMIT,
    UnreadableImage,
)
from doc_checker.images.refs import content_digest, pixels_ref, unreadable_ref
from doc_checker.model.document import PixelSource
from doc_checker.text.normalize import printable

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from doc_checker.extract.pdf.figures import VectorFigure
    from doc_checker.model.document import ImageRef

FIXED_PITCH_FLAG: Final = 1
FORCE_BOLD_FLAG: Final = 1 << 18
BOLD_WEIGHT: Final = 600
# Neither "code" alone, which would take "Arial Unicode MS" for code, nor
# "mono" before "type", as in "Monotype Corsiva".
MONO_NAME: Final = re.compile(
    r"courier|mono(?!type)|consol|menlo|typewriter|fixed|cursor|cmtt|lmtt"
    r"|inconsolata|sourcecode|firacode|cascadia",
    re.IGNORECASE,
)
BOLD_NAME: Final = re.compile(
    r"bold|black|heavy|semibold|demi|cmbx",
    re.IGNORECASE,
)
TURN_TOLERANCE: Final = 2.0  # Largest gap to a right angle, in degrees.
LABEL_LIMIT: Final = 40  # Most characters of a page label.
METADATA_LIMIT: Final = 200  # Most characters of a metadata value.
MAX_PAGE_FIGURES: Final = 64  # Most vector figures rendered on a page.

_SUBSET_PREFIX: Final = re.compile(r"^[A-Z]{6}\+")
_LINE_BREAKS: Final = frozenset({0x0A, 0x0D})
_HYPHEN_CODE: Final = 0x02  # PDFium reports a line end hyphen as U+0002.
_REPLACEMENT: Final = "\U0000fffd"
_HIGH_SURROGATES: Final = range(0xD800, 0xDC00)
_LOW_SURROGATES: Final = range(0xDC00, 0xE000)
_ASTRAL_START: Final = 0x10000  # The first character beyond U+FFFF.
_HALF_BITS: Final = 10  # The bits of a character that a surrogate half holds.
_ARTIFACT: Final = "Artifact"
_POINTS_PER_INCH: Final = 72.0
# The format of an image, by the last filter of its stream; another filter
# gives its name in lower case, without "decode", and no filter gives "raw".
_FILTER_FORMATS: Final = {
    "DCTDecode": "jpeg",
    "JPXDecode": "jpx",
    "JBIG2Decode": "jbig2",
    "CCITTFaxDecode": "ccitt",
    "FlateDecode": "flate",
}
_RAW_FORMAT: Final = "raw"
_FORMAT_LIMIT: Final = 20  # Most characters of the format of another filter.
_FILTER_SEPARATOR: Final = "\x1f"  # Joins the filters that the digest reads.
_FULL_TURN: Final = 360.0  # Degrees in a full turn.
_QUARTER_TURN: Final = 90.0  # Degrees in a quarter turn.
_QUARTERS: Final = 4  # Quarter turns in a full turn.
_FLIPPED: Final = 2  # Quarter turns that a negative font size adds.
_VECTOR_FORMAT: Final = "vector"  # The format of a vector figure.

# The left, right, bottom and top edges of the box of a character, then the x
# and the y of its origin, in points from the lower left corner of the crop
# box.
type _Box = tuple[float, float, float, float, float, float]
# The effective size, the mono flag and the bold flag of a character.
type _Style = tuple[float, bool, bool]
# The style of a character and its quarter turns, None for text at an angle.
type _Look = tuple[_Style, int | None]
# A character before its page is turned upright: its text, its box, its style,
# its quarter turns, None for text at an angle, whether it has no Unicode
# mapping, and whether it has no text object, as the space that PDFium
# generates between two objects.
type _Pending = tuple[str, _Box, _Style, int | None, bool, bool]


@dataclass(frozen=True, slots=True)
class PdfChar:
    """A character of a PDF page.

    Its position is on its page turned upright, as `PdfPageText` tells. The
    turn follows the direction of its baseline, so a glyph that slants on a
    level baseline, as in synthetic italics, is kept where it stands.

    Attributes:
        text: Exactly one character, which UTF-8 can encode. The two surrogate
            halves that PDFium gives for a character beyond U+FFFF are
            joined, and a lone half is replaced by U+FFFD.
        x0: Left edge of the character box, in points from the left of the
            turned crop box.
        x1: Right edge of the character box, in points from the left of the
            turned crop box.
        baseline: Vertical position of the baseline, in points above the
            bottom of the turned crop box.
        size: Effective font size, in points.
        mono: Whether the font is monospaced.
        bold: Whether the font is bold.
    """

    text: str
    x0: float
    x1: float
    baseline: float
    size: float
    mono: bool
    bold: bool


@dataclass(frozen=True, slots=True)
class PdfImage:
    """A raster image or a vector figure of a PDF page.

    Its box is on its page turned upright, as `PdfPageText` tells, in the
    coordinates of the characters.

    Attributes:
        ref: The image: its signature, or why its pixels cannot be read.
        x0: Left edge of the box, in points from the left of the turned crop
            box.
        x1: Right edge of the box.
        bottom: Bottom edge of the box, in points above the bottom of the
            turned crop box.
        top: Top edge of the box.
    """

    ref: ImageRef
    x0: float
    x1: float
    bottom: float
    top: float


@dataclass(frozen=True, slots=True)
class PdfPageText:
    """The characters of a PDF page and what was left out of them.

    The page is turned upright, by quarter turns, so that the baselines of
    most of its characters run from left to right, whatever rotation the page
    declares. A page whose text runs up the page, as a landscape page often
    does, is turned a quarter clockwise, and its width and height swap. The
    characters whose baseline runs at another angle, such as those of a
    slanted watermark, are left out. Glyphs that slant on a level baseline,
    as synthetic italics do, are kept.

    The coordinates are in points, with the y axis pointing up, and their
    origin is the lower left corner of the turned crop box, the crop box being
    the media box cut by the crop box.

    Attributes:
        index: Position of the page in the document, from 0.
        label: Page label, such as ``iv``, or the page number from 1. It is
            printable and has at most `LABEL_LIMIT` characters.
        width: Width of the turned crop box, in points.
        height: Height of the turned crop box, in points.
        chars: Kept characters, in the order of PDFium.
        artifacts: Number of characters dropped as artifacts.
        unmapped: Number of kept characters without Unicode mapping, with the
            lone surrogate halves that were replaced.
        image_coverage: Share of the page covered by images, from 0 to 1.
        angled: Number of characters left out because they do not follow the
            direction of the page.
        images: The raster images of the page and of its forms, in the order
            of PDFium, but those of the forms that are vector figures, then
            the vector figures, in the order of their objects, unless they
            were not read.
        figure_chars: Number of characters left out because they are drawn
            in a vector figure.
        unrendered: Number of vector figures not rendered because they come
            after the first `MAX_PAGE_FIGURES` of the page.
    """

    index: int
    label: str
    width: float
    height: float
    chars: tuple[PdfChar, ...]
    artifacts: int
    unmapped: int
    image_coverage: float
    angled: int
    images: tuple[PdfImage, ...] = ()
    figure_chars: int = 0
    unrendered: int = 0


@dataclass(frozen=True, slots=True)
class Bookmark:
    """An entry of the outline of a PDF document.

    Attributes:
        level: Depth of the entry in the outline, from 0.
        title: Title of the entry.
        page: Page the entry points to, from 0, if it points to one.
    """

    level: int
    title: str
    page: int | None


@dataclass(frozen=True, slots=True)
class _ObjectInfo:
    """What is read once for each text object of a page.

    An object has one font and one matrix, so all its characters share their
    style, their marks and their quarter turns, which `_turn` reads from the
    direction of their baseline.
    """

    mono: bool
    bold: bool
    artifact: bool
    turn: int | None


def font_style(name: str, flags: int, weight: int) -> tuple[bool, bool]:
    """Tell whether a font is monospaced and whether it is bold.

    Base fonts have neither flags nor weight, so their name tells the style.

    Args:
        name: Name of the font, with or without the ``ABCDEF+`` prefix of
            subset fonts.
        flags: Flags of the font descriptor.
        weight: Weight of the font, 0 if unknown.

    Returns:
        Whether the font is monospaced, and whether it is bold.
    """
    base = _SUBSET_PREFIX.sub("", name)
    mono = bool(flags & FIXED_PITCH_FLAG) or bool(MONO_NAME.search(base))
    bold = (
        bool(flags & FORCE_BOLD_FLAG)
        or weight >= BOLD_WEIGHT
        or bool(BOLD_NAME.search(base))
    )
    return mono, bold


def char_text(code: int, *, hyphen: bool, unmapped: bool) -> str:
    """Return the text of a character reported by PDFium.

    Args:
        code: Unicode code point of the character.
        hyphen: Whether the character is a hyphen that ends a line.
        unmapped: Whether the character has no Unicode mapping.

    Returns:
        The replacement character for an unmapped character, a hyphen for a
        line end hyphen, and else the character itself.
    """
    if unmapped:
        return _REPLACEMENT
    if hyphen and code == _HYPHEN_CODE:
        return "-"
    return chr(code)


def open_pdf(path: Path) -> pdfium.PdfDocument:
    """Open a PDF document.

    Args:
        path: The PDF file.

    Returns:
        The document, which is a context manager.

    Raises:
        InputError: If the file cannot be read, is protected by a password,
            or is not a valid PDF.
    """
    try:
        return pdfium.PdfDocument(path)
    except pdfium.PdfiumError as exc:
        if "password" in str(exc).casefold():
            reason = "the PDF is password protected"
        else:
            reason = "invalid or damaged PDF"
        msg = f"cannot read {path}: {reason}"
        raise InputError(msg) from None
    except OSError as exc:
        # For anything but a file, pypdfium2 raises a FileNotFoundError that
        # has no system message.
        msg = f"cannot read {path}: {exc.strerror or 'not a file'}"
        raise InputError(msg) from None


def _font(textpage: pdfium.PdfTextPage, index: int) -> tuple[str, int, int]:
    """Return the name, the descriptor flags and the weight of a font."""
    flags = ctypes.c_int()
    size = pdfium_c.FPDFText_GetFontInfo(textpage, index, None, 0, flags)
    buffer = ctypes.create_string_buffer(size)
    pdfium_c.FPDFText_GetFontInfo(textpage, index, buffer, size, flags)
    weight = pdfium_c.FPDFText_GetFontWeight(textpage, index)
    name = buffer.value.decode("utf-8", errors="replace")
    return name, flags.value, weight


def _mark_name(mark: pdfium_c.FPDF_PAGEOBJECTMARK) -> str:
    """Return the name of a content mark."""
    size = ctypes.c_ulong()
    pdfium_c.FPDFPageObjMark_GetName(mark, None, 0, size)
    # The name is UTF-16, and the size counts its terminator, in bytes.
    buffer = (pdfium_c.FPDF_WCHAR * (size.value // 2))()
    pdfium_c.FPDFPageObjMark_GetName(mark, buffer, size.value, size)
    return bytes(buffer).decode("utf-16-le").rstrip("\x00")


def _is_artifact(obj: pdfium_c.FPDF_PAGEOBJECT) -> bool:
    """Return whether a text object is marked as an artifact."""
    return any(
        _mark_name(pdfium_c.FPDFPageObj_GetMark(obj, position)) == _ARTIFACT
        for position in range(pdfium_c.FPDFPageObj_CountMarks(obj))
    )


def _object_info(
    textpage: pdfium.PdfTextPage,
    index: int,
    obj: pdfium_c.FPDF_PAGEOBJECT,
    cache: dict[int, _ObjectInfo],
) -> _ObjectInfo | None:
    """Return the facts about the text object of a character.

    The first character of an object reads them, and the others get them from
    the cache.

    Args:
        textpage: The text page of the character.
        index: Index of the character.
        obj: Its text object, as PDFium gives it, null if it has none.
        cache: Facts already read, by address of text object.

    Returns:
        The facts, or ``None`` for a character without a text object, such as
        the space that PDFium generates between two objects.
    """
    if not obj:
        return None
    address = ctypes.addressof(obj.contents)
    if address not in cache:
        mono, bold = font_style(*_font(textpage, index))
        cache[address] = _ObjectInfo(
            mono,
            bold,
            artifact=_is_artifact(obj),
            turn=_turn(textpage, index),
        )
    return cache[address]


def _turn(textpage: pdfium.PdfTextPage, index: int) -> int | None:
    """Return the quarter turns of the text of a character.

    The turns follow the direction of the baseline that the matrix of the
    character gives, clockwise, which is the angle of PDFium for a text that
    is not sheared. A shear, as in synthetic italics, slants the glyphs but
    leaves the baseline level, so it does not turn the text, whereas the
    angle of PDFium would count it. A negative font size, as in
    ``/F1 -60 Tf``, sets the text upside down, which the matrix does not
    tell, so it adds a half turn.

    Args:
        textpage: The text page of the character.
        index: Index of the character.

    Returns:
        The number of quarter turns, clockwise and from 0 to 3, of the right
        angle that is at most `TURN_TOLERANCE` from the direction of the
        baseline, or ``None`` for a baseline at an angle between two right
        angles.
    """
    radians = 0.0  # A matrix that cannot be read counts as no turn.
    matrix = pdfium_c.FS_MATRIX()
    if pdfium_c.FPDFText_GetMatrix(textpage, index, matrix):
        # The baseline runs along (a, b), so its clockwise angle is that of
        # (a, -b).
        radians = math.atan2(-matrix.b, matrix.a)
    degrees = math.degrees(radians) % _FULL_TURN
    quarters = round(degrees / _QUARTER_TURN)
    if abs(degrees - quarters * _QUARTER_TURN) > TURN_TOLERANCE:
        return None
    if pdfium_c.FPDFText_GetFontSize(textpage, index) < 0:
        quarters += _FLIPPED
    return quarters % _QUARTERS


def _effective_size(textpage: pdfium.PdfTextPage, index: int) -> float:
    """Return the font size of a character, scaled by its text matrix.

    PDF allows a negative font size, as in ``/F1 -60 Tf``, and PDFium reports
    it as is. Its absolute value is taken, so that the line box of the text is
    never upside down: the half turn that the sign gives the text is read by
    `_turn`.

    Args:
        textpage: The text page of the character.
        index: Index of the character.

    Returns:
        The font size in points, which is never negative.
    """
    size = abs(pdfium_c.FPDFText_GetFontSize(textpage, index))
    matrix = pdfium_c.FS_MATRIX()
    if pdfium_c.FPDFText_GetMatrix(textpage, index, matrix):
        size *= math.sqrt(abs(matrix.a * matrix.d - matrix.b * matrix.c))
    return size


def _char_box(
    textpage: pdfium.PdfTextPage,
    index: int,
    corner: tuple[float, float],
) -> _Box:
    """Return the box and the origin of a character.

    Args:
        textpage: The text page of the character.
        index: Index of the character.
        corner: The point of the page that is the origin of the result.

    Returns:
        The left, right, bottom and top edges of the box, then the x and the
        y of the origin, from the corner.
    """
    left, right, bottom, top = (ctypes.c_double() for _ in range(4))
    pdfium_c.FPDFText_GetCharBox(textpage, index, left, right, bottom, top)
    origin_x, origin_y = ctypes.c_double(), ctypes.c_double()
    pdfium_c.FPDFText_GetCharOrigin(textpage, index, origin_x, origin_y)
    x, y = corner
    return (
        left.value - x,
        right.value - x,
        bottom.value - y,
        top.value - y,
        origin_x.value - x,
        origin_y.value - y,
    )


def _upright(
    box: _Box,
    turn: int,
    size: tuple[float, float],
) -> tuple[float, float, float]:
    """Return where a character is once its page is turned upright.

    Args:
        box: The box and the origin of the character, as `_char_box` returns
            them from the lower left corner of the crop box.
        turn: The quarter turns, clockwise, of the text of the page.
        size: The width and the height of the crop box.

    Returns:
        The left edge, the right edge and the baseline of the character, from
        the lower left corner of the crop box turned so that the text of the
        page runs from left to right.
    """
    left, right, bottom, top, x, y = box
    width, height = size
    match turn:
        case 0:
            return left, right, y
        case 1:
            return height - top, height - bottom, x
        case 2:
            return width - right, width - left, height - y
        case _:  # Three quarters, as pdflscape sets a landscape page.
            return bottom, top, width - x


def _look(
    textpage: pdfium.PdfTextPage,
    index: int,
    info: _ObjectInfo | None,
    previous: _Look | None,
) -> _Look | None:
    """Return the style and the quarter turns of a character.

    PDFium generates some characters: the spaces between the words that a
    kerned string separates, and those between two text objects of a line,
    which have no text object. Their size is not reliable, as they do not have
    the text matrix of the text around them, so they take the style and the
    turns of the previous character.

    Args:
        textpage: The text page of the character.
        index: Index of the character.
        info: The facts about its text object, if it has one.
        previous: The style and the turns of the last character that was
            kept, if any.

    Returns:
        The size, the mono flag and the bold flag of the character, and its
        quarter turns as `_turn` tells them, or ``None`` for a generated
        character that follows no kept character, and has to be left out.
    """
    if info is None or pdfium_c.FPDFText_IsGenerated(textpage, index) == 1:
        return previous
    return (_effective_size(textpage, index), info.mono, info.bold), info.turn


def _text_at(
    textpage: pdfium.PdfTextPage,
    index: int,
    code: int,
) -> tuple[str, bool]:
    """Return the text of a character, and whether it has no Unicode map."""
    unmapped = pdfium_c.FPDFText_HasUnicodeMapError(textpage, index) == 1
    hyphen = pdfium_c.FPDFText_IsHyphen(textpage, index) == 1
    return char_text(code, hyphen=hyphen, unmapped=unmapped), unmapped


def _join_halves(high: str, low: str) -> str:
    """Return the character that a high and a low surrogate half make."""
    return chr(
        _ASTRAL_START
        + ((ord(high) - _HIGH_SURROGATES.start) << _HALF_BITS)
        + (ord(low) - _LOW_SURROGATES.start),
    )


def _join_surrogates(chars: list[PdfChar]) -> tuple[list[PdfChar], int]:
    """Join the surrogate halves of the characters beyond U+FFFF.

    PDFium reports such a character as two UTF-16 halves, one character each,
    and both have the box of the whole glyph. A high half followed by a low
    half make one character. Any other half is replaced by U+FFFD, so that the
    text can be encoded.

    Args:
        chars: The kept characters of a page.

    Returns:
        The characters, and the number of lone halves that were replaced.
    """
    joined: list[PdfChar] = []
    lone = 0
    high: PdfChar | None = None  # A high half that waits for its low half.
    for char in chars:
        code = ord(char.text)
        if high is not None:
            if code in _LOW_SURROGATES:
                text = _join_halves(high.text, char.text)
                joined.append(replace(high, text=text))
                high = None
                continue
            joined.append(replace(high, text=_REPLACEMENT))
            lone += 1
            high = None
        if code in _HIGH_SURROGATES:
            high = char
        elif code in _LOW_SURROGATES:
            joined.append(replace(char, text=_REPLACEMENT))
            lone += 1
        else:
            joined.append(char)
    if high is not None:
        joined.append(replace(high, text=_REPLACEMENT))
        lone += 1
    return joined, lone


@dataclass(frozen=True, slots=True)
class _FigureText:
    """Where the text drawn in the vector figures of a page lies.

    Attributes:
        objects: The addresses of the text objects of the forms that are
            figures.
        chars: The indices of the characters in the boxes of the clusters
            that are figures.
    """

    objects: frozenset[int]
    chars: frozenset[int]

    @classmethod
    def of(cls, figures: Sequence[VectorFigure]) -> _FigureText:
        """Gather where the text of the figures of a page lies.

        Args:
            figures: The vector figures of the page.

        Returns:
            The text objects of its forms, and the characters of its
            clusters.
        """
        return cls(
            frozenset().union(*(figure.text_objects for figure in figures)),
            frozenset().union(*(figure.text_chars for figure in figures)),
        )

    def holds(self, address: int | None, index: int) -> bool:
        """Tell whether a character is drawn in a figure.

        Args:
            address: The address of its text object, if it has one.
            index: Its index in the text page.

        Returns:
            Whether its text object is that of a form that is a figure, or
            it is in the box of a cluster that is one.
        """
        return index in self.chars or (
            address is not None and address in self.objects
        )


@dataclass(frozen=True, slots=True)
class _ReadChars:
    """The characters of a text page, and those left out of them.

    Attributes:
        chars: The kept characters, as they lie on the page.
        artifacts: The number of characters dropped as artifacts.
        figure_chars: The number of characters dropped as the text of a
            vector figure.
    """

    chars: list[_Pending]
    artifacts: int
    figure_chars: int


def _read_chars(
    textpage: pdfium.PdfTextPage,
    corner: tuple[float, float],
    figures: _FigureText,
) -> _ReadChars:
    """Read the characters of a text page, but not the line breaks.

    The characters of artifacts, such as running headers and footers, are
    dropped. So are those drawn in a vector figure, with a character without
    a text object, as PDFium generates between two objects, that follows one
    of them. The others are read as they lie on the page, with the quarter
    turns of their text, before the page is turned upright. A character
    generated with no kept character before it is dropped too.

    Args:
        textpage: The text page to read.
        corner: The lower left corner of the crop box, which is the origin of
            the coordinates.
        figures: Where the text of the vector figures of the page lies.

    Returns:
        The kept characters, and the numbers of characters dropped.
    """
    chars: list[_Pending] = []
    cache: dict[int, _ObjectInfo] = {}
    artifacts = figure_chars = 0
    previous: _Look | None = None
    in_figure = False  # Whether the last character was drawn in a figure.
    for index in range(textpage.count_chars()):
        code = pdfium_c.FPDFText_GetUnicode(textpage, index)
        if code in _LINE_BREAKS:
            continue
        obj = pdfium_c.FPDFText_GetTextObject(textpage, index)
        info = _object_info(textpage, index, obj, cache)
        if info is not None and info.artifact:
            artifacts += 1
            in_figure = False
            continue
        address = ctypes.addressof(obj.contents) if obj else None
        in_figure = (address is None and in_figure) or figures.holds(
            address,
            index,
        )
        if in_figure:
            figure_chars += 1
            continue
        box = _char_box(textpage, index, corner)
        look = _look(textpage, index, info, previous)
        if look is None:
            continue
        previous = look
        text, no_mapping = _text_at(textpage, index, code)
        chars.append((text, box, *look, no_mapping, info is None))
    return _ReadChars(chars, artifacts, figure_chars)


def _page_turn(chars: Sequence[_Pending]) -> int:
    """Return the quarter turns that the text of a page follows.

    Args:
        chars: The characters of the page, as `_read_chars` reads them.

    Returns:
        The quarter turns that the most characters have, those at an angle
        left aside: the smallest in a tie, and 0 for a page without any such
        character.
    """
    counts = Counter(turn for _, _, _, turn, _, _ in chars if turn is not None)
    return min(counts, key=lambda turn: (-counts[turn], turn), default=0)


@dataclass(frozen=True, slots=True)
class _Upright:
    """The characters of a page turned upright, and the size of the page.

    Attributes:
        chars: The characters that follow the turns of the page.
        width: Width of the turned crop box, in points.
        height: Height of the turned crop box, in points.
        unmapped: Number of these characters without Unicode mapping, with
            the lone surrogate halves that were replaced.
        angled: Number of characters left out because they do not follow the
            turns of the page.
        turn: The quarter turns, clockwise, of the text of the page.
    """

    chars: tuple[PdfChar, ...]
    width: float
    height: float
    unmapped: int
    angled: int
    turn: int


def _kept_chars(
    chars: Sequence[_Pending],
    turn: int,
    size: tuple[float, float],
) -> tuple[list[PdfChar], int]:
    """Make the characters that follow the turns of a page, upright.

    PDFium puts the space that it generates between two text objects at the
    origin of the character before it, moved by its advance along the x axis
    of the page, whatever the direction of the text: on a turned page, the
    space leaves the line. There, a character without a text object stands
    instead on the baseline of the character before it, at its right edge,
    with no width.

    Args:
        chars: The characters of the page, as `_read_chars` reads them.
        turn: The quarter turns of the page.
        size: The width and the height of the crop box.

    Returns:
        The characters that follow the turns of the page, where they lie on
        the turned page, and the number of them without Unicode mapping.
    """
    kept: list[PdfChar] = []
    unmapped = 0
    for text, box, style, char_turn, no_mapping, no_object in chars:
        if char_turn != turn:
            continue
        if no_object and turn:
            # It has the turns of the character before it, which is kept.
            previous = kept[-1]
            edges = (previous.x1, previous.x1, previous.baseline)
        else:
            edges = _upright(box, turn, size)
        kept.append(PdfChar(text, *edges, *style))
        if no_mapping:
            unmapped += 1
    return kept, unmapped


def _upright_page(
    chars: Sequence[_Pending],
    size: tuple[float, float],
) -> _Upright:
    """Turn a page upright, and leave out its characters at another angle.

    The page is turned by quarter turns, so that the baselines of most of its
    characters run from left to right. Each kept character is made once,
    where it lies on the turned page, and the halves of the characters beyond
    U+FFFF are joined.

    Args:
        chars: The characters of the page, as `_read_chars` reads them.
        size: The width and the height of the crop box.

    Returns:
        The kept characters and the size of the turned page, with the counts
        of what was left out or replaced.
    """
    turn = _page_turn(chars)
    kept, unmapped = _kept_chars(chars, turn, size)
    joined, lone = _join_surrogates(kept)
    width, height = size
    if turn % 2:  # A quarter turn, either way, swaps the sides of the page.
        width, height = height, width
    return _Upright(
        chars=tuple(joined),
        width=width,
        height=height,
        unmapped=unmapped + lone,
        angled=len(chars) - len(kept),
        turn=turn,
    )


def _overlap(first: tuple[float, float], second: tuple[float, float]) -> float:
    """Return the length of the overlap of two segments."""
    return max(0.0, min(first[1], second[1]) - max(first[0], second[0]))


def _page_matrix(image: pdfium.PdfImage) -> pdfium.PdfMatrix:
    """Return the matrix that sets an image object on its page.

    PDFium gives the matrix and the bounds of an object of a form in the
    space of that form, so the matrices of the forms that hold the object
    apply after its own.

    Args:
        image: The image object.

    Returns:
        The matrix from the unit square of the image to the page.
    """
    matrix = image.get_matrix()
    container = image.container
    while container is not None:
        matrix = matrix.multiply(container.get_matrix())
        container = container.container
    return matrix


def _upright_box(
    box: tuple[float, float, float, float],
    turn: int,
    size: tuple[float, float],
) -> tuple[float, float, float, float]:
    """Return where a box is once its page is turned upright, as `_upright`.

    Args:
        box: The left, bottom, right and top edges of the box, from the lower
            left corner of the crop box.
        turn: The quarter turns, clockwise, of the text of the page.
        size: The width and the height of the crop box.

    Returns:
        The left, right, bottom and top edges of the box, from the lower left
        corner of the crop box turned so that the text of the page runs from
        left to right.
    """
    left, bottom, right, top = box
    width, height = size
    match turn:
        case 0:
            return left, right, bottom, top
        case 1:
            return height - top, height - bottom, left, right
        case 2:
            return width - right, width - left, height - top, height - bottom
        case _:
            return bottom, top, width - right, width - left


def _image_format(filters: Sequence[str]) -> str:
    """Return the format of an image, by the last filter of its stream.

    Args:
        filters: The filters of the stream, in the order they apply.

    Returns:
        The format of a known filter, else the printable name of the filter
        in lower case and without ``decode``, or ``raw`` without a filter.
    """
    if not filters:
        return _RAW_FORMAT
    last = filters[-1]
    if last in _FILTER_FORMATS:
        return _FILTER_FORMATS[last]
    name = last.lower().removesuffix("decode")
    return printable(name, limit=_FORMAT_LIMIT) or _RAW_FORMAT


def _dpi(matrix: pdfium.PdfMatrix, size: tuple[int, int]) -> float | None:
    """Return the resolution at which an image is shown on its page.

    The matrix maps the width of the pixels along its first axis, and their
    height along its second, whatever the turn of the image.

    Args:
        matrix: The matrix from the unit square of the image to the page.
        size: The width and the height of the image, in pixels.

    Returns:
        The lower of the horizontal and the vertical resolutions, in dots per
        inch, or ``None`` for an image shown without a width or a height.
    """
    width, height = size
    shown_width = math.hypot(matrix.a, matrix.b) / _POINTS_PER_INCH
    shown_height = math.hypot(matrix.c, matrix.d) / _POINTS_PER_INCH
    if not (shown_width > 0 and shown_height > 0):
        return None
    dpi = min(width / shown_width, height / shown_height)
    return dpi if math.isfinite(dpi) else None


@dataclass(frozen=True, slots=True)
class _PagePlace:
    """Which page of its document a page is.

    Attributes:
        page: Its position, from 0.
        label: Its label, as `PdfPageText.label` tells it.
    """

    page: int
    label: str


@dataclass(frozen=True, slots=True)
class _ImagePlace:
    """Where an image object is in its document.

    Attributes:
        page: Position of its page, from 0.
        position: Its position among the image objects of the page, from 0.
        label: The label of its page, as `PdfPageText.label` tells it.
    """

    page: int
    position: int
    label: str


def _image_ref(
    image: pdfium.PdfImage,
    matrix: pdfium.PdfMatrix,
    *,
    place: _ImagePlace,
    shown: tuple[float, float],
) -> ImageRef:
    """Read the signature of a raster image object, then drop its pixels.

    The digest reads the bytes of the stream as they are written, and its
    filters, then the mode, the size and the hashes of the pixels, which the
    entries of its dictionary, such as a palette or a decode array, change.
    The pixels are decoded without the mask of the image, once its stream is
    checked.

    Args:
        image: The image object.
        matrix: The matrix from the unit square of the image to the page.
        place: Where it is in its document.
        shown: The width and the height of its box, in points.

    Returns:
        The image, or why its pixels cannot be read. An image that cannot be
        read is named after its place, such as ``page iv image 1``: the label
        of its page, and its position counted from 1.
    """
    data, filters = object_stream(image)
    image_format = _image_format(filters)
    stream = data + _FILTER_SEPARATOR.join(filters).encode()
    page, position = place.page, place.position
    try:
        decoded = decode_object(image, data, filters)
    except UnreadableImage as error:
        return unreadable_ref(
            name=f"page {place.label} image {position + 1}",
            format=image_format,
            digest=content_digest(stream),
            problem=error.reason,
            shown=shown,
            size=len(data),
        )
    pixels = decoded.pixels
    ref = pixels_ref(
        pixels,
        format=image_format,
        digest="",
        source=PixelSource(page=page, index=position),
        size=len(data),
        quality=decoded.quality,
        shown=shown,
        dpi=_dpi(matrix, pixels.size),
    )
    look = (
        f"{_FILTER_SEPARATOR}{pixels.mode} {pixels.width}x{pixels.height}"
        f" {ref.dhash:016x} {ref.phash:016x}"
    )
    return replace(ref, digest=content_digest(stream + look.encode()))


def _covered(
    bounds: tuple[float, float, float, float],
    crop: tuple[float, float, float, float],
) -> float:
    """Return the area of the part of a box that is in the crop box.

    Args:
        bounds: The left, bottom, right and top edges of the box, on the page.
        crop: The left, bottom, right and top edges of the crop box.

    Returns:
        The area, in square points.
    """
    x0, y0, x1, y1 = bounds
    left, bottom, right, top = crop
    width = _overlap((x0, x1), (left, right))
    return width * _overlap((y0, y1), (bottom, top))


def _pdf_image(
    image: pdfium.PdfImage,
    matrix: pdfium.PdfMatrix,
    *,
    crop: tuple[float, float, float, float],
    turn: int,
    place: _ImagePlace,
) -> PdfImage:
    """Read a raster image object, and its box on the page turned upright.

    Args:
        image: The image object.
        matrix: The matrix from the unit square of the image to the page.
        crop: The left, bottom, right and top edges of the crop box.
        turn: The quarter turns, clockwise, of the text of the page.
        place: Where it is in its document.

    Returns:
        The image.
    """
    left, bottom, right, top = crop
    x0, y0, x1, y1 = matrix.on_rect(0, 0, 1, 1)
    box = _upright_box(
        (x0 - left, y0 - bottom, x1 - left, y1 - bottom),
        turn,
        (right - left, top - bottom),
    )
    ref = _image_ref(
        image,
        matrix,
        place=place,
        shown=(box[1] - box[0], box[3] - box[2]),
    )
    return PdfImage(ref, *box)


@dataclass(frozen=True, slots=True)
class _FigurePlace:
    """Where a vector figure is in its document.

    Attributes:
        page: Position of its page, from 0.
        rank: Its position among the figures of the page, from 0.
        label: The label of its page, as `PdfPageText.label` tells it.
    """

    page: int
    rank: int
    label: str


def _figure_ref(
    page: pdfium.PdfPage,
    figure: VectorFigure,
    place: _FigurePlace,
    guard: RenderGuard | None,
) -> ImageRef:
    """Render a vector figure for its signature, then drop the render.

    A figure is not rendered without a guard, past the first figures of its
    page, which gives it the problem `PAGE_LIMIT`; on a page that `/Rotate`
    turns; or over an image that the checks refuse. Its reference then has
    no pixels; it is named after its place, such as ``page iv figure 1``:
    the label of its page and its rank counted from 1; and its digest is
    that of its page and its box, so that it is never taken for the same
    image on another page.

    Args:
        page: The page.
        figure: The figure.
        place: Where it is in its document.
        guard: The checks of the image objects of the page, or ``None`` not
            to render the figure.

    Returns:
        The reference of the figure.
    """
    left, bottom, right, top = figure.box
    shown = (right - left, top - bottom)
    problem = PAGE_LIMIT if guard is None else NOT_RENDERED
    if guard is not None and page.get_rotation() == 0:
        try:
            render = render_figure(page, figure.box, guard)
        except UnreadableImage as error:
            problem = error.reason
        else:
            return pixels_ref(
                render,
                format=_VECTOR_FORMAT,
                digest=content_digest(render.tobytes()),
                source=PixelSource(page=place.page, box=figure.box),
                shown=shown,
                vector=True,
            )
    return unreadable_ref(
        name=f"page {place.label} figure {place.rank + 1}",
        format=_VECTOR_FORMAT,
        digest=content_digest(f"{place.page} {figure.box!r}".encode()),
        problem=problem,
        shown=shown,
        vector=True,
    )


def _figure_image(
    page: pdfium.PdfPage,
    figure: VectorFigure,
    *,
    place: _FigurePlace,
    turn: int,
    guard: RenderGuard | None,
) -> PdfImage:
    """Read a vector figure, and its box on the page turned upright.

    Args:
        page: The page.
        figure: The figure.
        place: Where it is in its document.
        turn: The quarter turns, clockwise, of the text of the page.
        guard: The checks of the image objects of the page, or ``None`` not
            to render the figure.

    Returns:
        The figure, as `_figure_ref` reads it.
    """
    left, bottom, right, top = figure.box
    crop_left, crop_bottom, crop_right, crop_top = page.get_bbox()
    box = _upright_box(
        (
            left - crop_left,
            bottom - crop_bottom,
            right - crop_left,
            top - crop_bottom,
        ),
        turn,
        (crop_right - crop_left, crop_top - crop_bottom),
    )
    return PdfImage(_figure_ref(page, figure, place, guard), *box)


def _figure_images(
    page: pdfium.PdfPage,
    place: _PagePlace,
    *,
    turn: int,
    figures: Sequence[VectorFigure],
) -> list[PdfImage]:
    """Read the vector figures of a page, rendering the first ones only.

    The first `MAX_PAGE_FIGURES` figures are rendered, each one over image
    objects checked once for the whole page.

    Args:
        page: The page.
        place: Which page it is.
        turn: The quarter turns, clockwise, of the text of the page.
        figures: The vector figures of the page.

    Returns:
        The figures, with their boxes on the page turned upright.
    """
    guard = RenderGuard(page)
    return [
        _figure_image(
            page,
            figure,
            place=_FigurePlace(place.page, rank, place.label),
            turn=turn,
            guard=guard if rank < MAX_PAGE_FIGURES else None,
        )
        for rank, figure in enumerate(figures)
    ]


def _page_images(
    page: pdfium.PdfPage,
    place: _PagePlace,
    *,
    turn: int,
    read: bool,
    figures: Sequence[VectorFigure],
) -> tuple[float, tuple[PdfImage, ...]]:
    """Find the share of a page that images cover, and read its images.

    The images are cut to the crop box for the share, and overlapping images
    count twice, so the share is capped at 1. The image objects of a form
    that is a vector figure count in the share, and keep their place among
    the image objects, but are not decoded: the figure is rendered instead.

    Args:
        page: The page.
        place: Which page it is.
        turn: The quarter turns, clockwise, of the text of the page.
        read: Whether the images and the figures are read, or only the share
            of the images counted.
        figures: The vector figures of the page.

    Returns:
        The share of the crop box that images cover, and the images then the
        figures, with their boxes on the page turned upright, if they are
        read.
    """
    crop = page.get_bbox()
    covered = 0.0
    images: list[PdfImage] = []
    in_figures = frozenset().union(*(f.image_objects for f in figures))
    for position, image in enumerate(image_objects(page)):
        matrix = _page_matrix(image)
        covered += _covered(matrix.on_rect(0, 0, 1, 1), crop)
        if read and ctypes.addressof(image.raw.contents) not in in_figures:
            images.append(
                _pdf_image(
                    image,
                    matrix,
                    crop=crop,
                    turn=turn,
                    place=_ImagePlace(place.page, position, place.label),
                ),
            )
    if read and figures:
        images += _figure_images(page, place, turn=turn, figures=figures)
    left, bottom, right, top = crop
    area = (right - left) * (top - bottom)
    coverage = min(1.0, covered / area) if area > 0 else 0.0
    return coverage, tuple(images)


def _page_label(pdf: pdfium.PdfDocument, index: int) -> str:
    """Return the printable label of a page, else its number from 1."""
    label = printable(pdf.get_page_label(index), limit=LABEL_LIMIT)
    return label or str(index + 1)


def _read_open_page(
    pdf: pdfium.PdfDocument,
    index: int,
    pages: tuple[pdfium.PdfPage, pdfium.PdfTextPage],
    *,
    images: bool,
) -> PdfPageText:
    """Read a page that is open, as `read_page` does.

    Args:
        pdf: The document of the page.
        index: Position of the page, from 0.
        pages: The page, and its text page.
        images: Whether the images and the figures are read.

    Returns:
        What `read_page` returns.
    """
    page, textpage = pages
    label = _page_label(pdf, index)
    left, bottom, right, top = page.get_bbox()
    figures = vector_figures(page, textpage)
    read = _read_chars(textpage, (left, bottom), _FigureText.of(figures))
    upright = _upright_page(read.chars, (right - left, top - bottom))
    coverage, found = _page_images(
        page,
        _PagePlace(index, label),
        turn=upright.turn,
        read=images,
        figures=figures,
    )
    return PdfPageText(
        index=index,
        label=label,
        width=upright.width,
        height=upright.height,
        chars=upright.chars,
        artifacts=read.artifacts,
        unmapped=upright.unmapped,
        image_coverage=coverage,
        angled=upright.angled,
        images=found,
        figure_chars=read.figure_chars,
        unrendered=max(0, len(figures) - MAX_PAGE_FIGURES) if images else 0,
    )


def read_page(
    pdf: pdfium.PdfDocument,
    index: int,
    *,
    images: bool = True,
) -> PdfPageText:
    """Read the characters, the raster images and the figures of a page.

    Each image is decoded once, for its signature, and its pixels are
    dropped. An image whose pixels cannot be read is kept, with the reason.
    The vector figures are always found, and the text drawn in them is left
    out of the characters; each figure is rendered once, for its signature,
    if the images are read.

    Args:
        pdf: The document, opened by `open_pdf`.
        index: Position of the page, from 0.
        images: Whether the images and the figures are read. Without them,
            none is decoded or rendered, but the share of the page that the
            images cover is still found, and the text of the figures still
            left out.

    Returns:
        The characters of the page turned upright, with its printable label,
        the size of its turned crop box, the share of it that images cover,
        its images and its figures.

    Raises:
        InputError: If PDFium cannot load the page or read its text.
    """
    try:
        with (
            closing(pdf[index]) as page,
            closing(page.get_textpage()) as textpage,
        ):
            return _read_open_page(
                pdf,
                index,
                (page, textpage),
                images=images,
            )
    except pdfium.PdfiumError:
        msg = f"page {index + 1}: invalid or damaged PDF"
        raise InputError(msg) from None


def read_bookmarks(pdf: pdfium.PdfDocument) -> list[Bookmark]:
    """Read the outline of a document, in the order of the document.

    Args:
        pdf: The document, opened by `open_pdf`.

    Returns:
        The entries of the outline, parents before their children.
    """
    bookmarks: list[Bookmark] = []
    for item in pdf.get_toc():
        dest = item.get_dest()
        page = None if dest is None else dest.get_index()
        bookmarks.append(Bookmark(item.level, item.get_title(), page))
    return bookmarks


def _metadata(pdf: pdfium.PdfDocument, key: str) -> str | None:
    """Return a printable metadata value, if not empty."""
    return printable(pdf.get_metadata_value(key), limit=METADATA_LIMIT) or None


def read_metadata(pdf: pdfium.PdfDocument) -> tuple[str | None, str | None]:
    """Read the title and the generator of a document.

    Args:
        pdf: The document, opened by `open_pdf`.

    Returns:
        The title, and the generator, which is the producer or else the
        creator. Each is printable and has at most `METADATA_LIMIT`
        characters, and is ``None`` when the document has none.
    """
    generator = _metadata(pdf, "Producer") or _metadata(pdf, "Creator")
    return _metadata(pdf, "Title"), generator
