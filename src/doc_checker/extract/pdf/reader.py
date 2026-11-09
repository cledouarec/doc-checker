# Copyright 2026 Christophe Le Douarec
"""Read the characters, images, links, bookmarks and metadata of PDFs.

PDFium does the reading. Each character keeps its position and its style, so
that the next steps can rebuild lines, blocks and headings. A page is read
upright, turned so that the baselines of most of its characters run from left
to right, and its characters whose baseline runs at another angle are left
out. Its raster images keep their box on the upright page and their
signature, their pixels being dropped once read. Its vector figures are found
whether images are read or not, and the text drawn in them is left out and
counted; when images are read, each figure is rendered for its signature, and
the render dropped. Its link annotations are read on demand, with the text
under them and where they lead. Pages are read one at a time, so that the
caller never holds the characters of a whole document.
"""

import ctypes
import logging
import math
import ntpath
import posixpath
import re
import sys
from collections import Counter
from contextlib import closing
from dataclasses import dataclass, replace
from functools import partial
from itertools import islice
from pathlib import PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Final, Literal

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
from doc_checker.model.facts import PdfLink
from doc_checker.text.normalize import printable

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from doc_checker.extract.pdf.figures import VectorFigure
    from doc_checker.model.document import ImageRef
    from doc_checker.model.facts import LinkAction

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
MAX_PAGE_LINKS: Final = 10_000  # Most link annotations read on a page.
LINK_TEXT_LIMIT: Final = 200  # Most characters of the text of a link.
# Most bytes of the URI or the file path of a link that are read: a longer
# one is left unread, as no file path is that long, so that a string that
# many links share cannot make the reading copy it again and again.
LINK_TARGET_LIMIT: Final = 2_048
# Most characters of a page that the texts of its links examine: PDFium tests
# every character of the page for the text of each link, so the links after
# the first `LINK_TEXT_WORK` divided by the characters of the page keep no
# text.
LINK_TEXT_WORK: Final = 20_000_000
MAX_BOOKMARKS: Final = 100_000  # Most bookmarks read in an outline.
# Most bytes of the titles of an outline that PDFium is asked for: it decodes
# a title whole to tell its size, kept or not, so that one long title that
# many bookmarks share would cost its size again for each of them.
OUTLINE_TITLE_WORK: Final = 16_000_000
# Most bytes, in UTF-16, of a string of the document that is read: a longer
# one is left unread, as an empty string, so that a string that many objects
# share cannot make the reading copy it again and again.
BOOKMARK_TITLE_BYTES: Final = 4_096  # Of the title of a bookmark.
METADATA_BYTES: Final = 4_096  # Of a value of the document information.
LABEL_BYTES: Final = 1_024  # Of the label of a page.
# Most bytes of the page labels of a document that PDFium is asked for: it
# builds a label whole to tell its size, kept or not, so that one long prefix
# that many pages share would cost its size again for each of them.
PAGE_LABEL_WORK: Final = 16_000_000

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
# The text of a link is read up to these UTF-16 units, enough for
# `LINK_TEXT_LIMIT` characters once the runs of blanks are reduced.
_LINK_TEXT_UNITS: Final = 4 * LINK_TEXT_LIMIT
# A URL scheme, which a file path of a link may start with: two letters at
# least, so that a drive letter is not one.
_SCHEME: Final = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]+:")
_ROOTS: Final = ("/", "\\")  # What a path from a root starts with.
# The parent folder, which a normalized path out of its folder starts with,
# as its whole first component.
_PARENT: Final = ".."
_PARENT_PREFIXES: Final = ("../", "..\\")
# The platforms on which PDFium gives a file path in a form of its own, as
# `sys.platform` names them.
_WINDOWS: Final = "win32"
_MACOS: Final = "darwin"
_UTF8: Final = "utf-8"
_UTF16: Final = "utf-16-le"
_NUL_BYTES: Final = {_UTF8: 1, _UTF16: 2}  # Bytes of a final NUL, by encoding.
# The actions whose destination is a page of another file.
_OTHER_FILE_ACTIONS: Final = frozenset(
    {pdfium_c.PDFACTION_REMOTEGOTO, pdfium_c.PDFACTION_EMBEDDEDGOTO},
)

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
# A call to PDFium that copies a string into a buffer of a given size in
# bytes, if there is one, and returns the size of the string.
type _StringReader = Callable[[ctypes.Array[ctypes.c_char] | None, int], int]
# Where a link leads: the kind of its action, its target, and whether it is
# valid, as `read_links` tells them.
type LinkTarget = tuple[LinkAction, str, bool | None]
# The targets of the link annotations of a document already resolved, by
# address of their dictionary.
type LinkTargets = dict[int, LinkTarget]
# Why the walk of an outline stopped before its end: `MAX_BOOKMARKS`, or
# `OUTLINE_TITLE_WORK`; None if it read the whole outline.
type OutlineCut = Literal["bookmarks", "titles"] | None

# pypdfium2 logs the cycles and the depth cuts of an outline, which are no
# concern of the user of a command line: its messages are dropped.
_PYPDFIUM_LOGGER: Final = logging.getLogger("pypdfium2")
_PYPDFIUM_LOGGER.addHandler(logging.NullHandler())
_PYPDFIUM_LOGGER.propagate = False


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
        links: The link annotations of the page, in the order of PDFium, as
            `read_links` reads them, if they were read.
        more_links: Whether other link annotations follow the last one read.
        untexted_links: Number of these links whose text was not read, as
            they come after the work that `LINK_TEXT_WORK` allows.
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
    links: tuple[PdfLink, ...] = ()
    more_links: bool = False
    untexted_links: int = 0


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


def _link_text(
    textpage: pdfium.PdfTextPage,
    link: pdfium_c.FPDF_LINK,
    buffer: ctypes.Array[ctypes.c_ushort],
) -> str:
    """Return the text under a link annotation.

    Args:
        textpage: The text page of the page of the link.
        link: The link annotation.
        buffer: Room for the first UTF-16 units of the text, which this
            overwrites.

    Returns:
        The printable text in the box of the annotation, its runs of blanks
        reduced to one space, and cut at `LINK_TEXT_LIMIT` characters; or
        ``""`` if the box cannot be read.
    """
    rect = pdfium_c.FS_RECTF()
    if not pdfium_c.FPDFLink_GetAnnotRect(link, rect):
        return ""
    left, right = sorted((rect.left, rect.right))
    bottom, top = sorted((rect.bottom, rect.top))
    units = pdfium_c.FPDFText_GetBoundedText(
        textpage.raw,
        left,
        top,
        right,
        bottom,
        buffer,
        len(buffer),
    )
    data = bytes(buffer)[: 2 * max(0, units)]
    text = data.decode("utf-16-le", errors="replace")
    return printable(text, limit=LINK_TEXT_LIMIT)


def _goto(pdf: pdfium.PdfDocument, dest: pdfium_c.FPDF_DEST) -> LinkTarget:
    """Return the page of the document that a destination leads to.

    Args:
        pdf: The document.
        dest: The destination, which is not null.

    Returns:
        ``goto``, the page as ``page N`` with N from 1, and whether the
        document has that page.
    """
    index = pdfium_c.FPDFDest_GetDestPageIndex(pdf.raw, dest)
    return "goto", f"page {index + 1}", 0 <= index < len(pdf)


def _sized_string(
    read: _StringReader,
    *,
    limit: int,
    encoding: str,
) -> tuple[str, int]:
    """Read a string that PDFium copies, such as a URI or a title.

    PDFium copies a string whole or not at all, so the size is asked first,
    and a string over the limit is never copied.

    Args:
        read: Calls PDFium with a buffer and its size in bytes, and returns
            the size of the string with its final NUL; without a buffer, it
            only returns that size.
        limit: The most bytes of the string to read, its NUL left out.
        encoding: The encoding of the string: ``utf-8`` or ``utf-16-le``.

    Returns:
        The string, each byte or half character that cannot be decoded
        replaced by U+FFFD, or ``""`` if it is empty, cannot be read, or has
        more bytes than the limit; and the size that PDFium told, in bytes
        with the NUL.
    """
    nul = _NUL_BYTES[encoding]
    size = read(None, 0)
    if not nul < size <= limit + nul:
        return "", size
    buffer = ctypes.create_string_buffer(size)
    if read(buffer, size) != size:
        return "", size
    return buffer.raw[: size - nul].decode(encoding, errors="replace"), size


def _pdfium_string(read: _StringReader, *, limit: int, encoding: str) -> str:
    """Read a string that PDFium copies, as `_sized_string` does.

    Args:
        read: Calls PDFium, as `_sized_string` takes it.
        limit: The most bytes of the string to read, its NUL left out.
        encoding: The encoding of the string: ``utf-8`` or ``utf-16-le``.

    Returns:
        The string, as `_sized_string` returns it.
    """
    return _sized_string(read, limit=limit, encoding=encoding)[0]


def _file_exists(folder: Path, path: str) -> bool | None:
    """Tell whether the file that a link opens exists, if it is checked.

    Only a path relative to the folder of the PDF and that stays in it is
    checked, so that a document can neither probe the files of the machine
    nor make a check go through the network: a path with a URL scheme, a
    drive or a root, as POSIX or Windows tell them, or that leads out of the
    folder once ``..`` is resolved, as either tells it, is not. A name that
    only starts with two dots, such as ``..notes.pdf``, stays in the folder.

    Args:
        folder: The folder of the PDF.
        path: The path of the file, with slashes.

    Returns:
        Whether the file exists, or ``None`` for an empty path, a path that
        is not checked, or a path that cannot be checked.
    """
    if not path or _SCHEME.match(path) or path.startswith(_ROOTS):
        return None
    # A Windows anchor is a drive, a root, or both, as an absolute path has.
    if PurePosixPath(path).is_absolute() or PureWindowsPath(path).anchor:
        return None
    for normalized in (posixpath.normpath(path), ntpath.normpath(path)):
        if normalized == _PARENT or normalized.startswith(_PARENT_PREFIXES):
            return None
    try:
        return (folder / path).exists()
    except OSError, ValueError:
        return None


def pdf_file_path(raw: str, platform: str = sys.platform) -> str:
    r"""Undo the form that PDFium gives the path of a file on its platform.

    PDFium writes the file specification of an action as a path of the
    platform it runs on. On Windows, its slashes become backslashes, a
    leading pair of slashes loses one, a leading ``/X/`` becomes ``X:\``,
    and another leading slash gains a backslash before it. On macOS, its
    slashes become colons, and a leading ``/Mac`` loses its slash.
    Elsewhere, the path is left as it is.

    Args:
        raw: The path as PDFium gives it.
        platform: The platform PDFium runs on, as `sys.platform` names it.

    Returns:
        The path with slashes. On Windows, a leading pair of backslashes
        becomes one slash, a single leading backslash two slashes, and each
        other backslash a slash, so that ``/X/`` comes back as ``X:/``. On
        macOS, each colon becomes a slash. Elsewhere, the path as it is.
    """
    if platform == _WINDOWS:
        if raw.startswith("\\\\"):
            raw = "/" + raw[2:]
        elif raw.startswith("\\"):
            raw = "//" + raw[1:]
        return raw.replace("\\", "/")
    if platform == _MACOS:
        return raw.replace(":", "/")
    return raw


def file_link_target(
    raw: str,
    folder: Path,
    platform: str = sys.platform,
) -> tuple[str, bool | None]:
    """Return the path that a GoToR or a Launch action opens, and its check.

    Args:
        raw: The path as PDFium gives it, as `pdf_file_path` takes it.
        folder: The folder of the PDF, from which a relative path is
            checked.
        platform: The platform PDFium runs on, as `sys.platform` names it.

    Returns:
        The printable path with slashes, and whether its file exists, as
        `_file_exists` tells. On macOS, a path that PDFium gives with a
        colon is never checked, as a folder cannot be told there from a URL
        scheme.
    """
    path = pdf_file_path(raw, platform)
    if platform == _MACOS and ":" in raw:
        return printable(path), None
    return printable(path), _file_exists(folder, path)


def _link_target(
    pdf: pdfium.PdfDocument,
    link: pdfium_c.FPDF_LINK,
    folder: Path,
) -> LinkTarget:
    """Tell where a link annotation leads, and whether it can go there.

    The destination of a link, or else of its GoTo action, is a page of the
    document. PDFium also gives the destination of a GoToR or a GoToE
    action, which is a page of another file: a link with such an action is
    taken for its action.

    Args:
        pdf: The document.
        link: The link annotation.
        folder: The folder of the PDF, from which a relative file path is
            checked.

    Returns:
        The kind of action, its target and whether it is valid: ``goto``,
        the page as ``page N`` and whether the document has it, or ``""``
        and false for a destination that leads nowhere; ``uri``, the URI and
        ``None``; ``file``, the path of a GoToR or a Launch action and
        whether the file exists, as `file_link_target` tells; ``none``,
        ``""`` and false for a link without destination or action;
        ``other``, ``""`` and ``None`` for another action. A URI or a path
        of more than `LINK_TARGET_LIMIT` bytes is left unread, as ``""``.
    """
    action = pdfium_c.FPDFLink_GetAction(link)
    kind = pdfium_c.FPDFAction_GetType(action) if action else None
    if kind not in _OTHER_FILE_ACTIONS:
        dest = pdfium_c.FPDFLink_GetDest(pdf.raw, link)
        if dest:
            return _goto(pdf, dest)
    match kind:
        case None:
            return "none", "", False
        case pdfium_c.PDFACTION_GOTO:
            dest = pdfium_c.FPDFAction_GetDest(pdf.raw, action)
            return _goto(pdf, dest) if dest else ("goto", "", False)
        case pdfium_c.PDFACTION_URI:
            uri = _pdfium_string(
                partial(pdfium_c.FPDFAction_GetURIPath, pdf.raw, action),
                limit=LINK_TARGET_LIMIT,
                encoding=_UTF8,
            )
            return "uri", printable(uri), None
        case pdfium_c.PDFACTION_REMOTEGOTO | pdfium_c.PDFACTION_LAUNCH:
            path = _pdfium_string(
                partial(pdfium_c.FPDFAction_GetFilePath, action),
                limit=LINK_TARGET_LIMIT,
                encoding=_UTF8,
            )
            return "file", *file_link_target(path, folder)
        case _:
            return "other", "", None


def _resolved_target(
    pdf: pdfium.PdfDocument,
    link: pdfium_c.FPDF_LINK,
    folder: Path,
    targets: LinkTargets,
) -> LinkTarget:
    """Return where a link annotation leads, resolving it once only.

    An annotation that many references share, on one page or on many, is
    one dictionary of the document, so its address is its identity while the
    document is open.

    Args:
        pdf: The document.
        link: The link annotation.
        folder: The folder of the PDF.
        targets: The targets already resolved, which this fills.

    Returns:
        Its target, as `_link_target` tells it, shared by every reference.
    """
    address = ctypes.addressof(link.contents)
    target = targets.get(address)
    if target is None:
        target = targets[address] = _link_target(pdf, link, folder)
    return target


def _text_quota(textpage: pdfium.PdfTextPage) -> int:
    """Return how many links of a page get their text.

    Args:
        textpage: The text page of the page, open.

    Returns:
        `LINK_TEXT_WORK`, as the module holds it when called, divided by the
        number of characters of the page, as each text examines them all.
    """
    chars = pdfium_c.FPDFText_CountChars(textpage.raw)
    return LINK_TEXT_WORK // max(1, chars)


def read_links(  # ruff: ignore[too-many-arguments] - one keyword per bound and memory of the reading
    pdf: pdfium.PdfDocument,
    page: pdfium.PdfPage,
    textpage: pdfium.PdfTextPage,
    folder: Path,
    *,
    limit: int | None = None,
    targets: LinkTargets | None = None,
) -> tuple[tuple[PdfLink, ...], bool]:
    """Read the link annotations of a page, with their text and target.

    The annotations are read in the order of PDFium, the first
    `MAX_PAGE_LINKS` only, as the module holds it when called. The text of
    the first ones only is read, as `_text_quota` tells, and the others keep
    an empty text.

    Args:
        pdf: The document, opened by `open_pdf`.
        page: The page, open.
        textpage: Its text page, open.
        folder: The folder of the PDF, from which the relative path of a
            file that a link opens is checked.
        limit: The most annotations to read, if fewer than `MAX_PAGE_LINKS`.
        targets: The targets of the annotations of the document already
            resolved, which this fills, so that an annotation that many
            references share is resolved once. Without it, they are shared
            on the page only.

    Returns:
        The links, each with the printable text in its box, cut at
        `LINK_TEXT_LIMIT` characters, and where it leads, as `_link_target`
        tells; and whether other link annotations follow the last one read.
    """
    most = MAX_PAGE_LINKS if limit is None else min(limit, MAX_PAGE_LINKS)
    texts = _text_quota(textpage)
    resolved: LinkTargets = {} if targets is None else targets
    buffer = (ctypes.c_ushort * _LINK_TEXT_UNITS)()
    position = ctypes.c_int(0)
    link = pdfium_c.FPDF_LINK()
    links: list[PdfLink] = []
    for number in range(max(0, most)):
        if not pdfium_c.FPDFLink_Enumerate(page.raw, position, link):
            return tuple(links), False
        text = _link_text(textpage, link, buffer) if number < texts else ""
        target = _resolved_target(pdf, link, folder, resolved)
        links.append(PdfLink(text, *target))
    more = bool(pdfium_c.FPDFLink_Enumerate(page.raw, position, link))
    return tuple(links), more


def _sized_label(pdf: pdfium.PdfDocument, index: int) -> tuple[str, int]:
    """Read the label of a page, and the size that PDFium told for it.

    A label of more than `LABEL_BYTES` bytes is left unread.

    Args:
        pdf: The document.
        index: Position of the page, from 0.

    Returns:
        The printable label of the page, cut at `LABEL_LIMIT` characters,
        else its number from 1; and the size of the label in bytes, with its
        NUL, as PDFium told it.
    """
    label, size = _sized_string(
        partial(pdfium_c.FPDF_GetPageLabel, pdf.raw, index),
        limit=LABEL_BYTES,
        encoding=_UTF16,
    )
    return printable(label, limit=LABEL_LIMIT) or str(index + 1), size


def read_labels(pdf: pdfium.PdfDocument) -> tuple[list[str], bool]:
    """Read the label of each page of a document.

    The labels are read until the sizes asked, with their NUL, reach
    `PAGE_LABEL_WORK` bytes, as the module holds it when called; each page
    after them takes its number, as a page without label does.

    Args:
        pdf: The document, opened by `open_pdf`.

    Returns:
        The label of each page, in order, as `read_page` reads it, and
        whether pages were numbered for the limit.
    """
    labels: list[str] = []
    spent = 0  # The bytes of the labels asked.
    cut = False
    for index in range(len(pdf)):
        if spent >= PAGE_LABEL_WORK:
            cut = True
            labels.append(str(index + 1))
            continue
        label, size = _sized_label(pdf, index)
        spent += size
        labels.append(label)
    return labels, cut


def _read_open_page(
    index: int,
    pages: tuple[pdfium.PdfPage, pdfium.PdfTextPage],
    *,
    images: bool,
    label: str,
) -> PdfPageText:
    """Read a page that is open, as `read_page` does.

    Args:
        index: Position of the page, from 0.
        pages: The page, and its text page.
        images: Whether the images and the figures are read.
        label: The label of the page.

    Returns:
        What `read_page` returns.
    """
    page, textpage = pages
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


def read_page(  # ruff: ignore[too-many-arguments] - one keyword per part of the page that is read
    pdf: pdfium.PdfDocument,
    index: int,
    *,
    images: bool = True,
    link_folder: Path | None = None,
    link_limit: int | None = None,
    link_targets: LinkTargets | None = None,
    label: str | None = None,
) -> PdfPageText:
    """Read the characters, the raster images and the figures of a page.

    Each image is decoded once, for its signature, and its pixels are
    dropped. An image whose pixels cannot be read is kept, with the reason.
    The vector figures are always found, and the text drawn in them is left
    out of the characters; each figure is rendered once, for its signature,
    if the images are read. The link annotations are read, as `read_links`
    reads them, only with a folder to check their files from.

    Args:
        pdf: The document, opened by `open_pdf`.
        index: Position of the page, from 0.
        images: Whether the images and the figures are read. Without them,
            none is decoded or rendered, but the share of the page that the
            images cover is still found, and the text of the figures still
            left out.
        link_folder: The folder of the PDF, to read the links of the page,
            or ``None`` not to read them.
        link_limit: The most links to read, if fewer than `MAX_PAGE_LINKS`.
        link_targets: The targets of the link annotations of the document
            already resolved, as `read_links` takes them.
        label: The label of the page, as `read_labels` reads it, or
            ``None`` to read it here.

    Returns:
        The characters of the page turned upright, with its printable label,
        the size of its turned crop box, the share of it that images cover,
        its images, its figures and its links.

    Raises:
        InputError: If PDFium cannot load the page or read its text.
    """
    try:
        with (
            closing(pdf[index]) as page,
            closing(page.get_textpage()) as textpage,
        ):
            text = _read_open_page(
                index,
                (page, textpage),
                images=images,
                label=_sized_label(pdf, index)[0] if label is None else label,
            )
            if link_folder is None:
                return text
            links, more = read_links(
                pdf,
                page,
                textpage,
                link_folder,
                limit=link_limit,
                targets=link_targets,
            )
            return replace(
                text,
                links=links,
                more_links=more,
                untexted_links=max(0, len(links) - _text_quota(textpage)),
            )
    except pdfium.PdfiumError:
        msg = f"page {index + 1}: invalid or damaged PDF"
        raise InputError(msg) from None


def read_bookmarks(
    pdf: pdfium.PdfDocument,
) -> tuple[list[Bookmark], OutlineCut]:
    """Read the outline of a document, in the order of the document.

    The walk of pypdfium2 stops at a cycle, and at a depth of 15. It is cut
    after the first `MAX_BOOKMARKS` bookmarks, or once the sizes of the
    titles asked, with their NUL, reach `OUTLINE_TITLE_WORK` bytes, each
    limit as the module holds it when called. A title of more than
    `BOOKMARK_TITLE_BYTES` bytes is left unread, as ``""``, but its size
    counts.

    Args:
        pdf: The document, opened by `open_pdf`.

    Returns:
        The entries of the outline, parents before their children, and why
        bookmarks after them were left unread, if they were.
    """
    bookmarks: list[Bookmark] = []
    spent = 0  # The bytes of the titles asked.
    for item in islice(pdf.get_toc(), MAX_BOOKMARKS + 1):
        if len(bookmarks) == MAX_BOOKMARKS:
            return bookmarks, "bookmarks"
        if spent >= OUTLINE_TITLE_WORK:
            return bookmarks, "titles"
        title, size = _sized_string(
            partial(pdfium_c.FPDFBookmark_GetTitle, item.raw),
            limit=BOOKMARK_TITLE_BYTES,
            encoding=_UTF16,
        )
        spent += size
        dest = item.get_dest()
        page = None if dest is None else dest.get_index()
        bookmarks.append(Bookmark(item.level, title, page))
    return bookmarks, None


def _metadata(pdf: pdfium.PdfDocument, key: str) -> str | None:
    """Return a printable metadata value, if not empty.

    A value of more than `METADATA_BYTES` bytes is left unread.
    """
    value = _pdfium_string(
        partial(pdfium_c.FPDF_GetMetaText, pdf.raw, f"{key}\x00".encode()),
        limit=METADATA_BYTES,
        encoding=_UTF16,
    )
    return printable(value, limit=METADATA_LIMIT) or None


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
