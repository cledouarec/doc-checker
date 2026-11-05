# Copyright 2026 Christophe Le Douarec
"""Read the characters, bookmarks and metadata of PDF documents.

PDFium does the reading. Each character keeps its position and its style, so
that the next steps can rebuild lines, blocks and headings. A page is read
upright, turned so that the baselines of most of its characters run from left
to right, and its characters whose baseline runs at another angle are left
out. Pages are read one at a time, so that the caller never holds the
characters of a whole document.
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
from doc_checker.text.normalize import printable

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

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

_SUBSET_PREFIX: Final = re.compile(r"^[A-Z]{6}\+")
_LINE_BREAKS: Final = frozenset({0x0A, 0x0D})
_HYPHEN_CODE: Final = 0x02  # PDFium reports a line end hyphen as U+0002.
_REPLACEMENT: Final = "\U0000fffd"
_HIGH_SURROGATES: Final = range(0xD800, 0xDC00)
_LOW_SURROGATES: Final = range(0xDC00, 0xE000)
_ASTRAL_START: Final = 0x10000  # The first character beyond U+FFFF.
_HALF_BITS: Final = 10  # The bits of a character that a surrogate half holds.
_ARTIFACT: Final = "Artifact"
_IMAGE_DEPTH: Final = 2  # The images of the page and those of its forms.
_FULL_TURN: Final = 360.0  # Degrees in a full turn.
_QUARTER_TURN: Final = 90.0  # Degrees in a quarter turn.
_QUARTERS: Final = 4  # Quarter turns in a full turn.
_FLIPPED: Final = 2  # Quarter turns that a negative font size adds.

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
    cache: dict[int, _ObjectInfo],
) -> _ObjectInfo | None:
    """Return the facts about the text object of a character.

    The first character of an object reads them, and the others get them from
    the cache.

    Args:
        textpage: The text page of the character.
        index: Index of the character.
        cache: Facts already read, by address of text object.

    Returns:
        The facts, or ``None`` for a character without a text object, such as
        the space that PDFium generates between two objects.
    """
    obj = pdfium_c.FPDFText_GetTextObject(textpage, index)
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


def _read_chars(
    textpage: pdfium.PdfTextPage,
    corner: tuple[float, float],
) -> tuple[list[_Pending], int]:
    """Read the characters of a text page, but not the line breaks.

    The characters of artifacts, such as running headers and footers, are
    dropped. The others are read as they lie on the page, with the quarter
    turns of their text, before the page is turned upright. A character
    generated with no kept character before it is dropped too.

    Args:
        textpage: The text page to read.
        corner: The lower left corner of the crop box, which is the origin of
            the coordinates.

    Returns:
        The kept characters, and the number of artifact characters dropped.
    """
    chars: list[_Pending] = []
    cache: dict[int, _ObjectInfo] = {}
    artifacts = 0
    previous: _Look | None = None
    for index in range(textpage.count_chars()):
        code = pdfium_c.FPDFText_GetUnicode(textpage, index)
        if code in _LINE_BREAKS:
            continue
        info = _object_info(textpage, index, cache)
        if info is not None and info.artifact:
            artifacts += 1
            continue
        look = _look(textpage, index, info, previous)
        if look is None:
            continue
        previous = look
        text, no_mapping = _text_at(textpage, index, code)
        box = _char_box(textpage, index, corner)
        chars.append((text, box, *look, no_mapping, info is None))
    return chars, artifacts


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
    """

    chars: tuple[PdfChar, ...]
    width: float
    height: float
    unmapped: int
    angled: int


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
    )


def _overlap(first: tuple[float, float], second: tuple[float, float]) -> float:
    """Return the length of the overlap of two segments."""
    return max(0.0, min(first[1], second[1]) - max(first[0], second[0]))


def _image_coverage(
    page: pdfium.PdfPage,
    box: tuple[float, float, float, float],
) -> float:
    """Return the share of a page that images cover, from 0 to 1.

    The images are cut to the crop box, and overlapping images count twice, so
    the share is capped at 1.

    Args:
        page: The page.
        box: The left, bottom, right and top edges of its crop box.

    Returns:
        The share of the crop box that images cover.
    """
    left, bottom, right, top = box
    area = (right - left) * (top - bottom)
    if area <= 0:
        return 0.0
    covered = 0.0
    for image in page.get_objects(
        filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE],
        max_depth=_IMAGE_DEPTH,
    ):
        x0, y0, x1, y1 = image.get_bounds()
        shared_width = _overlap((x0, x1), (left, right))
        shared_height = _overlap((y0, y1), (bottom, top))
        covered += shared_width * shared_height
    return min(1.0, covered / area)


def _page_label(pdf: pdfium.PdfDocument, index: int) -> str:
    """Return the printable label of a page, else its number from 1."""
    label = printable(pdf.get_page_label(index), limit=LABEL_LIMIT)
    return label or str(index + 1)


def read_page(pdf: pdfium.PdfDocument, index: int) -> PdfPageText:
    """Read the characters of a page.

    Args:
        pdf: The document, opened by `open_pdf`.
        index: Position of the page, from 0.

    Returns:
        The characters of the page turned upright, with its printable label,
        the size of its turned crop box and the share of it that images cover.

    Raises:
        InputError: If PDFium cannot load the page or read its text.
    """
    try:
        with (
            closing(pdf[index]) as page,
            closing(page.get_textpage()) as textpage,
        ):
            box = page.get_bbox()
            left, bottom, right, top = box
            chars, artifacts = _read_chars(textpage, (left, bottom))
            upright = _upright_page(chars, (right - left, top - bottom))
            return PdfPageText(
                index=index,
                label=_page_label(pdf, index),
                width=upright.width,
                height=upright.height,
                chars=upright.chars,
                artifacts=artifacts,
                unmapped=upright.unmapped,
                image_coverage=_image_coverage(page, box),
                angled=upright.angled,
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
