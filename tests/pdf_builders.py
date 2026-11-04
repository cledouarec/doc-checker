# Copyright 2026 Christophe Le Douarec
"""Builders of test PDF documents, made with reportlab and fpdf2.

No PDF is versioned: every test PDF is built here, in a temporary directory,
with the base fonts (Helvetica, Times and Courier) encoded in cp1252 and never
embedded. fpdf2 is LGPL licensed, so it is only used by these test builders.

The module also makes the characters and the pages that the PDF reader returns,
the line segments that are built from them and the lines that these make,
without any PDF, for the tests of the steps that work on them.
"""

import html
import io
import re
from itertools import chain, pairwise
from typing import TYPE_CHECKING, override

from fpdf import FPDF
from fpdf.enums import XPos, YPos
from fpdf.fonts import TextStyle
from PIL import Image as PILImage
from PIL import ImageDraw
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    Image,
    ListFlowable,
    ListItem,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Table,
)
from reportlab.platypus.tableofcontents import TableOfContents

from doc_checker.extract.pdf.layout import Line
from doc_checker.extract.pdf.reader import PdfChar, PdfPageText
from doc_checker.extract.pdf.segments import Segment
from tests.builders import DocSpec, Section, smart_typography

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

_SHY = "\xad"
_LONG_WORD = re.compile(r"[^\W\d_]{9,}")


def _hyphenate_word(match: re.Match[str]) -> str:
    word = match[0]
    # Cut every three letters, but keep at least three in the last piece.
    bounds = [0, *range(3, len(word) - 3, 3), len(word)]
    return _SHY.join(word[start:end] for start, end in pairwise(bounds))


def soft_hyphenate(text: str) -> str:
    """Insert a soft hyphen every three letters inside long words.

    Only the words of nine letters or more are cut, and the last piece keeps
    at least three letters.

    Args:
        text: The text to hyphenate.

    Returns:
        The text with a soft hyphen inside each long word.
    """
    return _LONG_WORD.sub(_hyphenate_word, text)


def pdf_typography(text: str, lang: str) -> str:
    """Apply the typography of a PDF generator to a text.

    This is the HTML typography, except that the narrow no-break spaces, which
    cp1252 cannot encode, become no-break spaces.

    Args:
        text: The text to typeset.
        lang: The language of the text, ``"en"`` or ``"fr"``.

    Returns:
        The typeset text.
    """
    return smart_typography(text, lang).replace("\U0000202f", "\xa0")


def scan_image() -> PILImage.Image:
    """Return the image of a scanned page: black lines on a white sheet."""
    image = PILImage.new("RGB", (400, 560), "white")
    draw = ImageDraw.Draw(image)
    for y in range(20, 540, 16):
        draw.line((20, y, 380, y), fill="black", width=3)
    return image


_W, _H = A4  # The page size, in points.
_PAGE_MARGIN = 25 * mm
_TOC_TITLES = {"en": "Contents", "fr": "Table des matières"}

_H1 = ParagraphStyle(
    "h1",
    fontName="Helvetica-Bold",
    fontSize=20,
    leading=24,
    spaceAfter=12,
)
_H2 = ParagraphStyle(
    "h2",
    fontName="Helvetica-Bold",
    fontSize=14,
    leading=18,
    spaceBefore=8,
    spaceAfter=6,
)
_BODY = ParagraphStyle(
    "body",
    fontName="Times-Roman",
    fontSize=11,
    leading=14,
    spaceAfter=7,
    alignment=TA_JUSTIFY,
    embeddedHyphenation=1,
    uriWasteReduce=0.3,
)
_CODE = ParagraphStyle(
    "code",
    fontName="Courier",
    fontSize=9,
    leading=11,
    spaceBefore=4,
    spaceAfter=8,
)
_NOTE = ParagraphStyle(
    "note",
    fontName="Times-Bold",
    fontSize=11,
    leading=14,
)


class _Heading(Paragraph):
    """A section heading, which carries its outline level."""

    def __init__(self, text: str, style: ParagraphStyle, level: int) -> None:
        super().__init__(text, style)
        self.level = level


class _Document(BaseDocTemplate):
    """An A4 document with a running header and footer on each page."""

    def __init__(
        self,
        path: Path,
        spec: DocSpec,
        *,
        artifacts: bool,
        outline: bool,
        toc: bool,
    ) -> None:
        super().__init__(
            str(path),
            pagesize=A4,
            title=spec.title,
            lang=spec.lang,
            leftMargin=_PAGE_MARGIN,
            rightMargin=_PAGE_MARGIN,
            topMargin=_PAGE_MARGIN,
            bottomMargin=_PAGE_MARGIN,
        )
        self.spec = spec
        self.artifacts = artifacts
        self.outline = outline
        self.toc = toc
        self.chapter = ""
        body = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
        )
        scan = Frame(
            0,
            0,
            _W,
            _H,
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        self.addPageTemplates(
            [
                PageTemplate("page", [body], onPage=self._draw_page),
                PageTemplate("scan", [scan]),
            ],
        )

    def _draw_page(self, canvas: Canvas, doc: BaseDocTemplate) -> None:
        if self.toc and doc.page == 1:
            canvas.addPageLabel(0, style="ROMAN_LOWER", start=1)
            canvas.addPageLabel(1, style="ARABIC", start=1)
        canvas.saveState()
        if self.artifacts:
            canvas.addLiteral("/Artifact BMC")
        canvas.setFont("Helvetica", 9)
        canvas.drawString(_PAGE_MARGIN, _H - 15 * mm, self.spec.title)
        canvas.drawRightString(_W - _PAGE_MARGIN, _H - 15 * mm, self.chapter)
        canvas.drawString(_PAGE_MARGIN, 12 * mm, "Draft")
        canvas.drawCentredString(_W / 2, 12 * mm, f"Page {doc.page}")
        if self.artifacts:
            canvas.addLiteral("EMC")
        canvas.restoreState()

    @override
    def beforeDocument(self) -> None:
        # multiBuild makes several passes over the same document.
        self.chapter = ""

    @override
    def afterFlowable(self, flowable: Flowable) -> None:
        if not isinstance(flowable, _Heading):
            return
        text = flowable.getPlainText()
        if flowable.level == 0:
            self.chapter = text
        if self.outline:
            key = f"h{self.seq.nextf('heading')}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=flowable.level)
        self.notify("TOCEntry", (flowable.level, text, self.page))


def _body_markup(text: str, lang: str) -> str:
    return html.escape(soft_hyphenate(pdf_typography(text, lang)), quote=False)


def _title_markup(text: str, lang: str) -> str:
    return html.escape(pdf_typography(text, lang), quote=False)


def _section_flowables(section: Section, lang: str) -> list[Flowable]:
    style = _H1 if section.level == 1 else _H2
    flowables: list[Flowable] = [
        _Heading(_title_markup(section.title, lang), style, section.level - 1),
    ]
    flowables += [
        Paragraph(_body_markup(text, lang), _BODY)
        for text in section.paragraphs
    ]
    if section.bullets:
        items = [
            ListItem(Paragraph(_body_markup(text, lang), _BODY))
            for text in section.bullets
        ]
        flowables.append(
            ListFlowable(items, bulletType="bullet", start="\U00002022"),
        )
    if section.code:
        lines = section.code.split("\n")
        numbered = "\n".join(
            f"{number}  {line}" for number, line in enumerate(lines, 1)
        )
        flowables.append(Preformatted(numbered, _CODE))
    if section.table:
        rows = [list(row) for row in section.table]
        flowables.append(Table(rows, hAlign="LEFT"))
    if section.note:
        flowables += [
            Paragraph("Note", _NOTE),
            Paragraph(_body_markup(section.note, lang), _BODY),
        ]
    return flowables


def _scan_flowable() -> Image:
    buffer = io.BytesIO()
    scan_image().save(buffer, format="PNG")
    buffer.seek(0)
    return Image(buffer, width=_W, height=_H)


def build_reportlab_pdf(  # ruff: ignore[too-many-arguments] - one keyword per PDF feature
    spec: DocSpec,
    out: Path,
    *,
    toc: bool = False,
    artifacts: bool = False,
    outline: bool = True,
    scanned_page: bool = False,
) -> Path:
    """Write the documentation as an A4 PDF made with reportlab.

    The PDF has one column, one page per section, a running header and footer,
    and the language and the title of the documentation.

    Args:
        spec: The documentation to write.
        out: The PDF file to write.
        toc: Whether to start with a table of contents page, numbered with
            roman page labels.
        artifacts: Whether to mark the running header and footer as artifacts.
        outline: Whether to add the section titles as bookmarks.
        scanned_page: Whether to end with a page that is only an image.

    Returns:
        The PDF file.
    """
    story: list[Flowable] = []
    if toc:
        story += [
            Paragraph(_TOC_TITLES[spec.lang], _H1),
            TableOfContents(dotsMinLevel=0),
            PageBreak(),
        ]
    for index, section in enumerate(spec.sections):
        if index:
            story.append(PageBreak())
        story += _section_flowables(section, spec.lang)
    if scanned_page:
        story += [NextPageTemplate("scan"), PageBreak(), _scan_flowable()]
    document = _Document(
        out,
        spec,
        artifacts=artifacts,
        outline=outline,
        toc=toc,
    )
    document.multiBuild(story)
    return out


# Layout of the fpdf2 documents, in millimetres.
_MARGIN_MM = 14
_GUTTER_MM = 7
_LINE_MM = 5.76  # The line height of the 12 pt body text.


class _TwoColumns(FPDF):
    """An A5 document with two text columns and a page count footer."""

    def __init__(self) -> None:
        # The string "A5" emits a UserWarning in fpdf2 2.9.
        super().__init__(format=(148, 210))
        self.core_fonts_encoding = "windows-1252"
        self.set_margins(_MARGIN_MM, _MARGIN_MM, _MARGIN_MM)
        self.set_auto_page_break(auto=True, margin=_MARGIN_MM)
        self.col_w = (148 - 2 * _MARGIN_MM - _GUTTER_MM) / 2
        self.col = 0
        self._set_column(0)

    def _set_column(self, col: int) -> None:
        self.col = col
        left = _MARGIN_MM + col * (self.col_w + _GUTTER_MM)
        self.set_left_margin(left)
        self.set_right_margin(self.w - left - self.col_w)
        self.set_x(left)

    def _next_column(self) -> bool:
        """Move to the next column and tell whether it needs a new page."""
        if self.col == 0:
            self._set_column(1)
            self.set_y(self.t_margin)
            return False
        self._set_column(0)
        return True

    @property
    @override
    def accept_page_break(self) -> bool:
        return self._next_column()

    def ensure_room(self, height: float) -> None:
        """Move to the next column, or page, unless the height fits."""
        if self.y + height > self.page_break_trigger and self._next_column():
            self.add_page()

    @override
    def footer(self) -> None:
        self.set_font("helvetica", size=8)
        self.set_y(-9)
        self.set_x(0)
        self.cell(self.w, 5, f"{self.page_no()}/{{nb}}", align="C")


def _write_text(pdf: FPDF, text: str, align: str) -> None:
    pdf.multi_cell(
        0,
        _LINE_MM,
        text,
        align=align,
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )


def _write_section(pdf: _TwoColumns, section: Section) -> None:
    for text in section.paragraphs:
        _write_text(pdf, soft_hyphenate(text), "J")
        pdf.ln(2.5)
    for text in section.bullets:
        _write_text(pdf, "\U00002013 " + soft_hyphenate(text), "L")
        pdf.ln(1.5)
    if section.code:
        pdf.set_font("courier", size=9)
        for line in section.code.split("\n"):
            pdf.cell(0, 4.2, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2.5)
    if section.table:
        pdf.set_font("helvetica", size=9.5)
        with pdf.table(width=pdf.col_w, col_widths=(1, 1), align="L") as table:
            for row in section.table:
                table.row(row)
        pdf.ln(2.5)
    if section.note:
        pdf.set_font("times", "B", 12)
        _write_text(pdf, "Note", "J")
        pdf.set_font("times", size=12)
        _write_text(pdf, soft_hyphenate(section.note), "J")
        pdf.ln(2.5)


def build_fpdf2_pdf(spec: DocSpec, out: Path) -> Path:
    """Write the documentation as a tagged two column A5 PDF made with fpdf2.

    The PDF has the language and the title of the documentation, bookmarks for
    the section titles, and the page number and count in the footer.

    Args:
        spec: The documentation to write.
        out: The PDF file to write.

    Returns:
        The PDF file.
    """
    pdf = _TwoColumns()
    pdf.set_lang(spec.lang)
    pdf.set_title(spec.title)
    pdf.set_section_title_styles(
        TextStyle(
            font_family="helvetica",
            font_style="B",
            font_size_pt=15,
            t_margin=3,
            b_margin=2,
        ),
        TextStyle(
            font_family="helvetica",
            font_style="B",
            font_size_pt=12.5,
            t_margin=2,
            b_margin=1.5,
        ),
    )
    pdf.add_page()
    for section in spec.sections:
        pdf.set_font("times", size=12)
        # Without room for the title, fpdf2 breaks the page inside it.
        pdf.ensure_room(12)
        pdf.start_section(section.title, level=section.level - 1)
        _write_section(pdf, section)
    pdf.output(out)
    return out


def build_raw_pdf(
    out: Path,
    operators: Sequence[str],
    *,
    encrypt: str | None = None,
) -> Path:
    """Write a one page PDF whose content stream ends with raw operators.

    The operators follow a dot drawn in Times-Roman. They can select ``/F1``,
    which reportlab declares on every page, as Helvetica.

    Args:
        out: The PDF file to write.
        operators: The PDF operators to append to the page content.
        encrypt: The user password to protect the PDF with, if any.

    Returns:
        The PDF file.
    """
    canvas = Canvas(str(out), pagesize=A4, encrypt=encrypt)
    canvas.setFont("Times-Roman", 10)
    canvas.drawString(10, 10, ".")
    for operator in operators:
        canvas.addLiteral(operator)
    canvas.save()
    return out


def build_scanned_pdf(out: Path, *, pages: int = 2) -> Path:
    """Write a PDF of full page images without any text, like a scan.

    Args:
        out: The PDF file to write.
        pages: The number of pages.

    Returns:
        The PDF file.
    """
    canvas = Canvas(str(out), pagesize=A4)
    image = ImageReader(scan_image())
    for _ in range(pages):
        canvas.drawImage(image, 0, 0, *A4)
        canvas.showPage()
    canvas.save()
    return out


def make_chars(  # ruff: ignore[too-many-arguments] - one keyword per attribute of a character
    text: str,
    x0: float,
    baseline: float,
    *,
    size: float = 10.0,
    advance: float | None = None,
    mono: bool = False,
    bold: bool = False,
) -> list[PdfChar]:
    """Lay out a text on one line, as the PDF reader would return it.

    Every character of the text gets a box, the blanks too. The box of the
    character ``i`` spans from ``x0 + i * a`` to ``x0 + (i + 1) * a``, where
    ``a`` is the advance.

    Args:
        text: The text of the line.
        x0: Left edge of the first character, in points.
        baseline: Position of the baseline, in points above the page bottom.
        size: Font size, in points.
        advance: Width of the box of a character, in points. Half the size if
            omitted.
        mono: Whether the font is monospaced.
        bold: Whether the font is bold.

    Returns:
        The characters of the text.
    """
    step = advance or 0.5 * size
    return [
        PdfChar(
            char,
            x0 + index * step,
            x0 + (index + 1) * step,
            baseline,
            size,
            mono,
            bold,
        )
        for index, char in enumerate(text)
    ]


def make_page(
    *runs: Sequence[PdfChar],
    index: int = 0,
    height: float = 842.0,
) -> PdfPageText:
    """Make an A4 wide page of the characters that the runs hold, in order.

    The page has no artifact, no character without Unicode mapping, no
    character at an angle and no image.

    Args:
        runs: The characters of the page, as lines made by `make_chars`.
        index: Position of the page in the document, from 0.
        height: Height of the page, in points.

    Returns:
        The page, labelled with its number from 1.
    """
    return PdfPageText(
        index=index,
        label=str(index + 1),
        width=595.0,
        height=height,
        chars=tuple(chain.from_iterable(runs)),
        artifacts=0,
        unmapped=0,
        image_coverage=0.0,
        angled=0,
    )


def make_segment(  # ruff: ignore[too-many-arguments] - one keyword per attribute of a segment
    text: str,
    x0: float,
    baseline: float,
    *,
    size: float = 10.0,
    width: float | None = None,
    page: int = 0,
    mono: bool = False,
    bold: bool = False,
) -> Segment:
    """Make a line segment, as the grouping of the characters returns it.

    Args:
        text: The text of the segment.
        x0: Left edge of the segment, in points.
        baseline: Position of the baseline, in points above the page bottom.
        size: Font size, in points.
        width: Width of the segment, in points. Half the size for each
            character of the text if omitted.
        page: Position of the page in the document, from 0.
        mono: Whether the font is monospaced.
        bold: Whether the font is bold.

    Returns:
        The segment.
    """
    return Segment(
        page=page,
        text=text,
        x0=x0,
        x1=x0 + (width or len(text) * 0.5 * size),
        baseline=baseline,
        size=size,
        mono=mono,
        bold=bold,
    )


def make_line(  # ruff: ignore[too-many-arguments] - one keyword per attribute of a line
    text: str,
    baseline: float,
    *,
    x0: float = 72.0,
    width: float | None = None,
    size: float = 10.0,
    page: int = 0,
    mono: bool = False,
    bold: bool = False,
) -> Line:
    """Make a line of text, as the joining of the segments returns it.

    Args:
        text: The text of the line.
        baseline: Position of the baseline, in points above the page bottom.
        x0: Left edge of the line, in points.
        width: Width of the line, in points. Half the size for each character
            of the text if omitted.
        size: Font size, in points.
        page: Position of the page in the document, from 0.
        mono: Whether the font is monospaced.
        bold: Whether the font is bold.

    Returns:
        The line.
    """
    return Line(
        page=page,
        text=text,
        x0=x0,
        x1=x0 + (width or len(text) * 0.5 * size),
        baseline=baseline,
        size=size,
        mono=mono,
        bold=bold,
    )
