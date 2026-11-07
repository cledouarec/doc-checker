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
import math
import re
from dataclasses import dataclass
from functools import partial
from itertools import chain, pairwise
from typing import TYPE_CHECKING, Final, override

from fpdf import FPDF
from fpdf.enums import XPos, YPos
from fpdf.fonts import TextStyle
from PIL import Image as PILImage
from PIL import ImageDraw
from pypdf import PdfReader, PdfWriter
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader, simpleSplit
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
    SimpleDocTemplate,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

from doc_checker.extract.pdf.layout import Line
from doc_checker.extract.pdf.reader import PdfChar, PdfPageText
from doc_checker.extract.pdf.segments import Segment
from tests.builders import (
    DocSpec,
    Figure,
    Section,
    figure_file,
    link_text,
    section_numbers,
    smart_typography,
)
from tests.image_builders import (
    FIGURE_SIZE,
    drawing,
    label_positions,
    labels,
    logo,
    shapes,
)

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence
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
    text = smart_typography(link_text(text), lang)
    return text.replace("\U0000202f", "\xa0")


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
_CAPTION = ParagraphStyle(
    "caption",
    fontName="Times-Italic",
    fontSize=10,
    leading=12,
    alignment=TA_CENTER,
    spaceBefore=4,
    spaceAfter=8,
)


class _VectorFigure(Flowable):
    """A figure drawn with paths, in the page or in a form."""

    def __init__(
        self,
        name: str,
        *,
        flipped: bool = False,
        form: bool = False,
    ) -> None:
        super().__init__()
        self.name = name
        self.flipped = flipped
        self.form = form
        self.width, self.height = FIGURE_SIZE

    @override
    def wrap(
        self,
        availWidth: float,
        availHeight: float,
    ) -> tuple[float, float]:
        return self.width, self.height

    def _shapes(self, canvas: Canvas) -> None:
        for shape in shapes(flipped=self.flipped):
            x0, y0, x1, y1 = shape.box
            canvas.setFillColor(HexColor(shape.fill))
            box = (x0, self.height - y1, x1, self.height - y0)
            if shape.kind == "rect":
                canvas.rect(*box[:2], x1 - x0, y1 - y0, stroke=0, fill=1)
            else:
                canvas.ellipse(*box, stroke=0, fill=1)
        canvas.setFillColor(HexColor("#000000"))
        canvas.setFont("Helvetica", 11)
        for label, (x, y) in zip(
            labels(self.name),
            label_positions(flipped=self.flipped),
            strict=True,
        ):
            canvas.drawString(x, self.height - y, label)

    @override
    def draw(self) -> None:
        canvas = self.canv
        if not self.form:
            self._shapes(canvas)
            return
        key = f"figure-{self.name}"
        canvas.beginForm(key)
        self._shapes(canvas)
        canvas.endForm()
        canvas.doForm(key)


def _figure_flowables(
    figure: Figure,
    k: int,
    *,
    forms: bool,
    reencode: bool,
) -> list[Flowable]:
    """Return the flowables of a figure: its picture and its caption."""
    flowables: list[Flowable] = []
    if figure.defect != "dropped":
        suffix, data = figure_file(figure)
        if suffix == "svg":
            other = figure.defect == "other"
            name = f"{figure.name}-other" if other else figure.name
            flowables.append(_VectorFigure(name, flipped=other, form=forms))
        else:
            if reencode and suffix == "png":
                buffer = io.BytesIO()
                PILImage.open(io.BytesIO(data)).convert("RGB").save(
                    buffer,
                    "JPEG",
                    quality=95,
                )
                data = buffer.getvalue()
            width, height = PILImage.open(io.BytesIO(data)).size
            flowables.append(
                Image(
                    io.BytesIO(data),
                    width=100 * mm,
                    height=100 * mm * height / width,
                ),
            )
    caption = html.escape(f"Figure {k}: {figure.caption}", quote=False)
    flowables.append(Paragraph(caption, _CAPTION))
    return flowables


class _Heading(Paragraph):
    """A section heading, which carries its outline level."""

    def __init__(self, text: str, style: ParagraphStyle, level: int) -> None:
        super().__init__(text, style)
        self.level = level


class _Document(BaseDocTemplate):
    """An A4 document with a running header and footer on each page."""

    def __init__(  # ruff: ignore[too-many-arguments] - one keyword per PDF feature
        self,
        path: Path,
        spec: DocSpec,
        *,
        artifacts: bool,
        outline: bool,
        outline_depth: int | None,
        toc: bool,
        logo_on_pages: bool,
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
        self.outline_depth = outline_depth
        self.toc = toc
        self.logo_on_pages = logo_on_pages
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
        if self.logo_on_pages:
            canvas.drawImage(
                ImageReader(logo()),
                _PAGE_MARGIN,
                _H - 12 * mm,
                width=60,
                height=20,
            )

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
        depth = self.outline_depth
        if self.outline and (depth is None or flowable.level < depth):
            key = f"h{self.seq.nextf('heading')}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=flowable.level)
        self.notify("TOCEntry", (flowable.level, text, self.page))


def _body_markup(text: str, lang: str) -> str:
    return html.escape(soft_hyphenate(pdf_typography(text, lang)), quote=False)


def _title_markup(text: str, lang: str) -> str:
    return html.escape(pdf_typography(text, lang), quote=False)


def _section_flowables(  # ruff: ignore[too-many-arguments] - one keyword per figure option
    section: Section,
    lang: str,
    *,
    number: str | None = None,
    chapter: int | None = None,
    figure_numbers: dict[str, int] | None = None,
    forms: bool = False,
    reencode: bool = False,
) -> list[Flowable]:
    style = _H1 if section.level == 1 else _H2
    title = _title_markup(section.title, lang)
    flowables: list[Flowable] = []
    if chapter is not None:
        flowables.append(Paragraph(f"Chapter {chapter}", _H2))
    if number is not None:
        title = f"{number} {title}"
    flowables.append(_Heading(title, style, section.level - 1))
    flowables += [
        Paragraph(_body_markup(text, lang), _BODY)
        for text in section.paragraphs
    ]
    for figure in section.figures:
        flowables += _figure_flowables(
            figure,
            (figure_numbers or {}).get(figure.name, 0),
            forms=forms,
            reencode=reencode,
        )
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
    numbered: bool = False,
    chapter_labels: bool = False,
    outline_depth: int | None = None,
    figure_forms: bool = False,
    logo: bool = False,
    reencode: bool = False,
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
        numbered: Whether the titles and their bookmarks start with their
            section number.
        chapter_labels: Whether each level 1 title is preceded by a
            ``Chapter N`` line, with no bookmark.
        outline_depth: The deepest title level that gets a bookmark, with
            1 for the chapters, or ``None`` for a bookmark on every title.
        figure_forms: Whether each vector figure is drawn in a form.
        logo: Whether each page has the logo at the top left.
        reencode: Whether each PNG figure is written as a JPEG of quality 95.

    Returns:
        The PDF file.
    """
    numbers = section_numbers(spec) if numbered else {}
    figure_numbers = {
        figure.name: k
        for k, figure in enumerate(
            (f for section in spec.sections for f in section.figures),
            1,
        )
    }
    chapters = 0
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
        chapters += section.level == 1
        story += _section_flowables(
            section,
            spec.lang,
            number=numbers.get(section.slug),
            chapter=chapters
            if chapter_labels and section.level == 1
            else None,
            figure_numbers=figure_numbers,
            forms=figure_forms,
            reencode=reencode,
        )
    if scanned_page:
        story += [NextPageTemplate("scan"), PageBreak(), _scan_flowable()]
    document = _Document(
        out,
        spec,
        artifacts=artifacts,
        outline=outline,
        outline_depth=outline_depth,
        toc=toc,
        logo_on_pages=logo,
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


def _write_figure(pdf: _TwoColumns, figure: Figure) -> None:
    """Write a figure: its picture, then its caption."""
    if figure.defect != "dropped":
        suffix, data = figure_file(figure)
        if suffix == "svg":
            ratio = FIGURE_SIZE[1] / FIGURE_SIZE[0]
        else:
            width, height = PILImage.open(io.BytesIO(data)).size
            ratio = height / width
        pdf.ensure_room(pdf.col_w * ratio + 12)
        alt = None if figure.defect == "noalt" else figure.alt
        pdf.image(io.BytesIO(data), w=pdf.col_w, alt_text=alt)
        pdf.set_x(pdf.l_margin)
    pdf.set_font("times", "I", 10)
    _write_text(pdf, figure.caption, "C")
    pdf.ln(2.5)
    pdf.set_font("times", size=12)


def _write_section(pdf: _TwoColumns, section: Section) -> None:
    if section.heading_as_text:
        pdf.set_font("times", size=12)
        _write_text(pdf, section.title, "J")
        pdf.ln(2.5)
    for text in map(link_text, section.paragraphs):
        _write_text(pdf, soft_hyphenate(text), "J")
        pdf.ln(2.5)
    for figure in section.figures:
        _write_figure(pdf, figure)
    for text in map(link_text, section.bullets):
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
        _write_text(pdf, soft_hyphenate(link_text(section.note)), "J")
        pdf.ln(2.5)


def build_fpdf2_pdf(spec: DocSpec, out: Path) -> Path:
    """Write the documentation as a tagged two column A5 PDF made with fpdf2.

    The PDF has the language and the title of the documentation, bookmarks for
    the section titles, and the page number and count in the footer. The
    title of a section with ``heading_as_text`` is a paragraph, with no
    bookmark.

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
        TextStyle(
            font_family="helvetica",
            font_style="B",
            font_size_pt=11,
            t_margin=2,
            b_margin=1,
        ),
    )
    pdf.add_page()
    for section in spec.sections:
        pdf.set_font("times", size=12)
        # Without room for the title, fpdf2 breaks the page inside it.
        pdf.ensure_room(12)
        if not section.heading_as_text:
            pdf.start_section(section.title, level=section.level - 1)
        _write_section(pdf, section)
    pdf.output(out)
    return out


def build_images_pdf(
    out: Path,
    *,
    pages: int,
    image_size: tuple[int, int] = (480, 320),
) -> Path:
    """Write a PDF with a line of text and an image on each page.

    Args:
        out: The PDF file to write.
        pages: The number of pages.
        image_size: Width and height of the images, in pixels, which are
            drawn in points.

    Returns:
        The PDF file.
    """
    canvas = Canvas(str(out), pagesize=A4)
    for number in range(1, pages + 1):
        canvas.setFont("Helvetica", 11)
        canvas.drawString(72, _H - 72, f"Text of page {number}")
        picture = ImageReader(drawing(f"page {number}", image_size))
        canvas.drawImage(
            picture,
            72,
            _H - 100 - image_size[1],
            width=image_size[0],
            height=image_size[1],
        )
        canvas.showPage()
    canvas.save()
    return out


def build_dense_paths_pdf(
    out: Path,
    *,
    paths: int,
    pitch: float = 10.0,
) -> Path:
    """Write a page of many small filled squares, none touching another.

    Args:
        out: The PDF file to write.
        paths: The number of squares of 2 by 2 points, 50 on each row.
        pitch: The distance between the squares of a row, and between rows.

    Returns:
        The PDF file.
    """
    height = 40 + pitch * math.ceil(paths / 50)
    canvas = Canvas(str(out), pagesize=(600, height))
    canvas.setFont("Helvetica", 11)
    canvas.drawString(10, height - 20, "Dense paths")
    for index in range(paths):
        row, column = divmod(index, 50)
        canvas.rect(
            10 + column * pitch,
            height - 40 - row * pitch - 2,
            2,
            2,
            stroke=0,
            fill=1,
        )
    canvas.showPage()
    canvas.save()
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


def pdf_stream(text: str) -> str:
    """Return a PDF stream object that holds an ASCII text.

    Args:
        text: The data of the stream.

    Returns:
        The stream object: its dictionary, which gives the length, then the
        data.
    """
    return f"<< /Length {len(text)} >>\nstream\n{text}\nendstream"


def to_unicode_cmap(mapping: Mapping[str, str]) -> str:
    """Return a ToUnicode CMap object, from byte codes to UTF-16 hex.

    Args:
        mapping: The text of each byte code, both in hexadecimal digits:
            ``{"41": "0041"}`` maps A to itself, and ``{"41": "D83DDE00"}``
            maps it to a character beyond U+FFFF, written as its two halves.
            The CMap has a single block of codes, which allows 100 at most.

    Returns:
        The stream object of the CMap.
    """
    pairs = "".join(f"<{code}> <{utf16}>\n" for code, utf16 in mapping.items())
    return pdf_stream(
        "/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n"
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) "
        "/Supplement 0 >> def\n"
        "/CMapName /Adobe-Identity-UCS def /CMapType 2 def\n"
        "1 begincodespacerange <00> <FF> endcodespacerange\n"
        f"{len(mapping)} beginbfchar\n{pairs}endbfchar\n"
        "endcmap CMapName currentdict /CMap defineresource pop end end",
    )


def build_single_page_pdf(
    out: Path,
    content: str,
    *,
    font: str = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    page: str = "/MediaBox [0 0 200 200]",
    extra: tuple[str, ...] = (),
) -> Path:
    """Write a one page PDF that shows a content stream in the font F1.

    Args:
        out: The PDF file to write.
        content: The content stream of the page, in ASCII.
        font: The object of the font F1.
        page: The entries of the page dictionary, such as its boxes and its
            rotation.
        extra: Objects to write after the font. They are numbered from 6, and
            the font can refer to them.

    Returns:
        The PDF file.
    """
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R {page} /Contents 4 0 R "
            "/Resources << /Font << /F1 5 0 R >> >> >>"
        ),
        pdf_stream(content),
        font,
        *extra,
    ]
    data = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(data))
        data += f"{number} 0 obj\n{body}\nendobj\n".encode("ascii")
    count = len(objects) + 1
    entries = "".join(f"{offset:010d} 00000 n \n" for offset in offsets)
    trailer = (
        f"xref\n0 {count}\n0000000000 65535 f \n{entries}"
        f"trailer\n<< /Size {count} /Root 1 0 R >>\n"
        f"startxref\n{len(data)}\n%%EOF\n"
    )
    out.write_bytes(data + trailer.encode("ascii"))
    return out


# Layout of the PDF of bullets, in points: the left edge of the sentences and
# of the bullets, the left edge of the text of the items, the baseline of the
# first sentence and the distance between two baselines.
_BULLETS_LEFT = 72
_BULLETS_TEXT = 90
_BULLETS_TOP = 700
_BULLETS_LEADING = 20
# The font has the encoding of cp1252, in which the code 95 is a bullet: with
# no encoding, PDFium drops a text object that holds that code alone. It gets
# its Unicode values from the CMap that follows it in the file.
_BULLETS_FONT = (
    "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
    "/Encoding /WinAnsiEncoding /ToUnicode 6 0 R >>"
)
# In that CMap, the codes from the space to the tilde are their ASCII
# characters. The code 95 is the bullet of cp1252, U+2022. The code B7, the
# middle dot of cp1252, is the private use character U+F0B7, which PDFium
# returns for the bullet of the Symbol font of Word.
_BULLETS_UNICODE = {
    **{f"{code:02X}": f"{code:04X}" for code in range(0x20, 0x7F)},
    "95": "2022",
    "B7": "F0B7",
    "D8": "F0D8",
    "FC": "F0FC",
    "F6": "F076",
}
# The code of the cp1252 font for each private use bullet that Word sets.
_BULLETS_CODES = {
    "\U0000f0b7": r"\267",
    "\U0000f0d8": r"\330",
    "\U0000f0fc": r"\374",
    "\U0000f076": r"\366",
}


def _show_text(text: str, x: float, y: float) -> str:
    """Return the operators that draw a text in Helvetica 11."""
    return f"BT /F1 11 Tf {x} {y} Td ({text}) Tj ET"


def build_bullets_pdf(
    items: Sequence[str],
    out: Path,
    *,
    word: bool,
    bullet: str = "\U0000f0b7",
    gap: bool = True,
) -> Path:
    """Write an A4 PDF of a bulleted list between two sentences.

    The text is Helvetica 11 in the cp1252 encoding, a line every 20 pt. Each
    bullet is a text object of its own, 72 pt from the left edge, and the text
    of its item is another one, 18 pt right of the bullet. The bullet of Word
    is the code B7, which the ToUnicode CMap of the font maps to U+F0B7, the
    private use character that PDFium returns for the Symbol font of Word. The
    other bullet is the code 95, the bullet of cp1252, which the CMap maps to
    U+2022. PDFium returns both kinds of bullet. The CMap also maps the codes
    D8, FC and F6 to U+F0D8, U+F0FC and U+F076, the arrow, the check mark and
    the diamond of Wingdings.

    Args:
        items: The text of each item, in ASCII and without parentheses or
            backslashes.
        out: The PDF file to write.
        word: Whether the bullets are those of Word rather than the others.
        bullet: With `word`, the character that PDFium reads for the bullet,
            one of the private use characters of the CMap. The PDF holds the
            code that the CMap maps to it.
        gap: Whether the bullet and the text of an item are two text objects.
            Otherwise they are one string in one text object, 72 pt from the
            left edge.

    Returns:
        The PDF file.
    """
    code = _BULLETS_CODES[bullet] if word else r"\225"
    shown = [
        _show_text("The requirements follow.", _BULLETS_LEFT, _BULLETS_TOP),
    ]
    for number, item in enumerate(items, 1):
        y = _BULLETS_TOP - _BULLETS_LEADING * number
        if gap:
            shown += [
                _show_text(code, _BULLETS_LEFT, y),
                _show_text(item, _BULLETS_TEXT, y),
            ]
        else:
            shown.append(_show_text(code + item, _BULLETS_LEFT, y))
    closing = _BULLETS_TOP - _BULLETS_LEADING * (len(items) + 1)
    shown.append(
        _show_text("Then install the package.", _BULLETS_LEFT, closing),
    )
    return build_single_page_pdf(
        out,
        "\n".join(shown),
        font=_BULLETS_FONT,
        page="/MediaBox [0 0 595 842]",
        extra=(to_unicode_cmap(_BULLETS_UNICODE),),
    )


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


# Layout of the PDF of lines, in points: the margin, the distance between two
# lines and the extra distance between two paragraphs.
_LINES_MARGIN = 72
_LINES_LEADING = 12
_LINES_PARAGRAPH_GAP = 6
# The slant of synthetic italics, in degrees.
_SYNTHETIC_SLANT = 12


def _draw_line(
    canvas: Canvas,
    y: float,
    line: str,
    slanted: re.Pattern[str] | None,
) -> None:
    """Draw a line of `build_lines_pdf`, with its slanted words.

    Args:
        canvas: The canvas of the page.
        y: The baseline of the line.
        line: The text of the line.
        slanted: The words to slant, as a pattern that captures them, if any.
    """
    # Splitting on a capturing pattern puts the slanted words at odd indexes,
    # between the runs of upright text.
    pieces = slanted.split(line) if slanted else [line]
    x = _LINES_MARGIN
    for index, piece in enumerate(pieces):
        if index % 2:
            canvas.saveState()
            canvas.translate(x, y)
            canvas.skew(0, _SYNTHETIC_SLANT)
            canvas.drawString(0, 0, piece)
            canvas.restoreState()
        else:
            canvas.drawString(x, y, piece)
        x += canvas.stringWidth(piece)


def build_lines_pdf(
    paragraphs: Sequence[Sequence[str]],
    out: Path,
    *,
    slanted: Collection[str] = (),
) -> Path:
    """Write an A4 PDF whose lines are broken exactly where they are given.

    The text is Times-Roman 10, drawn from the top left corner, 72 pt from the
    edges of the page. There are 12 pt between two lines and 6 pt more between
    two paragraphs. A line that would start below the bottom margin goes to
    the top of a new page.

    The slanted words are set in synthetic italics, as LibreOffice, Chrome or
    cairo set the italics of a font that has none: their upright glyphs are
    sheared by 12 degrees, on the baseline of the line.

    Args:
        paragraphs: The paragraphs, each given line by line.
        out: The PDF file to write.
        slanted: The words to set in synthetic italics, wherever they stand
            as whole words.

    Returns:
        The PDF file.
    """
    pattern = None
    if slanted:
        words = "|".join(re.escape(word) for word in sorted(slanted))
        pattern = re.compile(rf"\b({words})\b")
    canvas = Canvas(str(out), pagesize=A4)
    canvas.setFont("Times-Roman", 10)
    y = _H - _LINES_MARGIN
    for lines in paragraphs:
        for line in lines:
            if y < _LINES_MARGIN:
                canvas.showPage()
                canvas.setFont("Times-Roman", 10)
                y = _H - _LINES_MARGIN
            _draw_line(canvas, y, line, pattern)
            y -= _LINES_LEADING
        y -= _LINES_PARAGRAPH_GAP
    canvas.save()
    return out


def build_turned_pdf(
    pages: Sequence[Sequence[str]],
    out: Path,
    *,
    turned: Collection[int] = (),
    tab: float | None = None,
) -> Path:
    """Write an A4 PDF of lines, with some of its pages turned to landscape.

    Each page holds its lines in Times-Roman 10, a line every 12 pt, 72 pt from
    the left and top edges of the page as it is read. A turned page is set as
    the LaTeX package pdflscape sets one: its text runs up the page, and the
    page has a rotation of 90 degrees, which turns it upright in a viewer.

    With a tab, the text that follows the tab character of a line is a text
    object of its own, which starts `tab` points right of the start of the
    line, with no space glyph between the two objects, as the next cell of a
    table in LaTeX. PDFium generates the space between them.

    Args:
        pages: The lines of each page.
        out: The PDF file to write.
        turned: Positions of the turned pages, from 0.
        tab: Where the text after the tab of a line starts, in points from
            the start of the line. Without it, each line is drawn whole.

    Returns:
        The PDF file.
    """
    canvas = Canvas(str(out), pagesize=A4)
    for index, lines in enumerate(pages):
        # reportlab keeps the rotation of a page for the next pages.
        canvas.setPageRotation(90 if index in turned else 0)
        canvas.saveState()
        if index in turned:
            canvas.translate(_W, 0)
            canvas.rotate(90)
            top = _W - _LINES_MARGIN
        else:
            top = _H - _LINES_MARGIN
        canvas.setFont("Times-Roman", 10)
        for number, line in enumerate(lines):
            y = top - _LINES_LEADING * number
            if tab is not None and "\t" in line:
                cell, rest = line.split("\t", 1)
                canvas.drawString(_LINES_MARGIN, y, cell)
                canvas.drawString(_LINES_MARGIN + tab, y, rest)
            else:
                canvas.drawString(_LINES_MARGIN, y, line)
        canvas.restoreState()
        canvas.showPage()
    canvas.save()
    return out


_GUIDE_BODY = ParagraphStyle(
    "guide",
    fontName="Times-Roman",
    fontSize=11,
    leading=14,
    spaceAfter=7,
)
_GUIDE_TITLE = ParagraphStyle(
    "guide-title",
    fontName="Helvetica-Bold",
    fontSize=18,
    leading=22,
    spaceAfter=10,
)
_GUIDE_ENTRIES = ("Preface", "Installation", "Usage", "Limits")


def build_contents_pdf(out: Path, *, contents: bool) -> Path:
    """Write a short A4 guide, with a contents page before it or without.

    The guide opens with a plain paragraph, which is the first block after the
    contents page. The last entry of the contents does not end like a
    sentence, so it looks like the start of a sentence that goes on.

    Args:
        out: The PDF file to write.
        contents: Whether to start with a contents page, titled ``Contents``,
            whose entries run to their page numbers along dot leaders.

    Returns:
        The PDF file.
    """
    story: list[Flowable] = []
    if contents:
        story.append(Paragraph("Contents", _GUIDE_TITLE))
        story += [
            Paragraph(f"{entry} . . . . . . . . . . 2", _GUIDE_BODY)
            for entry in _GUIDE_ENTRIES
        ]
        story.append(PageBreak())
    story += [
        Paragraph(
            "This guide explains how to install, configure and run the "
            "service on a cluster.",
            _GUIDE_BODY,
        ),
        Paragraph("Installation", _GUIDE_TITLE),
        Paragraph(
            "Install the package with the default configuration before you "
            "start.",
            _GUIDE_BODY,
        ),
        Paragraph("Usage", _GUIDE_TITLE),
        Paragraph("Run the service with its default options.", _GUIDE_BODY),
    ]
    SimpleDocTemplate(str(out), pagesize=A4, title="Example").build(story)
    return out


# The words of the pseudo English text of `filler_sections`.
FILLER_WORDS: Final = (
    "service",
    "request",
    "client",
    "server",
    "cache",
    "option",
    "value",
    "table",
    "project",
    "export",
    "install",
    "configure",
    "restart",
    "directory",
    "archive",
    "module",
    "package",
    "network",
    "latency",
    "storage",
    "volume",
    "backup",
    "restore",
    "account",
    "member",
    "owner",
    "schedule",
    "worker",
    "queue",
    "message",
    "timeout",
    "retry",
    "policy",
    "limit",
    "quota",
    "rule",
    "report",
    "document",
    "section",
    "chapter",
    "figure",
    "listing",
    "example",
    "manual",
    "guide",
    "release",
    "version",
    "upgrade",
    "migrate",
    "deploy",
    "cluster",
    "node",
    "replica",
    "shard",
)


def _filler_sentence(section: int, paragraph: int, index: int) -> str:
    length = 8 + (section + 2 * paragraph + 3 * index) % 9
    start = 11 * section + 7 * paragraph + 5 * index
    words = [
        FILLER_WORDS[(start + 3 * word + word * word) % len(FILLER_WORDS)]
        for word in range(length)
    ]
    return " ".join(words).capitalize() + "."


def filler_sections(
    count: int,
    paragraphs: int,
    sentences: int,
) -> list[list[str]]:
    """Make sections of paragraphs of pseudo English, the same on every call.

    Each paragraph starts with a mark that no other paragraph has, such as
    ``S1P3`` for the paragraph 3 of the section 1, both counted from 0. Its
    sentences follow, one blank apart. A sentence has 8 to 16 words of
    `FILLER_WORDS`, which follow from its position, without any randomness.

    Args:
        count: The number of sections.
        paragraphs: The number of paragraphs of each section.
        sentences: The number of sentences of each paragraph.

    Returns:
        The paragraphs of each section.
    """
    return [
        [
            f"S{section}P{paragraph} "
            + " ".join(
                _filler_sentence(section, paragraph, index)
                for index in range(sentences)
            )
            for paragraph in range(paragraphs)
        ]
        for section in range(count)
    ]


@dataclass(frozen=True, slots=True)
class ProseStyle:
    """The layout of the pages of `build_prose_pdf`, in points.

    Attributes:
        leading: The distance between the baselines of two lines of a
            paragraph.
        space_after: The extra space after each paragraph.
        first_indent: The indent of the first line of each paragraph.
        bottom_margin: The distance between the text and the bottom edge of
            the page.
        page_number_y: The baseline of the page number, centred on each
            page, or None for pages without a number.
        watermark: Whether a DRAFT watermark crosses each page at 45
            degrees.
        roman_pages: The number of first pages whose number is a lower case
            roman numeral, as in a book with front matter. The arabic
            numbers start again at 1 after them.
        page_labels: Whether the PDF has page labels that tell those numbers:
            roman lower case on the first pages, then arabic from 1.
    """

    leading: float = 12.0
    space_after: float = 6.0
    first_indent: float = 0.0
    bottom_margin: float = _PAGE_MARGIN
    page_number_y: float | None = None
    watermark: bool = False
    roman_pages: int = 0
    page_labels: bool = False


_ROMAN_FOLIOS: Final = ("i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix")


def _draw_prose_page(
    canvas: Canvas,
    doc: BaseDocTemplate,
    *,
    style: ProseStyle,
) -> None:
    """Draw what the style adds to a page of `build_prose_pdf`, besides text.

    Args:
        canvas: The canvas of the page.
        doc: The document, which gives the number of the page.
        style: The layout of the pages.
    """
    canvas.saveState()
    if style.page_labels and doc.page == 1:
        canvas.addPageLabel(0, style="ROMAN_LOWER", start=1)
        if style.roman_pages:
            canvas.addPageLabel(style.roman_pages, style="ARABIC", start=1)
    if style.page_number_y is not None:
        folio = (
            _ROMAN_FOLIOS[doc.page - 1]
            if doc.page <= style.roman_pages
            else str(doc.page - style.roman_pages)
        )
        canvas.setFont("Times-Roman", 10)
        canvas.drawCentredString(_W / 2, style.page_number_y, folio)
    if style.watermark:
        canvas.translate(_W / 2, _H / 2)
        canvas.rotate(45)
        canvas.setFont("Helvetica", 72)
        canvas.drawCentredString(0, 0, "DRAFT")
    canvas.restoreState()


def build_prose_pdf(
    sections: Sequence[Sequence[str]],
    out: Path,
    style: ProseStyle | None = None,
) -> Path:
    """Write sections of paragraphs as an A4 PDF made with reportlab.

    The PDF has one column, 25 mm from the left, right and top edges of the
    page and `bottom_margin` from the bottom one, and no header. Its only
    footer is the page number, if the style gives a baseline for it, in
    Times-Roman 10. A watermark, if the style asks for one, is ``DRAFT`` in
    Helvetica 72, centred on the page. The section ``i``, counted from 0,
    opens with the title ``Chapter title i`` in Helvetica-Bold 16. Its
    paragraphs follow, justified, in Times-Roman 10.

    Args:
        sections: The paragraphs of each section.
        out: The PDF file to write.
        style: The layout of the pages. The default `ProseStyle` if
            omitted.

    Returns:
        The PDF file.
    """
    if style is None:
        style = ProseStyle()
    heading = ParagraphStyle(
        "prose-heading",
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        spaceBefore=10,
        spaceAfter=8,
    )
    body = ParagraphStyle(
        "prose",
        fontName="Times-Roman",
        fontSize=10,
        leading=style.leading,
        spaceAfter=style.space_after,
        firstLineIndent=style.first_indent,
        alignment=TA_JUSTIFY,
    )
    doc = BaseDocTemplate(
        str(out),
        pagesize=A4,
        title="Example",
        leftMargin=_PAGE_MARGIN,
        rightMargin=_PAGE_MARGIN,
        topMargin=_PAGE_MARGIN,
        bottomMargin=style.bottom_margin,
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height)
    doc.addPageTemplates(
        [
            PageTemplate(
                "prose",
                [frame],
                onPage=partial(_draw_prose_page, style=style),
            ),
        ],
    )
    story: list[Flowable] = []
    for index, texts in enumerate(sections):
        story.append(Paragraph(f"Chapter title {index}", heading))
        story += [Paragraph(text, body) for text in texts]
    doc.build(story)
    return out


_REFERENCE = ParagraphStyle(
    "reference",
    fontName="Times-Roman",
    fontSize=10,
    leading=12,
    leftIndent=18,
    firstLineIndent=-18,
)
_REFERENCES_TITLE = ParagraphStyle(
    "references-title",
    fontName="Helvetica-Bold",
    fontSize=16,
    leading=20,
    spaceAfter=8,
)


def build_references_pdf(
    entries: Sequence[str],
    out: Path,
    *,
    right_margin: float,
) -> Path:
    """Write a list of references with hanging indents as an A4 PDF.

    The list is titled ``References``. Each entry is a paragraph in
    Times-Roman 10, whose lines after the first are indented by 18 pt, with no
    space between two entries. The left margin is 25 mm.

    Args:
        entries: The texts of the entries.
        out: The PDF file to write.
        right_margin: The right margin, in points, which sets where the lines
            of the entries break.

    Returns:
        The PDF file.
    """
    story: list[Flowable] = [Paragraph("References", _REFERENCES_TITLE)]
    story += [Paragraph(text, _REFERENCE) for text in entries]
    SimpleDocTemplate(
        str(out),
        pagesize=A4,
        title="Example",
        leftMargin=_PAGE_MARGIN,
        rightMargin=right_margin,
    ).build(story)
    return out


_TABLE_TEXT = ParagraphStyle(
    "table-text",
    fontName="Times-Roman",
    fontSize=10,
    leading=12,
    spaceAfter=6,
)


def build_table_pdf(
    out: Path,
    *,
    margin: float,
    rows: Sequence[Sequence[str]] | None = None,
    padding: float | None = None,
) -> Path:
    """Write a long table as an A4 PDF.

    A sentence introduces the table and another one ends the text, both in
    Times-Roman 10. The table runs over several pages, and its header row is
    not repeated on the next pages. Its first column is 120 pt wide, and the
    other ones share 300 pt. Its cells are in Helvetica 10 on a 12 pt
    leading, which their padding sets apart.

    Args:
        out: The PDF file to write.
        margin: The distance between the text and each edge of the page, in
            points.
        rows: The rows of the table, the header row first. If omitted, a
            table of settings: the header row ``Option`` and ``Default``, and
            160 rows, the row ``i``, counted from 0, holding a setting name
            made of a letter and a number, such as ``setting_c0``, and the
            number ``10 * i + 7``.
        padding: The space above and below the text of each cell, in
            points. The 3 pt of reportlab if omitted.

    Returns:
        The PDF file.
    """
    if rows is None:
        rows = [
            ("Option", "Default"),
            *(
                (f"setting_{chr(97 + i % 26)}{i // 26}", str(10 * i + 7))
                for i in range(160)
            ),
        ]
    others = len(rows[0]) - 1
    widths = [120, *[300 / others] * others]
    style = None
    if padding is not None:
        style = TableStyle(
            [
                ("TOPPADDING", (0, 0), (-1, -1), padding),
                ("BOTTOMPADDING", (0, 0), (-1, -1), padding),
            ],
        )
    story: list[Flowable] = [
        Paragraph(
            "The settings of the service are listed below with their "
            "default values.",
            _TABLE_TEXT,
        ),
        Table(rows, hAlign="LEFT", colWidths=widths, style=style),
        Paragraph(
            "All the values can be changed in the configuration file.",
            _TABLE_TEXT,
        ),
    ]
    SimpleDocTemplate(
        str(out),
        pagesize=A4,
        title="Settings",
        leftMargin=margin,
        rightMargin=margin,
        topMargin=margin,
        bottomMargin=margin,
    ).build(story)
    return out


# Layout of the PDF of a listing, in points: the left edge of the text and of
# the code, the baseline of the first line of code and the distance between the
# baselines of two lines of code.
_LISTING_LEFT = 72
_LISTING_TOP = 740
_LISTING_LEADING = 12


def build_listing_pdf(
    code: Sequence[str],
    out: Path,
    *,
    numbers: bool,
    numbersep: float = 6.0,
    page_lines: int | None = None,
) -> Path:
    """Write an A4 PDF of a code listing between two sentences.

    The sentences are in Times-Roman 11, 72 pt from the left edge. The code is
    in Courier 10, 72 pt from the left edge too, a line every 12 pt, and an
    empty line leaves a blank. With numbers, each line, an empty one too, has
    its number in Times-Roman 6, right aligned `numbersep` before the code:
    this is how the LaTeX package listings sets the numbers of the lines in a
    tiny type. A number is drawn before its code, as LaTeX does, so that it
    opens the text of its line.

    Args:
        code: The lines of code, the first one being the line 1.
        out: The PDF file to write.
        numbers: Whether to number the lines.
        numbersep: The distance between the right edge of the numbers and the
            code, in points. The 6 pt default leaves the number in the same
            segment as its code; two sizes of the code, 20 pt, or more set it
            in a segment of its own.
        page_lines: The number of lines of code on a page, the empty ones
            included. The code goes on at the top of the next page after
            them, as high as it starts on the first page, and the last
            sentence follows the last line, on the last page. All the code
            is on the first page if omitted.

    Returns:
        The PDF file.
    """
    canvas = Canvas(str(out), pagesize=A4)
    canvas.setFont("Times-Roman", 11)
    canvas.drawString(
        _LISTING_LEFT,
        _LISTING_TOP + 20,
        "The program prints the working directory.",
    )
    per_page = page_lines or max(len(code), 1)
    row = -1  # The row of the last line of code, counted from the page top.
    for number, text in enumerate(code, 1):
        row = (number - 1) % per_page
        if number > 1 and row == 0:
            canvas.showPage()
        y = _LISTING_TOP - _LISTING_LEADING * row
        if numbers:
            canvas.setFont("Times-Roman", 6)
            canvas.drawRightString(_LISTING_LEFT - numbersep, y, str(number))
        if text:
            canvas.setFont("Courier", 10)
            canvas.drawString(_LISTING_LEFT, y, text)
    canvas.setFont("Times-Roman", 11)
    canvas.drawString(
        _LISTING_LEFT,
        _LISTING_TOP - _LISTING_LEADING * (row + 2),
        "Run it from any directory.",
    )
    canvas.save()
    return out


# Layout of the PDF of numbered sections, in points: the left edge of the
# text, the baseline of the first title, the distance between the baselines
# of two lines of a title and of a paragraph, and the extra space after a
# title and after a paragraph.
_TITLES_LEFT = 72
_TITLES_TOP = 780
_TITLE_LEADING = 17
_TITLE_SPACE = 9
_TEXT_LEADING = 13
_TEXT_SPACE = 14
_TITLE_FONT = ("Helvetica-Bold", 13)
_TEXT_FONT = ("Times-Roman", 10)


def build_numbered_pdf(
    sections: Sequence[tuple[str, str]],
    out: Path,
    *,
    title_width: float,
    numbered: bool = True,
) -> Path:
    """Write numbered sections on one A4 page, with no outline.

    Each section is a title in Helvetica-Bold 13, its number first, broken
    into lines no wider than `title_width`, then the same sentence on three
    lines in Times-Roman 10. Only the style of the titles tells them apart,
    so a title broken into more lines than a heading has is a paragraph.

    Args:
        sections: The number and the title of each section.
        out: The PDF file to write.
        title_width: The widest line of a title, in points.
        numbered: Whether the titles start with their numbers.

    Returns:
        The PDF file.
    """
    canvas = Canvas(str(out), pagesize=A4)
    y = _TITLES_TOP
    for number, title in sections:
        canvas.setFont(*_TITLE_FONT)
        heading = f"{number} {title}" if numbered else title
        lines = simpleSplit(heading, *_TITLE_FONT, title_width)
        for line in lines:
            canvas.drawString(_TITLES_LEFT, y, line)
            y -= _TITLE_LEADING
        y -= _TITLE_SPACE
        canvas.setFont(*_TEXT_FONT)
        sentence = f"The step {title.lower()} is described with each option."
        for _ in range(3):
            canvas.drawString(_TITLES_LEFT, y, sentence)
            y -= _TEXT_LEADING
        y -= _TEXT_SPACE
    canvas.save()
    return out


def swap_pages(source: Path, out: Path, first: int, second: int) -> Path:
    """Copy a PDF with two of its pages swapped.

    Only the pages are copied: the new PDF has no bookmarks, and neither the
    title nor the language of the source.

    Args:
        source: The PDF file to copy.
        out: The PDF file to write.
        first: Position of one of the two pages, from 0.
        second: Position of the other page, from 0.

    Returns:
        The PDF file.
    """
    pages = PdfReader(source).pages
    order = list(range(len(pages)))
    order[first], order[second] = order[second], order[first]
    writer = PdfWriter()
    for index in order:
        writer.add_page(pages[index])
    writer.write(out)
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
