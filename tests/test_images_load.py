# Copyright 2026 Christophe Le Douarec
"""Tests for the pixels read again on demand."""

from typing import TYPE_CHECKING, Literal

import pypdfium2.raw as pdfium_c
import pytest

from doc_checker.config import PdfConfig
from doc_checker.extract.pdf.loader import load_pdf
from doc_checker.images.load import PixelLoader, least_length
from doc_checker.images.pixels import TOO_LARGE, UnreadableImage, open_raster
from doc_checker.images.refs import raster_ref
from doc_checker.model.document import PixelSource
from tests.builders import FIGURES_EN
from tests.image_builders import drawing, encode, progressive_jpeg
from tests.pdf_builders import build_image_stream_pdf, build_reportlab_pdf

if TYPE_CHECKING:
    from pathlib import Path


def test_loader_reads_data_files_and_pdf_images(tmp_path: Path) -> None:
    """Data URLs, files and PDF image objects are read again."""
    data = encode(drawing("setup"))
    (tmp_path / "a.png").write_bytes(data)
    in_data = raster_ref(data, name="", source=PixelSource(data=data))
    in_file = raster_ref(
        data,
        name="a.png",
        source=PixelSource(file=tmp_path / "a.png"),
    )
    pdf = build_reportlab_pdf(FIGURES_EN, tmp_path / "a.pdf")
    doc = load_pdf(pdf, PdfConfig(), lang="en")
    in_pdf = next(b.image for b in doc.blocks if b.image is not None)
    with PixelLoader(pdf) as loader:
        sizes = [loader.load(ref) for ref in (in_data, in_file, in_pdf)]
    assert [image.size for image in sizes if image is not None] == [
        (400, 250),
        (400, 250),
        (400, 250),
    ]


def test_loader_survives_a_vanished_file(tmp_path: Path) -> None:
    """A file removed since the extraction gives no pixels, quietly."""
    data = encode(drawing("setup"))
    gone = raster_ref(
        data,
        name="gone.png",
        source=PixelSource(file=tmp_path / "gone.png"),
    )
    with PixelLoader(tmp_path) as loader:
        assert loader.load(gone) is None


def test_closed_loader_reads_nothing(tmp_path: Path) -> None:
    """A closed loader gives no pixels and does not open the PDF again."""
    pdf = build_reportlab_pdf(FIGURES_EN, tmp_path / "a.pdf")
    doc = load_pdf(pdf, PdfConfig(), lang="en")
    ref = next(b.image for b in doc.blocks if b.image is not None)
    loader = PixelLoader(pdf)
    with loader:
        assert loader.load(ref) is not None
    assert loader.load(ref) is None


@pytest.mark.parametrize(
    ("bpp", "colorspace", "bits"),
    [
        (1, pdfium_c.FPDF_COLORSPACE_UNKNOWN, 1),  # An image mask.
        (8, pdfium_c.FPDF_COLORSPACE_DEVICEGRAY, 2),  # 2, 4 or 8 bits.
        (8, pdfium_c.FPDF_COLORSPACE_DEVICERGB, 3),  # 1 or 2 bits each.
        (24, pdfium_c.FPDF_COLORSPACE_DEVICERGB, 12),  # 4, 8 or 16 bits.
        (24, pdfium_c.FPDF_COLORSPACE_DEVICEGRAY, 16),
        (24, pdfium_c.FPDF_COLORSPACE_ICCBASED, 12),  # 1, 3 or 4 colors.
        (24, pdfium_c.FPDF_COLORSPACE_DEVICEN, 9),  # Up to 32 colors.
        (0, pdfium_c.FPDF_COLORSPACE_UNKNOWN, 0),  # Not known: no check.
    ],
)
def test_least_length_takes_the_fewest_bits(
    bpp: int,
    colorspace: int,
    bits: int,
) -> None:
    """The bytes a stream needs are those of the fewest bits it may have."""
    width, height = 401, 3
    assert least_length((width, height), bpp, colorspace) == (
        (width * bits + 7) // 8 * height
    )


@pytest.mark.parametrize("fmt", ["JPEG", "MPO"])
def test_jpeg_of_too_many_scans_is_too_large(
    fmt: Literal["JPEG", "MPO"],
) -> None:
    """A JPEG of too many scans is refused before any decoder runs, MPO too."""
    with pytest.raises(UnreadableImage) as caught:
        open_raster(progressive_jpeg(70, fmt=fmt))
    assert caught.value.reason == TOO_LARGE
    image = open_raster(progressive_jpeg(0, fmt=fmt))
    assert (image.size, image.format) == ((64, 64), fmt)


def test_pdf_jpeg_of_too_many_scans_is_too_large(tmp_path: Path) -> None:
    """A DCT stream of too many scans is refused before PDFium decodes it."""
    problems = []
    for extra in (70, 0):
        pdf = build_image_stream_pdf(
            tmp_path / f"s{extra}.pdf",
            progressive_jpeg(extra),
            filters="/DCTDecode",
            size=(64, 64),
        )
        doc = load_pdf(pdf, PdfConfig(), lang="en")
        image = next(b.image for b in doc.blocks if b.image is not None)
        problems.append(image.problem)
    assert problems == [TOO_LARGE, None]
