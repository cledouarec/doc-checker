# Copyright 2026 Christophe Le Douarec
"""Decode the image objects of a PDF safely, and read pixels again on demand.

The extraction decodes each image once, for its signature, and drops its
pixels. They are read again from their `PixelSource` only for the measures of
a pair of images that changed and for the thumbnails of the reports. An image
that can no longer be read, such as a file removed since the extraction,
gives no pixels, without any error. A vector figure of a PDF is rendered by
PDFium, from the box of its page, for its signature and again on demand.

Before PDFium decodes an image object, its stream is checked within bounds.
Its filters must form a chain that generators write. The size that its
dictionary, or the header of its JPEG or JPEG 2000, announces must stay under
the pixel limit, and that header must announce at most four components; a
JPEG must decode without error; and a Flate or raw stream of pixels must hold
the bytes that they need. A wrapped JPEG or JPEG 2000 is checked in the bytes
that PDFium gives its decoder, once their length is known to be bounded: by
counting what a Flate layer inflates to, or by the length of the stream of an
ASCII layer. An image that fails is never decoded by PDFium, nor is a figure
rendered over it.
"""

import ctypes
import io
import warnings
from contextlib import closing
from dataclasses import dataclass
from itertools import islice
from typing import TYPE_CHECKING, Final, Self

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from PIL import Image

from doc_checker.images.objects import (
    Box,
    PageObject,
    form_children,
    form_matrix,
    object_bounds,
    page_children,
)
from doc_checker.images.pixels import (
    CANNOT_DECODE,
    MAX_PIXELS,
    TOO_LARGE,
    UnreadableImage,
    jpeg_quality,
    open_raster,
    read_image_file,
)
from doc_checker.images.streams import (
    CODEC_CHAINS,
    DCT,
    FLATE,
    JPX,
    PIXEL_CHAINS,
    WRAPPED_CHAINS,
    CodecHeader,
    full_filters,
    inflated_length,
    jpeg_header,
    jpx_header,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from types import TracebackType

    from doc_checker.model.document import ImageRef, PixelSource

IMAGE_DEPTH: Final = 2  # The images of a page and those of its forms.
RENDER_SCALE: Final = 2.0  # Pixels for each point of a figure: 144 DPI.
RENDER_SIDE_MAX: Final = 2048  # Most pixels on a side of a figure render.
# The pixels around the box of a render whose objects are checked too: PDFium
# draws the objects that meet its pixels, which round the box.
_RENDER_MARGIN: Final = 2
# The most bytes of a JPEG or a JPEG 2000 for each pixel that its dictionary
# announces: four components, twice their decoded size; and the bytes added
# for its header, room for large ICC or XML segments.
_CODEC_SPREAD: Final = 8
_CODEC_SLACK: Final = 1 << 20
# The longest stream of a codec wrapped in an ASCII layer: PDFium decodes it
# whole for the checks, so a longer one is refused.
_ASCII_LONGEST: Final = 16_000_000
_JPEG_DRAFT: Final = 8  # A JPEG is checked at an eighth of its size.
# The most components of a JPEG or a JPEG 2000: gray, RGB or CMYK, which the
# generators write. Their decoders hold every component at the full size.
_MOST_COMPONENTS: Final = 4
_SOI: Final = b"\xff\xd8"
# The components that each colour space family of PDFium can have.
_FAMILY_COMPONENTS: Final = {
    pdfium_c.FPDF_COLORSPACE_DEVICEGRAY: (1,),
    pdfium_c.FPDF_COLORSPACE_CALGRAY: (1,),
    pdfium_c.FPDF_COLORSPACE_SEPARATION: (1,),
    pdfium_c.FPDF_COLORSPACE_INDEXED: (1,),
    pdfium_c.FPDF_COLORSPACE_DEVICERGB: (3,),
    pdfium_c.FPDF_COLORSPACE_CALRGB: (3,),
    pdfium_c.FPDF_COLORSPACE_LAB: (3,),
    pdfium_c.FPDF_COLORSPACE_DEVICECMYK: (4,),
    pdfium_c.FPDF_COLORSPACE_ICCBASED: (1, 3, 4),
}
_ANY_COMPONENTS: Final = tuple(range(1, 33))  # DeviceN, or not known.
_DEPTHS: Final = (1, 2, 4, 8, 16)  # The bits of a component.
_BYTE: Final = 8
# PDFium decodes an image of up to 8 bits a pixel to 8 bits, and others to 24.
_SHORT_PIXEL: Final = 8


@dataclass(frozen=True, slots=True)
class DecodedImage:
    """The pixels of a PDF image object, and the quality of its JPEG.

    Attributes:
        pixels: The pixels, without the mask of the image.
        quality: The estimated quality of a JPEG, ``None`` for another
            image.
    """

    pixels: Image.Image
    quality: int | None


def image_objects(page: pdfium.PdfPage) -> Iterator[pdfium.PdfImage]:
    """Yield the image objects of a PDF page, in the order of their index.

    Args:
        page: The page.

    Yields:
        The image objects of the page and those of its forms, in the order
        that `PixelSource.index` counts.
    """
    yield from page.get_objects(
        filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE],
        max_depth=IMAGE_DEPTH,
    )


def object_stream(image: pdfium.PdfImage) -> tuple[bytes, tuple[str, ...]]:
    """Return the stream of an image object, as written, and its filters.

    Args:
        image: The image object.

    Returns:
        The bytes of the stream, and the full names of its filters, in the
        order they apply.
    """
    data = bytes(image.get_data(decode_simple=False))
    return data, full_filters(image.get_filters())


def _checked_header(
    header: CodecHeader | None,
    announced: int,
) -> CodecHeader:
    """Check what the header of a JPEG or a JPEG 2000 announces.

    Args:
        header: The header, or ``None`` if it cannot be read.
        announced: The pixels that the dictionary of the image announces.

    Returns:
        The header.

    Raises:
        UnreadableImage: `CANNOT_DECODE` if the header cannot be read or
            announces more than `_MOST_COMPONENTS`; `TOO_LARGE` if it
            announces more pixels than `MAX_PIXELS`, or than the dictionary.
    """
    if header is None:
        raise UnreadableImage(CANNOT_DECODE)
    pixels = header.width * header.height
    if pixels > MAX_PIXELS or pixels > announced:
        raise UnreadableImage(TOO_LARGE)
    if header.components > _MOST_COMPONENTS:
        raise UnreadableImage(CANNOT_DECODE)
    return header


def _checked_wrapped(
    image: pdfium.PdfImage,
    data: bytes,
    chain: tuple[str, ...],
    size: tuple[int, int],
) -> int | None:
    """Check a JPEG or JPEG 2000 that one filter wraps.

    The bytes that PDFium gives its decoder are bounded first: a Flate layer
    is inflated here, and dropped, up to what any image of its size needs;
    an ASCII layer has a stream of at most `_ASCII_LONGEST` bytes. PDFium
    then decodes the layer, and the codec is checked in the bytes it gives,
    which are the raw stream when its layer gives none.

    Args:
        image: The image object.
        data: Its stream, as written.
        chain: Its filters, a wrapping filter then a codec.
        size: The width and the height that its dictionary announces.

    Returns:
        The quality of a JPEG, ``None`` for a JPEG 2000.

    Raises:
        UnreadableImage: If the layer is too long, or if the codec is
            refused.
    """
    wrapper, codec = chain
    announced = size[0] * size[1]
    if wrapper == FLATE:
        bound = announced * _CODEC_SPREAD + _CODEC_SLACK
        if inflated_length([data], bound + 1) > bound:
            raise UnreadableImage(CANNOT_DECODE)
    elif len(data) > _ASCII_LONGEST:
        raise UnreadableImage(CANNOT_DECODE)
    return _checked_codec(
        bytes(image.get_data(decode_simple=True)),
        codec,
        announced,
    )


def _checked_codec(stream: bytes, codec: str, announced: int) -> int | None:
    """Check a JPEG or JPEG 2000 before PDFium decodes it.

    Args:
        stream: The codec stream, as PDFium decodes it.
        codec: ``DCTDecode`` or ``JPXDecode``.
        announced: The pixels that the dictionary of the image announces.

    Returns:
        The quality of a JPEG, ``None`` for a JPEG 2000.
    """
    if codec == DCT:
        return _checked_jpeg(stream, announced)
    _checked_header(jpx_header(stream), announced)
    return None


def _draft_decoded(jpeg: bytes, header: CodecHeader) -> int | None:
    """Decode a JPEG at an eighth of its size, and estimate its quality.

    The decoder still reads every byte of the scans, so a truncated JPEG
    fails as it would at full size, but at a fraction of the cost.

    Args:
        jpeg: The JPEG, maybe after other bytes.
        header: What its frame header announces.

    Returns:
        The quality, as the luminance table tells, or ``None`` without one.
    """
    draft = (
        max(1, header.width // _JPEG_DRAFT),
        max(1, header.height // _JPEG_DRAFT),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        stream = io.BytesIO(jpeg[jpeg.find(_SOI) :])
        with Image.open(stream, formats=["JPEG"]) as image:
            quality = jpeg_quality(image)
            image.draft(image.mode, draft)
            image.load()
    return quality


def _checked_jpeg(jpeg: bytes, announced: int) -> int | None:
    """Check a JPEG before PDFium decodes it, and estimate its quality.

    Args:
        jpeg: The JPEG, as PDFium decodes it.
        announced: The pixels that the dictionary of the image announces.

    Returns:
        The quality of the JPEG, as its luminance table tells, or ``None``
        without one.

    Raises:
        UnreadableImage: If its frame header cannot be read or announces too
            many pixels or components, or if the JPEG does not decode
            without error.
    """
    header = _checked_header(jpeg_header(jpeg), announced)
    try:
        return _draft_decoded(jpeg, header)
    except Exception:  # ruff: ignore[blind-except] - any decoder failure on an untrusted JPEG means it cannot be decoded
        raise UnreadableImage(CANNOT_DECODE) from None


def least_length(size: tuple[int, int], bpp: int, colorspace: int) -> int:
    """Return the fewest decoded bytes that the pixels of an image need.

    The dictionary of the image is not read: the bits of each pixel are the
    fewest that its colour space family and the bits of a pixel that PDFium
    decodes it to allow. Each row is padded to a whole byte.

    Args:
        size: The width and the height of the image, in pixels.
        bpp: The bits of a pixel once PDFium decodes it: 1, 8 or 24, or 0
            when PDFium cannot tell.
        colorspace: The colour space family, as PDFium tells it.

    Returns:
        The fewest bytes, 0 when PDFium cannot tell the bits of a pixel.
    """
    width, height = size
    bits = bpp
    if bpp > 1:
        counts = _FAMILY_COMPONENTS.get(colorspace, _ANY_COMPONENTS)
        bits = min(
            count * depth
            for count in counts
            for depth in _DEPTHS
            if count * depth > 1
            and (count * depth > _SHORT_PIXEL) == (bpp > _SHORT_PIXEL)
        )
    return (width * bits + _BYTE - 1) // _BYTE * height


def _check_length(
    image: pdfium.PdfImage,
    data: bytes,
    chain: tuple[str, ...],
    size: tuple[int, int],
) -> None:
    """Check that a Flate or raw stream of pixels holds the bytes they need.

    PDFium decodes a truncated or corrupt stream without error, and leaves
    blank the pixels that it lacks. Only the bytes needed are inflated, and
    dropped. The other chains of pixels are not checked: a stream cut short
    there gives a partly blank image, which the comparison sees.

    Args:
        image: The image object.
        data: The stream, as written.
        chain: Its filters, a chain of pixels.
        size: The width and the height that its dictionary announces.

    Raises:
        UnreadableImage: If the decoded stream is shorter than its pixels
            need.
    """
    if chain not in {(), (FLATE,)}:
        return
    metadata = image.get_metadata()
    least = least_length(size, metadata.bits_per_pixel, metadata.colorspace)
    length = len(data) if not chain else inflated_length([data], least)
    if length < least:
        raise UnreadableImage(CANNOT_DECODE)


def _checked_stream(
    image: pdfium.PdfImage,
    data: bytes,
    filters: tuple[str, ...],
    size: tuple[int, int],
) -> int | None:
    """Check the stream of an image object before PDFium decodes it.

    Args:
        image: The image object.
        data: The stream, as written.
        filters: The full names of its filters.
        size: The width and the height that its dictionary announces.

    Returns:
        The quality of a JPEG, ``None`` for another image.

    Raises:
        UnreadableImage: If the stream is refused, and among them each chain
            of filters that generators do not write.
    """
    if filters in PIXEL_CHAINS:
        _check_length(image, data, filters, size)
        return None
    if filters in WRAPPED_CHAINS:
        return _checked_wrapped(image, data, filters, size)
    if filters not in CODEC_CHAINS:
        raise UnreadableImage(CANNOT_DECODE)
    if filters[-1] in {DCT, JPX}:
        return _checked_codec(data, filters[-1], size[0] * size[1])
    return None


def decode_object(
    image: pdfium.PdfImage,
    data: bytes,
    filters: tuple[str, ...],
) -> DecodedImage:
    """Check the stream of a PDF image object, then decode its pixels.

    Args:
        image: The image object.
        data: Its stream, as written.
        filters: The full names of its filters, as `object_stream` gives
            them.

    Returns:
        The pixels, without the mask of the image, and the quality of a
        JPEG.

    Raises:
        UnreadableImage: If the dictionary of the object, or the header of
            its JPEG or JPEG 2000, announces more than `MAX_PIXELS`, which
            refuses it before any decoding; or if its stream is short, its
            JPEG does not decode, or PDFium cannot decode it.
    """
    try:
        quality = _checked_object(image, data, filters)
        pixels = image.get_bitmap(render=False).to_pil()
    except pdfium.PdfiumError:
        raise UnreadableImage(CANNOT_DECODE) from None
    return DecodedImage(pixels, quality)


def _checked_object(
    image: pdfium.PdfImage,
    data: bytes,
    filters: tuple[str, ...],
) -> int | None:
    """Check a PDF image object, as `decode_object` does, without decoding it.

    Args:
        image: The image object.
        data: Its stream, as written.
        filters: The full names of its filters.

    Returns:
        The quality of a JPEG, ``None`` for another image.

    Raises:
        UnreadableImage: If the dictionary announces more than `MAX_PIXELS`,
            or if the stream is refused.
    """
    size = image.get_px_size()
    if size[0] * size[1] > MAX_PIXELS:
        raise UnreadableImage(TOO_LARGE)
    return _checked_stream(image, data, filters, size)


def _may_show(bounds: Box | None, box: Box) -> bool:
    """Tell whether an object may be drawn in a box.

    Args:
        bounds: The bounds of the object on the page, or ``None`` if they
            are not known.
        box: The box.

    Returns:
        False only for bounds known to lie outside the box. Bounds that are
        not known, or not numbers, may show in it.
    """
    if bounds is None:
        return True
    left, bottom, right, top = bounds
    box_left, box_bottom, box_right, box_top = box
    return not (
        right < box_left
        or left > box_right
        or top < box_bottom
        or bottom > box_top
    )


class RenderGuard:
    """The image objects that a render of a page may decode, and their checks.

    The objects of the page are walked once, at any depth, for its image
    objects and their bounds on the page; each image is checked at most once,
    as `decode_object` checks it, when a render first meets it.
    """

    def __init__(self, page: pdfium.PdfPage) -> None:
        """Keep the page, whose objects are walked when a render needs them.

        Args:
            page: The page.
        """
        self._page = page
        self._images: list[tuple[pdfium.PdfImage, Box | None]] | None = None
        self._refused: dict[int, str | None] = {}

    def _walk(self) -> list[tuple[pdfium.PdfImage, Box | None]]:
        """Find the image objects of the page, at any depth.

        Returns:
            Each image object, and its bounds on the page, ``None`` if they
            are not known.
        """
        page = self._page
        images: list[tuple[pdfium.PdfImage, Box | None]] = []
        # Each form to walk, None for the page, with the matrix from its
        # space to the page, None if it is not known.
        pending: list[tuple[PageObject | None, pdfium.PdfMatrix | None]] = [
            (None, pdfium.PdfMatrix()),
        ]
        while pending:
            form, matrix = pending.pop()
            children = (
                page_children(page) if form is None else (form_children(form))
            )
            for _, child in children:
                kind = pdfium_c.FPDFPageObj_GetType(child)
                if kind == pdfium_c.FPDF_PAGEOBJ_FORM:
                    pending.append((child, form_matrix(child, matrix)))
                elif kind == pdfium_c.FPDF_PAGEOBJ_IMAGE:
                    bounds = object_bounds(child)
                    if bounds is not None and matrix is not None:
                        bounds = matrix.on_rect(*bounds)
                    else:
                        bounds = None
                    images.append((pdfium.PdfObject(child, page=page), bounds))
        return images

    def check(self, box: Box) -> None:
        """Check the image objects that may show in a box of the page.

        Args:
            box: The box, in the coordinates of PDFium for the page.

        Raises:
            UnreadableImage: With the reason of the first image refused.
        """
        if self._images is None:
            self._images = self._walk()
        for image, bounds in self._images:
            if not _may_show(bounds, box):
                continue
            address = ctypes.addressof(image.raw.contents)
            if address not in self._refused:
                self._refused[address] = _refusal(image)
            reason = self._refused[address]
            if reason is not None:
                raise UnreadableImage(reason)


def _refusal(image: pdfium.PdfImage) -> str | None:
    """Return why the checks refuse an image object, ``None`` if they pass.

    Args:
        image: The image object.

    Returns:
        The reason, as `UnreadableImage` gives it.
    """
    try:
        data, filters = object_stream(image)
        _checked_object(image, data, filters)
    except UnreadableImage as error:
        return error.reason
    except pdfium.PdfiumError:
        return CANNOT_DECODE
    return None


def render_figure(
    page: pdfium.PdfPage,
    box: Box,
    guard: RenderGuard | None = None,
) -> Image.Image:
    """Render a box of a page, as a vector figure shows there.

    The scale is `RENDER_SCALE`, or less, so that no side of the render has
    more than `RENDER_SIDE_MAX` pixels. Before PDFium renders, every image
    object that it may decode there, or a few pixels around, is checked as
    `decode_object` checks it. The annotations of the page are not drawn.

    Args:
        page: The page, which `/Rotate` does not turn.
        box: The left, bottom, right and top edges of the box, in the
            coordinates of PDFium for the page, within its crop box.
        guard: The checks of the image objects of the page, shared by the
            renders of its figures, or ``None`` for checks of its own.

    Returns:
        The render, in RGB.

    Raises:
        UnreadableImage: If an image object that may show in the box is
            refused, or if PDFium cannot render the box.
    """
    page_left, page_bottom, page_right, page_top = page.get_bbox()
    left, bottom, right, top = box
    longest = max(right - left, top - bottom)
    scale = min(RENDER_SCALE, RENDER_SIDE_MAX / longest)
    margin = _RENDER_MARGIN / scale
    around = (left - margin, bottom - margin, right + margin, top + margin)
    try:
        (guard or RenderGuard(page)).check(around)
        bitmap = page.render(
            scale=scale,
            crop=(
                left - page_left,
                bottom - page_bottom,
                page_right - right,
                page_top - top,
            ),
            draw_annots=False,
        )
    except pdfium.PdfiumError, ValueError:
        raise UnreadableImage(CANNOT_DECODE) from None
    return bitmap.to_pil()


class PixelLoader:
    """Read the pixels of the images of a document again.

    A PDF is opened at the first image that it is asked for, and closed with
    the loader, which is a context manager. A closed loader reads nothing.
    """

    def __init__(self, path: Path) -> None:
        """Keep the document whose images are read.

        Args:
            path: The PDF file, or the HTML file or site directory, whose
                images give the references to load.
        """
        self._path = path
        self._pdf: pdfium.PdfDocument | None = None
        self._closed = False

    def __enter__(self) -> Self:
        """Return the loader.

        Returns:
            The loader itself.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the PDF, if one was opened."""
        self.close()

    def close(self) -> None:
        """Close the PDF, if one was opened, and read nothing more."""
        self._closed = True
        if self._pdf is not None:
            self._pdf.close()
            self._pdf = None

    def load(self, ref: ImageRef) -> Image.Image | None:
        """Read the pixels of an image again, or render a vector figure.

        Args:
            ref: The image, as the extraction found it.

        Returns:
            Its pixels, or ``None`` if it has no source, can no longer be
            read, or if the loader is closed.
        """
        source = ref.source
        if source is None or self._closed:
            return None
        try:
            return self._read(source)
        except UnreadableImage, pdfium.PdfiumError, OSError:
            return None

    def _read(self, source: PixelSource) -> Image.Image | None:
        """Read the pixels of a source, raising what stops it.

        Args:
            source: Where the pixels are.

        Returns:
            The pixels, or ``None`` for a source that tells no pixels.
        """
        if source.data is not None:
            return open_raster(source.data)
        if source.file is not None:
            return open_raster(read_image_file(source.file))
        if source.page is None:
            return None
        if source.index is not None:
            return self._pdf_pixels(source.page, source.index)
        if source.box is not None:
            return self._pdf_render(source.page, source.box)
        return None

    def _pdf_page(self, page: int) -> pdfium.PdfPage:
        """Open a page of the PDF, and the PDF first if it is not open.

        Args:
            page: Position of the page, from 0.

        Returns:
            The page, which the caller closes.
        """
        if self._pdf is None:
            self._pdf = pdfium.PdfDocument(self._path)
        return self._pdf[page]

    def _pdf_pixels(self, page: int, index: int) -> Image.Image | None:
        """Read the pixels of an image object of the PDF, with its checks.

        Args:
            page: Position of the page, from 0.
            index: Position of the image among the image objects of the page.

        Returns:
            The pixels, or ``None`` if the page has no such image. An image
            that `decode_object` refuses raises its `UnreadableImage`.
        """
        with closing(self._pdf_page(page)) as pdf_page:
            image = next(islice(image_objects(pdf_page), index, None), None)
            if image is None:
                return None
            data, filters = object_stream(image)
            return decode_object(image, data, filters).pixels

    def _pdf_render(self, page: int, box: Box) -> Image.Image:
        """Render a vector figure of the PDF again, as `render_figure` does.

        Args:
            page: Position of the page, from 0.
            box: The box of the figure, in the coordinates of PDFium for the
                page.

        Returns:
            The render. A box that `render_figure` refuses raises its
            `UnreadableImage`.
        """
        with closing(self._pdf_page(page)) as pdf_page:
            return render_figure(pdf_page, box)
