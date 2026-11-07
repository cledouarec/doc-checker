# Copyright 2026 Christophe Le Douarec
"""Builders of test images: drawings, encodings, defects, forged files."""

import io
import random
import struct
import zlib
from dataclasses import dataclass
from hashlib import blake2b
from typing import Literal

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from doc_checker.images.pixels import flatten

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _color(rng: random.Random) -> tuple[int, int, int]:
    """Draw a color."""
    return (rng.randrange(256), rng.randrange(256), rng.randrange(256))


def drawing(name: str, size: tuple[int, int] = (400, 250)) -> Image.Image:
    """Draw shapes and a title, the same every time for a name.

    Args:
        name: Name of the drawing, which seeds the shapes and is written.
        size: Width and height in pixels.

    Returns:
        An RGB image.
    """
    seed = int.from_bytes(blake2b(name.encode()).digest()[:8])
    rng = random.Random(seed)  # ruff: ignore[suspicious-non-cryptographic-random-usage] - test drawing, not security
    width, height = size
    # A side of 20 pixels or less, as a logo has, puts the shapes at 0.
    left, top = max(1, width - 20), max(1, height - 20)
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    for _ in range(12):
        x, y = rng.randrange(left), rng.randrange(top)
        draw.rectangle(
            (x, y, x + rng.randrange(20, 160), y + rng.randrange(20, 120)),
            fill=_color(rng),
        )
    for _ in range(6):
        x, y = rng.randrange(left), rng.randrange(top)
        draw.ellipse(
            (x, y, x + rng.randrange(20, 140), y + rng.randrange(20, 100)),
            fill=_color(rng),
        )
    for _ in range(4):
        draw.line(
            (
                rng.randrange(width),
                rng.randrange(height),
                rng.randrange(width),
                rng.randrange(height),
            ),
            fill=_color(rng),
            width=3,
        )
    draw.text((10, 10), name, fill="black")
    return image


def encode(image: Image.Image, fmt: str = "PNG", **options: object) -> bytes:
    """Encode an image.

    Args:
        image: The image; JPEG and BMP get it flattened on white.
        fmt: Pillow format name.
        **options: Options of the Pillow encoder.

    Returns:
        The encoded bytes.
    """
    if fmt.upper() in {"JPEG", "BMP"}:
        image = flatten(image)
    buffer = io.BytesIO()
    image.save(buffer, fmt, **options)
    return buffer.getvalue()


def scaled(image: Image.Image, factor: float) -> Image.Image:
    """Resize an image by a factor, with LANCZOS.

    Args:
        image: The image.
        factor: Scale of both sides.

    Returns:
        The resized image.
    """
    size = (round(image.width * factor), round(image.height * factor))
    return image.resize(size, Image.Resampling.LANCZOS)


def pixelated(image: Image.Image, factor: float = 0.4) -> Image.Image:
    """Lose resolution, then enlarge to the original size by pixels.

    Args:
        image: The image.
        factor: Scale of the intermediate image.

    Returns:
        An image of the original size.
    """
    return scaled(image, factor).resize(image.size, Image.Resampling.NEAREST)


def blurred(image: Image.Image, radius: float = 2.0) -> Image.Image:
    """Blur an image.

    Args:
        image: The image.
        radius: Radius of the Gaussian blur.

    Returns:
        The blurred image.
    """
    return image.filter(ImageFilter.GaussianBlur(radius))


def grayed(image: Image.Image) -> Image.Image:
    """Remove the colors of an image, keeping it RGB.

    Args:
        image: The image.

    Returns:
        An RGB image of grays.
    """
    return image.convert("L").convert("RGB")


def _chunk(kind: bytes, body: bytes) -> bytes:
    """Write a PNG chunk with its CRC."""
    crc = zlib.crc32(kind + body)
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", crc)


def bomb_png(width: int, height: int) -> bytes:
    """Forge a PNG that announces a size and holds one row of pixels.

    Args:
        width: Announced width.
        height: Announced height.

    Returns:
        The bytes of the file.
    """
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    row = zlib.compress(b"\x00" + b"\x00" * (width * 3))
    return (
        _PNG_SIGNATURE
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", row)
        + _chunk(b"IEND", b"")
    )


def jpeg_announcing(width: int, height: int) -> bytes:
    """Forge a JPEG of the setup drawing whose frame header announces a size.

    Args:
        width: Announced width.
        height: Announced height.

    Returns:
        The bytes of a JPEG of 400 by 250 pixels, of quality 90, whose SOF0
        segment announces this size.
    """
    data = encode(drawing("setup"), "JPEG", quality=90)
    start = data.index(b"\xff\xc0") + 5  # Length and precision come first.
    size = struct.pack(">HH", height, width)
    return data[:start] + size + data[start + len(size) :]


def jpx_announcing(width: int, height: int) -> bytes:
    """Forge a JPEG 2000 codestream whose SIZ marker announces a size.

    Args:
        width: Announced width.
        height: Announced height.

    Returns:
        The bytes of a codestream of the setup drawing, whose ``Xsiz`` and
        ``Ysiz`` announce this size, with no offset.
    """
    data = encode(drawing("setup"), "JPEG2000", no_jp2=True)
    start = 8  # After SOC, SIZ, Lsiz and Rsiz.
    size = struct.pack(">II", width, height)
    return data[:start] + size + data[start + len(size) :]


def jpx_of_components(count: int) -> bytes:
    """Forge a JPEG 2000 codestream whose SIZ marker announces components.

    Args:
        count: Announced number of components, 3 or more.

    Returns:
        The bytes of a codestream of the setup drawing, of 400 by 250
        pixels, whose ``Csiz`` announces `count` components, the ones added
        sampled as the first.
    """
    data = encode(drawing("setup"), "JPEG2000", no_jp2=True)
    length = int.from_bytes(data[4:6])  # Lsiz, after SOC and SIZ.
    first = data[42:45]  # Ssiz, XRsiz and YRsiz of the first component.
    siz = data[6:40] + struct.pack(">H", count) + data[42:51]
    siz += first * (count - 3)
    rest = data[4 + length :]
    return data[:4] + struct.pack(">H", len(siz) + 2) + siz + rest


def jpeg_of_components(count: int) -> bytes:
    """Forge a JPEG whose frame header announces components.

    Args:
        count: Announced number of components, 3 or more.

    Returns:
        The bytes of a JPEG of the setup drawing, of 400 by 250 pixels and
        quality 90, whose SOF0 segment announces `count` components, the
        ones added sampled and quantized as the last.
    """
    data = encode(drawing("setup"), "JPEG", quality=90)
    start = data.index(b"\xff\xc0") + 2  # The length of the frame header.
    end = start + int.from_bytes(data[start : start + 2])
    last = data[end - 3 : end]  # Identifier, sampling and table.
    added = b"".join(
        bytes([last[0] + number]) + last[1:] for number in range(1, count - 2)
    )
    frame = data[start + 2 : start + 7] + bytes([count])
    frame += data[start + 8 : end] + added
    return (
        data[:start] + struct.pack(">H", len(frame) + 2) + frame + data[end:]
    )


def cmyk_encoded(fmt: Literal["JPEG", "JPEG2000"]) -> bytes:
    """Encode the setup drawing in CMYK, in four components.

    Args:
        fmt: ``JPEG``, of quality 90, or ``JPEG2000``, a bare codestream.

    Returns:
        The encoded bytes.
    """
    cmyk = drawing("setup").convert("CMYK")
    buffer = io.BytesIO()
    if fmt == "JPEG":
        cmyk.save(buffer, fmt, quality=90)
    else:
        # The encoder takes no CMYK: the four bands go in as RGBA.
        Image.merge("RGBA", cmyk.split()).save(buffer, fmt, no_jp2=True)
    return buffer.getvalue()


PREDICTOR_COLUMNS = 16384  # Bytes of a row of the decoy, after its tag byte.


def jpeg_behind_a_predictor(width: int, height: int) -> bytes:
    """Forge bytes that hold one JPEG, and another once a predictor runs.

    The bytes are a JPEG of the setup drawing of 400 by 250 pixels, whose
    APP15 segment holds a second JPEG that announces a size. Run through a
    PNG predictor of `PREDICTOR_COLUMNS` columns, which drops the tag byte
    that starts each row, they start with the second JPEG instead: a first
    row of zeros that ends with 0xFF, then 0xD8 as the tag of the next row.

    Args:
        width: Announced width of the hidden JPEG.
        height: Announced height of the hidden JPEG.

    Returns:
        The bytes, before the predictor runs.
    """
    hidden = jpeg_announcing(width, height)
    app15 = b"\xff\xef" + struct.pack(">H", len(hidden) + 2) + hidden
    first_row = bytes(PREDICTOR_COLUMNS - 1) + b"\xff"
    return (
        b"\x00"
        + first_row
        + b"\xd8"
        + app15
        + encode(
            drawing("setup"),
            "JPEG",
            quality=90,
        )[2:]
    )


def ccitt_group4(image: Image.Image) -> bytes:
    """Encode a picture as one strip of CCITT Group 4, black on white.

    Args:
        image: The picture, made bilevel.

    Returns:
        The bytes of the strip, as a PDF stream holds them.
    """
    bilevel = image.convert("1")
    buffer = io.BytesIO()
    bilevel.save(
        buffer,
        "TIFF",
        compression="group4",
        strip_size=1 << 30,
    )
    with Image.open(buffer) as tiff:
        offset = tiff.tag_v2[273][0]  # StripOffsets.
        length = tiff.tag_v2[279][0]  # StripByteCounts.
    return buffer.getvalue()[offset : offset + length]


FIGURE_SIZE = (300, 200)
_LABEL_AT = ((20, 50), (210, 50))


@dataclass(frozen=True)
class Shape:
    """A rectangle or an ellipse of a vector figure.

    Attributes:
        kind: The shape.
        box: Left, top, right and bottom, in SVG coordinates, on a 300 by
            200 canvas.
        fill: The fill color, ``#rrggbb``.
    """

    kind: Literal["rect", "ellipse"]
    box: tuple[int, int, int, int]
    fill: str


def shapes(*, flipped: bool = False) -> tuple[Shape, ...]:
    """Return the five touching shapes of a figure.

    Args:
        flipped: Whether to flip each box vertically, for another drawing.

    Returns:
        The shapes, which form one cluster of vector paths.
    """
    found = (
        Shape("rect", (10, 20, 100, 70), "#4477aa"),
        Shape("rect", (100, 40, 200, 46), "#ccbb44"),
        Shape("rect", (200, 20, 290, 70), "#ee6677"),
        Shape("rect", (147, 46, 153, 92), "#66ccee"),
        Shape("ellipse", (100, 90, 200, 150), "#228833"),
    )
    if not flipped:
        return found
    height = FIGURE_SIZE[1]
    return tuple(
        Shape(
            s.kind,
            (s.box[0], height - s.box[3], s.box[2], height - s.box[1]),
            s.fill,
        )
        for s in found
    )


def labels(name: str) -> tuple[str, str]:
    """Return the two labels of a figure."""
    return (f"{name} in", f"{name} out")


def label_positions(*, flipped: bool = False) -> tuple[tuple[int, int], ...]:
    """Return where the two labels are written, at their baseline."""
    if not flipped:
        return _LABEL_AT
    # The second label starts further left, to stay within the shapes.
    return ((20, FIGURE_SIZE[1] - 50), (160, FIGURE_SIZE[1] - 50))


def figure_svg(name: str, *, flipped: bool = False) -> str:
    """Write a figure as an SVG.

    Args:
        name: The name, which the labels carry.
        flipped: Whether to draw the flipped shapes and labels.

    Returns:
        The SVG document.
    """
    parts = [
        (
            '<svg xmlns="http://www.w3.org/2000/svg" width="300" '
            'height="200" viewBox="0 0 300 200">'
        ),
    ]
    for shape in shapes(flipped=flipped):
        x0, y0, x1, y1 = shape.box
        if shape.kind == "rect":
            parts.append(
                f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" '
                f'height="{y1 - y0}" fill="{shape.fill}"/>',
            )
        else:
            parts.append(
                f'<ellipse cx="{(x0 + x1) / 2:g}" cy="{(y0 + y1) / 2:g}" '
                f'rx="{(x1 - x0) / 2:g}" ry="{(y1 - y0) / 2:g}" '
                f'fill="{shape.fill}"/>',
            )
    parts.extend(
        f'<text x="{x}" y="{y}" font-size="14">{label}</text>'
        for label, (x, y) in zip(
            labels(name),
            label_positions(flipped=flipped),
            strict=True,
        )
    )
    parts.append("</svg>")
    return "".join(parts)


def figure_drawing(name: str, *, flipped: bool = False) -> Image.Image:
    """Draw the shapes and the labels of a figure with Pillow.

    Args:
        name: The name, which the labels carry.
        flipped: Whether to draw the flipped shapes and labels.

    Returns:
        A 300 by 200 RGB image on white.
    """
    image = Image.new("RGB", FIGURE_SIZE, "white")
    draw = ImageDraw.Draw(image)
    for shape in shapes(flipped=flipped):
        if shape.kind == "rect":
            draw.rectangle(shape.box, fill=shape.fill)
        else:
            draw.ellipse(shape.box, fill=shape.fill)
    font = ImageFont.load_default(14)
    for label, at in zip(
        labels(name),
        label_positions(flipped=flipped),
        strict=True,
    ):
        draw.text(at, label, fill="black", font=font, anchor="ls")
    return image


def logo() -> Image.Image:
    """Draw a 60 by 20 logo: white letters on a blue ground."""
    image = Image.new("RGB", (60, 20), "#1f77b4")
    ImageDraw.Draw(image).text((6, 4), "LOGO", fill="white")
    return image
