# Copyright 2026 Christophe Le Douarec
"""Identity of SVG images: source digest, texts and size.

SVG come from untrusted documents: they are parsed without resolving
entities, loading a DTD or touching the network, never rendered, and a
script inside one is never part of its identity.
"""

import re
from dataclasses import dataclass
from typing import Final

from lxml import etree

from doc_checker.images.pixels import (
    CANNOT_DECODE,
    TOO_LARGE,
    UnreadableImage,
)
from doc_checker.images.refs import content_digest

SVG_MAX_BYTES: Final = 5_000_000
ICON_MAX_SIDE: Final = 48.0
ICON_CLASSES: Final = frozenset({"twemoji", "emoji", "gemoji"})

_NAME_SEPARATOR: Final = re.compile(r"[}:]")
_LENGTH: Final = re.compile(r"(?i)\s*(\d+(?:\.\d+)?|\.\d+)\s*(?:px)?\s*")
_NUMBER: Final = re.compile(r"(?i)[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?")
# The marks of a CDATA section that the HTML parser keeps in a text: those of
# the comment that `keep_cdata` of the HTML site makes of it, then its own.
_CDATA_MARKERS: Final = ("<!--[CDATA[", "]]-->", "<![CDATA[", "]]>")


@dataclass(frozen=True, slots=True)
class SvgIdentity:
    """What identifies an SVG image.

    Attributes:
        digest: Digest of the canonical form of its source.
        texts: Texts of its `text` elements, in order.
        width: Width in pixels, if known.
        height: Height in pixels, if known.
    """

    digest: str
    texts: tuple[str, ...]
    width: float | None
    height: float | None


def _squash(text: str | None) -> str:
    """Reduce the white space of a text.

    Args:
        text: Text, or `None`.

    Returns:
        The text without CDATA markers (the HTML parser keeps them as
        text, as they are or as `_CDATA_MARKERS` gives them) and with runs
        of blanks reduced to one space.
    """
    plain = text or ""
    for marker in _CDATA_MARKERS:
        plain = plain.replace(marker, "")
    return " ".join(plain.split())


def _local(name: str) -> str:
    """Take the lower case local name of an element or attribute name.

    Args:
        name: Name, possibly in `{namespace}name` or `prefix:name` form.

    Returns:
        The part after the last `}` or `:`, in lower case.
    """
    return _NAME_SEPARATOR.split(name)[-1].lower()


def _is_element(node: etree._Element) -> bool:
    """Tell an element from a comment, instruction or entity.

    Args:
        node: Node of the tree.

    Returns:
        Whether the node is an element.
    """
    return isinstance(node.tag, str)


def _cdata(node: etree._Element) -> str:
    """Read a CDATA section the HTML parser turned into a comment.

    Args:
        node: Node of the tree.

    Returns:
        The text of the section, or an empty string for any other node.
    """
    text = node.text if node.tag is etree.Comment else None
    if text and text.startswith("[CDATA[") and text.endswith("]]"):
        return text[len("[CDATA[") : -len("]]")]
    return ""


def _is_script(node: etree._Element) -> bool:
    """Tell a `script` element.

    Args:
        node: Node of the tree.

    Returns:
        Whether the node is a script element.
    """
    return _is_element(node) and _local(node.tag) == "script"


def _attributes(node: etree._Element) -> list[tuple[str, str]]:
    """List the attributes that belong to the drawing, sorted.

    Args:
        node: Element.

    Returns:
        Pairs of lower case local name and squashed value, with handlers
        and namespace declarations left out.
    """
    pairs = []
    for key, value in node.attrib.items():
        name = _local(str(key))
        full = str(key).lower()
        if (
            name.startswith("on")
            or full == "xmlns"
            or full.startswith("xmlns:")
        ):
            continue
        pairs.append((name, _squash(str(value))))
    return sorted(pairs)


def _lines(node: etree._Element, out: list[str]) -> None:
    """Append the canonical lines of an element and its descendants.

    Args:
        node: Element.
        out: Lines written so far.
    """
    out.append(f"<{_local(node.tag)}")
    out.extend(f"@{name}={value}" for name, value in _attributes(node))
    pending = node.text or ""
    for child in node:
        if _is_element(child) and not _is_script(child):
            _flush(pending, out)
            _lines(child, out)
            pending = child.tail or ""
        else:
            pending += _cdata(child) + (child.tail or "")
    _flush(pending, out)


def _flush(text: str, out: list[str]) -> None:
    """Append a text line when it is not empty.

    Args:
        text: Text.
        out: Lines written so far.
    """
    squashed = _squash(text)
    if squashed:
        out.append(f"#{squashed}")


def _flow(node: etree._Element, parts: list[str]) -> None:
    """Gather the text of an element, without comments or scripts.

    Args:
        node: Element.
        parts: The pieces of text gathered so far, to which the text of the
            element and its descendants is appended.
    """
    parts.append(node.text or "")
    for child in node:
        if _is_element(child) and not _is_script(child):
            _flow(child, parts)
        else:
            parts.append(_cdata(child))
        parts.append(child.tail or "")


def _texts(root: etree._Element) -> tuple[str, ...]:
    """Collect the texts of the outermost `text` elements, in order.

    The text of a `text` element holds that of its `tspan` and `textPath`
    elements, and of any `text` element nested in it, which no valid SVG
    has: each piece of text is read once.

    Args:
        root: Root of the SVG.

    Returns:
        Non-empty squashed texts.
    """
    found: list[str] = []

    def visit(node: etree._Element) -> None:
        if _local(node.tag) == "text":
            parts: list[str] = []
            _flow(node, parts)
            text = _squash("".join(parts))
            if text:
                found.append(text)
            return
        for child in node:
            if _is_element(child) and not _is_script(child):
                visit(child)

    visit(root)
    return tuple(found)


def _attribute(node: etree._Element, name: str) -> str | None:
    """Read an attribute by its lower case local name.

    Args:
        node: Element.
        name: Lower case local name.

    Returns:
        The value, or `None`.
    """
    for key, value in node.attrib.items():
        if _local(str(key)) == name:
            return str(value)
    return None


def _length(value: str | None) -> float | None:
    """Read a length in pixels.

    Args:
        value: Attribute value, such as `16` or `16px`.

    Returns:
        The number, or `None` for another unit or a missing value.
    """
    match = _LENGTH.fullmatch(value or "")
    return float(match.group(1)) if match else None


def _size(node: etree._Element) -> tuple[float | None, float | None]:
    """Read the size of an SVG.

    Args:
        node: The `svg` element.

    Returns:
        Width and height from the attributes, else from the `viewBox`.
    """
    width = _length(_attribute(node, "width"))
    height = _length(_attribute(node, "height"))
    if width is not None and height is not None:
        return width, height
    numbers = _NUMBER.findall(_attribute(node, "viewbox") or "")
    try:
        *_, box_width, box_height = numbers
    except ValueError:
        return width, height
    return (
        width if width is not None else float(box_width),
        height if height is not None else float(box_height),
    )


def svg_identity(root: etree._Element) -> SvgIdentity:
    """Identify an SVG, whichever parser read it.

    Args:
        root: The `svg` element, from lxml's HTML or XML parser.

    Returns:
        Its digest, texts and size.
    """
    out: list[str] = []
    _lines(root, out)
    digest = content_digest("\n".join(out).encode("utf-8"))
    width, height = _size(root)
    return SvgIdentity(digest, _texts(root), width, height)


def parse_svg(data: bytes) -> SvgIdentity:
    """Parse an SVG file safely and identify it.

    Args:
        data: Content of the file.

    Returns:
        Its identity.

    Raises:
        UnreadableImage: When it is too large, not well formed or not an
            SVG.
    """
    if len(data) > SVG_MAX_BYTES:
        raise UnreadableImage(TOO_LARGE)
    parser = etree.XMLParser(
        resolve_entities=False,
        no_network=True,
        load_dtd=False,
        huge_tree=False,
        remove_comments=True,
        remove_pis=True,
    )
    try:
        root = etree.fromstring(data, parser)
    except (etree.XMLSyntaxError, ValueError) as error:
        raise UnreadableImage(CANNOT_DECODE) from error
    if _local(root.tag) != "svg":
        raise UnreadableImage(CANNOT_DECODE)
    return svg_identity(root)


def is_icon(element: etree._Element) -> bool:
    """Tell an icon from a figure.

    Args:
        element: An `svg` element.

    Returns:
        Whether it, or an ancestor, has an emoji class, or its sides are
        both at most `ICON_MAX_SIDE`.
    """
    for node in (element, *element.iterancestors()):
        classes = (_attribute(node, "class") or "").lower().split()
        if ICON_CLASSES.intersection(classes):
            return True
    width, height = _size(element)
    return (
        width is not None
        and height is not None
        and width <= ICON_MAX_SIDE
        and height <= ICON_MAX_SIDE
    )
