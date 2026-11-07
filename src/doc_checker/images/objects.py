# Copyright 2026 Christophe Le Douarec
"""Walk the objects of a PDF page with PDFium, and read their bounds.

The extraction finds the vector figures of a page among its objects, and
the renders of these figures check the image objects that they may decode:
both walk the objects of the page and of its forms, and read their bounds,
with these helpers.
"""

import ctypes
from typing import TYPE_CHECKING

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

if TYPE_CHECKING:
    from collections.abc import Iterator

# The left, bottom, right and top edges of a box, in the coordinates of
# PDFium for a page, or for the space of a form.
type Box = tuple[float, float, float, float]
# An object of a page or of a form, as PDFium gives it.
type PageObject = pdfium_c.FPDF_PAGEOBJECT


def object_bounds(obj: PageObject) -> Box | None:
    """Return the bounds of a page object, in the space of its container.

    Args:
        obj: The page object.

    Returns:
        Its left, bottom, right and top edges, or ``None`` if PDFium cannot
        tell them. They may not be finite numbers.
    """
    left, bottom, right, top = (ctypes.c_float() for _ in range(4))
    if not pdfium_c.FPDFPageObj_GetBounds(obj, left, bottom, right, top):
        return None
    return left.value, bottom.value, right.value, top.value


def page_children(page: pdfium.PdfPage) -> Iterator[tuple[int, PageObject]]:
    """Yield the objects of a page itself, not those of its forms.

    Args:
        page: The page.

    Yields:
        The position of each object among the objects of the page, and the
        object, unless PDFium gives none there.
    """
    for position in range(pdfium_c.FPDFPage_CountObjects(page)):
        obj = pdfium_c.FPDFPage_GetObject(page, position)
        if obj:
            yield position, obj


def form_children(form: PageObject) -> Iterator[tuple[int, PageObject]]:
    """Yield the objects of a form itself, not those of the forms it holds.

    Args:
        form: The form object.

    Yields:
        The position of each object among the objects of the form, and the
        object, unless PDFium gives none there.
    """
    for position in range(pdfium_c.FPDFFormObj_CountObjects(form)):
        obj = pdfium_c.FPDFFormObj_GetObject(form, position)
        if obj:
            yield position, obj


def form_objects(form: PageObject) -> Iterator[tuple[PageObject, int]]:
    """Yield the objects of a form, at any depth, with their types.

    Args:
        form: The form object.

    Yields:
        Each object of the form and of the forms it holds, a form before its
        objects, and its type, a `FPDF_PAGEOBJ_*` constant of PDFium.
    """
    pending = [form]
    while pending:
        current = pending.pop()
        for _, child in form_children(current):
            kind = pdfium_c.FPDFPageObj_GetType(child)
            yield child, kind
            if kind == pdfium_c.FPDF_PAGEOBJ_FORM:
                pending.append(child)


def form_matrix(
    form: PageObject,
    outer: pdfium.PdfMatrix | None,
) -> pdfium.PdfMatrix | None:
    """Return the matrix from the space of a form to the page.

    Args:
        form: The form object.
        outer: The matrix from the space that holds the form to the page,
            ``None`` if it is not known.

    Returns:
        The matrix, or ``None`` if it is not known.
    """
    matrix = pdfium_c.FS_MATRIX()
    if outer is None or not pdfium_c.FPDFPageObj_GetMatrix(form, matrix):
        return None
    return pdfium.PdfMatrix.from_raw(matrix).multiply(outer)
