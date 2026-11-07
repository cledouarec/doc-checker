# Copyright 2026 Christophe Le Douarec
"""Thresholds and verdict on a pair of images with pixels."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from doc_checker.compare.drafts import ARROW
from doc_checker.images.pixels import LOSSLESS_FORMATS
from doc_checker.model.findings import Severity

if TYPE_CHECKING:
    from doc_checker.images.metrics import Measures
    from doc_checker.model.document import ImageRef

RESOLUTION_MIN: Final = 0.7
RESOLUTION_CRITICAL: Final = 0.4
SSIM_MIN: Final = 0.90
SHARPNESS_MIN: Final = 0.6
QUALITY_DROP: Final = 20
COLOR_MIN: Final = 0.02
COLOR_KEPT: Final = 0.5
DPI_ENOUGH: Final = 300.0

_FULL_QUALITY: Final = 100


@dataclass(frozen=True, slots=True)
class Verdict:
    """What a pair of images deserves.

    Attributes:
        rule: Rule of the finding, None for no finding.
        severity: Severity overriding the rule's, None to keep it.
        detail: Measures behind the finding.
        pattern: Short cause, to group findings.
    """

    rule: str | None
    severity: Severity | None
    detail: str | None
    pattern: str | None


def _quality(image: ImageRef) -> int | None:
    """Give the JPEG quality of an image, 100 for a lossless one.

    Args:
        image: An image.

    Returns:
        The quality, or None when unknown.
    """
    if image.format in LOSSLESS_FORMATS:
        return _FULL_QUALITY
    if image.format == "jpeg":
        return image.quality
    return None


def _lost(before: float, after: float) -> bool:
    """Tell whether the colour of an image went away.

    Args:
        before: Colorfulness of the reference.
        after: Colorfulness of the new image.

    Returns:
        True if the colour was there and is mostly gone.
    """
    return before >= COLOR_MIN and after < before * COLOR_KEPT


def judge(
    ref: ImageRef,
    new: ImageRef,
    measures: Measures,
    *,
    pdf: bool,
) -> Verdict:
    """Judge the change from a reference image to a new one.

    Args:
        ref: Reference image.
        new: New image.
        measures: Measures of the pair.
        pdf: Whether both images come from PDF files.

    Returns:
        No finding for identical pixels, `image.degraded` when something
        got worse, else `image.reencoded`.
    """
    if measures.identical:
        return Verdict(None, None, None, None)
    ref_q, new_q = _quality(ref), _quality(new)
    gap = None if ref_q is None or new_q is None else new_q - ref_q
    sharper = measures.sharpness > 1 / SHARPNESS_MIN
    # The effective scale counts unless the new image is sharper or of a
    # higher quality, when only its pixels count.
    effective = (
        not (sharper or (gap is not None and gap > 0))
        and measures.effective < measures.scale
    )
    res = measures.effective if effective else measures.scale
    quality_better = gap is not None and gap >= QUALITY_DROP
    scale_better = res > 1 / RESOLUTION_MIN
    better = (
        scale_better
        or sharper
        or quality_better
        or _lost(measures.new_color, measures.ref_color)
        or (new.alpha is True and ref.alpha is False)
    )
    worse = {
        "resolution": res < RESOLUTION_MIN
        and not (pdf and new.dpi is not None and new.dpi >= DPI_ENOUGH),
        "jpeg quality": gap is not None and -gap >= QUALITY_DROP,
        "color": _lost(measures.ref_color, measures.new_color),
        "transparency": ref.alpha is True and new.alpha is False,
        "sharpness": measures.sharpness < SHARPNESS_MIN
        and not (scale_better or quality_better),
    }
    causes = sorted(name for name, bad in worse.items() if bad)
    ssim_decides = not causes and measures.ssim < SSIM_MIN and not better
    if not causes and not ssim_decides:
        return Verdict(
            "image.reencoded",
            None,
            f"{ref.format.upper()}{ARROW}{new.format.upper()}, "
            f"SSIM {measures.ssim:.2f}",
            f"{ref.format}{ARROW}{new.format}",
        )
    if ssim_decides:
        causes = ["ssim"]
    parts = {
        "resolution": (
            f"{'effective resolution' if effective else 'resolution'} "
            f"x{res:.2f} "
            f"({ref.width}x{ref.height}{ARROW}{new.width}x{new.height} px)"
        ),
        "ssim": f"SSIM {measures.ssim:.2f}",
        "sharpness": f"sharpness x{measures.sharpness:.2f}",
        "jpeg quality": f"JPEG quality {ref_q}{ARROW}{new_q}",
        "color": "colors lost",
        "transparency": "transparency lost",
    }
    # The parts come in their order above, whatever the order of the causes.
    detail = "; ".join(part for name, part in parts.items() if name in causes)
    critical = worse["resolution"] and measures.scale < RESOLUTION_CRITICAL
    return Verdict(
        "image.degraded",
        Severity.CRITICAL if critical else None,
        detail,
        "+".join(causes),
    )
