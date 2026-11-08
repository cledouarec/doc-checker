# Copyright 2026 Christophe Le Douarec
"""Format-neutral data shared by the Markdown and HTML reports.

The module also sets up the templates of both reports, once, and makes the
thumbnails that both show.
"""

import os
import urllib.parse
from pathlib import PurePath
from typing import TYPE_CHECKING, Final, Literal, Self

from jinja2 import (
    Environment,
    PackageLoader,
    StrictUndefined,
    select_autoescape,
)

from doc_checker.images.load import PixelLoader
from doc_checker.images.thumbnail import make_thumbnail
from doc_checker.model.document import BlockKind
from doc_checker.model.findings import Category, Severity

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path
    from types import TracebackType

    from doc_checker.compare.engine import CompareResult, CompareStats
    from doc_checker.images.thumbnail import Thumbnail
    from doc_checker.model.document import ImageRef
    from doc_checker.model.findings import Finding, Side

MAX_OCCURRENCES_SHOWN: Final = 100
CATEGORY_TITLES: Final[Mapping[Category, str]] = {
    Category.CONTENT: "Content",
    Category.TEXT: "Text",
    Category.STRUCTURE: "Structure",
    Category.LINK: "Links",
    Category.IMAGE: "Images",
    Category.ACCESSIBILITY: "Accessibility",
}
SEVERITY_TITLES: Final[Mapping[Severity, str]] = {
    Severity.CRITICAL: "Critical",
    Severity.MAJOR: "Major",
    Severity.MINOR: "Minor",
    Severity.INFO: "Info",
}
# What stands for a value that a document does not tell.
UNKNOWN: Final = "-"
NOT_COMPARED: Final = "not compared"  # Images left out by --no-images.
# The templates of the reports. Only the HTML one escapes the texts it
# shows: the Markdown renderer escapes them itself, with md_text, md_code
# and md_cell.
TEMPLATES: Final = Environment(
    loader=PackageLoader("doc_checker", "report/templates"),
    autoescape=select_autoescape(
        enabled_extensions=("html.j2",),
        default_for_string=False,
    ),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
)
_PERCENT: Final = 100


def matched_percent(stats: CompareStats) -> str:
    """Return the share of the reference blocks found, in percent.

    Args:
        stats: Figures of a comparison.

    Returns:
        The share with one decimal, such as ``97.5``; ``100.0`` when the
        reference has no block.
    """
    share = stats.matched / stats.ref_blocks if stats.ref_blocks else 1.0
    return f"{share * _PERCENT:.1f}"


def relative_link(
    target: Path,
    fragment: str | None,
    report_dir: Path,
) -> str:
    """Return a link from the report directory to a source page.

    Args:
        target: Source file.
        fragment: Fragment of the link, if any: an anchor of the source file,
            or ``page=N`` for a page of a PDF.
        report_dir: Directory of the report.

    Returns:
        A percent encoded relative URL, or a file URI when no relative path
        exists, such as across Windows drives. The fragment is percent encoded
        too, but for its equal signs.
    """
    try:
        relative = PurePath(os.path.relpath(target, report_dir)).as_posix()
        link = urllib.parse.quote(relative, safe="/")
    except ValueError:
        link = target.resolve().as_uri()
    if not fragment:
        return link
    return f"{link}#{urllib.parse.quote(fragment, safe='=')}"


def is_code(finding: Finding) -> bool:
    """Tell whether a side of a finding is a code block.

    Args:
        finding: Any finding.

    Returns:
        True if the reference or the new side is code.
    """
    return any(
        side is not None and side.kind is BlockKind.CODE
        for side in (finding.ref, finding.new)
    )


class Thumbnails:
    """The thumbnails of the sides of the findings, made once for both reports.

    The pixels of an image are read again, at the first demand of its
    thumbnail, by a `PixelLoader` of its document, which opens with that
    first demand. This object is a context manager that closes the loaders;
    once closed, it reads no more pixels.
    """

    def __init__(self, result: CompareResult) -> None:
        """Keep the documents whose images are read.

        Args:
            result: Comparison result.
        """
        self._paths = {"ref": result.ref.path, "new": result.new.path}
        self._loaders: dict[str, PixelLoader] = {}
        self._made: dict[tuple[str, ImageRef], Thumbnail | None] = {}
        self._closed = False

    def __enter__(self) -> Self:
        """Return the thumbnails.

        Returns:
            This object itself.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the loaders."""
        self.close()

    def close(self) -> None:
        """Close the loaders, and read no more pixels."""
        self._closed = True
        for loader in self._loaders.values():
            loader.close()
        self._loaders.clear()

    def get(
        self,
        side: Side | None,
        which: Literal["ref", "new"],
    ) -> Thumbnail | None:
        """Return the thumbnail of the image of a side.

        Args:
            side: A side of a finding, if any.
            which: The document of the side.

        Returns:
            The thumbnail, the same one each time for an image of a document;
            ``None`` without a side, without an image, for an image without
            pixels, such as an SVG, or for one whose pixels can no longer be
            read.
        """
        image = None if side is None else side.image
        if image is None or not image.readable:
            return None
        key = (which, image)
        if key not in self._made:
            if self._closed:
                return None
            self._made[key] = self._make(image, which)
        return self._made[key]

    def _make(
        self,
        image: ImageRef,
        which: Literal["ref", "new"],
    ) -> Thumbnail | None:
        """Read the pixels of an image again and make its thumbnail.

        Args:
            image: An image with pixels.
            which: Its document.

        Returns:
            The thumbnail, or ``None`` if the pixels cannot be read or put
            in a thumbnail.
        """
        loader = self._loaders.get(which)
        if loader is None:
            loader = self._loaders[which] = PixelLoader(self._paths[which])
        pixels = loader.load(image)
        if pixels is None:
            return None
        try:
            return make_thumbnail(pixels)
        except OSError, ValueError:
            return None


class ReportContext:
    """Raw values of a report, with no formatting of any output format."""

    def __init__(self, result: CompareResult, report_dir: Path) -> None:
        """Keep the result and the directory the report is written to.

        Args:
            result: Comparison result.
            report_dir: Directory of the report, for the relative links.
        """
        self.result = result
        self.report_dir = report_dir

    def inputs(self) -> list[tuple[str, str, str]]:
        """Return the rows of the inputs table, as raw values."""
        ref, new, stats = self.result.ref, self.result.new, self.result.stats
        return [
            ("Path", str(ref.path), str(new.path)),
            ("Format", ref.format, new.format),
            ("Pages", str(len(ref.parts)), str(len(new.parts))),
            (
                "Generator",
                ref.generator or UNKNOWN,
                new.generator or UNKNOWN,
            ),
            ("Language", ref.lang or UNKNOWN, new.lang or UNKNOWN),
            ("Blocks", str(stats.ref_blocks), str(stats.new_blocks)),
            (
                ("Images", str(stats.ref_images), str(stats.new_images))
                if stats.images
                else ("Images", NOT_COMPARED, NOT_COMPARED)
            ),
        ]

    def matched(self) -> str:
        """Return the share of reference blocks found, in percent."""
        return matched_percent(self.result.stats)

    def matched_images(self) -> str | None:
        """Return the reference images matched, out of all of them.

        Returns:
            Such as ``3 of 4 reference images``, or None when the images
            were not compared or neither document has one.
        """
        stats = self.result.stats
        if not stats.images or not (stats.ref_images or stats.new_images):
            return None
        return f"{stats.matched_images} of {stats.ref_images} reference images"

    def categories(self) -> list[Category]:
        """Return the categories that have findings, in a stable order."""
        present = {finding.category for finding in self.result.findings}
        return [category for category in Category if category in present]

    def findings_of(self, severity: Severity) -> list[Finding]:
        """Return the findings of a severity, in document order."""
        return [f for f in self.result.findings if f.severity is severity]

    def summary(self) -> list[tuple[Severity, list[int], int]]:
        """Return one row per severity: severity, counts, total."""
        if not self.result.findings:
            return []
        rows = []
        for severity in sorted(Severity, reverse=True):
            findings = self.findings_of(severity)
            counts = [
                sum(finding.category is category for finding in findings)
                for category in self.categories()
            ]
            rows.append((severity, counts, len(findings)))
        return rows

    def normalization(self) -> list[tuple[str, int, int]]:
        """Return the normalization counters of both documents."""
        ref = self.result.ref.normalization
        new = self.result.new.normalization
        keys = sorted(
            key for key in ref.keys() | new.keys() if ref[key] or new[key]
        )
        return [(key, ref[key], new[key]) for key in keys]

    def chrome(self) -> list[str]:
        """Return the texts removed as page chrome, without duplicates."""
        texts = [*self.result.ref.chrome, *self.result.new.chrome]
        return list(dict.fromkeys(texts))

    def warnings(self) -> list[tuple[str, str]]:
        """Return the extraction warnings: document label, text."""
        ref, new = self.result.ref, self.result.new
        return [
            *(("Reference", warning) for warning in ref.warnings),
            *(("New", warning) for warning in new.warnings),
        ]

    def sections(
        self,
    ) -> list[tuple[Severity, list[tuple[Category, list[Finding]]]]]:
        """Return the findings by severity, then by category.

        Returns:
            The severities from the highest to the lowest, each with its
            categories in the order of `Category` and the findings of each
            in document order. Empty severities and categories are left out.
        """
        sections = []
        for severity in sorted(Severity, reverse=True):
            findings = self.findings_of(severity)
            by_category = [
                (category, group)
                for category in Category
                if (group := [f for f in findings if f.category is category])
            ]
            if by_category:
                sections.append((severity, by_category))
        return sections

    def source(
        self,
        side: Side | None,
        which: Literal["ref", "new"],
    ) -> tuple[str, str] | None:
        """Return the label and the relative link of the source of a side."""
        if side is None:
            return None
        doc = self.result.ref if which == "ref" else self.result.new
        target = doc.resolve(side.source)
        url = relative_link(target, side.source.fragment, self.report_dir)
        return side.source.href, url

    @staticmethod
    def shown(finding: Finding) -> list[Finding]:
        """Return the occurrences of a group that the report lists."""
        return list(finding.occurrences[:MAX_OCCURRENCES_SHOWN])

    @staticmethod
    def hidden(finding: Finding) -> int:
        """Return the number of occurrences that the report leaves out."""
        return max(len(finding.occurrences) - MAX_OCCURRENCES_SHOWN, 0)
