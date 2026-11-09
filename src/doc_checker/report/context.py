# Copyright 2026 Christophe Le Douarec
"""Format-neutral data shared by the Markdown and HTML reports.

The reports of a comparison and of an audit are the same templates: each
shows one column per document, `Column`, which the result gives by
`columns`. The module also sets up the templates of both reports, once, and
makes the thumbnails that both show.
"""

import os
import urllib.parse
from dataclasses import dataclass
from pathlib import PurePath
from typing import TYPE_CHECKING, Final, Literal, Self

from jinja2 import (
    Environment,
    PackageLoader,
    StrictUndefined,
    select_autoescape,
)

from doc_checker.audit.engine import AuditResult
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
    from doc_checker.model.document import Document, ImageRef
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
IMAGES_NOT_READ: Final = "not read"  # Images an audit did not read.
NOT_RUN: Final = "-"  # Checks of an audit that did not run.
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
# Whether a PDF is tagged; an HTML document does not tell.
_TAGGED: Final[Mapping[bool | None, str]] = {
    True: "yes",
    False: "no",
    None: UNKNOWN,
}


@dataclass(frozen=True, slots=True)
class Column:
    """A document that the reports show in a column of their own.

    Attributes:
        which: The side of the findings that the document gives.
        label: Title of the column, such as ``"Reference"``.
        document: The document.
    """

    which: Literal["ref", "new"]
    label: str
    document: Document


def columns(result: CompareResult | AuditResult) -> tuple[Column, ...]:
    """Return the columns of the reports of a result.

    Args:
        result: Result of a comparison or of an audit.

    Returns:
        The reference, then the new document, for a comparison; the audited
        document, whose findings put their side in ``ref``, for an audit.
    """
    if isinstance(result, AuditResult):
        return (Column("ref", "Document", result.doc),)
    return (
        Column("ref", "Reference", result.ref),
        Column("new", "New", result.new),
    )


def side_of(finding: Finding, which: Literal["ref", "new"]) -> Side | None:
    """Return the side of a finding that a column shows.

    Args:
        finding: Any finding.
        which: The side of the column.

    Returns:
        The reference side or the new side of the finding, if it has one.
    """
    return finding.ref if which == "ref" else finding.new


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

    def __init__(self, result: CompareResult | AuditResult) -> None:
        """Keep the documents whose images are read.

        Args:
            result: Result of a comparison or of an audit.
        """
        self._paths = {
            column.which: column.document.path for column in columns(result)
        }
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


def _compare_inputs(
    result: CompareResult,
) -> list[tuple[str, tuple[str, ...]]]:
    """Return the rows of the inputs table of a comparison."""
    ref, new, stats = result.ref, result.new, result.stats
    return [
        ("Path", (str(ref.path), str(new.path))),
        ("Format", (ref.format, new.format)),
        ("Pages", (str(len(ref.parts)), str(len(new.parts)))),
        ("Generator", (ref.generator or UNKNOWN, new.generator or UNKNOWN)),
        ("Language", (ref.lang or UNKNOWN, new.lang or UNKNOWN)),
        ("Blocks", (str(stats.ref_blocks), str(stats.new_blocks))),
        (
            ("Images", (str(stats.ref_images), str(stats.new_images)))
            if stats.images
            else ("Images", (NOT_COMPARED, NOT_COMPARED))
        ),
    ]


def _audit_inputs(result: AuditResult) -> list[tuple[str, tuple[str, ...]]]:
    """Return the rows of the inputs table of an audit."""
    doc, stats = result.doc, result.stats
    images = str(stats.images) if stats.images_read else IMAGES_NOT_READ
    return [
        ("Path", (str(doc.path),)),
        ("Format", (doc.format,)),
        ("Pages", (str(stats.pages),)),
        ("Generator", (doc.generator or UNKNOWN,)),
        ("Language", (doc.lang or UNKNOWN,)),
        ("Tagged", (_TAGGED[doc.tagged],)),
        ("Blocks", (str(stats.blocks),)),
        ("Images", (images,)),
    ]


class ReportContext:
    """Raw values of a report, with no formatting of any output format.

    Attributes:
        result: Result of a comparison or of an audit.
        report_dir: Directory of the report, for the relative links.
        command: The command whose result the report shows.
        columns: The documents of the report, one column each.
    """

    def __init__(
        self,
        result: CompareResult | AuditResult,
        report_dir: Path,
    ) -> None:
        """Keep the result and the directory the report is written to.

        Args:
            result: Result of a comparison or of an audit.
            report_dir: Directory of the report, for the relative links.
        """
        self.result = result
        self.report_dir = report_dir
        self.command: Literal["compare", "audit"] = (
            "audit" if isinstance(result, AuditResult) else "compare"
        )
        self.columns = columns(result)
        self._documents = {
            column.which: column.document for column in self.columns
        }

    def inputs(self) -> list[tuple[str, tuple[str, ...]]]:
        """Return the rows of the inputs table, as raw values.

        Returns:
            One row per property: its label, then its value in each column.
            An audit also tells whether its document is tagged, ``yes`` or
            ``no`` for a PDF, and says its images were not read when they
            were left out.
        """
        if isinstance(self.result, AuditResult):
            return _audit_inputs(self.result)
        return _compare_inputs(self.result)

    def headline(self) -> dict[str, object]:
        """Return the values of the line under the inputs table.

        Returns:
            For a comparison, ``matched``, ``matched_images`` and ``moved``;
            for an audit, ``checks`` and ``profile``.
        """
        if isinstance(self.result, AuditResult):
            return {"checks": self.checks(), "profile": self.profile()}
        return {
            "matched": self.matched(),
            "matched_images": self.matched_images(),
            "moved": self.result.stats.moved,
        }

    def checks(self) -> str:
        """Return the families of checks an audit ran, such as ``a11y``."""
        return ", ".join(self._audit().stats.checks) or NOT_RUN

    def profile(self) -> str:
        """Return the profile of an audit, ``screen`` or ``print``."""
        return self._audit().stats.profile

    def matched(self) -> str:
        """Return the share of reference blocks found, in percent."""
        return matched_percent(self._compare().stats)

    def matched_images(self) -> str | None:
        """Return the reference images matched, out of all of them.

        Returns:
            Such as ``3 of 4 reference images``, or None when the images
            were not compared or neither document has one.
        """
        stats = self._compare().stats
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

    def normalization(self) -> list[tuple[str, tuple[int, ...]]]:
        """Return the normalization counters of the documents.

        Returns:
            The counters that a document does not leave at zero, by name,
            each with its count in each column.
        """
        counters = [column.document.normalization for column in self.columns]
        keys = sorted(
            {
                key
                for counter in counters
                for key, count in counter.items()
                if count
            },
        )
        return [
            (key, tuple(counter[key] for counter in counters)) for key in keys
        ]

    def chrome(self) -> list[str]:
        """Return the texts removed as page chrome, without duplicates."""
        texts = [
            text for column in self.columns for text in column.document.chrome
        ]
        return list(dict.fromkeys(texts))

    def warnings(self) -> list[tuple[str, str]]:
        """Return the extraction warnings: label of the column, text."""
        return [
            (column.label, warning)
            for column in self.columns
            for warning in column.document.warnings
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
        target = self._documents[which].resolve(side.source)
        url = relative_link(target, side.source.fragment, self.report_dir)
        return side.source.href, url

    def _compare(self) -> CompareResult:
        """Return the result of a comparison, which the report must show.

        Raises:
            TypeError: If the report shows an audit.
        """
        if isinstance(self.result, AuditResult):
            msg = "an audit has no figures of a comparison"
            raise TypeError(msg)
        return self.result

    def _audit(self) -> AuditResult:
        """Return the result of an audit, which the report must show.

        Raises:
            TypeError: If the report shows a comparison.
        """
        if not isinstance(self.result, AuditResult):
            msg = "a comparison has no figures of an audit"
            raise TypeError(msg)
        return self.result

    @staticmethod
    def shown(finding: Finding) -> list[Finding]:
        """Return the occurrences of a group that the report lists."""
        return list(finding.occurrences[:MAX_OCCURRENCES_SHOWN])

    @staticmethod
    def hidden(finding: Finding) -> int:
        """Return the number of occurrences that the report leaves out."""
        return max(len(finding.occurrences) - MAX_OCCURRENCES_SHOWN, 0)
