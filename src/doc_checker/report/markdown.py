# Copyright 2026 Christophe Le Douarec
"""Markdown report of a comparison."""

import os
import urllib.parse
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Final

from jinja2 import Environment, PackageLoader, StrictUndefined

from doc_checker.model.findings import Category, Severity

if TYPE_CHECKING:
    from collections.abc import Sequence

    from doc_checker.compare.engine import CompareResult
    from doc_checker.model.document import Document
    from doc_checker.model.findings import DiffSegment, Finding, Side

_ABSENT: Final = "\U00002014"
_PERCENT: Final = 100
_SECTION_TITLES: Final = {
    Severity.CRITICAL: "Critical",
    Severity.MAJOR: "Major",
    Severity.MINOR: "Minor",
    Severity.INFO: "Info",
}
_MARKUP: Final = {
    "delete": ("<del>", "</del>"),
    "insert": ("<ins>", "</ins>"),
    "equal": ("", ""),
}
_ENVIRONMENT: Final = Environment(
    loader=PackageLoader("doc_checker", "report/templates"),
    autoescape=False,  # ruff: ignore[jinja2-autoescape-false] - Markdown output, cells escaped by md_cell
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
)


def _escape(text: str) -> str:
    """Return the text safe for a Markdown table cell."""
    for raw, escaped in (
        ("&", "&amp;"),
        ("<", "&lt;"),
        (">", "&gt;"),
        ("|", "\\|"),
        ("\n", "<br>"),
    ):
        text = text.replace(raw, escaped)
    return text


def md_cell(segments: Sequence[DiffSegment]) -> str:
    """Return diff segments as the content of a Markdown table cell.

    Args:
        segments: Unchanged, deleted and inserted pieces of text.

    Returns:
        Escaped text, deletions in ``<del>`` and insertions in ``<ins>``.
    """
    parts = []
    for segment in segments:
        opening, closing = _MARKUP[segment.op]
        parts.append(f"{opening}{_escape(segment.text)}{closing}")
    return "".join(parts)


def relative_link(target: Path, anchor: str | None, report_dir: Path) -> str:
    """Return a link from the report directory to a source page.

    Args:
        target: Source file.
        anchor: Anchor in the source file, if any.
        report_dir: Directory of the report.

    Returns:
        A percent encoded relative URL, or a file URI when no relative path
        exists, such as across Windows drives.
    """
    try:
        relative = PurePath(os.path.relpath(target, report_dir)).as_posix()
        link = urllib.parse.quote(relative, safe="/")
    except ValueError:
        link = target.resolve().as_uri()
    return f"{link}#{anchor}" if anchor else link


class _Context:
    """Values and helpers used by the report template."""

    def __init__(self, result: CompareResult, report_dir: Path) -> None:
        self.result = result
        self.report_dir = report_dir

    def link(self, side: Side | None, which: str) -> str:
        """Return the Markdown link to the source of a side."""
        if side is None:
            return _ABSENT
        doc: Document = self.result.ref if which == "ref" else self.result.new
        target = doc.resolve(side.source)
        url = relative_link(target, side.source.anchor, self.report_dir)
        return f"[{side.source.href}]({url})"

    @staticmethod
    def cell(side: Side | None) -> str:
        """Return the table cell of a side."""
        return _ABSENT if side is None else md_cell(side.segments)

    def inputs(self) -> list[tuple[str, str, str]]:
        """Return the rows of the inputs table."""
        ref, new, stats = self.result.ref, self.result.new, self.result.stats
        return [
            ("Path", f"`{ref.path}`", f"`{new.path}`"),
            ("Format", ref.format, new.format),
            ("Pages", str(len(ref.parts)), str(len(new.parts))),
            ("Generator", ref.generator or "-", new.generator or "-"),
            ("Language", ref.lang or "-", new.lang or "-"),
            ("Blocks", str(stats.ref_blocks), str(stats.new_blocks)),
        ]

    def categories(self) -> list[Category]:
        """Return the categories that have findings, in a stable order."""
        present = {finding.category for finding in self.result.findings}
        return [category for category in Category if category in present]

    def summary(self) -> list[tuple[str, list[int], int]]:
        """Return one row per severity: label, counts, total."""
        if not self.result.findings:
            return []
        rows = []
        for severity in sorted(Severity, reverse=True):
            findings = self.findings_of(severity)
            counts = [
                sum(finding.category is category for finding in findings)
                for category in self.categories()
            ]
            rows.append((severity.label, counts, len(findings)))
        return rows

    def findings_of(self, severity: Severity) -> list[Finding]:
        """Return the findings of a severity, in document order."""
        return [f for f in self.result.findings if f.severity is severity]

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

    def sections(self) -> list[tuple[str, list[Finding]]]:
        """Return the findings grouped by severity, most severe first."""
        return [
            (_SECTION_TITLES[severity], findings)
            for severity in sorted(Severity, reverse=True)
            if (findings := self.findings_of(severity))
        ]

    def matched(self) -> str:
        """Return the share of reference blocks found, in percent."""
        stats = self.result.stats
        share = stats.matched / stats.ref_blocks if stats.ref_blocks else 1.0
        return f"{share * _PERCENT:.1f}"


def render_markdown(result: CompareResult, path: Path) -> Path:
    """Write the Markdown report of a comparison.

    Args:
        result: Comparison result.
        path: Report file to write.

    Returns:
        The written file.
    """
    context = _Context(result, path.parent)
    text = _ENVIRONMENT.get_template("report.md.j2").render(
        inputs=context.inputs(),
        matched=context.matched(),
        moved=result.stats.moved,
        categories=context.categories(),
        summary=context.summary(),
        normalization=context.normalization(),
        chrome=context.chrome(),
        sections=context.sections(),
        link=context.link,
        cell=context.cell,
    )
    path.write_text(text, encoding="utf-8")
    return path
