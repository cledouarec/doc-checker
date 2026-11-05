# Copyright 2026 Christophe Le Douarec
"""Markdown report of a comparison."""

import os
import re
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
_UNKNOWN: Final = "-"
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
    autoescape=False,  # ruff: ignore[jinja2-autoescape-false] - Markdown output, texts escaped by md_text, md_code and md_cell
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
)
# The characters of HTML and of the inline syntax of Markdown.
_ESCAPES: Final = str.maketrans(
    {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        **{char: f"\\{char}" for char in "\\`*_[]|~"},
    },
)
_LINE_BREAK_RE: Final = re.compile(r"\r\n|[\r\n]")
# What opens a heading, a bullet or a thematic break, then what opens a
# numbered list, after the blanks that a line may start with.
_OPENING_MARK_RE: Final = re.compile(r"^([ \t]*)([#+-])")
_OPENING_NUMBER_RE: Final = re.compile(r"^([ \t]*\d+)([.)])")
_BACKTICKS_RE: Final = re.compile(r"`+")
_PIPE_RE: Final = re.compile(r"(\\*)\|")
_BACKSLASH: Final = "\\"


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


def md_text(text: str) -> str:
    """Return a text of a document as safe text in a line of Markdown.

    The result is one line that cannot start a tag, an emphasis, a code span,
    a link, a table cell, a heading or a list. It still holds the controls of
    the text: see `printable` for them.

    Args:
        text: Any text that comes from a document.

    Returns:
        The text with each line break turned into a space, the ampersand and
        the angle brackets turned into entities, and a backslash put before
        each backslash, backtick, asterisk, underscore, square bracket, pipe
        and tilde. A backslash also comes before a ``#``, a ``-`` or a ``+``
        that opens the text, and before the ``.`` or the ``)`` of a number
        that opens it, whether or not blanks come first. The parentheses are
        left as they are.
    """
    text = _LINE_BREAK_RE.sub(" ", text).translate(_ESCAPES)
    text = _OPENING_MARK_RE.sub(r"\1\\\2", text)
    return _OPENING_NUMBER_RE.sub(r"\1\\\2", text)


def md_code(text: str) -> str:
    """Return a text of a document as a Markdown code span.

    Args:
        text: Any text that comes from a document, such as a path.

    Returns:
        The text on one line between fences of backticks, which are one more
        than the longest run of backticks in the text, and with a space inside
        each fence if the text holds a backtick. In a table cell, use
        `md_cell_code`.
    """
    text = _LINE_BREAK_RE.sub(" ", text)
    longest = max(map(len, _BACKTICKS_RE.findall(text)), default=0)
    fence = "`" * (longest + 1)
    blank = " " if longest else ""
    return f"{fence}{blank}{text}{blank}{fence}"


def _hide_pipe(match: re.Match[str]) -> str:
    """Return a pipe with the backslashes that keep it in its table cell."""
    run = match.group(1)
    return run + _BACKSLASH * (1 + len(run) % 2) + "|"


def md_cell_code(text: str) -> str:
    """Return a text of a document as a code span in a Markdown table cell.

    A table ends a cell at each pipe, even in a code span, unless an odd number
    of backslashes come right before it, and the cell then loses the last one.
    So a pipe gets one more backslash, or two more if an odd number of them
    come before it already, in which case the span shows one backslash more
    than the text.

    Args:
        text: Any text that comes from a document, such as a path.

    Returns:
        The code span of `md_code`, whose pipes cannot end the cell.
    """
    return _PIPE_RE.sub(_hide_pipe, md_code(text))


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


def _optional_text(value: str | None) -> str:
    """Return a text of a document that may be missing, as a table cell."""
    return md_text(value) if value else _UNKNOWN


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
        url = relative_link(target, side.source.fragment, self.report_dir)
        return f"[{md_text(side.source.href)}]({url})"

    @staticmethod
    def cell(side: Side | None) -> str:
        """Return the table cell of a side."""
        return _ABSENT if side is None else md_cell(side.segments)

    def inputs(self) -> list[tuple[str, str, str]]:
        """Return the rows of the inputs table."""
        ref, new, stats = self.result.ref, self.result.new, self.result.stats
        return [
            ("Path", md_cell_code(str(ref.path)), md_cell_code(str(new.path))),
            ("Format", ref.format, new.format),
            ("Pages", str(len(ref.parts)), str(len(new.parts))),
            (
                "Generator",
                _optional_text(ref.generator),
                _optional_text(new.generator),
            ),
            ("Language", _optional_text(ref.lang), _optional_text(new.lang)),
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

    def warnings(self) -> list[tuple[str, str]]:
        """Return the extraction warnings: document label, text."""
        ref, new = self.result.ref, self.result.new
        return [
            *(("Reference", warning) for warning in ref.warnings),
            *(("New", warning) for warning in new.warnings),
        ]

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
        warnings=context.warnings(),
        sections=context.sections(),
        link=context.link,
        cell=context.cell,
        md_text=md_text,
    )
    path.write_text(text, encoding="utf-8")
    return path
