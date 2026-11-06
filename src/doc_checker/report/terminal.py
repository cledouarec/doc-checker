# Copyright 2026 Christophe Le Douarec
"""Text rendered in the terminal."""

import re
from typing import TYPE_CHECKING, Any, Final

import click

from doc_checker.model.findings import Severity
from doc_checker.text.normalize import BIDI_CLASS, CONTROL_CLASS

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from doc_checker.compare.engine import CompareResult
    from doc_checker.model.document import Block, Document, Link

_MAX_TEXT: Final = 200
_NO_GENERATOR: Final = "-"
_LINK_KIND: Final = "link"
_MISSING: Final = " (missing)"
_ELLIPSIS: Final = "\U00002026"
_REPLACEMENT: Final = "\U0000fffd"
# What `printable` blanks out, but the tab and the line feed.
_HIDDEN_RE: Final = re.compile(f"[{CONTROL_CLASS}{BIDI_CLASS}]")
_NEWLINE_MARK: Final = " \U000023ce "
_TOP_CRITICAL: Final = 5
_QUOTE_CHARS: Final = 60
_PERCENT: Final = 100
_STYLES: Final[dict[Severity, dict[str, Any]]] = {
    Severity.CRITICAL: {"fg": "red", "bold": True},
    Severity.MAJOR: {"fg": "yellow"},
    Severity.MINOR: {"fg": "blue"},
    Severity.INFO: {"dim": True},
}


def visible(text: str) -> str:
    """Return a text of a document without what acts on a terminal.

    Args:
        text: A text that comes from a document, before any styling: the
            escape sequences of the styles would be replaced too.

    Returns:
        The text, where each control character and each bidirectional
        override or isolate is replaced by U+FFFD. Tabs and line feeds are
        kept, so a text shown on one line has to be put on one line too.
    """
    return _HIDDEN_RE.sub(_REPLACEMENT, text)


def _one_line(text: str) -> str:
    """Return a text of a document on one line, with its controls visible."""
    return visible(" ".join(text.split()))


def _shorten(text: str, limit: int) -> str:
    """Return the text on one line, cut to ``limit`` characters."""
    text = visible(text).replace("\n", _NEWLINE_MARK)
    return text if len(text) <= limit else text[:limit] + _ELLIPSIS


def _link_line(link: Link) -> str:
    """Return the line of a link: its text, its target, if it is missing."""
    missing = _MISSING if link.url is None and not link.found else ""
    return (
        f"{'':>5}  {_LINK_KIND:<10}  "
        f"{_one_line(link.text)} -> {_one_line(link.target)}{missing}"
    )


def _block_lines(index: int, block: Block) -> list[str]:
    """Return the line of a block, then the line of each of its links."""
    line = (
        f"{index:>5}  {block.kind.value:<10}  {_one_line(block.source.href)}  "
        f"{_shorten(block.text, _MAX_TEXT)}"
    )
    return [line, *map(_link_line, block.links)]


def render_blocks(doc: Document) -> str:
    """Return a listing of the normalized blocks of a document.

    Args:
        doc: Loaded document.

    Returns:
        A header line, then one line per warning and one line per block,
        followed by one line per link of the block: its text and its
        `Link.target`, then ``(missing)`` for an internal link whose page
        does not exist. The header tells the generator, a dash if it is
        unknown, and whether a PDF is tagged. The values that come from the
        document are on one line and have their controls made visible, as
        `visible` does; an excerpt of a block shows its line breaks as
        marks.
    """
    header = (
        f"{_one_line(str(doc.path))}: {doc.format}, {len(doc.parts)} page(s), "
        f"generator {_one_line(doc.generator or _NO_GENERATOR)}, "
        f"lang {_one_line(str(doc.lang))}"
    )
    if doc.tagged is not None:
        header += ", tagged yes" if doc.tagged else ", tagged no"
    lines = [header]
    lines.extend(f"warning: {_one_line(warning)}" for warning in doc.warnings)
    for index, block in enumerate(doc.blocks):
        lines.extend(_block_lines(index, block))
    return "\n".join(lines)


def _styled(severity: Severity, text: str) -> str:
    """Return the text in the color of a severity."""
    return click.style(text, **_STYLES[severity])


def _described(doc: Document) -> str:
    """Return the path, the page count and the generator of a document."""
    generator = _one_line(doc.generator or _NO_GENERATOR)
    return f"{_one_line(str(doc.path))} ({len(doc.parts)} pages, {generator})"


def render_summary(result: CompareResult, reports: Sequence[Path]) -> str:
    """Return the short summary of a comparison shown in the terminal.

    Args:
        result: Comparison result.
        reports: Report files written.

    Returns:
        Styled text; ``click.echo`` removes the styles when not in a TTY. The
        warnings of both documents follow the critical findings. The
        values that come from the documents are on one line and have their
        controls made visible, as `visible` does; an excerpt of a block
        shows its line breaks as marks.
    """
    ref, new, stats = result.ref, result.new, result.stats
    share = stats.matched / stats.ref_blocks if stats.ref_blocks else 1.0
    counts = ", ".join(
        f"{sum(f.severity is severity for f in result.findings)} "
        f"{_styled(severity, severity.label)}"
        for severity in sorted(Severity, reverse=True)
    )
    lines = [
        f"compare  ref={_described(ref)}  new={_described(new)}",
        (
            f"blocks {stats.ref_blocks} <-> {stats.new_blocks} | "
            f"matched {share * _PERCENT:.1f}% | moved {stats.moved}"
        ),
        f"findings: {counts}",
    ]
    critical = [
        finding
        for finding in result.findings
        if finding.severity is Severity.CRITICAL
    ]
    for finding in critical[:_TOP_CRITICAL]:
        side = finding.ref or finding.new
        location = _one_line(side.source.href) if side else ""
        quote = _shorten(side.text, _QUOTE_CHARS) if side else ""
        label = _styled(Severity.CRITICAL, "CRITICAL")
        count = (
            f" ({len(finding.occurrences)} occurrences)"
            if finding.occurrences
            else ""
        )
        lines.append(
            f'  {label} {finding.rule}  {location}  "{quote}"{count}',
        )
    tag = click.style("warning", fg="yellow")
    lines.extend(
        f"{tag}: {which}: {_one_line(warning)}"
        for which, document in (("ref", ref), ("new", new))
        for warning in document.warnings
    )
    lines.append("reports: " + ", ".join(str(path) for path in reports))
    return "\n".join(lines)
