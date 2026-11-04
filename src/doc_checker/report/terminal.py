# Copyright 2026 Christophe Le Douarec
"""Text rendered in the terminal."""

from typing import TYPE_CHECKING, Any, Final

import click

from doc_checker.model.findings import Severity

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from doc_checker.compare.engine import CompareResult
    from doc_checker.model.document import Document

_MAX_TEXT: Final = 200
_ELLIPSIS: Final = "\U00002026"
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


def _shorten(text: str, limit: int) -> str:
    """Return the text on one line, cut to ``limit`` characters."""
    text = text.replace("\n", _NEWLINE_MARK)
    return text if len(text) <= limit else text[:limit] + _ELLIPSIS


def render_blocks(doc: Document) -> str:
    """Return a listing of the normalized blocks of a document.

    Args:
        doc: Loaded document.

    Returns:
        A header line followed by one line per block.
    """
    lines = [
        (
            f"{doc.path}: {doc.format}, {len(doc.parts)} page(s), "
            f"profile {doc.generator}, lang {doc.lang}"
        ),
    ]
    lines.extend(
        f"{index:>5}  {block.kind.value:<10}  {block.source.href}  "
        f"{_shorten(block.text, _MAX_TEXT)}"
        for index, block in enumerate(doc.blocks)
    )
    return "\n".join(lines)


def _styled(severity: Severity, text: str) -> str:
    """Return the text in the color of a severity."""
    return click.style(text, **_STYLES[severity])


def render_summary(result: CompareResult, reports: Sequence[Path]) -> str:
    """Return the short summary of a comparison shown in the terminal.

    Args:
        result: Comparison result.
        reports: Report files written.

    Returns:
        Styled text; ``click.echo`` removes the styles when not in a TTY.
    """
    ref, new, stats = result.ref, result.new, result.stats
    share = stats.matched / stats.ref_blocks if stats.ref_blocks else 1.0
    counts = ", ".join(
        f"{sum(f.severity is severity for f in result.findings)} "
        f"{_styled(severity, severity.label)}"
        for severity in sorted(Severity, reverse=True)
    )
    lines = [
        (
            f"compare  ref={ref.path} ({len(ref.parts)} pages, "
            f"{ref.generator})  new={new.path} ({len(new.parts)} pages, "
            f"{new.generator})"
        ),
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
        location = side.source.href if side else ""
        quote = _shorten(side.text, _QUOTE_CHARS) if side else ""
        label = _styled(Severity.CRITICAL, "CRITICAL")
        lines.append(f'  {label} {finding.rule}  {location}  "{quote}"')
    lines.append("reports: " + ", ".join(str(path) for path in reports))
    return "\n".join(lines)
