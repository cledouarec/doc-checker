# Copyright 2026 Christophe Le Douarec
"""Text rendered in the terminal."""

import re
from typing import TYPE_CHECKING, Any, Final

import click

from doc_checker.images.refs import describe
from doc_checker.model.findings import Severity
from doc_checker.report.context import (
    IMAGES_NOT_READ,
    NOT_RUN,
    UNKNOWN,
    matched_percent,
)
from doc_checker.text.normalize import BIDI_CLASS, CONTROL_CLASS, truncate

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from doc_checker.audit.engine import AuditResult
    from doc_checker.compare.engine import CompareResult
    from doc_checker.model.document import Block, Document, Link
    from doc_checker.model.findings import Finding

_MAX_TEXT: Final = 200
_LINK_KIND: Final = "link"
_MISSING: Final = " (missing)"
_REPLACEMENT: Final = "\U0000fffd"
# What `printable` blanks out, but the tab and the line feed.
_HIDDEN_RE: Final = re.compile(f"[{CONTROL_CLASS}{BIDI_CLASS}]")
_NEWLINE_MARK: Final = " \U000023ce "
_TOP_FINDINGS: Final = 5
_QUOTE_CHARS: Final = 60
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
    return truncate(visible(text).replace("\n", _NEWLINE_MARK), limit)


def _link_line(link: Link) -> str:
    """Return the line of a link: its text, its target, if it is missing."""
    missing = _MISSING if link.url is None and not link.found else ""
    return (
        f"{'':>5}  {_LINK_KIND:<10}  "
        f"{_one_line(link.text)} -> {_one_line(link.target)}{missing}"
    )


def _block_lines(index: int, block: Block) -> list[str]:
    """Return the line of a block, then the line of each of its links.

    The line of an image describes it, as `describe` does.
    """
    content = (
        _shorten(block.text, _MAX_TEXT)
        if block.image is None
        else _one_line(describe(block.image))
    )
    line = (
        f"{index:>5}  {block.kind.value:<10}  {_one_line(block.source.href)}  "
        f"{content}"
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
        does not exist. The line of an image describes it, as `describe`
        does. The header tells the generator, a dash if it is unknown, and
        whether a PDF is tagged. The values that come from the document are
        on one line and have their controls made visible, as `visible` does;
        an excerpt of a block shows its line breaks as marks.
    """
    header = (
        f"{_one_line(str(doc.path))}: {doc.format}, {len(doc.parts)} page(s), "
        f"generator {_one_line(doc.generator or UNKNOWN)}, "
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
    generator = _one_line(doc.generator or UNKNOWN)
    return f"{_one_line(str(doc.path))} ({len(doc.parts)} pages, {generator})"


def _counts(findings: Sequence[Finding]) -> str:
    """Return the line that counts the findings of each severity."""
    counts = ", ".join(
        f"{sum(f.severity is severity for f in findings)} "
        f"{_styled(severity, severity.label)}"
        for severity in sorted(Severity, reverse=True)
    )
    return f"findings: {counts}"


def _top_line(finding: Finding) -> str:
    """Return the line of a finding listed under the counts.

    It tells the severity, the rule, the place and an excerpt of the first
    side of the finding, then the occurrences of a group.
    """
    side = finding.ref or finding.new
    location = _one_line(side.source.href) if side else ""
    quote = _shorten(side.text, _QUOTE_CHARS) if side else ""
    label = _styled(finding.severity, finding.severity.name)
    count = (
        f" ({len(finding.occurrences)} occurrences)"
        if finding.occurrences
        else ""
    )
    return f'  {label} {finding.rule}  {location}  "{quote}"{count}'


def _reports_line(reports: Sequence[Path]) -> str:
    """Return the line that lists the reports written."""
    return "reports: " + ", ".join(str(path) for path in reports)


def render_summary(result: CompareResult, reports: Sequence[Path]) -> str:
    """Return the short summary of a comparison shown in the terminal.

    Args:
        result: Comparison result.
        reports: Report files written.

    Returns:
        Styled text; ``click.echo`` removes the styles when not in a TTY.
        A line counts the images when a document has some. The warnings of
        both documents follow the critical findings. The values that come
        from the documents are on one line and have their controls made
        visible, as `visible` does; an excerpt of a block shows its line
        breaks as marks.
    """
    ref, new, stats = result.ref, result.new, result.stats
    lines = [
        f"compare  ref={_described(ref)}  new={_described(new)}",
        (
            f"blocks {stats.ref_blocks} <-> {stats.new_blocks} | "
            f"matched {matched_percent(stats)}% | moved {stats.moved}"
        ),
    ]
    if stats.ref_images or stats.new_images:
        lines.append(
            f"images {stats.ref_images} <-> {stats.new_images} | "
            f"matched {stats.matched_images}",
        )
    lines.append(_counts(result.findings))
    critical = [
        finding
        for finding in result.findings
        if finding.severity is Severity.CRITICAL
    ]
    lines.extend(map(_top_line, critical[:_TOP_FINDINGS]))
    tag = click.style("warning", fg="yellow")
    lines.extend(
        f"{tag}: {which}: {_one_line(warning)}"
        for which, document in (("ref", ref), ("new", new))
        for warning in document.warnings
    )
    lines.append(_reports_line(reports))
    return "\n".join(lines)


def render_audit_summary(result: AuditResult, reports: Sequence[Path]) -> str:
    """Return the short summary of an audit shown in the terminal.

    Args:
        result: Audit result.
        reports: Report files written.

    Returns:
        Styled text; ``click.echo`` removes the styles when not in a TTY.
        The document, its counts, the families of checks and the profile,
        the findings counted by severity, then, if the highest severity
        found is major or critical, the first five findings of that
        severity, as `render_summary` lists the critical ones; then the
        warnings of the document and the reports. The values that come from
        the document are on one line and have their controls made visible,
        as `visible` does; an excerpt of a block shows its line breaks as
        marks.
    """
    stats = result.stats
    images = str(stats.images) if stats.images_read else IMAGES_NOT_READ
    links = NOT_RUN if stats.links is None else str(stats.links)
    checks = ", ".join(stats.checks) or NOT_RUN
    lines = [
        f"audit  doc={_described(result.doc)}",
        (
            f"blocks {stats.blocks} | images {images} | links {links}"
            f" | checks {checks} | profile {stats.profile}"
        ),
        _counts(result.findings),
    ]
    highest = max((f.severity for f in result.findings), default=None)
    if highest is not None and highest >= Severity.MAJOR:
        top = [f for f in result.findings if f.severity is highest]
        lines.extend(map(_top_line, top[:_TOP_FINDINGS]))
    tag = click.style("warning", fg="yellow")
    lines.extend(
        f"{tag}: {_one_line(warning)}" for warning in result.doc.warnings
    )
    lines.append(_reports_line(reports))
    return "\n".join(lines)
