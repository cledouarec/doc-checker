# Copyright 2026 Christophe Le Douarec
"""Markdown report of a comparison.

The thumbnails that the report shows are WebP files in a directory next to
it, named by their content. A file already there is kept, and none is ever
deleted.
"""

import difflib
import re
from typing import TYPE_CHECKING, Final, Literal

from doc_checker.compare.aggregate import shown_pattern
from doc_checker.report.context import (
    CATEGORY_TITLES,
    SEVERITY_TITLES,
    TEMPLATES,
    UNKNOWN,
    ReportContext,
    Thumbnails,
    is_code,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from doc_checker.compare.engine import CompareResult
    from doc_checker.images.thumbnail import Thumbnail
    from doc_checker.model.findings import DiffSegment, Finding, Side

# The directory of the thumbnails, next to the report.
ASSETS_DIR: Final = "report_assets"
_IMAGE_LABELS: Final = {"ref": "Reference image", "new": "New image"}
_ABSENT: Final = "\U00002014"
_MARKUP: Final = {
    "delete": ("<del>", "</del>"),
    "insert": ("<ins>", "</ins>"),
    "equal": ("", ""),
}
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


def _backtick_fence(text: str, minimum: int = 1) -> str:
    """Return backticks that are more than any run of backticks in a text."""
    longest = max(map(len, _BACKTICKS_RE.findall(text)), default=0)
    return "`" * max(longest + 1, minimum)


def md_fence(lines: Sequence[str]) -> str:
    """Return the fence of a block of code in Markdown.

    Args:
        lines: Lines of the block.

    Returns:
        Backticks that are more than any run of backticks in the lines, and
        three at least.
    """
    return _backtick_fence("\n".join(lines), minimum=3)


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
    fence = _backtick_fence(text)
    blank = " " if "`" in text else ""
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
        Escaped text, with a ``<br>`` for each line break, deletions in
        ``<del>`` and insertions in ``<ins>``.
    """
    parts = []
    for segment in segments:
        opening, closing = _MARKUP[segment.op]
        lines = _LINE_BREAK_RE.split(segment.text)
        text = "<br>".join(md_text(line) for line in lines)
        parts.append(f"{opening}{text}{closing}")
    return "".join(parts)


def diff_lines(finding: Finding) -> list[str]:
    """Return the lines that tell how the code of a finding changed.

    Args:
        finding: A finding with a code side.

    Returns:
        The lines of both sides, each after a two character prefix: two
        blanks for a line that both sides have, ``- `` for a line of the
        reference only and ``+ `` for a line of the new side only. Where
        lines changed, those of the reference come first. A missing side
        counts for no line.
    """
    ref = finding.ref.text.splitlines() if finding.ref else []
    new = finding.new.text.splitlines() if finding.new else []
    matcher = difflib.SequenceMatcher(None, ref, new, autojunk=False)
    lines: list[str] = []
    for tag, ref_start, ref_end, new_start, new_end in matcher.get_opcodes():
        if tag == "equal":
            lines += [f"  {line}" for line in ref[ref_start:ref_end]]
            continue
        lines += [f"- {line}" for line in ref[ref_start:ref_end]]
        lines += [f"+ {line}" for line in new[new_start:new_end]]
    return lines


def _plain(value: str) -> str:
    """Return a raw value of the inputs table, as text of a cell."""
    return value if value == UNKNOWN else md_text(value)


def _source(
    context: ReportContext,
    side: Side | None,
    which: Literal["ref", "new"],
) -> str:
    """Return the Markdown link to the source of a side."""
    source = context.source(side, which)
    if source is None:
        return _ABSENT
    href, url = source
    return f"[{md_text(href)}]({url})"


def _cell(side: Side | None) -> str:
    """Return the table cell of a side."""
    return _ABSENT if side is None else md_cell(side.segments)


def _count_of(categories: Sequence[tuple[object, Sequence[object]]]) -> int:
    """Return the number of findings in the categories of a severity."""
    return sum(len(findings) for _, findings in categories)


def _holds(target: Path, size: int) -> bool:
    """Tell whether a file exists with a given size.

    Args:
        target: The file.
        size: The size it must have, in bytes.

    Returns:
        Whether it is a file of that size.
    """
    try:
        return target.is_file() and target.stat().st_size == size
    except OSError:
        return False


class _Assets:
    """The thumbnails that a Markdown report shows, as files next to it."""

    def __init__(self, thumbnails: Thumbnails) -> None:
        """Start with no thumbnail shown.

        Args:
            thumbnails: Thumbnails of the sides of the findings.
        """
        self._thumbnails = thumbnails
        self._shown: dict[str, Thumbnail] = {}

    def __call__(self, side: Side | None, which: Literal["ref", "new"]) -> str:
        """Return the image that opens the table cell of a side.

        Args:
            side: A side of a finding, if any.
            which: The document of the side.

        Returns:
            The image of its thumbnail and a line break, or "" without one.
        """
        thumb = self._thumbnails.get(side, which)
        if thumb is None:
            return ""
        self._shown.setdefault(thumb.name, thumb)
        return f"![{_IMAGE_LABELS[which]}]({ASSETS_DIR}/{thumb.name})<br>"

    def write(self, directory: Path) -> None:
        """Write each thumbnail shown that the directory does not hold yet.

        A file of the name and of another size, which a run cut short may
        leave, is written again; the bytes of a thumbnail are the same at
        each run.

        Args:
            directory: Where the thumbnails go, created if some are shown.
        """
        if not self._shown:
            return
        directory.mkdir(exist_ok=True)
        for name, thumb in self._shown.items():
            target = directory / name
            if not _holds(target, len(thumb.data)):
                target.write_bytes(thumb.data)


def render_markdown(
    result: CompareResult,
    path: Path,
    thumbnails: Thumbnails | None = None,
) -> Path:
    """Write the Markdown report of a comparison, and its thumbnails.

    Args:
        result: Comparison result.
        path: Report file to write.
        thumbnails: Thumbnails of the sides of the findings, shared with the
            other report; ``None`` for thumbnails of this report only.

    Returns:
        The written file. The thumbnails that it shows are in `ASSETS_DIR`,
        next to it.
    """
    if thumbnails is None:
        with Thumbnails(result) as own:
            return render_markdown(result, path, own)
    context = ReportContext(result, path.parent)
    assets = _Assets(thumbnails)
    inputs = [
        (label, md_cell_code(ref), md_cell_code(new))
        if label == "Path"
        else (label, _plain(ref), _plain(new))
        for label, ref, new in context.inputs()
    ]
    text = TEMPLATES.get_template("report.md.j2").render(
        inputs=inputs,
        matched=context.matched(),
        matched_images=context.matched_images(),
        moved=result.stats.moved,
        categories=context.categories(),
        summary=[
            (SEVERITY_TITLES[severity], counts, total)
            for severity, counts, total in context.summary()
        ],
        normalization=context.normalization(),
        chrome=context.chrome(),
        warnings=context.warnings(),
        sections=[
            (SEVERITY_TITLES[severity], _count_of(categories), categories)
            for severity, categories in context.sections()
        ],
        category_titles=CATEGORY_TITLES,
        context=context,
        is_code=is_code,
        diff_lines=diff_lines,
        md_fence=md_fence,
        md_code=md_code,
        md_text=md_text,
        shown_pattern=shown_pattern,
        link=lambda side, which: _source(context, side, which),
        cell=_cell,
        image=assets,
    )
    assets.write(path.parent / ASSETS_DIR)
    path.write_text(text, encoding="utf-8")
    return path
