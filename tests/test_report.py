# Copyright 2026 Christophe Le Douarec
"""Tests for the Markdown report and the terminal summary."""

from dataclasses import replace
from typing import TYPE_CHECKING

import click
import pytest

from doc_checker.compare.engine import CompareResult, compare_documents
from doc_checker.config import Config
from doc_checker.errors import ConfigError
from doc_checker.model.document import BlockKind, SourceRef
from doc_checker.model.findings import DiffSegment
from doc_checker.pipeline import write_reports
from doc_checker.report.markdown import (
    md_cell,
    md_cell_code,
    md_code,
    md_text,
    relative_link,
    render_markdown,
)
from doc_checker.report.terminal import render_blocks, render_summary, visible
from tests.builders import PARAGRAPHS, make_block, make_document

if TYPE_CHECKING:
    from pathlib import Path

REF_TEXT = (
    "Install the package with the default configuration before you start."
)


@pytest.fixture
def result() -> CompareResult:
    """Compare documents with one missing block and one typo."""
    ref = make_document(make_block(REF_TEXT), make_block(PARAGRAPHS[0]))
    new = make_document(
        make_block(REF_TEXT.replace("configuration", "configuraiton")),
    )
    return compare_documents(ref, new, Config())


@pytest.fixture
def pdf_result(tmp_path: Path) -> CompareResult:
    """Compare a one-block PDF with a scan warning to an empty document."""
    pdf = tmp_path / "ref.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")
    block = replace(
        make_block(PARAGRAPHS[0]),
        source=SourceRef("ref.pdf", page=3),
    )
    ref = replace(
        make_document(block),
        path=pdf,
        format="pdf",
        warnings=["1 page(s) without text, probably scanned: 2"],
    )
    return compare_documents(ref, make_document(), Config())


def test_markdown_report_sections(
    tmp_path: Path,
    result: CompareResult,
) -> None:
    """The report has its sections and the highlighted differences."""
    path = render_markdown(result, tmp_path / "report.md")
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# doc-checker report: compare")
    for expected in (
        "## Inputs",
        "## Summary",
        "## Critical (1)",
        "### Block missing (`block.missing`)",
        "## Major (1)",
        "<ins>configuraiton</ins>",
        "<del>configuration</del>",
    ):
        assert expected in text


def test_report_without_findings(tmp_path: Path) -> None:
    """A comparison without findings says so."""
    doc = make_document(make_block(PARAGRAPHS[0]))
    same = compare_documents(doc, doc, Config())
    path = render_markdown(same, tmp_path / "r.md")
    assert "No findings." in path.read_text(encoding="utf-8")


def test_md_cell_escapes() -> None:
    """Table cells escape pipes, markup and newlines."""
    segments = (
        DiffSegment("equal", "a|b <c>\nd"),
        DiffSegment("delete", "x"),
    )
    assert md_cell(segments) == "a\\|b &lt;c&gt;<br>d<del>x</del>"


def test_relative_link_non_ascii(tmp_path: Path) -> None:
    """Links are relative to the report and percent encoded."""
    target = tmp_path / "ref" / "guide" / "café.html"
    link = relative_link(target, "x", tmp_path / "out")
    assert link == "../ref/guide/caf%C3%A9.html#x"


def test_terminal_summary(tmp_path: Path, result: CompareResult) -> None:
    """The terminal summary counts findings and lists critical ones."""
    text = click.unstyle(render_summary(result, [tmp_path / "report.md"]))
    assert "findings: 1 critical, 1 major, 0 minor, 0 info" in text
    assert "CRITICAL block.missing" in text
    assert "reports: " in text


def test_markdown_pdf_links_and_warnings(
    tmp_path: Path,
    pdf_result: CompareResult,
) -> None:
    """PDF sources link to their page and the warnings are listed."""
    (tmp_path / "out").mkdir()
    path = render_markdown(pdf_result, tmp_path / "out" / "report.md")
    text = path.read_text(encoding="utf-8")
    assert "[ref.pdf#page=3](../ref.pdf#page=3)" in text
    assert (
        "## Warnings\n\n"
        "- Reference: 1 page(s) without text, probably scanned: 2\n"
    ) in text


def test_terminal_summary_lists_warnings(pdf_result: CompareResult) -> None:
    """The terminal summary shows the warnings of the reference."""
    text = click.unstyle(render_summary(pdf_result, []))
    assert "warning: ref: 1 page(s) without text, probably scanned: 2" in text


def test_markdown_warnings_precede_the_findings(
    tmp_path: Path,
    pdf_result: CompareResult,
) -> None:
    """Warnings of both documents sit right before the severity sections."""
    ref = replace(pdf_result.ref, chrome=["Draft"])
    new = replace(
        pdf_result.new,
        warnings=["2 page(s) without text, probably scanned: 1, 4"],
    )
    both = replace(pdf_result, ref=ref, new=new)
    path = render_markdown(both, tmp_path / "report.md")
    text = path.read_text(encoding="utf-8")
    assert text.index("## Removed page chrome") < text.index("## Warnings")
    assert (
        "## Warnings\n\n"
        "- Reference: 1 page(s) without text, probably scanned: 2\n"
        "- New: 2 page(s) without text, probably scanned: 1, 4\n"
        "\n## Critical (1)\n"
    ) in text


def test_terminal_warnings_sit_between_findings_and_reports(
    tmp_path: Path,
    pdf_result: CompareResult,
) -> None:
    """Warnings follow the critical findings, reference first, then new."""
    new = replace(
        pdf_result.new,
        warnings=["1 page(s) without text, probably scanned: 1"],
    )
    both = replace(pdf_result, new=new)
    summary = render_summary(both, [tmp_path / "report.md"])
    lines = click.unstyle(summary).splitlines()
    assert lines[-4].startswith("  CRITICAL block.missing  ref.pdf#page=3  ")
    assert lines[-3:-1] == [
        "warning: ref: 1 page(s) without text, probably scanned: 2",
        "warning: new: 1 page(s) without text, probably scanned: 1",
    ]
    assert lines[-1].startswith("reports: ")
    assert "\x1b[33mwarning\x1b[0m: ref: " in summary


def test_no_warnings_are_reported_when_there_are_none(
    tmp_path: Path,
    result: CompareResult,
) -> None:
    """Documents without warnings add no section and no line."""
    path = render_markdown(result, tmp_path / "report.md")
    assert "Warnings" not in path.read_text(encoding="utf-8")
    assert "warning" not in click.unstyle(render_summary(result, []))


def test_render_blocks_pdf_header_and_warnings() -> None:
    """The listing of a PDF tells its generator, tagging and warnings."""
    doc = replace(
        make_document(make_block("Text here.")),
        format="pdf",
        parts=["1", "2"],
        generator="ReportLab",
        tagged=False,
        warnings=["1 page(s) without text, probably scanned: 2"],
    )
    lines = render_blocks(doc).splitlines()
    assert lines[0] == (
        "doc: pdf, 2 page(s), generator ReportLab, lang en, tagged no"
    )
    assert lines[1] == "warning: 1 page(s) without text, probably scanned: 2"


def test_render_blocks_header_tells_tagging() -> None:
    """Only the header of a PDF tells whether it is tagged."""
    html = make_document(make_block("Text here."))
    pdf = replace(html, format="pdf", tagged=True)
    assert render_blocks(html).splitlines()[0] == (
        "doc: html, 1 page(s), generator generic, lang en"
    )
    assert render_blocks(pdf).splitlines()[0].endswith(", tagged yes")


def test_missing_generator_is_a_dash() -> None:
    """A document without a known generator shows a dash."""
    doc = replace(make_document(make_block("Text here.")), generator=None)
    assert render_blocks(doc).splitlines()[0] == (
        "doc: html, 1 page(s), generator -, lang en"
    )
    result = compare_documents(doc, doc, Config())
    assert "(1 pages, -)" in click.unstyle(render_summary(result, []))


def test_write_reports_creates_output_dir(
    tmp_path: Path,
    result: CompareResult,
) -> None:
    """Reports are written in a newly created output directory."""
    cfg = Config(output=tmp_path / "out" / "nested")
    (path,) = write_reports(result, cfg)
    assert path == tmp_path / "out" / "nested" / "report.md"
    assert path.is_file()


def test_unwritable_report_is_a_config_error(
    tmp_path: Path,
    result: CompareResult,
) -> None:
    """A report file that cannot be written is a configuration error."""
    out = tmp_path / "out"
    (out / "report.md").mkdir(parents=True)
    with pytest.raises(ConfigError, match="cannot write reports"):
        write_reports(result, Config(output=out))


def test_terminal_shows_controls_as_replacement_characters() -> None:
    """Controls and bidi overrides of a document never reach the terminal."""
    doc = replace(
        make_document(make_block("Left\x9b2J\U0000202eright")),
        generator="Gen\x9berator",
    )
    text = render_blocks(doc)
    assert "\x9b" not in text
    assert "\U0000202e" not in text
    assert "Left\U0000fffd2J\U0000fffdright" in text
    assert visible("tab\there\nnext") == "tab\there\nnext"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("*a* _b_ [c]", "\\*a\\* \\_b\\_ \\[c\\]"),
        ("<b> & |x| ~y~ `z`", "&lt;b&gt; &amp; \\|x\\| \\~y\\~ \\`z\\`"),
        ("# title", "\\# title"),
        ("- item", "\\- item"),
        ("12. item", "12\\. item"),
        ("a\nb", "a b"),
        ("ref.pdf#page=3", "ref.pdf#page=3"),
        ("1 page(s) without text", "1 page(s) without text"),
    ],
)
def test_md_text(raw: str, expected: str) -> None:
    """Markdown and HTML characters of a document text are escaped."""
    assert md_text(raw) == expected


def test_md_code() -> None:
    """A code span is fenced by more backticks than it holds."""
    assert md_code("out/report") == "`out/report`"
    assert md_code("a`b") == "`` a`b ``"


def test_markdown_escapes_document_texts(tmp_path: Path) -> None:
    """Chrome texts and warnings of a document are escaped."""
    ref = replace(
        make_document(make_block(PARAGRAPHS[0])),
        chrome=["<script>alert(1)</script>"],
        warnings=["*bold* [x](javascript:alert(1))"],
    )
    result = compare_documents(ref, make_document(), Config())
    path = render_markdown(result, tmp_path / "r.md")
    text = path.read_text(encoding="utf-8")
    assert "<script>" not in text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in text
    assert "\\*bold\\* \\[x\\](javascript:alert(1))" in text


def test_relative_link_quotes_the_fragment(tmp_path: Path) -> None:
    """A fragment is percent encoded, except the equal sign of a page."""
    target = tmp_path / "a.html"
    assert relative_link(target, 'x" y', tmp_path).endswith("#x%22%20y")
    assert relative_link(target, "page=3", tmp_path).endswith("#page=3")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("+ item", "\\+ item"),
        ("3) item", "3\\) item"),
        ("## two", "\\## two"),
        ("\n# title", " \\# title"),
        ("a\\b", "a\\\\b"),
        ("x\r\ny\rz", "x y z"),
    ],
)
def test_md_text_keeps_a_text_on_one_line_and_closed(
    raw: str,
    expected: str,
) -> None:
    """A text neither breaks its line nor opens a list or a heading."""
    assert md_text(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("a\nb", "`a b`"), ("``", "``` `` ```"), ("`", "`` ` ``")],
)
def test_md_code_fences_and_lines(raw: str, expected: str) -> None:
    """A code span stays on one line and its fence is never too short."""
    assert md_code(raw) == expected


def test_markdown_escapes_inputs_and_link_labels(tmp_path: Path) -> None:
    """Paths are code spans, and generators, languages and labels escaped."""
    ref = replace(
        make_document(make_block(PARAGRAPHS[0], file="a_b.html")),
        path=tmp_path / "o`ut",
        generator="<Gen> *x*",
        lang=None,
    )
    result = compare_documents(ref, make_document(), Config())
    path = render_markdown(result, tmp_path / "r.md")
    text = path.read_text(encoding="utf-8")
    assert f"| Path | `` {tmp_path}/o`ut `` | `doc` |" in text
    assert "| Generator | &lt;Gen&gt; \\*x\\* | generic |" in text
    assert "| Language | - | en |" in text
    assert "- Reference: [a\\_b.html](o%60ut/a_b.html)\n" in text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("out/report", "`out/report`"),
        ("a|b", "`a\\|b`"),
        ("a\\|b", "`a\\\\\\|b`"),
        ("a\\\\|b", "`a\\\\\\|b`"),
        ("a`|b", "`` a`\\|b ``"),
        ("a\nb", "`a b`"),
    ],
)
def test_md_cell_code(raw: str, expected: str) -> None:
    """A pipe of a cell code span follows an odd number of backslashes."""
    assert md_cell_code(raw) == expected


def test_markdown_keeps_a_hostile_path_in_its_cell(tmp_path: Path) -> None:
    """A pipe in a path ends neither its cell nor its code span."""
    ref = replace(
        make_document(make_block(PARAGRAPHS[0])),
        path=tmp_path / "a|<img src=x onerror=alert(1)>",
    )
    new = replace(make_document(), path=tmp_path / "b\\|<b>")
    result = compare_documents(ref, new, Config())
    path = render_markdown(result, tmp_path / "r.md")
    text = path.read_text(encoding="utf-8")
    assert (
        f"| Path | `{tmp_path}/a\\|<img src=x onerror=alert(1)>` "
        f"| `{tmp_path}/b\\\\\\|<b>` |\n"
    ) in text


def test_markdown_escapes_finding_details(
    tmp_path: Path,
    result: CompareResult,
) -> None:
    """The detail of a finding is escaped like the other texts."""
    first, *others = result.findings
    detailed = replace(first, detail="*3* blocks <b>missing</b>")
    path = render_markdown(
        replace(result, findings=[detailed, *others]),
        tmp_path / "r.md",
    )
    text = path.read_text(encoding="utf-8")
    assert "\n\\*3\\* blocks &lt;b&gt;missing&lt;/b&gt;\n" in text


def test_summary_shows_controls_as_replacement_characters() -> None:
    """Generators, locations and warnings of a document are made visible."""
    ref = replace(
        make_document(make_block(PARAGRAPHS[0], file="a\x9bb.html")),
        generator="Gen\x9berator",
        warnings=["odd\x07page"],
    )
    result = compare_documents(ref, make_document(), Config())
    summary = render_summary(result, [])
    assert "\x9b" not in summary
    assert "\x07" not in summary
    assert "ref=doc (1 pages, Gen\U0000fffderator)" in summary
    plain = click.unstyle(summary)
    assert "CRITICAL block.missing  a\U0000fffdb.html  " in plain
    assert "\x1b[33mwarning\x1b[0m: ref: odd\U0000fffdpage" in summary


def test_render_blocks_keeps_document_values_on_their_line(
    tmp_path: Path,
) -> None:
    """A line break of a document never starts a line of the listing."""
    doc = replace(
        make_document(
            make_block(PARAGRAPHS[0], file="a\nb.html"),
            make_block("line one\nline two", BlockKind.CODE),
        ),
        path=tmp_path / "do\nc",
        generator="Gen\r\nerator",
        lang="e\nn",
        warnings=["first\nsecond"],
    )
    lines = render_blocks(doc).splitlines()
    assert len(lines) == 4
    assert lines[0] == (
        f"{tmp_path}/do c: html, 1 page(s), generator Gen erator, lang e n"
    )
    assert lines[1] == "warning: first second"
    assert " a b.html  " in lines[2]
    assert "line one \U000023ce line two" in lines[3]


def test_render_summary_keeps_document_values_on_their_line(
    tmp_path: Path,
) -> None:
    """A line break of a document never starts a line of the summary."""
    ref = replace(
        make_document(make_block(PARAGRAPHS[0], file="a\nb.html")),
        path=tmp_path / "do\nc",
        generator="Gen\r\nerator",
        warnings=["first\nsecond"],
    )
    result = compare_documents(ref, make_document(), Config())
    lines = click.unstyle(render_summary(result, [])).splitlines()
    assert len(lines) == 6
    assert lines[0] == (
        f"compare  ref={tmp_path}/do c (1 pages, Gen erator)  "
        "new=doc (1 pages, generic)"
    )
    assert lines[3].startswith("  CRITICAL block.missing  a b.html  ")
    assert lines[4] == "warning: ref: first second"


@pytest.mark.parametrize(
    "char",
    [
        "\x00",
        "\x08",
        "\x0b",
        "\x0d",
        "\x1b",
        "\x1f",
        "\x7f",
        "\x85",
        "\x9b",
        "\x9f",
        "\U0000202a",
        "\U0000202e",
        "\U00002066",
        "\U00002069",
    ],
)
def test_visible_replaces_controls_and_bidi_characters(char: str) -> None:
    """Each control and bidi character is replaced, ends of ranges too."""
    assert visible(f"a{char}b") == "a\U0000fffdb"


@pytest.mark.parametrize(
    "char",
    [
        "\t",
        "\n",
        " ",
        "~",
        "\xa0",
        "\U00002029",
        "\U0000202f",
        "\U00002065",
        "\U0000206a",
    ],
)
def test_visible_keeps_the_other_characters(char: str) -> None:
    """The tab, the line feed and what lies next to the ranges are kept."""
    assert visible(f"a{char}b") == f"a{char}b"
