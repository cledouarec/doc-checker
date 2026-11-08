# Copyright 2026 Christophe Le Douarec
"""Tests for the Markdown report and the terminal summary."""

import re
from dataclasses import replace
from typing import TYPE_CHECKING

import click
import pytest

from doc_checker.compare.engine import CompareResult, compare_documents
from doc_checker.config import Config, HtmlConfig
from doc_checker.errors import ConfigError
from doc_checker.extract.html import load_html
from doc_checker.images.refs import raster_ref
from doc_checker.model.document import BlockKind, Link, PixelSource, SourceRef
from doc_checker.model.findings import DiffSegment
from doc_checker.pipeline import write_reports
from doc_checker.report.context import MAX_OCCURRENCES_SHOWN, relative_link
from doc_checker.report.markdown import (
    md_cell,
    md_cell_code,
    md_code,
    md_text,
    render_markdown,
)
from doc_checker.report.terminal import render_blocks, render_summary, visible
from tests.builders import (
    FIGURES_EN,
    PARAGRAPHS,
    build_sphinx_like,
    image_result,
    make_block,
    make_document,
    make_image_block,
)
from tests.image_builders import drawing, encode

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
        "#### Block missing (`block.missing`)",
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


def test_render_blocks_lists_links() -> None:
    """The listing of a document shows the targets of its links."""
    links = (
        Link(
            "guide",
            "guide/",
            page="guide/index.html",
            fragment="x",
            found=True,
        ),
        Link("gone", "gone.html", page="gone.html"),
        Link("site", "https://example.org", url="//example.org"),
    )
    block = replace(make_block("See the guide."), links=links)
    lines = render_blocks(make_document(block)).splitlines()
    prefix = f"{'':>5}  {'link':<10}  "
    assert lines[2:] == [
        f"{prefix}guide -> guide/index.html#x",
        f"{prefix}gone -> gone.html (missing)",
        f"{prefix}site -> https://example.org",
    ]


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
    """Both reports are written in a newly created output directory."""
    out = tmp_path / "out" / "nested"
    paths = write_reports(result, Config(output=out))
    assert paths == [out / "report.html", out / "report.md"]
    assert all(path.is_file() for path in paths)


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


def test_terminal_counts_a_group_once() -> None:
    """A critical group is one line that tells its occurrences."""
    texts = [f"Paragraph {i} says that the backup is safe." for i in range(6)]
    ref = make_document(*[make_block(t) for t in texts])
    new = make_document(
        *[make_block(t.replace("is safe", "is not safe")) for t in texts],
    )
    summary = click.unstyle(
        render_summary(compare_documents(ref, new, Config()), []),
    )
    assert "findings: 1 critical, 0 major, 0 minor, 0 info" in summary
    assert "(6 occurrences)" in summary


def test_markdown_sections_by_category(
    tmp_path: Path,
    result: CompareResult,
) -> None:
    """Each severity has a section per category."""
    path = render_markdown(result, tmp_path / "report.md")
    text = path.read_text(encoding="utf-8")
    assert "## Critical (1)\n\n### Content (1)\n\n#### Block missing" in text
    assert "## Major (1)\n\n### Text (1)\n\n#### Typo introduced" in text


def test_markdown_code_is_a_diff(tmp_path: Path) -> None:
    """A difference in code is shown as a diff block."""
    ref = make_document(
        make_block("pip install example\nexample --version", BlockKind.CODE),
    )
    new = make_document(
        make_block("pip install example\nexample --verbose", BlockKind.CODE),
    )
    result = compare_documents(ref, new, Config())
    path = render_markdown(result, tmp_path / "r.md")
    text = path.read_text(encoding="utf-8")
    assert (
        "```diff\n  pip install example\n- example --version\n"
        "+ example --verbose\n```"
    ) in text


def _accent_result() -> CompareResult:
    """Compare six paragraphs that each lose one accent."""
    texts = [f"Le paragraphe {n} décrit le service." for n in range(6)]
    ref = make_document(*[make_block(t) for t in texts])
    new = make_document(*[make_block(t.replace("é", "e")) for t in texts])
    return compare_documents(ref, new, Config())


# A row of the table of occurrences starts with the identifier in code.
OCCURRENCE_ROW = re.compile(r"^\| `[0-9a-f]{12}` \|", re.MULTILINE)


def test_markdown_groups(tmp_path: Path) -> None:
    """A group tells its occurrences, its pattern and lists them."""
    path = render_markdown(_accent_result(), tmp_path / "r.md")
    text = path.read_text(encoding="utf-8")
    assert "- Occurrences: 6\n- Pattern: é\U00002192e\n" in text
    assert "<details><summary>6 occurrences</summary>" in text
    assert len(OCCURRENCE_ROW.findall(text)) == 6


def test_markdown_caps_occurrences(tmp_path: Path) -> None:
    """A group lists a hundred occurrences at most."""
    block = make_block("Back to the top of the page.")
    result = compare_documents(
        make_document(*[block] * 120),
        make_document(),
        Config(),
    )
    path = render_markdown(result, tmp_path / "r.md")
    text = path.read_text(encoding="utf-8")
    assert len(OCCURRENCE_ROW.findall(text)) == MAX_OCCURRENCES_SHOWN
    assert "and 20 more occurrences" in text


def test_markdown_escapes_the_text_of_changed_cells(tmp_path: Path) -> None:
    """A changed block never forms a link or an emphasis in its cell."""
    ref = make_document(make_block("Open the *page* for the settings now."))
    new = make_document(
        make_block("Open the [x](javascript:alert(1)) for the settings now."),
    )
    result = compare_documents(ref, new, Config())
    path = render_markdown(result, tmp_path / "r.md")
    text = path.read_text(encoding="utf-8")
    assert "\\[x\\](javascript:alert(1))" in text
    assert "[x](javascript" not in text
    assert "*page*" not in text.replace("\\*page\\*", "")


def test_markdown_summary_names_categories_and_severities(
    tmp_path: Path,
    result: CompareResult,
) -> None:
    """The summary table names its rows and columns as the sections do."""
    path = render_markdown(result, tmp_path / "report.md")
    text = path.read_text(encoding="utf-8")
    assert "| Severity | Content | Text | Total |\n" in text
    assert "| Critical | 1 | 0 | 1 |\n| Major | 0 | 1 | 1 |\n" in text


# A paragraph longer than the pattern of a group may show.
LONG_TEXT = (
    "Back to the top of the page, where the contents list every section "
    "of this guide and of its appendices."
)


def test_markdown_group_of_blocks_shows_their_text(tmp_path: Path) -> None:
    """A group of missing blocks shows the text of the first one, cut."""
    result = compare_documents(
        make_document(*[make_block(LONG_TEXT)] * 6),
        make_document(),
        Config(),
    )
    path = render_markdown(result, tmp_path / "r.md")
    text = path.read_text(encoding="utf-8")
    assert f"- Pattern: {LONG_TEXT[:80]}\U00002026\n" in text


def test_markdown_diff_deletes_before_it_adds(tmp_path: Path) -> None:
    """The lines of the reference come before the new ones in a diff."""
    code = "pip install example\nexample --version"
    result = compare_documents(
        make_document(make_block(code, BlockKind.CODE)),
        make_document(make_block(code.replace("\n", " "))),
        Config(),
    )
    path = render_markdown(result, tmp_path / "r.md")
    text = path.read_text(encoding="utf-8")
    assert (
        "```diff\n- pip install example\n- example --version\n"
        "+ pip install example example --version\n```"
    ) in text


def _write(result: CompareResult, out: Path) -> dict[str, bytes]:
    """Write both reports and return every file written, by name."""
    write_reports(result, Config(output=out))
    return {
        path.relative_to(out).as_posix(): path.read_bytes()
        for path in sorted(out.rglob("*"))
        if path.is_file()
    }


def test_reports_are_byte_stable(tmp_path: Path) -> None:
    """Two runs write the same files, byte for byte."""
    result = image_result()
    first = _write(result, tmp_path / "a")
    second = _write(result, tmp_path / "b")
    assert first == second
    assets = [name for name in first if name.startswith("report_assets/")]
    assert len(assets) == 2


def test_markdown_assets_exist(tmp_path: Path) -> None:
    """Each image of the Markdown report is a file next to it."""
    path = render_markdown(image_result(), tmp_path / "report.md")
    text = path.read_text(encoding="utf-8")
    names = re.findall(r"\]\((report_assets/[0-9a-f]{16}\.webp)\)", text)
    assert len(names) == 2
    assert all((tmp_path / name).is_file() for name in names)


def test_markdown_assets_are_kept_or_mended(tmp_path: Path) -> None:
    """A thumbnail written whole is kept, a cut one rewritten, others stay."""
    first = render_markdown(image_result(), tmp_path / "report.md")
    text = first.read_text(encoding="utf-8")
    names = re.findall(r"!\[[A-Za-z]+ image\]\(report_assets/(.+?)\)", text)
    assets = tmp_path / "report_assets"
    kept, cut = (assets / name for name in names)
    whole, data = kept.stat().st_mtime_ns, cut.read_bytes()
    cut.write_bytes(data[: len(data) // 2])
    (assets / "0000000000000000.webp").write_bytes(b"older")
    again = render_markdown(image_result(), first)
    assert (kept.stat().st_mtime_ns, cut.read_bytes()) == (whole, data)
    assert (assets / "0000000000000000.webp").read_bytes() == b"older"
    assert again.read_text(encoding="utf-8") == text
    assert len(list(assets.iterdir())) == 3


def test_reports_tell_whether_images_were_compared(tmp_path: Path) -> None:
    """The reports count the images matched, or say they were not compared."""
    files = _write(image_result(), tmp_path / "on")
    report = files["report.md"].decode()
    assert "| Images | 1 | 1 |" in report
    assert "of reference blocks, 1 of 1 reference images \xb7" in report
    html_report = files["report.html"].decode()
    assert "of reference blocks, 1 of 1 reference images &middot;" in (
        html_report
    )
    block = make_block(PARAGRAPHS[0])
    off = compare_documents(
        make_document(block),
        make_document(block),
        Config(images=False),
    )
    files = _write(off, tmp_path / "off")
    report = files["report.md"].decode()
    assert "| Images | not compared | not compared |" in report
    assert "reference images" not in report
    html_report = files["report.html"].decode()
    assert html_report.count('<td class="quote">not compared</td>') == 2


def test_report_with_many_image_findings_stays_bounded(tmp_path: Path) -> None:
    """Many missing images make one group and a bounded report."""
    blocks = []
    for number in range(150):
        data = encode(drawing(f"image {number}", size=(200, 120)))
        ref = raster_ref(
            data,
            name=f"{number}.png",
            source=PixelSource(data=data),
        )
        blocks += [
            make_block(f"Paragraph {number} talks about its own topic here."),
            make_image_block(ref),
        ]
    texts = [block for block in blocks if block.kind is not BlockKind.IMAGE]
    result = compare_documents(
        make_document(*blocks),
        make_document(*texts),
        Config(),
    )
    (group,) = [f for f in result.findings if f.rule == "image.missing"]
    assert len(group.occurrences) == 150
    out = tmp_path / "out"
    write_reports(result, Config(output=out))
    assert (out / "report.html").stat().st_size < 3_000_000
    assert len(list((out / "report_assets").iterdir())) == 100


def test_extract_lists_images(tmp_path: Path) -> None:
    """The listing of a document describes its images."""
    site = build_sphinx_like(FIGURES_EN, tmp_path / "s")
    doc = load_html(site, HtmlConfig(), lang="en")
    text = render_blocks(doc)
    assert "image       index.html#installation  _images/setup.png" in text
    assert '_images/setup.png, 400x250 px, PNG, alt "Installer steps"' in text
