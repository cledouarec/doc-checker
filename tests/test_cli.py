# Copyright 2026 Christophe Le Douarec
"""Tests for the command line interface."""

import re
from typing import TYPE_CHECKING

import click
import lxml.html
import pytest
from click.testing import CliRunner

from doc_checker import __version__
from doc_checker.cli import main
from tests.builders import (
    FIGURES_EN,
    SAMPLE_EN,
    build_mkdocs_like,
    build_sphinx_like,
    figure_defect,
    replace_text,
)
from tests.pdf_builders import (
    build_fpdf2_pdf,
    build_raw_pdf,
    build_reportlab_pdf,
    build_scanned_pdf,
)

if TYPE_CHECKING:
    from pathlib import Path

# A URI scheme, such as http: or file:, that opens a link.
SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.-]*:", re.IGNORECASE)


def test_main_without_arguments_shows_help() -> None:
    """Running the CLI without a command prints the help and succeeds."""
    result = CliRunner().invoke(main, [])
    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_version_flag_prints_version() -> None:
    """The ``--version`` flag prints the package version."""
    result = CliRunner().invoke(main, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_unknown_command_is_usage_error() -> None:
    """An unknown command is reported as a usage error."""
    result = CliRunner().invoke(main, ["nope"])
    assert result.exit_code == 2


def test_extract_lists_blocks(tmp_path: Path) -> None:
    """The extract command prints the normalized blocks."""
    site = build_mkdocs_like(SAMPLE_EN, tmp_path / "site")
    result = CliRunner().invoke(main, ["extract", str(site)])
    assert result.exit_code == 0, result.output
    assert "heading" in result.output
    assert "Install the package" in result.output


def test_extract_pdf_lists_blocks(tmp_path: Path) -> None:
    """The extract command prints the blocks of a PDF with their pages."""
    pdf = build_reportlab_pdf(SAMPLE_EN, tmp_path / "doc.pdf")
    result = CliRunner().invoke(main, ["extract", str(pdf)])
    assert result.exit_code == 0, result.output
    assert ": pdf, 9 page(s), generator ReportLab" in result.output
    assert "tagged no" in result.output
    assert "doc.pdf#page=2" in result.output


def test_extract_password_protected_pdf_exits_3(tmp_path: Path) -> None:
    """A PDF protected by a user password exits with code 3."""
    pdf = build_raw_pdf(
        tmp_path / "secret.pdf",
        ["BT /F1 10 Tf 72 700 Td (Hi) Tj ET"],
        encrypt="secret",
    )
    result = CliRunner().invoke(main, ["extract", str(pdf)])
    assert result.exit_code == 3
    assert "password protected" in result.output


def test_extract_unsupported_input_exits_3(tmp_path: Path) -> None:
    """An unsupported input exits with code 3."""
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    result = CliRunner().invoke(main, ["extract", str(tmp_path / "notes.txt")])
    assert result.exit_code == 3
    assert "unsupported input" in result.output


def test_extract_empty_site_exits_3(tmp_path: Path) -> None:
    """A directory without pages exits with code 3."""
    result = CliRunner().invoke(main, ["extract", str(tmp_path)])
    assert result.exit_code == 3
    assert "no HTML pages" in result.output


def test_extract_bad_config_exits_2(tmp_path: Path) -> None:
    """An invalid configuration exits with code 2."""
    conf = tmp_path / "c.toml"
    conf.write_text("[general]\nlang = 'de'\n", encoding="utf-8")
    site = build_mkdocs_like(SAMPLE_EN, tmp_path / "site")
    result = CliRunner().invoke(main, ["extract", str(site), "-c", str(conf)])
    assert result.exit_code == 2


def test_compare_writes_both_reports(tmp_path: Path) -> None:
    """The comparison writes the HTML and the Markdown reports."""
    ref = build_sphinx_like(SAMPLE_EN, tmp_path / "ref")
    new = build_mkdocs_like(SAMPLE_EN, tmp_path / "new")
    out = tmp_path / "out"
    args = ["compare", str(ref), str(new), "-o", str(out)]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert (out / "report.html").is_file()
    assert (out / "report.md").is_file()
    assert "report.html" in result.output


def test_compare_html_report_references_no_external_resource(
    tmp_path: Path,
) -> None:
    """The HTML report loads nothing and links to no other site."""
    ref = build_sphinx_like(SAMPLE_EN, tmp_path / "ref")
    new = build_mkdocs_like(
        replace_text(SAMPLE_EN, "100 requests", "10 requests"),
        tmp_path / "new",
    )
    out = tmp_path / "out"
    args = ["compare", str(ref), str(new), "-o", str(out)]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    html = (out / "report.html").read_text(encoding="utf-8")
    tree = lxml.html.document_fromstring(html)
    assert not tree.xpath("//link | //@src")
    hrefs = tree.xpath("//@href")
    assert hrefs
    assert not [
        href
        for href in hrefs
        if SCHEME_RE.match(href) or href.startswith("//")
    ]
    style = tree.findtext(".//style")
    assert "url(" not in style
    assert "@import" not in style


def test_compare_writes_image_thumbnails(tmp_path: Path) -> None:
    """A degraded figure appears in both reports with its thumbnails."""
    ref = build_sphinx_like(FIGURES_EN, tmp_path / "ref")
    new = build_mkdocs_like(
        figure_defect(FIGURES_EN, "setup", "x0.4"),
        tmp_path / "new",
    )
    out = tmp_path / "out"
    args = ["compare", str(ref), str(new), "-o", str(out)]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert "images 4 <-> 4 | matched 4" in click.unstyle(result.output)
    html = (out / "report.html").read_text(encoding="utf-8")
    assert html.count('class="thumb"') == 2
    assert len(list((out / "report_assets").iterdir())) == 2
    tree = lxml.html.document_fromstring(html)
    assert not tree.xpath("//link | //svg")
    sources = tree.xpath("//@src")
    assert len(sources) == 2
    assert all(s.startswith("data:image/webp;base64,") for s in sources)


def test_compare_markdown_only(tmp_path: Path) -> None:
    """The format option chooses the reports."""
    site = build_mkdocs_like(SAMPLE_EN, tmp_path / "s")
    out = tmp_path / "out"
    args = ["compare", str(site), str(site), "-o", str(out), "--format", "md"]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert not (out / "report.html").exists()


def test_compare_mixed_formats_exits_3(tmp_path: Path) -> None:
    """Comparing builds of different formats exits with code 3."""
    ref = build_mkdocs_like(SAMPLE_EN, tmp_path / "ref")
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.7")
    result = CliRunner().invoke(main, ["compare", str(ref), str(pdf)])
    assert result.exit_code == 3
    assert "same format" in result.output


def test_compare_pdf_writes_report_with_page_links(tmp_path: Path) -> None:
    """Comparing two PDFs writes a report that links to their pages."""
    ref = build_reportlab_pdf(SAMPLE_EN, tmp_path / "ref.pdf")
    new = build_fpdf2_pdf(
        replace_text(SAMPLE_EN, "100 requests", "10 requests"),
        tmp_path / "new.pdf",
    )
    out = tmp_path / "out"
    args = ["compare", str(ref), str(new), "-o", str(out)]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert "CRITICAL text.number_changed  ref.pdf#page=9" in result.output
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "[ref.pdf#page=9](../ref.pdf#page=9)" in report


def test_compare_pdf_with_html_exits_3(tmp_path: Path) -> None:
    """Comparing a PDF with an HTML build exits with code 3."""
    pdf = build_reportlab_pdf(SAMPLE_EN, tmp_path / "ref.pdf")
    site = build_mkdocs_like(SAMPLE_EN, tmp_path / "site")
    result = CliRunner().invoke(main, ["compare", str(pdf), str(site)])
    assert result.exit_code == 3
    assert "same format" in result.output


def test_compare_scanned_pdf_warns(tmp_path: Path) -> None:
    """Comparing a scanned PDF warns about its pages without text."""
    scan = build_scanned_pdf(tmp_path / "scan.pdf", pages=2)
    new = build_fpdf2_pdf(SAMPLE_EN, tmp_path / "new.pdf")
    args = ["compare", str(scan), str(new), "-o", str(tmp_path / "out")]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert (
        "warning: ref: 2 page(s) without text, probably scanned: 1, 2"
        in result.output
    )


def test_compare_unknown_format_exits_2(tmp_path: Path) -> None:
    """An unknown report format is a usage error."""
    site = build_mkdocs_like(SAMPLE_EN, tmp_path / "s")
    args = ["compare", str(site), str(site), "--format", "pdf"]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 2
    assert "unknown format" in result.output


def test_compare_missing_path_exits_2(tmp_path: Path) -> None:
    """A missing input path is a usage error."""
    args = ["compare", str(tmp_path / "a"), str(tmp_path / "b")]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 2
    assert "does not exist" in result.output


def test_no_images_option_is_accepted(tmp_path: Path) -> None:
    """Both commands accept the option that leaves the images out."""
    site = build_mkdocs_like(SAMPLE_EN, tmp_path / "s")
    out = tmp_path / "out"
    for args in (
        ["extract", str(site), "--no-images"],
        ["compare", str(site), str(site), "-o", str(out), "--no-images"],
    ):
        result = CliRunner().invoke(main, args)
        assert result.exit_code == 0, result.output


def test_audit_writes_both_reports(tmp_path: Path) -> None:
    """An audit of a site exits 0 and writes self-contained reports."""
    site = build_sphinx_like(FIGURES_EN, tmp_path / "site", split="chapters")
    out = tmp_path / "out"
    result = CliRunner().invoke(main, ["audit", str(site), "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert result.output.startswith("audit  doc=")
    assert (out / "report.md").is_file()
    tree = lxml.html.parse(out / "report.html")
    assert not tree.xpath("//*[@src and not(starts-with(@src, 'data:'))]")
    assert not tree.xpath("//*[starts-with(@href, 'http')]")


def test_audit_of_an_untagged_pdf(tmp_path: Path) -> None:
    """The report of a reportlab PDF tells that it is not tagged."""
    pdf = build_reportlab_pdf(SAMPLE_EN, tmp_path / "doc.pdf")
    out = tmp_path / "out"
    result = CliRunner().invoke(
        main,
        ["audit", str(pdf), "-o", str(out), "--checks", "a11y"],
    )
    assert result.exit_code == 0, result.output
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "`a11y.pdf_untagged`" in report


def test_audit_unknown_check_exits_2(tmp_path: Path) -> None:
    """An unknown check family is a usage error."""
    site = build_mkdocs_like(SAMPLE_EN, tmp_path / "site")
    result = CliRunner().invoke(
        main,
        ["audit", str(site), "--checks", "grammar"],
    )
    assert result.exit_code == 2
    assert "unknown check 'grammar'" in result.output


def test_audit_unreadable_input_exits_3(tmp_path: Path) -> None:
    """A password protected PDF cannot be audited."""
    pdf = build_raw_pdf(tmp_path / "secret.pdf", [], encrypt="secret")
    result = CliRunner().invoke(main, ["audit", str(pdf)])
    assert result.exit_code == 3


AUDIT_SUMMARY_CASES = [
    ([], "checks images, structure, a11y, text | profile screen"),
    (
        ["--profile", "print"],
        "checks images, structure, a11y, text | profile print",
    ),
    (["--checks", "a11y,images"], "checks a11y, images | profile screen"),
    (
        ["--checks", "a11y,a11y,structure"],
        "| checks a11y, structure | profile",
    ),
    (["--no-images"], "| checks structure, a11y, text | profile screen"),
    (["--checks", "a11y"], "| links - | checks a11y | profile screen"),
    (["--checks", "images", "--no-images"], "| links - | checks - |"),
]


@pytest.mark.parametrize(("args", "expected"), AUDIT_SUMMARY_CASES)
def test_audit_options_reach_the_summary(
    tmp_path: Path,
    args: list[str],
    expected: str,
) -> None:
    """The check families and the profile of the command line are shown."""
    site = build_mkdocs_like(SAMPLE_EN, tmp_path / "site")
    cmd = ["audit", str(site), "-o", str(tmp_path / "out"), *args]
    result = CliRunner().invoke(main, cmd)
    assert result.exit_code == 0, result.output
    assert expected in result.output


def test_audit_no_images_reads_no_image(tmp_path: Path) -> None:
    """With --no-images the audit still runs and says images are not read."""
    site = build_sphinx_like(FIGURES_EN, tmp_path / "site", split="chapters")
    cmd = ["audit", str(site), "-o", str(tmp_path / "out"), "--no-images"]
    result = CliRunner().invoke(main, cmd)
    assert result.exit_code == 0, result.output
    assert "images not read" in result.output


def test_audit_empty_checks_exits_2(tmp_path: Path) -> None:
    """An empty list of check families is a usage error."""
    site = build_mkdocs_like(SAMPLE_EN, tmp_path / "site")
    result = CliRunner().invoke(main, ["audit", str(site), "--checks", ","])
    assert result.exit_code == 2
    assert "no check in" in result.output


def test_audit_command_line_overrides_the_config_file(tmp_path: Path) -> None:
    """--checks and --profile win over the checks and profile of a file."""
    site = build_mkdocs_like(SAMPLE_EN, tmp_path / "site")
    conf = tmp_path / "conf.toml"
    conf.write_text(
        '[general]\nchecks = ["images"]\nprofile = "print"\n',
        encoding="utf-8",
    )
    base = ["audit", str(site), "-o", str(tmp_path / "out"), "-c", str(conf)]
    kept = CliRunner().invoke(main, base)
    assert "checks images | profile print" in kept.output
    over = CliRunner().invoke(
        main,
        [*base, "--checks", "a11y", "--profile", "screen"],
    )
    assert "checks a11y | profile screen" in over.output


def test_no_images_help_names_what_each_command_does() -> None:
    """The audit leaves the images out of its checks, compare of its own."""
    audit = CliRunner().invoke(main, ["audit", "--help"]).output
    compare = CliRunner().invoke(main, ["compare", "--help"]).output
    assert "extraction and the checks." in " ".join(audit.split())
    assert "extraction and the comparison." in " ".join(compare.split())
    assert "comparison" not in audit
