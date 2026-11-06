# Copyright 2026 Christophe Le Douarec
"""End to end comparison of documentation builds."""

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

import pytest

from doc_checker.compare.aggregate import GROUP_MIN_OCCURRENCES
from doc_checker.config import DEFAULT_HTML_EXCLUDE, Config, HtmlConfig
from doc_checker.model.findings import Finding, Severity
from doc_checker.pipeline import run_compare
from tests.builders import (
    SAMPLE_EN,
    SAMPLE_FR,
    DocSpec,
    add_paragraph,
    build_docusaurus_like,
    build_mkdocs_like,
    build_pandoc_like,
    build_sphinx_like,
    demote_heading,
    drop_paragraph,
    drop_section,
    map_texts,
    move_section,
    number_titles,
    replace_text,
    set_level,
    uncode,
    unlist,
    untable,
)

if TYPE_CHECKING:
    from pathlib import Path

BANNER = "Docs for version 2.0."
# The sample with numbered titles, which every builder writes.
NUMBERED_EN = number_titles(SAMPLE_EN)
C = Severity.CRITICAL
MA = Severity.MAJOR
MI = Severity.MINOR
IN = Severity.INFO

type Transform = Callable[[DocSpec], DocSpec]

# Defects injected into a documentation, each with the rule and the severity
# of the one finding it must give. The PDF comparison reuses them.
DEFECTS: list[tuple[DocSpec, Transform, str, Severity]] = [
    (
        SAMPLE_EN,
        lambda s: drop_paragraph(s, "Each project has an owner"),
        "block.missing",
        C,
    ),
    (
        SAMPLE_EN,
        lambda s: drop_section(s, "exports"),
        "section.missing",
        C,
    ),
    (
        SAMPLE_EN,
        lambda s: replace_text(
            s,
            "installer downloads",
            "installer donwloads",
        ),
        "text.typo",
        MA,
    ),
    (
        SAMPLE_EN,
        lambda s: replace_text(s, "100 requests", "10 requests"),
        "text.number_changed",
        C,
    ),
    (
        SAMPLE_EN,
        lambda s: replace_text(s, "Do not delete", "Do delete"),
        "text.negation_changed",
        C,
    ),
    (
        SAMPLE_EN,
        lambda s: replace_text(s, "You must restart", "You may restart"),
        "text.modality_changed",
        MA,
    ),
    (
        SAMPLE_EN,
        lambda s: replace_text(
            s,
            "efficient scheduler",
            "ecient scheduler",
        ),
        "text.ligature_lost",
        MA,
    ),
    (
        SAMPLE_EN,
        lambda s: replace_text(s, "1,000 jobs", "1000 jobs"),
        "text.number_format",
        IN,
    ),
    (
        SAMPLE_EN,
        lambda s: add_paragraph(
            s,
            "exports",
            "Exports can also be scheduled every night.",
        ),
        "block.added",
        MI,
    ),
    (
        SAMPLE_FR,
        lambda s: replace_text(s, "pendant l'été", "pendant l'ete"),
        "text.accent_lost",
        MA,
    ),
    (
        SAMPLE_FR,
        lambda s: replace_text(s, "Ne supprimez pas", "Supprimez"),
        "text.negation_changed",
        C,
    ),
    (
        SAMPLE_FR,
        lambda s: replace_text(
            s,
            "pendant l'été",
            "pendant l'\xc3\xa9t\xc3\xa9",
        ),
        "text.encoding_issue",
        C,
    ),
    (
        SAMPLE_FR,
        lambda s: replace_text(s, "1 000 fichiers", "100 fichiers"),
        "text.number_changed",
        C,
    ),
]


def _compare(
    tmp_path: Path,
    ref_spec: DocSpec,
    new_spec: DocSpec,
    *,
    split: str = "single",
) -> list[Finding]:
    ref = build_sphinx_like(ref_spec, tmp_path / "ref", split=split)  # type: ignore[arg-type]
    new = build_mkdocs_like(new_spec, tmp_path / "new", banner=BANNER)
    return run_compare(ref, new, Config()).findings


@pytest.mark.parametrize("spec", [SAMPLE_EN, SAMPLE_FR])
@pytest.mark.parametrize("split", ["single", "chapters"])
def test_generator_only_differences_produce_no_finding(
    tmp_path: Path,
    spec: DocSpec,
    split: str,
) -> None:
    """The same content built by two generators has no finding."""
    findings = _compare(tmp_path, spec, spec, split=split)
    assert [f.rule for f in findings if f.severity >= MI] == []


@pytest.mark.parametrize(("spec", "transform", "rule", "severity"), DEFECTS)
def test_injected_defect_is_reported(
    tmp_path: Path,
    spec: DocSpec,
    transform: Transform,
    rule: str,
    severity: Severity,
) -> None:
    """Each injected defect is reported once, with its severity."""
    findings = _compare(tmp_path, spec, transform(spec))
    relevant = [
        (f.rule, f.severity)
        for f in findings
        if f.severity >= MI or f.rule == rule
    ]
    assert relevant == [(rule, severity)]


# Changes of structure, each with the one finding of at least minor
# severity that it must give.
STRUCTURE_DEFECTS: list[tuple[DocSpec, Transform, str, Severity]] = [
    (
        SAMPLE_EN,
        lambda s: demote_heading(s, "exports"),
        "structure.heading_lost",
        MA,
    ),
    # The paragraph keeps the number that the heading sets apart.
    (
        NUMBERED_EN,
        lambda s: demote_heading(s, "exports"),
        "structure.heading_lost",
        MA,
    ),
    (
        SAMPLE_EN,
        lambda s: unlist(s, "installation"),
        "structure.list_lost",
        MI,
    ),
    (SAMPLE_EN, lambda s: untable(s, "projects"), "structure.table_lost", MA),
    (
        SAMPLE_EN,
        lambda s: uncode(s, "installation"),
        "structure.code_lost",
        MA,
    ),
    (
        SAMPLE_EN,
        lambda s: set_level(s, "exports", 3),
        "structure.heading_level",
        MI,
    ),
    (
        SAMPLE_EN,
        lambda s: replace_text(s, "(#command-line)", "(#command-lines)"),
        "link.broken",
        C,
    ),
    (
        SAMPLE_EN,
        lambda s: replace_text(s, "[retried](#first-steps)", "retried"),
        "link.lost",
        MA,
    ),
    (
        SAMPLE_EN,
        lambda s: replace_text(
            s,
            "[configuration](#first-steps)",
            "[configuration](#limits)",
        ),
        "link.target_changed",
        MA,
    ),
]


@pytest.mark.parametrize(
    ("spec", "transform", "rule", "severity"),
    STRUCTURE_DEFECTS,
)
def test_structure_defect_is_reported(
    tmp_path: Path,
    spec: DocSpec,
    transform: Transform,
    rule: str,
    severity: Severity,
) -> None:
    """Each change of structure is reported once, with its severity."""
    findings = _compare(tmp_path, spec, transform(spec))
    assert [(f.rule, f.severity) for f in findings if f.severity >= MI] == [
        (rule, severity),
    ]


TITLES = ("Install", "Requirements", "Download", "Configure", "Usage")
NUMBERS = ("1", "1.1", "1.2", "1.3", "2")
# The position of the title that loses its heading.
LOST = 2


def _page(path: Path, body: str) -> Path:
    """Write a single HTML page whose main element holds a body."""
    path.write_text(
        '<!DOCTYPE html><html lang="en"><head><title>Manual</title></head>'
        f"<body><main>{body}</main></body></html>",
        encoding="utf-8",
    )
    return path


def _text(title: str) -> str:
    """Return the paragraph of a section."""
    return f"<p>The {title.lower()} step of the tool is described in full.</p>"


def _numbered_sections(lost: int | None, *, numbered: bool = True) -> str:
    """Return numbered sections, one title in bold text if ``lost`` is set."""
    parts = []
    for index, (number, title) in enumerate(zip(NUMBERS, TITLES, strict=True)):
        text = f"{number} {title}" if numbered else title
        heading = f"<p><strong>{text}</strong></p>" if index == lost else None
        parts += [heading or f"<h2>{text}</h2>", _text(title)]
    return "".join(parts)


def _labelled_chapters(lost: int | None) -> str:
    """Return labelled chapters, one title in bold text if ``lost`` is set."""
    parts = []
    for index, title in enumerate(TITLES[:3]):
        heading = f"<p><b>{title}</b></p>" if index == lost else None
        parts += [
            f"<p>Chapter {index + 1}</p>",
            heading or f"<h1>{title}</h1>",
            _text(title),
        ]
    return "".join(parts)


# Pages whose titles carry numbers, which the HTML of a word processor
# keeps in the text of a title that is no heading.
NUMBERED_PAGES: dict[str, Callable[[int | None], str]] = {
    "number": _numbered_sections,
    "label": _labelled_chapters,
}


@pytest.mark.parametrize("pages", list(NUMBERED_PAGES))
def test_numbered_title_written_as_text_only_loses_its_kind(
    tmp_path: Path,
    pages: str,
) -> None:
    """The number of a title that lost its heading is no text change."""
    body = NUMBERED_PAGES[pages]
    kept = _page(tmp_path / "kept.html", body(None))
    lost = _page(tmp_path / "lost.html", body(LOST))
    forward = run_compare(kept, lost, Config()).findings
    swapped = run_compare(lost, kept, Config()).findings
    assert [(f.rule, f.severity) for f in forward if f.severity >= MI] == [
        ("structure.heading_lost", MA),
    ]
    assert [(f.rule, f.severity) for f in swapped if f.severity >= MI] == []


def test_numbered_title_written_as_text_against_plain_titles(
    tmp_path: Path,
) -> None:
    """A numbered build that wrote a title as text only loses its kind."""
    titles = _numbered_sections(None, numbered=False)
    plain = _page(tmp_path / "plain.html", titles)
    lost = _page(tmp_path / "lost.html", _numbered_sections(LOST))
    forward = run_compare(plain, lost, Config()).findings
    swapped = run_compare(lost, plain, Config()).findings
    assert [(f.rule, f.severity) for f in forward if f.severity >= MI] == [
        ("structure.heading_lost", MA),
    ]
    assert [(f.rule, f.severity) for f in swapped if f.severity >= MI] == []


def _left_out(*chapters: str) -> Config:
    """Return a configuration that leaves chapters out of Sphinx and MkDocs."""
    patterns = [p for c in chapters for p in (f"{c}.html", f"{c}/**")]
    return Config(html=HtmlConfig(exclude=(*DEFAULT_HTML_EXCLUDE, *patterns)))


@pytest.mark.parametrize("order", [(0, 1), (1, 0)], ids=["forward", "swapped"])
def test_links_to_left_out_pages_follow_their_names(
    tmp_path: Path,
    order: tuple[int, int],
) -> None:
    """A link to a page left out leads to it in both URL layouts."""
    sites = (
        build_sphinx_like(SAMPLE_EN, tmp_path / "sphinx", split="chapters"),
        build_mkdocs_like(SAMPLE_EN, tmp_path / "mkdocs"),
    )
    cfg = _left_out("reference")
    result = run_compare(sites[order[0]], sites[order[1]], cfg)
    pages = {
        link.page
        for doc in (result.ref, result.new)
        for block in doc.blocks
        for link in block.links
    }
    assert {"reference.html", "reference/index.html"} <= pages
    assert result.stats.matched > 0
    assert [
        (f.rule, f.severity) for f in result.findings if f.severity >= MI
    ] == []


def test_link_moved_between_left_out_pages_is_reported(tmp_path: Path) -> None:
    """A link that leads to another page left out is a changed target."""
    ref_spec = replace_text(SAMPLE_EN, "(#command-line)", "(../reference/)")
    new_spec = replace_text(
        SAMPLE_EN,
        "(#command-line)",
        "(../getting-started/)",
    )
    ref = build_mkdocs_like(ref_spec, tmp_path / "ref")
    new = build_mkdocs_like(new_spec, tmp_path / "new")
    cfg = _left_out("reference", "getting-started")
    findings = run_compare(ref, new, cfg).findings
    assert [(f.rule, f.severity) for f in findings if f.severity >= MI] == [
        ("link.target_changed", MA),
    ]


def test_moved_section_is_info_only(tmp_path: Path) -> None:
    """A section moved elsewhere is only reported as moved."""
    moved = move_section(SAMPLE_EN, "limits", after="installation")
    findings = _compare(tmp_path, SAMPLE_EN, moved)
    assert findings
    assert {f.rule for f in findings} == {"block.moved"}


type SiteBuilder = Callable[[DocSpec, Path], Path]

# Each entry builds one documentation as two sites whose markup differs.
MARKUPS: dict[str, tuple[SiteBuilder, SiteBuilder]] = {
    "sphinx-chapters/pandoc": (
        partial(build_sphinx_like, split="chapters"),
        build_pandoc_like,
    ),
    "mkdocs/pandoc": (build_mkdocs_like, build_pandoc_like),
    "sphinx-numbered/mkdocs": (
        partial(build_sphinx_like, numbered=True),
        build_mkdocs_like,
    ),
    "pandoc-numbered/sphinx-chapters": (
        partial(build_pandoc_like, numbered=True),
        partial(build_sphinx_like, split="chapters"),
    ),
    "docusaurus/mkdocs": (build_docusaurus_like, build_mkdocs_like),
    "docusaurus-base/sphinx-chapters": (
        partial(build_docusaurus_like, base="/example/"),
        partial(build_sphinx_like, split="chapters"),
    ),
    "docusaurus-base/pandoc": (
        partial(build_docusaurus_like, base="/example/"),
        build_pandoc_like,
    ),
}


@pytest.mark.parametrize("order", [(0, 1), (1, 0)], ids=["forward", "swapped"])
@pytest.mark.parametrize("spec", [SAMPLE_EN, SAMPLE_FR], ids=["en", "fr"])
@pytest.mark.parametrize("markup", list(MARKUPS))
def test_markup_only_differences_produce_no_finding(
    tmp_path: Path,
    markup: str,
    spec: DocSpec,
    order: tuple[int, int],
) -> None:
    """One documentation in two markups has no finding, both ways."""
    builders = MARKUPS[markup]
    sites = (
        builders[0](spec, tmp_path / "first"),
        builders[1](spec, tmp_path / "second"),
    )
    result = run_compare(sites[order[0]], sites[order[1]], Config())
    assert result.stats.matched > 0
    assert [
        (f.rule, f.severity) for f in result.findings if f.severity >= MI
    ] == []


def test_accents_lost_everywhere_are_one_finding(tmp_path: Path) -> None:
    """A generator that loses every acute accent gives one group."""
    plain = map_texts(SAMPLE_FR, lambda text: text.replace("é", "e"))
    findings = _compare(tmp_path, SAMPLE_FR, plain)
    (group,) = [f for f in findings if f.severity >= MI]
    assert (group.rule, group.severity) == ("text.accent_lost", MA)
    assert len(group.occurrences) >= GROUP_MIN_OCCURRENCES
