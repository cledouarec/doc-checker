# Copyright 2026 Christophe Le Douarec
"""End to end tests of the audit on generated documentations."""

import shutil
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from doc_checker.config import AuditThresholds, Config, SpellingConfig
from doc_checker.images.pixels import open_raster
from doc_checker.model.findings import Category, Finding, Severity
from doc_checker.pipeline import run_audit
from tests.builders import (
    FIGURES_EN,
    SAMPLE_EN,
    SAMPLE_FR,
    build_docusaurus_like,
    build_mkdocs_like,
    build_pandoc_like,
    build_sphinx_like,
    figure_defect,
    plant,
    replace_text,
    rule_counts,
    sentence,
)
from tests.image_builders import blurred, drawing, encode, photo, pixelated
from tests.pdf_builders import (
    build_fpdf2_pdf,
    build_reportlab_pdf,
    build_single_page_pdf,
    build_tagged_pdf,
)

if TYPE_CHECKING:
    from pathlib import Path

type Build = Callable[[Path], Path]

CONFIG = Config()

CLEAN_BUILDS: list[tuple[str, Build, dict[str, int]]] = [
    ("sphinx", lambda out: build_sphinx_like(FIGURES_EN, out), {}),
    (
        "sphinx-chapters",
        lambda out: build_sphinx_like(FIGURES_EN, out, split="chapters"),
        {},
    ),
    (
        "mkdocs",
        lambda out: build_mkdocs_like(FIGURES_EN, out),
        {"a11y.svg_title": 1},
    ),
    ("pandoc", lambda out: build_pandoc_like(FIGURES_EN, out), {}),
    ("docusaurus", lambda out: build_docusaurus_like(FIGURES_EN, out), {}),
    (
        "sphinx-fr",
        lambda out: build_sphinx_like(SAMPLE_FR, out, split="chapters"),
        {},
    ),
    (
        "reportlab",
        lambda out: build_reportlab_pdf(FIGURES_EN, out.with_suffix(".pdf")),
        {"a11y.pdf_untagged": 1, "image.low_dpi": 3},
    ),
    (
        "reportlab-fr",
        lambda out: build_reportlab_pdf(SAMPLE_FR, out.with_suffix(".pdf")),
        {"a11y.pdf_untagged": 1},
    ),
    (
        "fpdf2",
        lambda out: build_fpdf2_pdf(FIGURES_EN, out.with_suffix(".pdf")),
        {"a11y.alt_missing": 1},
    ),
    ("tagged", lambda out: build_tagged_pdf(out.with_suffix(".pdf")), {}),
    ("mkdocs-fr", lambda out: build_mkdocs_like(SAMPLE_FR, out), {}),
    (
        "fpdf2-fr",
        lambda out: build_fpdf2_pdf(SAMPLE_FR, out.with_suffix(".pdf")),
        {"a11y.alt_missing": 1},
    ),
]


@pytest.mark.parametrize(
    ("name", "build", "expected"),
    CLEAN_BUILDS,
    ids=[case[0] for case in CLEAN_BUILDS],
)
def test_clean_builds_give_only_their_documented_findings(
    tmp_path: Path,
    name: str,
    build: Build,
    expected: dict[str, int],
) -> None:
    """No finding of minor or more but the documented true ones, counted."""
    result = run_audit(build(tmp_path / name), CONFIG)
    minor = rule_counts(result.findings, minimum=Severity.MINOR)
    assert minor == Counter(expected)


HEADERLINK = (
    '<a class="headerlink" href="#first-steps" '
    'title="Link to this heading">\xb6</a>'
)
FIRST_STEPS = '<section id="first-steps">'
GUIDE = "getting-started.html"


def _before_first_steps(html: str) -> tuple[str, str, str]:
    return (GUIDE, FIRST_STEPS, html + FIRST_STEPS)


def _blocky() -> bytes:
    once = open_raster(encode(photo("street"), "JPEG", quality=8))
    return encode(once)


type Edit = tuple[str, str, str]
type Picture = Callable[[], bytes] | None

HTML_DEFECTS: list[tuple[str, tuple[Edit, ...], Picture, dict[str, int]]] = [
    (
        "heading_skipped",
        (
            (
                GUIDE,
                f"<h2>First steps{HEADERLINK}</h2>",
                f"<h4>First steps{HEADERLINK}</h4>",
            ),
        ),
        None,
        {"structure.heading_skipped": 1},
    ),
    (
        "empty_section",
        (_before_first_steps('<section id="e"><h2>Empty</h2></section>'),),
        None,
        {"structure.empty_section": 1},
    ),
    (
        "duplicate_heading",
        (
            _before_first_steps(
                '<section id="again"><h2>Installation</h2>'
                "<p>Install again after an update of the tool.</p></section>",
            ),
        ),
        None,
        {"structure.duplicate_heading": 1},
    ),
    (
        "unresolved_xref",
        (
            _before_first_steps(
                '<p>See <span class="xref std std-ref">missing-label</span>'
                " for the details.</p>",
            ),
        ),
        None,
        {"structure.unresolved_reference": 1},
    ),
    (
        "unresolved_mark",
        (_before_first_steps("<p>The limits are in Section ?? below.</p>"),),
        None,
        {"structure.unresolved_reference": 1},
    ),
    (
        "markdown_reference",
        (_before_first_steps("<p>See [the details][missing] for more.</p>"),),
        None,
        {"structure.unresolved_reference": 1},
    ),
    (
        "page_missing",
        (_before_first_steps('<p>See <a href="nothere.html">it</a>.</p>'),),
        None,
        {"link.page_missing": 1},
    ),
    (
        "anchor_missing",
        (
            _before_first_steps(
                '<p>See <a href="reference.html#nowhere">the limits</a>.</p>',
            ),
        ),
        None,
        {"link.anchor_missing": 1},
    ),
    (
        "file_missing",
        (_before_first_steps('<p>Get <a href="files/t.zip">it</a>.</p>'),),
        None,
        {"link.file_missing": 1},
    ),
    (
        "link_text",
        (
            _before_first_steps(
                '<p>For the options <a href="reference.html#limits">'
                "click here</a>.</p>",
            ),
        ),
        None,
        {"a11y.link_text": 1},
    ),
    (
        "link_empty",
        (_before_first_steps('<p>Back <a href="index.html"></a>.</p>'),),
        None,
        {"a11y.link_empty": 1},
    ),
    (
        "table_headers",
        (
            (
                "user-guide.html",
                (
                    '<th class="head"><p>Option</p></th>'
                    '<th class="head"><p>Default</p></th>'
                ),
                "<td><p>Option</p></td><td><p>Default</p></td>",
            ),
        ),
        None,
        {"a11y.table_headers": 1},
    ),
    (
        "alt_missing",
        ((GUIDE, 'alt="Installer steps" ', ""),),
        None,
        {"a11y.alt_missing": 1},
    ),
    (
        "alt_filename",
        ((GUIDE, 'alt="Installer steps"', 'alt="_images/setup.png"'),),
        None,
        {"a11y.alt_filename": 1},
    ),
    (
        "svg_title",
        (
            _before_first_steps(
                '<svg xmlns="http://www.w3.org/2000/svg" width="300" '
                'height="200"><rect width="300" height="200" '
                'fill="#4682b4"/></svg>',
            ),
        ),
        None,
        {"a11y.svg_title": 1, "structure.figure_caption": 1},
    ),
    (
        "lang_missing",
        (("reference.html", '<html lang="en">', "<html>"),),
        None,
        {"a11y.lang_missing": 1},
    ),
    (
        "title_missing",
        (
            (
                "reference.html",
                "<title>Reference - Example</title>",
                "<title></title>",
            ),
        ),
        None,
        {"a11y.title_missing": 1},
    ),
    (
        "figure_caption",
        (
            _before_first_steps(
                '<p><img alt="Setup again" src="_images/setup.png" /></p>',
            ),
        ),
        None,
        {"structure.figure_caption": 1},
    ),
    (
        "broken_image",
        (
            _before_first_steps(
                '<p><img alt="Gone" src="_images/gone.png" /></p>',
            ),
        ),
        None,
        {"image.broken": 1, "structure.figure_caption": 1},
    ),
    (
        "upscaled",
        (
            (
                GUIDE,
                'src="_images/setup.png" />',
                'src="_images/setup.png" width="800" height="500" />',
            ),
        ),
        None,
        {"image.upscaled": 1},
    ),
    (
        "blurry",
        (),
        lambda: encode(blurred(drawing("setup"), 2.0)),
        {"image.blurry": 1},
    ),
    (
        "pixelated",
        (),
        lambda: encode(pixelated(drawing("setup"))),
        {"image.pixelated": 1},
    ),
    ("blocky", (), _blocky, {"image.blocky": 1}),
    (
        "jpeg_quality",
        (),
        lambda: encode(photo("street"), "JPEG", quality=20),
        {"image.jpeg_quality": 1},
    ),
    (
        "photo_png",
        (),
        lambda: encode(photo("street")),
        {"image.photo_png": 1},
    ),
]


@pytest.mark.parametrize(
    ("name", "edits", "picture", "expected"),
    HTML_DEFECTS,
    ids=[case[0] for case in HTML_DEFECTS],
)
def test_each_planted_html_problem_gives_its_findings(
    tmp_path: Path,
    name: str,
    edits: tuple[Edit, ...],
    picture: Picture,
    expected: dict[str, int],
) -> None:
    """A planted problem adds exactly its findings to the clean build."""
    site = build_sphinx_like(FIGURES_EN, tmp_path / name, split="chapters")
    clean = rule_counts(run_audit(site, CONFIG).findings)
    for file, old, new in edits:
        plant(site, file, old, new)
    if picture is not None:
        (site / "_images" / "setup.png").write_bytes(picture())
    planted = rule_counts(run_audit(site, CONFIG).findings)
    assert planted - clean == Counter(expected)
    assert not clean - planted


TOCTREE_SECTION = (
    '<section id="user-guide">\n'
    '<h2>User guide<a class="headerlink" href="#user-guide" '
    'title="Link to this heading">\xb6</a></h2>\n'
    '<div class="toctree-wrapper compound">\n<ul>\n'
    '<li class="toctree-l1"><a class="reference internal" '
    'href="reference.html">Reference</a></li>\n'
    "</ul>\n</div>\n</section>\n"
)
PICTURE_SECTIONS = (
    '<section id="diagram">\n<h2>Diagram</h2>\n<figure>'
    '<img alt="The steps" src="_images/setup.png" /></figure>\n</section>\n'
    '<section id="screen">\n<h2>Screen</h2>\n'
    '<img alt="The screen" src="_images/setup.png" />\n</section>\n'
)


def test_a_section_that_holds_a_toctree_is_not_empty(tmp_path: Path) -> None:
    """A Sphinx section whose body is a toctree is no empty section."""
    site = build_sphinx_like(FIGURES_EN, tmp_path / "s", split="chapters")
    clean = rule_counts(run_audit(site, CONFIG).findings)
    plant(site, *_before_first_steps(TOCTREE_SECTION))
    planted = rule_counts(run_audit(site, CONFIG).findings)
    assert planted == clean


def test_picture_sections_are_not_empty_without_images(
    tmp_path: Path,
) -> None:
    """Under --no-images, a section of a figure or an image is not empty."""
    cfg = replace(CONFIG, images=False)
    site = build_sphinx_like(FIGURES_EN, tmp_path / "s", split="chapters")
    clean = rule_counts(run_audit(site, cfg).findings)
    plant(site, *_before_first_steps(PICTURE_SECTIONS))
    planted = rule_counts(run_audit(site, cfg).findings)
    assert planted == clean


def test_heavy_image_with_a_lower_threshold(tmp_path: Path) -> None:
    """A large photograph above the configured weight is heavy."""
    cfg = replace(CONFIG, audit=AuditThresholds(heavy_minor=500_000))
    site = build_sphinx_like(FIGURES_EN, tmp_path / "s", split="chapters")
    clean = rule_counts(run_audit(site, cfg).findings)
    big = encode(photo("heavy", (1600, 1200)), "JPEG", quality=95)
    (site / "_images" / "setup.png").write_bytes(big)
    planted = rule_counts(run_audit(site, cfg).findings)
    assert planted - clean == Counter({"image.heavy": 1})


def test_orphan_page(tmp_path: Path) -> None:
    """A page that no other page links to is an orphan."""
    site = build_sphinx_like(FIGURES_EN, tmp_path / "s", split="chapters")
    clean = rule_counts(run_audit(site, CONFIG).findings)
    shutil.copy(site / "reference.html", site / "old-notes.html")
    planted = rule_counts(run_audit(site, CONFIG).findings)
    assert planted - clean == Counter({"structure.orphan_page": 1})


def test_declared_language_of_the_wrong_text(tmp_path: Path) -> None:
    """A French site that declares English is reported once."""
    site = build_sphinx_like(SAMPLE_FR, tmp_path / "s", split="chapters")
    clean = rule_counts(run_audit(site, CONFIG).findings)
    for page in sorted(site.glob("*.html")):
        plant(site, page.name, '<html lang="fr">', '<html lang="en">')
    planted = rule_counts(run_audit(site, CONFIG).findings)
    assert planted - clean == Counter({"a11y.lang_mismatch": 1})


PDF_DEFECTS: list[tuple[str, dict[str, object], dict[str, int]]] = [
    ("lang", {"lang": None}, {"a11y.lang_missing": 1}),
    ("title", {"title": None}, {"a11y.title_missing": 1}),
    ("alt", {"figure_alt": None}, {"a11y.alt_missing": 1}),
    ("alt_file", {"figure_alt": "setup.png"}, {"a11y.alt_filename": 1}),
    ("headers", {"table_headers": False}, {"a11y.table_headers": 1}),
    (
        "skip",
        {"headings": ((1, "Guide"), (2, "Install"), (4, "Deep"))},
        {"structure.heading_skipped": 1},
    ),
    (
        "destination",
        {"links": (("the target", "name:missing"),)},
        {"link.destination_invalid": 1},
    ),
    (
        "file",
        {"links": (("the manual", "launch:missing.pdf"),)},
        {"link.file_missing": 1},
    ),
    (
        "click",
        {"links": (("click here", "page:2"),)},
        {"a11y.link_text": 1},
    ),
    ("untagged", {"tagged": False}, {"a11y.pdf_untagged": 1}),
]


@pytest.mark.parametrize(
    ("name", "options", "expected"),
    PDF_DEFECTS,
    ids=[case[0] for case in PDF_DEFECTS],
)
def test_each_planted_pdf_problem_gives_its_findings(
    tmp_path: Path,
    name: str,
    options: dict[str, object],
    expected: dict[str, int],
) -> None:
    """A planted problem adds exactly its findings to the tagged PDF."""
    clean = rule_counts(
        run_audit(build_tagged_pdf(tmp_path / "clean.pdf"), CONFIG).findings,
    )
    path = build_tagged_pdf(tmp_path / f"{name}.pdf", **options)
    planted = rule_counts(run_audit(path, CONFIG).findings)
    assert planted - clean == Counter(expected)
    assert not clean - planted


def test_question_marks_of_an_unresolved_reference_in_pdf(
    tmp_path: Path,
) -> None:
    """LaTeX-like question marks in a PDF paragraph are reported."""
    clean = build_reportlab_pdf(SAMPLE_EN, tmp_path / "a.pdf")
    spec = replace_text(
        SAMPLE_EN,
        "must be [retried](#first-steps) later.",
        "must be retried later, as Section ?? explains.",
    )
    planted = build_reportlab_pdf(spec, tmp_path / "b.pdf")
    before = rule_counts(run_audit(clean, CONFIG).findings)
    after = rule_counts(run_audit(planted, CONFIG).findings)
    assert after - before == Counter({"structure.unresolved_reference": 1})


def test_scanned_page_in_pdf(tmp_path: Path) -> None:
    """A scanned page is reported, and so is its low resolution."""
    clean = build_reportlab_pdf(SAMPLE_EN, tmp_path / "a.pdf")
    planted = build_reportlab_pdf(
        SAMPLE_EN,
        tmp_path / "b.pdf",
        scanned_page=True,
    )
    before = rule_counts(run_audit(clean, CONFIG).findings)
    after = rule_counts(run_audit(planted, CONFIG).findings)
    assert after - before == Counter(
        {"a11y.scanned_page": 1, "image.low_dpi": 1},
    )


def test_unmapped_characters_in_pdf(tmp_path: Path) -> None:
    """A glyph without Unicode mapping is reported once."""
    content = "BT /F1 12 Tf 20 100 Td (ABC) Tj ET"
    clean = build_single_page_pdf(tmp_path / "a.pdf", content)
    planted = build_single_page_pdf(
        tmp_path / "b.pdf",
        content,
        font=(
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
            "/Encoding << /Differences [66 /nosuchglyph] >> >>"
        ),
    )
    before = rule_counts(run_audit(clean, CONFIG).findings)
    after = rule_counts(run_audit(planted, CONFIG).findings)
    assert after - before == Counter({"a11y.unmapped_chars": 1})


def test_low_resolution_figure_in_pdf(tmp_path: Path) -> None:
    """A figure reduced to 0.4 falls under 96 dpi: one major finding."""
    clean = build_reportlab_pdf(FIGURES_EN, tmp_path / "a.pdf")
    spec = figure_defect(FIGURES_EN, "setup", "x0.4")
    planted = build_reportlab_pdf(spec, tmp_path / "b.pdf")
    major = Severity.MAJOR
    before = rule_counts(run_audit(clean, CONFIG).findings, minimum=major)
    after = rule_counts(run_audit(planted, CONFIG).findings, minimum=major)
    assert after - before == Counter({"image.low_dpi": 1})


TEXT_SPECS = {"en": FIGURES_EN, "fr": SAMPLE_FR}
MISSPELLED: Edit = (GUIDE, "installer downloads", "installer donwloads")
REPEATED: Edit = (GUIDE, "with the default", "with the the default")
START = "prise-en-main.html"
ATTENTION = "Attention\xa0: le mode"

TEXT_DEFECTS: list[tuple[str, str, tuple[Edit, ...], dict[str, int]]] = [
    ("en", "unknown_word", (MISSPELLED,), {"text.unknown_word": 1}),
    ("en", "repeated", (REPEATED,), {"text.repeated_word": 1}),
    (
        "en",
        "space_before",
        (("user-guide.html", "has an owner who", "has an owner , who"),),
        {"text.space_before_punctuation": 1},
    ),
    (
        "en",
        "long_minor",
        (_before_first_steps(f"<p>{sentence(45)}</p>"),),
        {"text.long_sentence": 1},
    ),
    (
        "en",
        "long_major",
        (_before_first_steps(f"<p>{sentence(65)}</p>"),),
        {"text.long_sentence": 1},
    ),
    (
        "en",
        "ordinal",
        (_before_first_steps("<p>Restart the service a 2th time.</p>"),),
        {"text.ordinal": 1},
    ),
    (
        "en",
        "unit",
        ((GUIDE, "2 GB of free", "2GB of free"),),
        {"text.unit_space": 1},
    ),
    (
        "en",
        "double",
        ((GUIDE, "before you start.", "before you start.."),),
        {"text.double_punctuation": 1},
    ),
    (
        "en",
        "block_language",
        (
            _before_first_steps(
                "<p>Le service redémarre tout seul après chaque mise à jour, "
                "et vous pouvez suivre son état depuis la page des réglages "
                "du projet.</p>",
            ),
        ),
        {"text.block_language": 1},
    ),
    (
        "en",
        "name",
        (_before_first_steps("<p>Zorblax exports the reports.</p>"),),
        {"text.unknown_name": 1},
    ),
    (
        "en",
        "jargon",
        (
            _before_first_steps(
                "<p>Run frobnicate first, then frobnicate again; "
                "frobnicate stops.</p>",
            ),
        ),
        {"text.jargon": 1},
    ),
    (
        "fr",
        "space_missing",
        ((START, ATTENTION, "Attention: le mode"),),
        {"text.space_missing": 1},
    ),
    (
        "fr",
        "breakable",
        ((START, ATTENTION, "Attention : le mode"),),
        {"text.space_breakable": 1},
    ),
    (
        "fr",
        "quotes",
        ((START, "\xab\xa0strict\xa0\xbb", '"strict"'),),
        {"text.quotes": 1},
    ),
    (
        "fr",
        "unknown_word",
        (("guide.html", "peut inviter", "peut invitter"),),
        {"text.unknown_word": 1},
    ),
    (
        "fr",
        "repeated",
        ((START, "redémarrer le service", "redémarrer le le service"),),
        {"text.repeated_word": 1},
    ),
    (
        "fr",
        "ordinal",
        ((START, "Cette procédure a été", "La 2ème procédure a été"),),
        {"text.ordinal": 1},
    ),
]


@pytest.mark.parametrize(
    ("lang", "name", "edits", "expected"),
    TEXT_DEFECTS,
    ids=[f"{case[0]}-{case[1]}" for case in TEXT_DEFECTS],
)
def test_each_planted_text_problem_gives_its_findings(
    tmp_path: Path,
    lang: str,
    name: str,
    edits: tuple[Edit, ...],
    expected: dict[str, int],
) -> None:
    """A planted text problem adds exactly its findings to the clean build."""
    spec = TEXT_SPECS[lang]
    site = build_sphinx_like(spec, tmp_path / name, split="chapters")
    clean = rule_counts(run_audit(site, CONFIG).findings)
    for file, old, new in edits:
        plant(site, file, old, new)
    planted = rule_counts(run_audit(site, CONFIG).findings)
    assert planted - clean == Counter(expected)
    assert not clean - planted


def test_long_sentences_take_the_severity_of_their_length(
    tmp_path: Path,
) -> None:
    """A sentence of 45 words is minor, one of 65 words is major."""
    site = build_sphinx_like(FIGURES_EN, tmp_path / "s", split="chapters")
    for count in (45, 65):
        plant(site, *_before_first_steps(f"<p>{sentence(count)}</p>"))
    findings = run_audit(site, CONFIG).findings
    long = [f.severity for f in findings if f.rule == "text.long_sentence"]
    assert sorted(long) == [Severity.MINOR, Severity.MAJOR]


def test_french_spacing_in_a_pdf(tmp_path: Path) -> None:
    """In a PDF, a colon after a space is no finding; a glued colon is one."""
    clean = build_reportlab_pdf(SAMPLE_FR, tmp_path / "a.pdf")
    spec = replace_text(SAMPLE_FR, "Attention : le mode", "Attention: le mode")
    planted = build_reportlab_pdf(spec, tmp_path / "b.pdf")
    before = rule_counts(run_audit(clean, CONFIG).findings)
    after = rule_counts(run_audit(planted, CONFIG).findings)
    assert not [rule for rule in before if rule.startswith("text.")]
    assert after - before == Counter({"text.space_missing": 1})
    assert not before - after


def test_misspelled_word_in_a_pdf(tmp_path: Path) -> None:
    """A misspelled word of a PDF paragraph is reported once."""
    clean = build_reportlab_pdf(SAMPLE_EN, tmp_path / "a.pdf")
    spec = replace_text(SAMPLE_EN, *MISSPELLED[1:])
    planted = build_reportlab_pdf(spec, tmp_path / "b.pdf")
    before = rule_counts(run_audit(clean, CONFIG).findings)
    after = rule_counts(run_audit(planted, CONFIG).findings)
    assert after - before == Counter({"text.unknown_word": 1})
    assert not before - after


def test_personal_words_in_the_configuration(tmp_path: Path) -> None:
    """A personal word of the configuration is no unknown word."""
    cfg = replace(CONFIG, spelling=SpellingConfig(("donwloads",)))
    site = build_sphinx_like(FIGURES_EN, tmp_path / "s", split="chapters")
    clean = rule_counts(run_audit(site, cfg).findings)
    plant(site, *MISSPELLED)
    planted = rule_counts(run_audit(site, cfg).findings)
    assert planted == clean


def test_text_audit_is_deterministic(tmp_path: Path) -> None:
    """Two audits of a site with text problems give the same findings."""
    site = build_sphinx_like(FIGURES_EN, tmp_path / "s", split="chapters")
    plant(site, *MISSPELLED)
    plant(site, *REPEATED)
    findings = run_audit(site, CONFIG).findings
    rules = {f.rule for f in findings}
    assert {"text.unknown_word", "text.repeated_word"} <= rules
    first = [f.id for f in findings]
    second = [f.id for f in run_audit(site, CONFIG).findings]
    assert first == second


def test_audit_is_deterministic(tmp_path: Path) -> None:
    """Two audits of one build give the same findings in the same order."""
    site = build_sphinx_like(FIGURES_EN, tmp_path / "s", split="chapters")
    plant(site, GUIDE, 'alt="Installer steps" ', "")
    first = [f.id for f in run_audit(site, CONFIG).findings]
    second = [f.id for f in run_audit(site, CONFIG).findings]
    assert first == second


def test_plant_replaces_a_text_found_once(tmp_path: Path) -> None:
    """Plant changes a text found once, and refuses one absent or repeated."""
    (tmp_path / "a.html").write_text("<p>one two one</p>", encoding="utf-8")
    plant(tmp_path, "a.html", "two", "three")
    text = (tmp_path / "a.html").read_text(encoding="utf-8")
    assert text == "<p>one three one</p>"
    for old in ("one", "four"):
        with pytest.raises(ValueError, match="expected once"):
            plant(tmp_path, "a.html", old, "x")


def _finding(rule: str, severity: Severity, *, size: int = 0) -> Finding:
    one = Finding(rule, severity, Category.LINK, rule, None, None)
    return replace(one, occurrences=(one,) * size)


def test_rule_counts_counts_a_group_once_from_a_severity() -> None:
    """A group counts as one finding, and lower severities are left out."""
    findings = [
        _finding("link.page_missing", Severity.MAJOR, size=5),
        _finding("link.page_missing", Severity.MAJOR),
        _finding("a11y.link_text", Severity.MINOR),
        _finding("structure.figure_caption", Severity.INFO),
    ]
    every = {
        "link.page_missing": 2,
        "a11y.link_text": 1,
        "structure.figure_caption": 1,
    }
    assert rule_counts(findings) == Counter(every)
    major = rule_counts(findings, minimum=Severity.MAJOR)
    assert major == Counter({"link.page_missing": 2})
