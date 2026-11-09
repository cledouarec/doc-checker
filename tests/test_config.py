# Copyright 2026 Christophe Le Douarec
"""Tests for configuration loading and command line overrides."""

import re
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from doc_checker.config import (
    CliOverrides,
    Config,
    PdfConfig,
    SpellingConfig,
    apply_overrides,
    load_config,
    rule_disabled,
)
from doc_checker.errors import ConfigError
from doc_checker.model.findings import Category, Severity
from doc_checker.rules import get_rule

if TYPE_CHECKING:
    from pathlib import Path


def test_defaults_without_files(tmp_path: Path) -> None:
    """Without any configuration file, the defaults apply."""
    cfg = load_config(None, cwd=tmp_path)
    assert cfg.lang == "auto"
    assert cfg.formats == ("html", "md")
    assert cfg.min_severity is Severity.INFO
    assert cfg.output == tmp_path / "doc-checker-report"
    assert (cfg.compare.match, cfg.compare.moved_match) == (0.6, 0.8)
    assert "**/_static/**" in cfg.html.exclude


def test_formats_may_be_html_only(tmp_path: Path) -> None:
    """The HTML report alone may be asked for."""
    (tmp_path / "doc-checker.toml").write_text(
        '[general]\nformats = ["html"]\n',
        encoding="utf-8",
    )
    assert load_config(None, cwd=tmp_path).formats == ("html",)


def test_project_file_is_loaded(tmp_path: Path) -> None:
    """A doc-checker.toml file in the working directory is used."""
    (tmp_path / "doc-checker.toml").write_text(
        '[general]\nlang = "fr"\n[html]\nexclude = ["drafts/**"]\n'
        '[severity]\n"block.added" = "info"\n',
        encoding="utf-8",
    )
    cfg = load_config(None, cwd=tmp_path)
    assert cfg.lang == "fr"
    assert cfg.html.exclude == ("drafts/**",)
    assert cfg.severity == {"block.added": Severity.INFO}


def test_pyproject_section_is_used(tmp_path: Path) -> None:
    """The [tool.doc-checker] table of pyproject.toml is used."""
    (tmp_path / "pyproject.toml").write_text(
        '[tool.doc-checker.general]\nmin_severity = "major"\n',
        encoding="utf-8",
    )
    assert load_config(None, cwd=tmp_path).min_severity is Severity.MAJOR


def test_explicit_config_wins(tmp_path: Path) -> None:
    """An explicit configuration file wins over the project file."""
    (tmp_path / "doc-checker.toml").write_text(
        '[general]\nlang = "fr"\n',
        encoding="utf-8",
    )
    other = tmp_path / "other.toml"
    other.write_text('[general]\nlang = "en"\n', encoding="utf-8")
    assert load_config(other, cwd=tmp_path).lang == "en"


def test_output_is_relative_to_config_file(tmp_path: Path) -> None:
    """A relative output directory is resolved from the config file."""
    conf = tmp_path / "sub" / "c.toml"
    conf.parent.mkdir()
    conf.write_text('[general]\noutput = "out"\n', encoding="utf-8")
    assert load_config(conf, cwd=tmp_path).output == tmp_path / "sub" / "out"


def test_pdf_defaults(tmp_path: Path) -> None:
    """Without a [pdf] section, the PDF defaults apply."""
    assert load_config(None, cwd=tmp_path).pdf == PdfConfig(
        header_band=0.08,
        footer_band=0.08,
        repeat_share=0.3,
        compare_toc=False,
    )


def test_pdf_defaults_when_section_is_absent(tmp_path: Path) -> None:
    """A configuration file without a [pdf] section keeps the defaults."""
    (tmp_path / "doc-checker.toml").write_text(
        '[general]\nlang = "fr"\n',
        encoding="utf-8",
    )
    assert load_config(None, cwd=tmp_path).pdf == PdfConfig()


def test_pdf_section_is_loaded(tmp_path: Path) -> None:
    """The keys of the [pdf] section replace the PDF defaults."""
    (tmp_path / "doc-checker.toml").write_text(
        "[pdf]\nheader_band = 0.1\nfooter_band = 0.05\nrepeat_share = 0.5\n"
        "compare_toc = true\n",
        encoding="utf-8",
    )
    assert load_config(None, cwd=tmp_path).pdf == PdfConfig(
        header_band=0.1,
        footer_band=0.05,
        repeat_share=0.5,
        compare_toc=True,
    )


def test_pdf_ranges_include_their_upper_bound(tmp_path: Path) -> None:
    """A band can cover half of the page and a share all of the pages."""
    (tmp_path / "doc-checker.toml").write_text(
        "[pdf]\nheader_band = 0.5\nfooter_band = 0.5\nrepeat_share = 1\n",
        encoding="utf-8",
    )
    assert load_config(None, cwd=tmp_path).pdf == PdfConfig(
        header_band=0.5,
        footer_band=0.5,
        repeat_share=1.0,
    )


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("[pdf]\nheader_band = 0.7\n", "pdf.header_band: must be in ]0, 0.5]"),
        ("[pdf]\nfooter_band = 0\n", "pdf.footer_band: must be in ]0, 0.5]"),
        ("[pdf]\nrepeat_share = 1.5\n", "pdf.repeat_share: must be in ]0, 1]"),
    ],
)
def test_pdf_range_errors_name_the_valid_range(
    tmp_path: Path,
    content: str,
    message: str,
) -> None:
    """An out-of-range PDF number is rejected with the range it must be in."""
    conf = tmp_path / "c.toml"
    conf.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigError, match=re.escape(message)):
        load_config(conf, cwd=tmp_path)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("[general]\nlangue = 'fr'\n", "general.langue"),
        ("[general]\nlang = 'de'\n", "general.lang"),
        ("[general]\nformats = ['pdf']\n", "general.formats"),
        ("[severity]\n'nope.rule' = 'info'\n", "nope.rule"),
        ("[ignore]\npatterns = ['(']\n", "ignore.patterns"),
        ("[thresholds.compare]\nmatch = 1.5\n", "thresholds.compare.match"),
        ("[html]\npage_order = 'random'\n", "html.page_order"),
        ("[pdf]\nheader_band = 0.7\n", "pdf.header_band"),
        ("[pdf]\nrepeat_share = 0\n", "pdf.repeat_share"),
        ("[pdf]\ncompare_toc = 'yes'\n", "pdf.compare_toc"),
        ("[pdf]\nmargin = 1\n", "pdf.margin"),
        ("not toml = = \n", "invalid TOML"),
    ],
)
def test_invalid_config_raises(
    tmp_path: Path,
    content: str,
    message: str,
) -> None:
    """Invalid configurations are rejected with the offending key."""
    conf = tmp_path / "c.toml"
    conf.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigError, match=re.escape(message)):
        load_config(conf, cwd=tmp_path)


def test_apply_overrides(tmp_path: Path) -> None:
    """Command line options override or extend the configuration."""
    overrides = CliOverrides(
        lang="en",
        min_severity=Severity.MAJOR,
        exclude=("old/**",),
        disable=("text.case_*",),
        ignore_moves=True,
    )
    cfg = apply_overrides(load_config(None, cwd=tmp_path), overrides)
    assert cfg.lang == "en"
    assert cfg.min_severity is Severity.MAJOR
    assert cfg.html.exclude[-1] == "old/**"
    assert "**/_static/**" in cfg.html.exclude
    assert rule_disabled(cfg, "text.case_changed")
    assert rule_disabled(cfg, "block.moved")
    assert not rule_disabled(cfg, "block.missing")


def test_unknown_disable_pattern_is_rejected(tmp_path: Path) -> None:
    """A disable pattern that matches no rule is rejected."""
    cfg = load_config(None, cwd=tmp_path)
    with pytest.raises(ConfigError, match="no rule matches"):
        apply_overrides(cfg, CliOverrides(disable=("nope.*",)))


IMAGE_RULES = {
    "image.missing": Severity.CRITICAL,
    "image.added": Severity.MINOR,
    "image.replaced": Severity.MAJOR,
    "image.degraded": Severity.MAJOR,
    "image.reencoded": Severity.INFO,
    "image.rasterized": Severity.MINOR,
    "image.vectorized": Severity.INFO,
    "image.alt_lost": Severity.MINOR,
}


def test_image_rules_have_the_severities_of_the_spec() -> None:
    """The eight image rules are registered in the image category."""
    assert {r: get_rule(r).severity for r in IMAGE_RULES} == IMAGE_RULES
    assert {get_rule(r).category for r in IMAGE_RULES} == {Category.IMAGE}


def test_images_can_be_turned_off(tmp_path: Path) -> None:
    """Images are on by default, off by the file or by the option."""
    assert load_config(None, cwd=tmp_path).images is True
    off = apply_overrides(Config(), CliOverrides(no_images=True))
    assert off.images is False
    (tmp_path / "doc-checker.toml").write_text(
        "[general]\nimages = false\n",
        encoding="utf-8",
    )
    assert load_config(None, cwd=tmp_path).images is False


AUDIT_RULES = {
    "image.low_dpi": Severity.MINOR,
    "image.upscaled": Severity.MINOR,
    "image.blurry": Severity.MINOR,
    "image.jpeg_quality": Severity.MINOR,
    "image.blocky": Severity.MINOR,
    "image.pixelated": Severity.MINOR,
    "image.heavy": Severity.MINOR,
    "image.photo_png": Severity.INFO,
    "image.broken": Severity.MAJOR,
    "structure.heading_skipped": Severity.MINOR,
    "structure.empty_section": Severity.MINOR,
    "structure.duplicate_heading": Severity.MINOR,
    "structure.unresolved_reference": Severity.MAJOR,
    "structure.figure_caption": Severity.INFO,
    "structure.table_caption": Severity.INFO,
    "structure.orphan_page": Severity.MINOR,
    "link.page_missing": Severity.MAJOR,
    "link.anchor_missing": Severity.MAJOR,
    "link.file_missing": Severity.MAJOR,
    "link.destination_invalid": Severity.MAJOR,
    "a11y.alt_missing": Severity.MAJOR,
    "a11y.alt_filename": Severity.MINOR,
    "a11y.svg_title": Severity.MINOR,
    "a11y.lang_missing": Severity.MAJOR,
    "a11y.lang_mismatch": Severity.MINOR,
    "a11y.table_headers": Severity.MINOR,
    "a11y.pdf_untagged": Severity.MAJOR,
    "a11y.title_missing": Severity.MINOR,
    "a11y.unmapped_chars": Severity.MINOR,
    "a11y.scanned_page": Severity.MAJOR,
    "a11y.link_empty": Severity.MAJOR,
    "a11y.link_text": Severity.MINOR,
}
AUDIT_FAMILIES = {
    "image": Category.IMAGE,
    "structure": Category.STRUCTURE,
    "link": Category.LINK,
    "a11y": Category.ACCESSIBILITY,
}


def test_audit_rules_have_their_severities_and_categories() -> None:
    """The 32 audit rules are registered, and none is critical."""
    assert {r: get_rule(r).severity for r in AUDIT_RULES} == AUDIT_RULES
    for rule in AUDIT_RULES:
        family = rule.split(".", maxsplit=1)[0]
        assert get_rule(rule).category is AUDIT_FAMILIES[family]


def test_audit_settings_have_the_defaults_of_the_spec() -> None:
    """All families, the screen profile and the thresholds of the spec."""
    cfg = Config()
    assert cfg.checks == ("images", "structure", "a11y", "text")
    assert cfg.profile == "screen"
    assert cfg.audit.dpi("screen") == (96.0, 150.0)
    assert cfg.audit.dpi("print") == (150.0, 300.0)
    assert (cfg.audit.heavy_minor, cfg.audit.heavy_major) == (
        1_000_000,
        3_000_000,
    )
    assert (cfg.audit.long_sentence_minor, cfg.audit.long_sentence_major) == (
        40,
        60,
    )


def test_audit_settings_are_read_from_the_file(tmp_path: Path) -> None:
    """The families, the profile and the thresholds come from the file."""
    (tmp_path / "doc-checker.toml").write_text(
        '[general]\nchecks = ["a11y", "images", "a11y"]\n'
        'profile = "print"\n'
        "[thresholds.audit]\ndpi_minor = 200\nheavy_minor = 500000\n"
        "blur = 0.05\n",
        encoding="utf-8",
    )
    cfg = load_config(None, cwd=tmp_path)
    assert (cfg.checks, cfg.profile) == (("a11y", "images"), "print")
    audit = cfg.audit
    assert (audit.dpi_minor, audit.heavy_minor, audit.blur) == (
        200.0,
        500_000,
        0.05,
    )


@pytest.mark.parametrize(
    ("toml", "message"),
    [
        ('[general]\nchecks = ["grammar"]\n', "unknown check 'grammar'"),
        ("[general]\nchecks = []\n", "checks: expected at least one check"),
        (
            '[spelling]\nfiles = ["missing.txt"]\n',
            "missing.txt: cannot read file",
        ),
        (
            "[spelling]\nwords = 3\n",
            "spelling.words: expected a list of strings",
        ),
        ('[spelling]\nlang = "fr"\n', "spelling.lang: unknown key"),
        (
            "[thresholds.audit]\nlong_sentence_minor = 70\n",
            (
                "thresholds.audit.long_sentence_minor: must be below "
                "long_sentence_major"
            ),
        ),
        (
            '[general]\nprofile = "paper"\n',
            "profile: 'paper' is not one of screen, print",
        ),
        (
            "[thresholds.audit]\ndpi_major = 200\n",
            "thresholds.audit.dpi_major: must be below dpi_minor",
        ),
        (
            "[thresholds.audit]\nblur = 0\n",
            "thresholds.audit.blur: must be in ]0, 1]",
        ),
        (
            "[thresholds.audit]\nheavy_major = true\n",
            "thresholds.audit.heavy_major: expected an integer",
        ),
        (
            "[thresholds.audit]\njpeg_quality_minor = 120\n",
            "thresholds.audit.jpeg_quality_minor: must be in [1, 100]",
        ),
        (
            "[thresholds.audit]\nupscale = 0.9\n",
            "thresholds.audit.upscale: must be above 1",
        ),
        (
            "[thresholds.audit]\nsharpness = 1\n",
            "thresholds.audit.sharpness: unknown key",
        ),
    ],
)
def test_bad_audit_settings_are_config_errors(
    tmp_path: Path,
    toml: str,
    message: str,
) -> None:
    """Each wrong audit setting is an error that names its key."""
    (tmp_path / "doc-checker.toml").write_text(toml, encoding="utf-8")
    with pytest.raises(ConfigError, match=re.escape(message)):
        load_config(None, cwd=tmp_path)


def test_command_line_overrides_checks_and_profile() -> None:
    """The options replace the families and the profile of the file."""
    overrides = CliOverrides(checks=("images",), profile="print")
    cfg = apply_overrides(Config(), overrides)
    assert (cfg.checks, cfg.profile) == (("images",), "print")


@pytest.mark.parametrize(
    ("toml", "message"),
    [
        (
            "[thresholds.audit]\nupscale = nan\n",
            "thresholds.audit.upscale: expected a finite number",
        ),
        (
            "[thresholds.audit]\ndpi_minor = inf\n",
            "thresholds.audit.dpi_minor: expected a finite number",
        ),
        (
            "[thresholds.audit]\nblockiness = -inf\n",
            "thresholds.audit.blockiness: expected a finite number",
        ),
        (
            "[thresholds.audit]\nblur = nan\n",
            "thresholds.audit.blur: must be in ]0, 1]",
        ),
        (
            "[thresholds.audit]\ndpi_major = 0\n",
            "thresholds.audit.dpi_major: must be above 0",
        ),
        (
            "[thresholds.audit]\nprint_dpi_minor = -5\n",
            "thresholds.audit.print_dpi_minor: must be above 0",
        ),
        (
            "[thresholds.audit]\nheavy_minor = 0\n",
            "thresholds.audit.heavy_minor: must be above 0",
        ),
        (
            "[thresholds.audit]\ndpi_major = true\n",
            "thresholds.audit.dpi_major: expected a number",
        ),
        (
            "[thresholds.audit]\nheavy_minor = false\n",
            "thresholds.audit.heavy_minor: expected an integer",
        ),
        (
            "[thresholds.audit]\nheavy_minor = 1.5\n",
            "thresholds.audit.heavy_minor: expected an integer",
        ),
        (
            "[thresholds.audit]\ndpi_major = 150\n",
            "thresholds.audit.dpi_major: must be below dpi_minor",
        ),
        (
            "[thresholds.audit]\nprint_dpi_major = 300\n",
            "thresholds.audit.print_dpi_major: must be below print_dpi_minor",
        ),
        (
            "[thresholds.audit]\nheavy_minor = 3000000\n",
            "thresholds.audit.heavy_minor: must be below heavy_major",
        ),
        (
            "[thresholds.audit]\njpeg_quality_major = 50\n",
            "jpeg_quality_major: must be below jpeg_quality_minor",
        ),
    ],
)
def test_audit_thresholds_reject_degenerate_values(
    tmp_path: Path,
    toml: str,
    message: str,
) -> None:
    """Non-finite, null, mistyped and tied thresholds are errors."""
    (tmp_path / "doc-checker.toml").write_text(toml, encoding="utf-8")
    with pytest.raises(ConfigError, match=re.escape(message)):
        load_config(None, cwd=tmp_path)


def test_spelling_words_come_from_the_file_and_its_word_files(
    tmp_path: Path,
) -> None:
    """Words of the table, then of each listed file, relative to it."""
    (tmp_path / "jargon.txt").write_text("kubectl\n", encoding="utf-8")
    (tmp_path / "doc-checker.toml").write_text(
        '[spelling]\nwords = ["Kubernetes"]\nfiles = ["jargon.txt"]\n'
        "[thresholds.audit]\nlong_sentence_minor = 30\n",
        encoding="utf-8",
    )
    cfg = load_config(None, cwd=tmp_path)
    assert cfg.spelling.words == ("Kubernetes", "kubectl")
    assert cfg.audit.long_sentence_minor == 30


def test_command_line_word_files_extend_the_words(tmp_path: Path) -> None:
    """--words adds the words of its files after the configured ones."""
    path = tmp_path / "more.txt"
    path.write_text("frobnicate\n", encoding="utf-8")
    cfg = replace(Config(), spelling=SpellingConfig(("Kubernetes",)))
    cfg = apply_overrides(cfg, CliOverrides(words=(path,)))
    assert cfg.spelling.words == ("Kubernetes", "frobnicate")


def test_word_files_resolve_against_the_config_folder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A word file is read next to the config file, not in the cwd."""
    folder = tmp_path / "conf"
    folder.mkdir()
    (folder / "words.txt").write_text("kubectl\n", encoding="utf-8")
    config = folder / "doc-checker.toml"
    config.write_text('[spelling]\nfiles = ["words.txt"]\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    cfg = load_config(config, cwd=tmp_path)
    assert cfg.spelling.words == ("kubectl",)


def test_word_files_may_have_an_absolute_path(tmp_path: Path) -> None:
    """An absolute path in files is used as it is."""
    words = tmp_path / "words.txt"
    words.write_text("kubectl\n", encoding="utf-8")
    other = tmp_path / "other"
    other.mkdir()
    config = other / "doc-checker.toml"
    config.write_text(
        f"[spelling]\nfiles = ['{words}']\n",
        encoding="utf-8",
    )
    cfg = load_config(config, cwd=tmp_path)
    assert cfg.spelling.words == ("kubectl",)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"\xff\xfe\x00bad\n", "cannot read file: not UTF-8 text"),
        (b"two words\n", ":1: expected one word per line"),
    ],
)
def test_bad_word_files_are_config_errors(
    tmp_path: Path,
    content: bytes,
    message: str,
) -> None:
    """A word file that is not UTF-8 or has a blank in a word is refused."""
    (tmp_path / "words.txt").write_bytes(content)
    (tmp_path / "doc-checker.toml").write_text(
        '[spelling]\nfiles = ["words.txt"]\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match=re.escape(message)):
        load_config(None, cwd=tmp_path)


def test_a_missing_command_line_word_file_is_a_config_error(
    tmp_path: Path,
) -> None:
    """--words with an unreadable file is a configuration error."""
    overrides = CliOverrides(words=(tmp_path / "nope.txt",))
    with pytest.raises(ConfigError, match=r"nope\.txt: cannot read file"):
        apply_overrides(Config(), overrides)


TEXT_RULES = {
    "text.typo_fixed": Severity.INFO,
    "text.unknown_word": Severity.MINOR,
    "text.unknown_name": Severity.INFO,
    "text.jargon": Severity.INFO,
    "text.space_missing": Severity.MINOR,
    "text.space_breakable": Severity.INFO,
    "text.space_before_punctuation": Severity.MINOR,
    "text.quotes": Severity.INFO,
    "text.ordinal": Severity.MINOR,
    "text.unit_space": Severity.MINOR,
    "text.double_punctuation": Severity.MINOR,
    "text.repeated_word": Severity.MINOR,
    "text.long_sentence": Severity.MINOR,
    "text.block_language": Severity.INFO,
}


def test_text_rules_have_their_severities_and_category() -> None:
    """The text rules of the audit and the typo fix are registered."""
    assert {r: get_rule(r).severity for r in TEXT_RULES} == TEXT_RULES
    assert {get_rule(r).category for r in TEXT_RULES} == {Category.TEXT}
