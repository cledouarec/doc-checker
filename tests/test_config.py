# Copyright 2026 Christophe Le Douarec
"""Tests for configuration loading and command line overrides."""

import re
from typing import TYPE_CHECKING

import pytest

from doc_checker.config import (
    CliOverrides,
    apply_overrides,
    load_config,
    rule_disabled,
)
from doc_checker.errors import ConfigError
from doc_checker.model.findings import Severity

if TYPE_CHECKING:
    from pathlib import Path


def test_defaults_without_files(tmp_path: Path) -> None:
    """Without any configuration file, the defaults apply."""
    cfg = load_config(None, cwd=tmp_path)
    assert cfg.lang == "auto"
    assert cfg.formats == ("md",)
    assert cfg.min_severity is Severity.INFO
    assert cfg.output == tmp_path / "doc-checker-report"
    assert (cfg.compare.match, cfg.compare.moved_match) == (0.6, 0.8)
    assert "**/_static/**" in cfg.html.exclude


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
