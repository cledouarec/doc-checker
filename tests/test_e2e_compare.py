# Copyright 2026 Christophe Le Douarec
"""End to end comparison of documentation builds."""

from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from doc_checker.config import Config
from doc_checker.model.findings import Finding, Severity
from doc_checker.pipeline import run_compare
from tests.builders import (
    SAMPLE_EN,
    SAMPLE_FR,
    DocSpec,
    add_paragraph,
    build_mkdocs_like,
    build_sphinx_like,
    drop_paragraph,
    drop_section,
    move_section,
    replace_text,
)

if TYPE_CHECKING:
    from pathlib import Path

BANNER = "Docs for version 2.0."
C = Severity.CRITICAL
MA = Severity.MAJOR
MI = Severity.MINOR
IN = Severity.INFO

type Transform = Callable[[DocSpec], DocSpec]


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


@pytest.mark.parametrize(
    ("spec", "transform", "rule", "severity"),
    [
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
    ],
)
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


def test_moved_section_is_info_only(tmp_path: Path) -> None:
    """A section moved elsewhere is only reported as moved."""
    moved = move_section(SAMPLE_EN, "limits", after="installation")
    findings = _compare(tmp_path, SAMPLE_EN, moved)
    assert findings
    assert {f.rule for f in findings} == {"block.moved"}
