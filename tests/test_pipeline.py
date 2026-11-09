# Copyright 2026 Christophe Le Douarec
"""Tests for the end to end runs of the audit."""

from dataclasses import replace
from typing import TYPE_CHECKING

from doc_checker.config import Config
from doc_checker.model.findings import Severity
from doc_checker.pipeline import run_audit
from tests.builders import FIGURES_EN
from tests.pdf_builders import build_reportlab_pdf

if TYPE_CHECKING:
    from pathlib import Path


IMAGES_ONLY = Config(checks=("images",))


def test_print_profile_raises_the_resolution_needed(tmp_path: Path) -> None:
    """Images at 102 dpi are minor for the screen and major for print."""
    pdf = build_reportlab_pdf(FIGURES_EN, tmp_path / "doc.pdf")
    for cfg, severity in (
        (IMAGES_ONLY, Severity.MINOR),
        (replace(IMAGES_ONLY, profile="print"), Severity.MAJOR),
    ):
        findings = run_audit(pdf, cfg).findings
        low = {f.severity for f in findings if f.rule == "image.low_dpi"}
        assert low == {severity}
