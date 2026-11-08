# Copyright 2026 Christophe Le Douarec
"""End to end runs shared by the command line and the tests."""

from typing import TYPE_CHECKING

from doc_checker.compare.engine import compare_documents
from doc_checker.errors import ConfigError, InputError
from doc_checker.extract import detect_format, load_document
from doc_checker.report.context import Thumbnails
from doc_checker.report.html import render_html
from doc_checker.report.markdown import render_markdown

if TYPE_CHECKING:
    from pathlib import Path

    from doc_checker.compare.engine import CompareResult
    from doc_checker.config import Config


def run_compare(ref_path: Path, new_path: Path, cfg: Config) -> CompareResult:
    """Load two documentation builds and compare them.

    Args:
        ref_path: Reference build.
        new_path: New build.
        cfg: Run configuration.

    Returns:
        The comparison result.

    Raises:
        InputError: If the builds have different formats or cannot be read.
    """
    ref_format = detect_format(ref_path)
    new_format = detect_format(new_path)
    if ref_format != new_format:
        msg = (
            f"REF is {ref_format} but NEW is {new_format}: "
            "both must have the same format"
        )
        raise InputError(msg)
    ref = load_document(ref_path, cfg)
    new = load_document(new_path, cfg)
    return compare_documents(ref, new, cfg)


def _render_reports(result: CompareResult, cfg: Config) -> list[Path]:
    """Write the reports of the formats of a run, with shared thumbnails.

    Args:
        result: Comparison result.
        cfg: Run configuration, with the output directory and formats.

    Returns:
        The written report files.
    """
    cfg.output.mkdir(parents=True, exist_ok=True)
    written = []
    with Thumbnails(result) as thumbnails:
        if "html" in cfg.formats:
            html = cfg.output / "report.html"
            written.append(render_html(result, html, thumbnails))
        if "md" in cfg.formats:
            markdown = cfg.output / "report.md"
            written.append(render_markdown(result, markdown, thumbnails))
    return written


def write_reports(result: CompareResult, cfg: Config) -> list[Path]:
    """Write the reports of a comparison in the output directory.

    Args:
        result: Comparison result.
        cfg: Run configuration, with the output directory and formats.

    Returns:
        The written report files: ``report.html``, then ``report.md``,
        for the formats of the configuration. Both show the same
        thumbnails, made once.

    Raises:
        ConfigError: If the reports cannot be written.
    """
    try:
        written = _render_reports(result, cfg)
    except OSError as exc:
        msg = f"cannot write reports to {cfg.output}: {exc.strerror}"
        raise ConfigError(msg) from None
    return written
