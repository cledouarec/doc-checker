# Copyright 2026 Christophe Le Douarec
"""End to end runs shared by the command line and the tests."""

import errno
import os
from typing import TYPE_CHECKING

from doc_checker.audit.engine import audit_document
from doc_checker.compare.engine import compare_documents
from doc_checker.errors import ConfigError, InputError
from doc_checker.extract import detect_format, load_document
from doc_checker.report.context import Thumbnails
from doc_checker.report.html import render_html
from doc_checker.report.markdown import render_markdown

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from doc_checker.audit.engine import AuditResult
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


def run_audit(path: Path, cfg: Config) -> AuditResult:
    """Load a documentation build with its audit facts and audit it.

    Args:
        path: The build, a site directory, an HTML file or a PDF.
        cfg: Run configuration.

    Returns:
        The audit result.
    """
    return audit_document(load_document(path, cfg, audit=True), cfg)


def _render_reports(
    result: CompareResult | AuditResult,
    cfg: Config,
) -> list[Path]:
    """Write the reports of the formats of a run, with shared thumbnails.

    Args:
        result: Comparison or audit result.
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


def write_reports(
    result: CompareResult | AuditResult,
    cfg: Config,
) -> list[Path]:
    """Write the reports of a comparison or an audit in the output directory.

    Args:
        result: Comparison or audit result.
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


def check_wordlist_path(path: Path, output: Path) -> None:
    """Tell, before a run, whether a word list can be written to a path.

    Its folder must exist, or be the output folder or a folder above it,
    which the run creates before it writes the list.

    Args:
        path: The file of the word list.
        output: The output folder of the run.

    Raises:
        ConfigError: If the path is a folder, or if its folder does not
            exist and the run does not create it.
    """
    folder = path.parent
    if path.is_dir():
        reason = os.strerror(errno.EISDIR)
    elif folder.is_dir():
        return
    elif folder.exists():
        reason = os.strerror(errno.ENOTDIR)
    elif output.resolve().is_relative_to(folder.resolve()):
        return
    else:
        reason = os.strerror(errno.ENOENT)
    msg = f"cannot write the word list to {path}: {reason}"
    raise ConfigError(msg)


def make_output_folder(cfg: Config) -> None:
    """Create the output folder of a run, and the folders above it.

    Args:
        cfg: Run configuration, with the output folder.

    Raises:
        ConfigError: If the folder cannot be created.
    """
    try:
        cfg.output.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        msg = f"cannot write reports to {cfg.output}: {exc.strerror}"
        raise ConfigError(msg) from None


def write_wordlist(words: Sequence[str], path: Path) -> Path:
    """Write a list of words, one per line, in UTF-8.

    Args:
        words: The words, as `AuditResult.words` lists them.
        path: The file to write; it is written even when there is no word.

    Returns:
        The file written.

    Raises:
        ConfigError: If the file cannot be written.
    """
    try:
        path.write_text(
            "".join(f"{word}\n" for word in words),
            encoding="utf-8",
            newline="\n",
        )
    except OSError as exc:
        msg = f"cannot write the word list to {path}: {exc.strerror}"
        raise ConfigError(msg) from None
    return path
