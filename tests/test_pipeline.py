# Copyright 2026 Christophe Le Douarec
"""Tests for the end to end runs of the audit."""

import errno
import os
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from doc_checker.config import Config
from doc_checker.errors import ConfigError
from doc_checker.model.findings import Severity
from doc_checker.pipeline import (
    check_wordlist_path,
    make_output_folder,
    run_audit,
    write_wordlist,
)
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


def test_write_wordlist(tmp_path: Path) -> None:
    """One word per line, UTF-8, with newline ends; empty for no word."""
    path = tmp_path / "w.txt"
    assert write_wordlist(("caf\xe9", "Zorblax"), path) == path
    assert path.read_bytes() == "caf\xe9\nZorblax\n".encode()
    write_wordlist((), path)
    assert path.read_bytes() == b""
    missing = tmp_path / "missing" / "w.txt"
    with pytest.raises(ConfigError, match="missing"):
        write_wordlist(("a",), missing)


def test_check_wordlist_path(tmp_path: Path) -> None:
    """A list goes in a folder that exists or that the run creates."""
    out = tmp_path / "a" / "b" / "out"
    for path in (
        tmp_path / "w.txt",
        out / "w.txt",
        tmp_path / "a" / "w.txt",
        out / ".." / "w.txt",
    ):
        check_wordlist_path(path, out)
    (tmp_path / "file").write_text("", encoding="utf-8")
    for path, code in (
        (tmp_path, errno.EISDIR),
        (tmp_path / "missing" / "w.txt", errno.ENOENT),
        (out / "sub" / "w.txt", errno.ENOENT),
        (tmp_path / "file" / "w.txt", errno.ENOTDIR),
    ):
        message = f"cannot write the word list to {path}: {os.strerror(code)}"
        with pytest.raises(ConfigError) as caught:
            check_wordlist_path(path, out)
        assert str(caught.value) == message


def test_make_output_folder(tmp_path: Path) -> None:
    """The output folder and those above it are made; a file refuses."""
    out = tmp_path / "a" / "b"
    make_output_folder(replace(Config(), output=out))
    make_output_folder(replace(Config(), output=out))
    assert out.is_dir()
    (tmp_path / "file").write_text("", encoding="utf-8")
    blocked = tmp_path / "file" / "out"
    with pytest.raises(ConfigError, match="cannot write reports to"):
        make_output_folder(replace(Config(), output=blocked))
