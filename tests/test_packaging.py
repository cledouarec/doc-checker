# Copyright 2026 Christophe Le Douarec
"""Tests for the metadata and the licence notices of the package."""

import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import] - runs uv to build the package
import tarfile
import tomllib
import zipfile
from pathlib import Path
from typing import Final

import pytest
from packaging.requirements import Requirement

ROOT: Final = Path(__file__).parents[1]
PROJECT: Final = tomllib.loads(
    (ROOT / "pyproject.toml").read_text(encoding="utf-8"),
)["project"]
NOTICES: Final = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
TRANSITIVE: Final = (
    "MarkupSafe",
    "pdfminer.six",
    "charset-normalizer",
    "cryptography",
    "cffi",
    "pycparser",
)


def _name(requirement: str) -> str:
    """Return the distribution name of a requirement."""
    return Requirement(requirement).name


def _table_names() -> set[str]:
    """Return the first cell of every table row of the notices, casefolded."""
    names = set()
    for line in NOTICES.splitlines():
        if line.startswith("|"):
            cell = line.split("|")[1].strip()
            names.add(cell.split(" ", 1)[0].casefold())
    return names


@pytest.mark.parametrize(
    ("requirement", "name"),
    [
        ("click>=8.5", "click"),
        ("spylls>=0.1.7,<0.2", "spylls"),
        ("rapidfuzz~=3.14", "rapidfuzz"),
        ("pdfplumber[extra]>=0.11", "pdfplumber"),
        ("pdfminer.six==20260107", "pdfminer.six"),
        ("cffi>=2.0 ; platform_python_implementation != 'PyPy'", "cffi"),
    ],
)
def test_requirement_name_handles_operators_extras_and_markers(
    requirement: str,
    name: str,
) -> None:
    """The distribution name is read whatever follows it."""
    assert _name(requirement) == name


def test_licence_expression_declares_the_bundled_dictionary() -> None:
    """The wheel declares the MPL-2.0 data and ships every licence file."""
    assert PROJECT["license"] == "Apache-2.0 AND MPL-2.0"
    assert PROJECT["license-files"] == [
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        "src/doc_checker/spell/data/fr/LICENSE",
    ]
    for name in PROJECT["license-files"]:
        assert (ROOT / name).is_file(), name


def test_notices_name_every_dependency() -> None:
    """Each dependency, direct or not, and the bundled data are named."""
    names = [_name(r) for r in PROJECT["dependencies"]]
    names += [_name(r) for r in PROJECT["optional-dependencies"]["pdf-tables"]]
    table = _table_names()
    for name in (*names, *TRANSITIVE):
        assert name.casefold() in table, name
    assert "colorama" in NOTICES
    for fact in ("Grammalecte", "MPL-2.0", "LGPL-2.1", "SOURCE.txt"):
        assert fact in NOTICES


def test_tables_extra_stays_optional() -> None:
    """The tables library is an extra, never a base dependency."""
    assert "pdfplumber" not in {_name(r) for r in PROJECT["dependencies"]}
    assert [
        _name(r) for r in PROJECT["optional-dependencies"]["pdf-tables"]
    ] == [
        "pdfplumber",
    ]


SDIST_ROOT_FILES: Final = {
    "doc_checker-0.1.0",
    "doc_checker-0.1.0/LICENSE",
    "doc_checker-0.1.0/PKG-INFO",
    "doc_checker-0.1.0/README.md",
    "doc_checker-0.1.0/THIRD_PARTY_NOTICES.md",
    "doc_checker-0.1.0/pyproject.toml",
    "doc_checker-0.1.0/pyproject.toml.orig",
    "doc_checker-0.1.0/src",
    "doc_checker-0.1.0/src/doc_checker",
}
DIST_INFO: Final = "doc_checker-0.1.0.dist-info/"
LICENCE_FILES: Final = (
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "src/doc_checker/spell/data/fr/LICENSE",
)


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not on PATH")
def test_built_sdist_and_wheel_hold_the_licences(tmp_path: Path) -> None:
    """The sdist and the wheel carry exactly the expected files."""
    subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] - fixed arguments, uv found on PATH
        [  # ruff: ignore[start-process-with-partial-path] - uv is looked up on PATH on purpose
            "uv",
            "build",
            "--offline",
            "--out-dir",
            str(tmp_path),
            str(ROOT),
        ],
        check=True,
        capture_output=True,
    )
    with tarfile.open(tmp_path / "doc_checker-0.1.0.tar.gz") as sdist:
        names = set(sdist.getnames())
    prefix = "doc_checker-0.1.0/src/doc_checker/"
    assert {n for n in names if not n.startswith(prefix)} == SDIST_ROOT_FILES
    for name in (
        "fr-toutesvariantes.aff",
        "fr-toutesvariantes.dic",
        "LICENSE",
    ):
        assert f"{prefix}spell/data/fr/{name}" in names
    with zipfile.ZipFile(
        tmp_path / "doc_checker-0.1.0-py3-none-any.whl",
    ) as wheel:
        files = {n for n in wheel.namelist() if not n.endswith("/")}
        metadata = wheel.read(f"{DIST_INFO}METADATA").decode("utf-8")
    info = {
        n.removeprefix(DIST_INFO) for n in files if n.startswith(DIST_INFO)
    }
    assert info == {
        "METADATA",
        "RECORD",
        "WHEEL",
        "entry_points.txt",
        *(f"licenses/{n}" for n in LICENCE_FILES),
    }
    assert all(n.startswith(("doc_checker/", DIST_INFO)) for n in files)
    assert "doc_checker/spell/data/fr/LICENSE" in files
    assert "doc_checker/spell/data/fr/SOURCE.txt" in files
    assert "License-Expression: Apache-2.0 AND MPL-2.0" in metadata
    for name in LICENCE_FILES:
        assert f"License-File: {name}" in metadata
