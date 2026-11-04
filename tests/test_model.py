# Copyright 2026 Christophe Le Douarec
"""Tests for the document, finding and rule models."""

from collections import Counter
from typing import TYPE_CHECKING

import pytest

from doc_checker.model.document import Document, SourceRef
from doc_checker.model.findings import (
    DiffSegment,
    Severity,
    Side,
    finding_id,
)
from doc_checker.rules import RULES, get_rule

if TYPE_CHECKING:
    from pathlib import Path


def _document(path: Path) -> Document:
    return Document(
        path=path,
        format="html",
        blocks=[],
        parts=[],
        lang=None,
        declared_lang=None,
        generator=None,
        normalization=Counter(),
        chrome=[],
        chrome_keys=frozenset(),
        warnings=[],
    )


def test_severity_parse_is_case_insensitive() -> None:
    """Severity names are parsed without regard to case."""
    assert Severity.parse("Major") is Severity.MAJOR


def test_severity_parse_rejects_unknown() -> None:
    """An unknown severity name raises a ``ValueError``."""
    with pytest.raises(ValueError, match="unknown severity"):
        Severity.parse("fatal")


def test_severity_order_and_label() -> None:
    """Severities are ordered and expose a lowercase label."""
    assert Severity.CRITICAL > Severity.MAJOR > Severity.MINOR > Severity.INFO
    assert Severity.MINOR.label == "minor"


def test_source_href() -> None:
    """A source reference renders its file and optional anchor."""
    assert SourceRef("guide/a.html", anchor="x").href == "guide/a.html#x"
    assert SourceRef("a.html").href == "a.html"


def test_side_text_joins_segments() -> None:
    """The text of a side is the concatenation of its segments."""
    side = Side(
        SourceRef("a.html"),
        (DiffSegment("equal", "a "), DiffSegment("delete", "b")),
    )
    assert side.text == "a b"


def test_finding_id_is_stable_and_short() -> None:
    """Finding identifiers are stable, short and depend on the rule."""
    side = Side(SourceRef("a.html", anchor="s"), (DiffSegment("delete", "x"),))
    first = finding_id("block.missing", side, None)
    assert first == finding_id("block.missing", side, None)
    assert len(first) == 12
    assert first != finding_id("block.added", side, None)


def test_document_resolve(tmp_path: Path) -> None:
    """Sources resolve inside a site directory or to the single file."""
    page = tmp_path / "index.html"
    page.write_text("<p>x</p>", encoding="utf-8")
    site = _document(tmp_path)
    single = _document(page)
    assert site.resolve(SourceRef("guide/a.html")) == tmp_path / "guide/a.html"
    assert single.resolve(SourceRef("index.html")) == page


def test_missing_content_rules_are_critical() -> None:
    """Missing blocks and sections are critical by default."""
    assert get_rule("block.missing").severity is Severity.CRITICAL
    assert get_rule("section.missing").severity is Severity.CRITICAL


def test_rule_registry_is_consistent() -> None:
    """Every rule is registered under its own identifier."""
    assert len(RULES) == 20
    assert all(spec.id == key for key, spec in RULES.items())
