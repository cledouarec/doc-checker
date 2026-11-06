# Copyright 2026 Christophe Le Douarec
"""Tests for the patterns of findings and their groups."""

import pytest

from doc_checker.compare.aggregate import (
    GROUP_MIN_OCCURRENCES,
    group_findings,
    text_pattern,
)
from doc_checker.model.document import SourceRef
from doc_checker.model.findings import (
    Category,
    DiffSegment,
    Finding,
    Severity,
    Side,
    finding_id,
    group_id,
)

ACCENT = "é\U00002192e"


@pytest.mark.parametrize(
    ("rule", "deleted", "inserted", "pattern"),
    [
        ("text.accent_lost", "été", "ete", ACCENT),
        ("text.accent_lost", "validée", "validee", ACCENT),
        ("text.accent_lost", "élève", "eleve", "è\U00002192e, " + ACCENT),
        (
            "text.ligature_lost",
            "configuration",
            "conguration",
            "fi\U00002192\U00002205",
        ),
        (
            "text.encoding_issue",
            "été",
            "\xc3\xa9t\xc3\xa9",
            "é\U00002192\xc3\xa9",
        ),
        (
            "text.punctuation_changed",
            "it, then",
            "it then",
            ",\U00002192\U00002205",
        ),
        ("text.case_changed", "Python", "python", "to lower case"),
        ("text.case_changed", "python", "PYTHON", "to upper case"),
        ("text.number_changed", "100", "10", "100\U0000219210"),
        (
            "text.words_missing",
            "with care",
            "",
            "with care\U00002192\U00002205",
        ),
    ],
)
def test_text_pattern(
    rule: str,
    deleted: str,
    inserted: str,
    pattern: str,
) -> None:
    """A difference has the pattern of the change it makes."""
    assert text_pattern(rule, deleted, inserted) == pattern


def _finding(
    text: str,
    pattern: str | None,
    *,
    rule: str = "text.accent_lost",
    severity: Severity = Severity.MAJOR,
) -> Finding:
    """Make a finding of the reference text, with a pattern."""
    side = Side(SourceRef("a.html"), (DiffSegment("delete", text),))
    return Finding(
        rule,
        severity,
        Category.TEXT,
        "Accent lost",
        side,
        None,
        id=finding_id(rule, side, None),
        pattern=pattern,
    )


def test_five_occurrences_of_a_pattern_are_grouped() -> None:
    """Five findings of one pattern become one group, in place."""
    accents = [_finding(f"word{i}", ACCENT) for i in range(6)]
    typo = _finding("teh", "the\U00002192teh", rule="text.typo")
    grouped = group_findings([accents[0], typo, *accents[1:]])
    assert [f.rule for f in grouped] == ["text.accent_lost", "text.typo"]
    group = grouped[0]
    assert group.occurrences == tuple(accents)
    assert group.id == group_id("text.accent_lost", ACCENT)
    assert group.detail == f"6 occurrences of {ACCENT}"
    assert (group.ref, group.pattern) == (accents[0].ref, ACCENT)


def test_fewer_occurrences_stay_apart() -> None:
    """Below five occurrences, the findings are kept as they are."""
    accents = [_finding(f"word{i}", ACCENT) for i in range(4)]
    assert group_findings(accents) == accents
    assert GROUP_MIN_OCCURRENCES == 5


def test_findings_without_a_pattern_are_never_grouped() -> None:
    """Moves and splits have no pattern and stay apart."""
    moves = [
        _finding(f"p{i}", None, rule="block.moved", severity=Severity.INFO)
        for i in range(7)
    ]
    assert group_findings(moves) == moves


def test_a_group_takes_the_highest_severity() -> None:
    """A group is as severe as its most severe occurrence."""
    words = [
        _finding(
            f"word{i}",
            "x\U00002192\U00002205",
            rule="text.words_missing",
            severity=Severity.MAJOR if i == 3 else Severity.MINOR,
        )
        for i in range(5)
    ]
    (group,) = group_findings(words)
    assert group.severity is Severity.MAJOR


def test_a_long_pattern_is_cut_in_the_detail() -> None:
    """The detail of a group shows at most 80 characters of the pattern."""
    pattern = "x" * 200
    group = group_findings([_finding(f"w{i}", pattern) for i in range(5)])[0]
    assert group.detail == "5 occurrences of " + "x" * 80 + "\U00002026"
