# Copyright 2026 Christophe Le Douarec
"""Patterns of findings, the groups of systematic patterns, and the findings.

A pattern says what a finding changes, such as the substitution of an
e-acute by an e for an accent that was lost. Once five findings of one rule
share a pattern, they are replaced by a single group that keeps them as its
occurrences.

`finalize_drafts` turns the drafts of a command, `compare` or `audit`, into
its findings: severities, filters, identifiers, then groups.
"""

import difflib
from collections import Counter
from typing import TYPE_CHECKING, Final

from doc_checker.compare.drafts import ARROW
from doc_checker.compare.textdiff import without_letter_hyphens
from doc_checker.config import rule_disabled
from doc_checker.model.findings import Finding, finding_id, group_id
from doc_checker.rules import get_rule
from doc_checker.text.normalize import truncate

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from doc_checker.compare.drafts import Draft
    from doc_checker.config import Config

GROUP_MIN_OCCURRENCES: Final = 5
PATTERN_MAX_CHARS: Final = 80
# The empty set, for what a change leaves out or brings in.
EMPTY: Final = "\U00002205"

# The rules whose pattern is the loose key of a block, which groups alike
# blocks but reads poorly: their groups show the text of a block instead.
_BLOCK_RULES: Final = frozenset({"block.missing", "block.added"})
_ACCENT_RULES: Final = frozenset({"text.accent_lost", "text.accent_changed"})
_EDIT_RULES: Final = frozenset(
    {
        "text.ligature_lost",
        "text.encoding_issue",
        "text.punctuation_changed",
    },
)


def _pair(deleted: str, inserted: str) -> str:
    """Return the pattern of one substitution."""
    return f"{deleted or EMPTY}{ARROW}{inserted or EMPTY}"


def _joined(pairs: Sequence[str]) -> str:
    """Return the distinct pairs, sorted and joined."""
    return ", ".join(sorted(set(pairs)))


def _accent_pairs(deleted: str, inserted: str) -> list[str] | None:
    """Return the differing characters of two texts of one length."""
    if len(deleted) != len(inserted):
        return None
    return [
        _pair(a, b) for a, b in zip(deleted, inserted, strict=True) if a != b
    ]


def _edit_pairs(deleted: str, inserted: str) -> list[str]:
    """Return the substitutions that turn one text into the other."""
    matcher = difflib.SequenceMatcher(None, deleted, inserted, autojunk=False)
    return [
        _pair(deleted[i1:i2], inserted[j1:j2])
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal"
    ]


def _case_pattern(deleted: str, inserted: str) -> str:
    """Return the direction of a change of case."""
    if inserted == deleted.lower():
        return "to lower case"
    if inserted == deleted.upper():
        return "to upper case"
    return "case"


def text_pattern(rule: str, deleted: str, inserted: str) -> str:
    """Return the pattern of a difference between two texts.

    Args:
        rule: Rule of the difference.
        deleted: Text of the reference that changed.
        inserted: Text of the new document that replaces it.

    Returns:
        The characters substituted for an accent, ligature, encoding or
        punctuation difference, the direction of a change of case, else
        the deleted text and the inserted one, joined by an arrow.
    """
    if rule in _ACCENT_RULES:
        pairs = _accent_pairs(
            without_letter_hyphens(deleted),
            without_letter_hyphens(inserted),
        )
        if pairs is not None:
            return _joined(pairs)
        return _joined(_edit_pairs(deleted, inserted))
    if rule in _EDIT_RULES:
        return _joined(_edit_pairs(deleted, inserted))
    if rule == "text.case_changed":
        return _case_pattern(deleted, inserted)
    return _pair(deleted, inserted)


def shown_pattern(finding: Finding) -> str:
    """Return the pattern of a finding, or of a group, as the reports show it.

    Args:
        finding: A finding with a pattern, or a group.

    Returns:
        The pattern of the finding, or of the first occurrence of the group,
        cut to `PATTERN_MAX_CHARS` characters. A missing or an added block
        shows its text instead of its pattern, its loose key.
    """
    first = finding.occurrences[0] if finding.occurrences else finding
    side = first.ref or first.new
    if first.rule in _BLOCK_RULES and side is not None:
        return truncate(side.text, PATTERN_MAX_CHARS)
    return truncate(first.pattern or "", PATTERN_MAX_CHARS)


def _group(occurrences: Sequence[Finding]) -> Finding:
    """Return the group that stands for findings of one rule and pattern."""
    first = occurrences[0]
    rule, pattern = first.rule, first.pattern or ""
    shown = shown_pattern(first)
    return Finding(
        rule,
        max(f.severity for f in occurrences),
        first.category,
        first.title,
        first.ref,
        first.new,
        f"{len(occurrences)} occurrences of {shown}",
        group_id(rule, pattern),
        pattern,
        tuple(occurrences),
    )


def group_findings(findings: Sequence[Finding]) -> list[Finding]:
    """Replace the systematic patterns by groups.

    Args:
        findings: Findings, in order.

    Returns:
        The findings, where those of one rule and pattern, at least
        `GROUP_MIN_OCCURRENCES` of them, are replaced by one group at the
        place of the first. Findings without a pattern are kept.
    """
    counts = Counter(
        (f.rule, f.pattern) for f in findings if f.pattern is not None
    )
    members: dict[tuple[str, str | None], list[Finding]] = {}
    for finding in findings:
        key = (finding.rule, finding.pattern)
        if counts[key] >= GROUP_MIN_OCCURRENCES:
            members.setdefault(key, []).append(finding)
    result: list[Finding] = []
    for finding in findings:
        key = (finding.rule, finding.pattern)
        if key not in members:
            result.append(finding)
        elif members[key][0] is finding:
            result.append(_group(members[key]))
    return result


def finalize_drafts(drafts: Iterable[Draft], cfg: Config) -> list[Finding]:
    """Return the findings, with overrides, filters, identifiers and groups.

    A finding is left out before the grouping if its identifier is ignored,
    or if the identifier of the group of its rule and pattern is: an ignored
    group stays ignored when fewer findings than a group needs are left.

    Args:
        drafts: The drafts of a command, in order.
        cfg: Run configuration.

    Returns:
        The findings of the drafts whose rule is enabled, whose severity,
        after the overrides of the configuration, reaches its minimum, and
        that are not ignored, in order, grouped by `group_findings`.
    """
    findings = []
    ignored = cfg.ignore.findings
    for draft in drafts:
        if rule_disabled(cfg, draft.rule):
            continue
        spec = get_rule(draft.rule)
        default = spec.severity if draft.severity is None else draft.severity
        severity = cfg.severity.get(draft.rule, default)
        if severity < cfg.min_severity:
            continue
        identifier = finding_id(draft.rule, draft.ref, draft.new)
        if identifier in ignored or (
            draft.pattern is not None
            and group_id(draft.rule, draft.pattern) in ignored
        ):
            continue
        findings.append(
            Finding(
                draft.rule,
                severity,
                spec.category,
                spec.title,
                draft.ref,
                draft.new,
                draft.detail,
                identifier,
                draft.pattern,
            ),
        )
    return group_findings(findings)
