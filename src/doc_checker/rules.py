# Copyright 2026 Christophe Le Douarec
"""Registry of the rules that produce findings."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from doc_checker.model.findings import Category, Severity

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True, slots=True)
class RuleSpec:
    """Static description of a rule.

    Attributes:
        id: Rule identifier, such as ``block.missing``.
        category: Family of the rule.
        severity: Default severity of its findings.
        title: Human readable title shown in reports.
    """

    id: str
    category: Category
    severity: Severity
    title: str


_CONTENT = Category.CONTENT
_TEXT = Category.TEXT
_STRUCTURE = Category.STRUCTURE
_LINK = Category.LINK
_IMAGE = Category.IMAGE
_SPECS: Final = (
    RuleSpec(
        "section.missing",
        _CONTENT,
        Severity.CRITICAL,
        "Section missing",
    ),
    RuleSpec("block.missing", _CONTENT, Severity.CRITICAL, "Block missing"),
    RuleSpec("block.added", _CONTENT, Severity.MINOR, "Block added"),
    RuleSpec("block.moved", _CONTENT, Severity.INFO, "Block moved"),
    RuleSpec(
        "block.split_merged",
        _CONTENT,
        Severity.INFO,
        "Block split or merged",
    ),
    RuleSpec(
        "text.encoding_issue",
        _TEXT,
        Severity.CRITICAL,
        "Encoding issue",
    ),
    RuleSpec(
        "text.number_changed",
        _TEXT,
        Severity.CRITICAL,
        "Number changed",
    ),
    RuleSpec(
        "text.negation_changed",
        _TEXT,
        Severity.CRITICAL,
        "Negation changed",
    ),
    RuleSpec("text.ligature_lost", _TEXT, Severity.MAJOR, "Ligature lost"),
    RuleSpec("text.accent_lost", _TEXT, Severity.MAJOR, "Accent lost"),
    RuleSpec("text.accent_changed", _TEXT, Severity.MINOR, "Accents changed"),
    RuleSpec("text.typo", _TEXT, Severity.MAJOR, "Typo introduced"),
    RuleSpec("text.word_replaced", _TEXT, Severity.MAJOR, "Word replaced"),
    RuleSpec(
        "text.modality_changed",
        _TEXT,
        Severity.MAJOR,
        "Modality changed",
    ),
    RuleSpec("text.words_missing", _TEXT, Severity.MAJOR, "Words missing"),
    RuleSpec("text.words_added", _TEXT, Severity.MINOR, "Words added"),
    RuleSpec("text.modified", _TEXT, Severity.MAJOR, "Text modified"),
    RuleSpec("text.case_changed", _TEXT, Severity.MINOR, "Case changed"),
    RuleSpec(
        "text.punctuation_changed",
        _TEXT,
        Severity.INFO,
        "Punctuation changed",
    ),
    RuleSpec(
        "text.number_format",
        _TEXT,
        Severity.INFO,
        "Number format changed",
    ),
    RuleSpec(
        "structure.heading_lost",
        _STRUCTURE,
        Severity.MAJOR,
        "Heading became text",
    ),
    RuleSpec(
        "structure.table_lost",
        _STRUCTURE,
        Severity.MAJOR,
        "Table became text",
    ),
    RuleSpec(
        "structure.code_lost",
        _STRUCTURE,
        Severity.MAJOR,
        "Code became text",
    ),
    RuleSpec("structure.list_lost", _STRUCTURE, Severity.MINOR, "List lost"),
    RuleSpec(
        "structure.kind_changed",
        _STRUCTURE,
        Severity.INFO,
        "Block kind changed",
    ),
    RuleSpec(
        "structure.heading_level",
        _STRUCTURE,
        Severity.MINOR,
        "Heading level changed",
    ),
    RuleSpec(
        "link.broken",
        _LINK,
        Severity.CRITICAL,
        "Internal link broken",
    ),
    RuleSpec("link.lost", _LINK, Severity.MAJOR, "Link lost"),
    RuleSpec(
        "link.target_changed",
        _LINK,
        Severity.MAJOR,
        "Link target changed",
    ),
    RuleSpec("image.missing", _IMAGE, Severity.CRITICAL, "Image missing"),
    RuleSpec("image.added", _IMAGE, Severity.MINOR, "Image added"),
    RuleSpec("image.replaced", _IMAGE, Severity.MAJOR, "Image replaced"),
    RuleSpec("image.degraded", _IMAGE, Severity.MAJOR, "Image degraded"),
    RuleSpec("image.reencoded", _IMAGE, Severity.INFO, "Image re-encoded"),
    RuleSpec(
        "image.rasterized",
        _IMAGE,
        Severity.MINOR,
        "Figure rasterized",
    ),
    RuleSpec(
        "image.vectorized",
        _IMAGE,
        Severity.INFO,
        "Figure vectorized",
    ),
    RuleSpec("image.alt_lost", _IMAGE, Severity.MINOR, "Alt text lost"),
)

RULES: Final[Mapping[str, RuleSpec]] = MappingProxyType(
    {spec.id: spec for spec in _SPECS},
)


def get_rule(rule_id: str) -> RuleSpec:
    """Return the specification of a rule.

    Args:
        rule_id: Rule identifier.

    Returns:
        The rule specification.

    Raises:
        KeyError: If the rule is unknown.
    """
    try:
        return RULES[rule_id]
    except KeyError:
        msg = f"unknown rule {rule_id!r}"
        raise KeyError(msg) from None
