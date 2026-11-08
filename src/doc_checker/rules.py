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
_A11Y = Category.ACCESSIBILITY
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
    RuleSpec(
        "image.low_dpi",
        _IMAGE,
        Severity.MINOR,
        "Low image resolution",
    ),
    RuleSpec(
        "image.upscaled",
        _IMAGE,
        Severity.MINOR,
        "Image shown larger than its size",
    ),
    RuleSpec(
        "image.blurry",
        _IMAGE,
        Severity.MINOR,
        "Blurry image",
    ),
    RuleSpec(
        "image.jpeg_quality",
        _IMAGE,
        Severity.MINOR,
        "Low JPEG quality",
    ),
    RuleSpec(
        "image.blocky",
        _IMAGE,
        Severity.MINOR,
        "JPEG blocks visible",
    ),
    RuleSpec(
        "image.pixelated",
        _IMAGE,
        Severity.MINOR,
        "Pixelated image",
    ),
    RuleSpec(
        "image.heavy",
        _IMAGE,
        Severity.MINOR,
        "Heavy image file",
    ),
    RuleSpec(
        "image.photo_png",
        _IMAGE,
        Severity.INFO,
        "Photo stored as PNG",
    ),
    RuleSpec(
        "image.broken",
        _IMAGE,
        Severity.MAJOR,
        "Image cannot be shown",
    ),
    RuleSpec(
        "structure.heading_skipped",
        _STRUCTURE,
        Severity.MINOR,
        "Heading level skipped",
    ),
    RuleSpec(
        "structure.empty_section",
        _STRUCTURE,
        Severity.MINOR,
        "Empty section",
    ),
    RuleSpec(
        "structure.duplicate_heading",
        _STRUCTURE,
        Severity.MINOR,
        "Identical sibling headings",
    ),
    RuleSpec(
        "structure.unresolved_reference",
        _STRUCTURE,
        Severity.MAJOR,
        "Unresolved reference",
    ),
    RuleSpec(
        "structure.figure_caption",
        _STRUCTURE,
        Severity.INFO,
        "Figure without caption",
    ),
    RuleSpec(
        "structure.table_caption",
        _STRUCTURE,
        Severity.INFO,
        "Table without caption",
    ),
    RuleSpec(
        "structure.orphan_page",
        _STRUCTURE,
        Severity.MINOR,
        "Orphan page",
    ),
    RuleSpec(
        "link.page_missing",
        _LINK,
        Severity.MAJOR,
        "Link to a missing page",
    ),
    RuleSpec(
        "link.anchor_missing",
        _LINK,
        Severity.MAJOR,
        "Link to a missing anchor",
    ),
    RuleSpec(
        "link.file_missing",
        _LINK,
        Severity.MAJOR,
        "Link to a missing file",
    ),
    RuleSpec(
        "link.destination_invalid",
        _LINK,
        Severity.MAJOR,
        "Link destination invalid",
    ),
    RuleSpec(
        "a11y.alt_missing",
        _A11Y,
        Severity.MAJOR,
        "Alt text missing",
    ),
    RuleSpec(
        "a11y.alt_filename",
        _A11Y,
        Severity.MINOR,
        "Alt text is a file name",
    ),
    RuleSpec(
        "a11y.svg_title",
        _A11Y,
        Severity.MINOR,
        "SVG without title",
    ),
    RuleSpec(
        "a11y.lang_missing",
        _A11Y,
        Severity.MAJOR,
        "Language not declared",
    ),
    RuleSpec(
        "a11y.lang_mismatch",
        _A11Y,
        Severity.MINOR,
        "Declared language differs from the text",
    ),
    RuleSpec(
        "a11y.table_headers",
        _A11Y,
        Severity.MINOR,
        "Table without headers",
    ),
    RuleSpec(
        "a11y.pdf_untagged",
        _A11Y,
        Severity.MAJOR,
        "PDF not tagged",
    ),
    RuleSpec(
        "a11y.title_missing",
        _A11Y,
        Severity.MINOR,
        "Title missing",
    ),
    RuleSpec(
        "a11y.unmapped_chars",
        _A11Y,
        Severity.MINOR,
        "Characters without Unicode mapping",
    ),
    RuleSpec(
        "a11y.scanned_page",
        _A11Y,
        Severity.MAJOR,
        "Scanned page",
    ),
    RuleSpec(
        "a11y.link_empty",
        _A11Y,
        Severity.MAJOR,
        "Link without text",
    ),
    RuleSpec(
        "a11y.link_text",
        _A11Y,
        Severity.MINOR,
        "Link text says nothing",
    ),
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
