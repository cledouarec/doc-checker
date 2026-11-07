# Copyright 2026 Christophe Le Douarec
"""Findings reported by the analyzers."""

import hashlib
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import TYPE_CHECKING, Literal, Self

if TYPE_CHECKING:
    from doc_checker.model.document import (
        BlockKind,
        ImageRef,
        SourceRef,
    )

_ID_SEPARATOR = "\x1f"
_ID_DIGEST_SIZE = 6


class Severity(IntEnum):
    """Severity of a finding, from the least to the most severe."""

    INFO = 0
    MINOR = 1
    MAJOR = 2
    CRITICAL = 3

    @property
    def label(self) -> str:
        """Lowercase name of the severity."""
        return self.name.lower()

    @classmethod
    def parse(cls, value: str) -> Self:
        """Parse a severity name, regardless of case.

        Args:
            value: Severity name such as ``"major"``.

        Returns:
            The matching severity.

        Raises:
            ValueError: If the name is not a known severity.
        """
        try:
            return cls[value.upper()]
        except KeyError:
            expected = ", ".join(member.label for member in cls)
            msg = f"unknown severity {value!r} (expected: {expected})"
            raise ValueError(msg) from None


class Category(StrEnum):
    """Family of a finding."""

    CONTENT = "content"
    TEXT = "text"
    STRUCTURE = "structure"
    LINK = "link"
    IMAGE = "image"


@dataclass(frozen=True, slots=True)
class DiffSegment:
    """A piece of text marked as unchanged, deleted or inserted."""

    op: Literal["equal", "delete", "insert"]
    text: str


@dataclass(frozen=True, slots=True)
class Side:
    """One side of a finding: where it is and what the text looks like.

    Attributes:
        source: Where the first block of the side comes from.
        segments: The text, cut into unchanged, deleted and inserted pieces.
        kind: Kind of the first block of the side, if known.
        image: Image of the first block of the side, if it is an image.
    """

    source: SourceRef
    segments: tuple[DiffSegment, ...]
    kind: BlockKind | None = None
    image: ImageRef | None = None

    @property
    def text(self) -> str:
        """Concatenated text of the segments."""
        return "".join(segment.text for segment in self.segments)


@dataclass(slots=True)
class Finding:
    """A problem found in a document or between two documents.

    Attributes:
        rule: Identifier of the rule that produced the finding.
        severity: Severity after configuration overrides.
        category: Family of the rule.
        title: Human readable title of the rule.
        ref: Reference side, if the finding involves the reference.
        new: New side, if the finding involves the new document.
        detail: Optional extra explanation.
        id: Stable identifier, usable to ignore the finding.
        pattern: What the finding changes; findings of one rule with the
            same pattern are grouped when there are enough of them.
        occurrences: The findings a group stands for, empty for a finding
            that is not a group.
    """

    rule: str
    severity: Severity
    category: Category
    title: str
    ref: Side | None
    new: Side | None
    detail: str | None = None
    id: str = ""
    pattern: str | None = None
    occurrences: tuple[Finding, ...] = ()


def finding_id(rule: str, ref: Side | None, new: Side | None) -> str:
    """Return a short identifier that is stable across runs.

    Args:
        rule: Rule identifier.
        ref: Reference side of the finding, if any.
        new: New side of the finding, if any.

    Returns:
        Twelve hexadecimal characters.
    """
    parts = [rule]
    for side in (ref, new):
        if side is None:
            parts.extend(("", "", ""))
        else:
            parts.extend(
                (side.source.file, side.source.anchor or "", side.text),
            )
    digest = hashlib.blake2b(
        _ID_SEPARATOR.join(parts).encode(),
        digest_size=_ID_DIGEST_SIZE,
    )
    return digest.hexdigest()


def group_id(rule: str, pattern: str) -> str:
    """Return the identifier of the group of a rule and a pattern.

    Args:
        rule: Rule identifier.
        pattern: Pattern shared by the findings of the group.

    Returns:
        Twelve hexadecimal characters.
    """
    digest = hashlib.blake2b(
        _ID_SEPARATOR.join(["group", rule, pattern]).encode(),
        digest_size=_ID_DIGEST_SIZE,
    )
    return digest.hexdigest()
