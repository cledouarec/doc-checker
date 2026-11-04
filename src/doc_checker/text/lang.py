# Copyright 2026 Christophe Le Douarec
"""Detection and resolution of the language of a document."""

import re
from typing import Final

from doc_checker.text.tokenize import TokenKind, tokenize

FR_STOPWORDS: Final[frozenset[str]] = frozenset(
    [
        "le",
        "la",
        "les",
        "des",
        "du",
        "de",
        "un",
        "une",
        "et",
        "est",
        "pour",
        "dans",
        "que",
        "qui",
        "sur",
        "pas",
        "avec",
        "vous",
        "nous",
        "ce",
        "cette",
        "sont",
        "par",
        "au",
        "aux",
        "ou",
        "il",
        "elle",
        "en",
    ],
)
EN_STOPWORDS: Final[frozenset[str]] = frozenset(
    [
        "the",
        "and",
        "is",
        "are",
        "of",
        "to",
        "in",
        "for",
        "that",
        "with",
        "this",
        "you",
        "on",
        "be",
        "it",
        "as",
        "by",
        "or",
        "not",
        "from",
        "can",
        "an",
        "at",
        "we",
        "your",
        "if",
        "will",
        "use",
    ],
)
MIN_EVIDENCE: Final = 5
AUTO: Final = "auto"

_SUBTAG_SEPARATOR: Final = re.compile(r"[-_]")


def detect_language(text: str) -> str | None:
    """Guess whether a text is French or English from its stopwords.

    Args:
        text: Text sample.

    Returns:
        ``"fr"``, ``"en"``, or ``None`` when the evidence is too weak or
        balanced.
    """
    words = [
        token.text.casefold()
        for token in tokenize(text)
        if token.kind is TokenKind.WORD
    ]
    french = sum(word in FR_STOPWORDS for word in words)
    english = sum(word in EN_STOPWORDS for word in words)
    if max(french, english) < MIN_EVIDENCE or french == english:
        return None
    return "fr" if french > english else "en"


def resolve_language(
    override: str,
    declared: str | None,
    sample: str,
) -> str | None:
    """Return the language to use for a document.

    Args:
        override: Language requested by the user, or ``"auto"``.
        declared: Language declared by the document, such as ``"en-US"``.
        sample: Text sample used when nothing else is known.

    Returns:
        The language code, or ``None`` if it cannot be determined.
    """
    if override != AUTO:
        return override
    if declared and declared.strip():
        return _SUBTAG_SEPARATOR.split(declared.strip())[0].lower()
    return detect_language(sample)
