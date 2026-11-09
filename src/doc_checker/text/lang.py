# Copyright 2026 Christophe Le Douarec
"""Detection and resolution of the language of a document."""

import re
from types import MappingProxyType
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
# The blocks whose text tells the language of a document: the loaders read
# its first blocks of text, the audit its first blocks of prose.
LANG_SAMPLE_BLOCKS: Final = 200
# The stopwords that a text needs before `strong_language` names its
# language, and how many times more than those of the other language.
LANG_EVIDENCE_MIN: Final = 15
LANG_RATIO: Final = 3.0
# The stopwords that the text of a block needs before `block_language` names
# its language.
BLOCK_EVIDENCE_MIN: Final = 6
# The names of the languages that a text can plainly read as.
LANGUAGE_NAMES: Final = MappingProxyType({"fr": "French", "en": "English"})
AUTO: Final = "auto"

# A language of letters, then subtags of letters and digits after hyphens.
_LANGUAGE_TAG: Final = re.compile(r"[A-Za-z]{1,8}(?:-[A-Za-z0-9]{1,8})*")
_SUBTAG_SEPARATOR: Final = re.compile(r"[-_]")


def _stopword_counts(text: str) -> tuple[int, int]:
    """Return the French and the English stopwords of a text."""
    words = [
        token.text.casefold()
        for token in tokenize(text)
        if token.kind is TokenKind.WORD
    ]
    french = sum(word in FR_STOPWORDS for word in words)
    english = sum(word in EN_STOPWORDS for word in words)
    return french, english


def detect_language(text: str) -> str | None:
    """Guess whether a text is French or English from its stopwords.

    Args:
        text: Text sample.

    Returns:
        ``"fr"``, ``"en"``, or ``None`` when the evidence is too weak or
        balanced.
    """
    french, english = _stopword_counts(text)
    if max(french, english) < MIN_EVIDENCE or french == english:
        return None
    return "fr" if french > english else "en"


def strong_language(
    text: str,
    *,
    minimum: int = LANG_EVIDENCE_MIN,
) -> str | None:
    """Tell whether a text is plainly French or plainly English.

    Args:
        text: Text sample.
        minimum: The stopwords of its language that the text needs.

    Returns:
        ``"fr"`` or ``"en"`` when the text has at least ``minimum``
        stopwords of that language and at least `LANG_RATIO` times as many
        as of the other one; else ``None``.
    """
    french, english = _stopword_counts(text)
    language, found, other = (
        ("fr", french, english)
        if french > english
        else ("en", english, french)
    )
    if found < minimum or found < LANG_RATIO * other:
        return None
    return language


def block_language(text: str) -> str | None:
    """Tell whether the text of a block is plainly French or English.

    Args:
        text: The text of the block.

    Returns:
        ``"fr"`` or ``"en"`` when the text has at least
        `BLOCK_EVIDENCE_MIN` stopwords of that language and at least
        `LANG_RATIO` times as many as of the other one, as
        `strong_language` counts them; else ``None``.
    """
    return strong_language(text, minimum=BLOCK_EVIDENCE_MIN)


def declared_language(declared: str | None) -> str | None:
    """Return the language that a document declares, if it is a language tag.

    A document can declare any text as its language, such as markup or control
    characters, and the reports show it, so only a language tag is kept.

    Args:
        declared: The language as the document declares it, such as
            ``"en-US"``, if it declares one.

    Returns:
        The language as declared, or ``None`` if the document declares none or
        if it is not a language tag: 1 to 8 letters, then subtags of 1 to 8
        letters or digits, each set off by a hyphen.
    """
    if declared is not None and _LANGUAGE_TAG.fullmatch(declared):
        return declared
    return None


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
