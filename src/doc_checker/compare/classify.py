# Copyright 2026 Christophe Le Douarec
"""Give a rule and a severity to each word level difference."""

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from rapidfuzz.distance import DamerauLevenshtein

from doc_checker.compare.textdiff import (
    Hunk,
    span_text,
    without_letter_hyphens,
)
from doc_checker.model.findings import Severity
from doc_checker.spell.dictionary import DICTIONARY_LANGUAGES
from doc_checker.spell.words import MAX_WORD_CHARS, skip_reason, words
from doc_checker.text.lexicon import is_modal, negation_count
from doc_checker.text.normalize import count_marks, strip_accents
from doc_checker.text.numbers import canonical_number
from doc_checker.text.tokenize import Token, TokenKind

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from doc_checker.spell.speller import Speller

TYPO_MAX_DISTANCE: Final = 2
SHORT_WORD_LEN: Final = 4
MANY_WORDS: Final = 3
# How far around a word its run of non-blank characters is read: a word in a
# longer run is no word that the speller reads.
RUN_SCAN_MAX: Final = 2 * MAX_WORD_CHARS

_LIGATURES: Final = ("ffi", "ffl", "ff", "fi", "fl")
_LIGATURE_GAPS: Final = ("", " ", "\U0000fffd")
_REPLACEMENT_CHARACTER: Final = "\U0000fffd"
_PRIVATE_USE: Final = re.compile(r"[\U0000e000-\U0000f8ff]")
# A UTF-8 lead byte read as cp1252, followed by continuation bytes.
_MOJIBAKE: Final = re.compile(
    r"[\xc2-\xf4]"
    r"[\x80-\xbf\U00000152\U00000153\U00000160\U00000161\U00000178"
    r"\U0000017d\U0000017e\U00000192\U000002c6\U000002dc\U00002013"
    r"\U00002014\U00002018-\U0000201e\U00002020-\U00002022\U00002026"
    r"\U00002030\U00002039\U0000203a\U000020ac\U00002122]+",
)


@dataclass(frozen=True, slots=True)
class HunkContext:
    """What a classification needs to know about the compared blocks.

    Attributes:
        ref_text: Strict key of the reference block.
        new_text: Strict key of the new block.
        ref_tokens: Tokens of the reference block.
        new_tokens: Tokens of the new block.
        ref_vocabulary: Casefolded words of the whole reference document.
        ref_lang: Language of the reference document.
        new_lang: Language of the new document.
        speller: The speller that confirms a typo, with the personal
            words; None to tell a typo by the reference vocabulary only.
        code_words: Casefolded words of the inline code of the compared
            blocks, which the speller does not judge.
    """

    ref_text: str
    new_text: str
    ref_tokens: Sequence[Token]
    new_tokens: Sequence[Token]
    ref_vocabulary: frozenset[str]
    ref_lang: str | None
    new_lang: str | None
    speller: Speller | None = None
    code_words: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class Classification:
    """Rule and severity given to a difference."""

    rule: str
    severity: Severity


@dataclass(frozen=True, slots=True)
class _View:
    """The two sides of a hunk."""

    op: str
    ref: str
    new: str
    ref_tokens: Sequence[Token]
    new_tokens: Sequence[Token]

    @staticmethod
    def words(tokens: Sequence[Token]) -> list[str]:
        """Return the texts of the word and number tokens."""
        return [t.text for t in tokens if t.kind is not TokenKind.PUNCT]

    def single_words(self) -> tuple[str, str] | None:
        """Return the two words of a one word replacement, if it is one."""
        if (
            self.op == "replace"
            and len(self.ref_tokens) == len(self.new_tokens) == 1
            and self.ref_tokens[0].kind is TokenKind.WORD
            and self.new_tokens[0].kind is TokenKind.WORD
        ):
            return self.ref_tokens[0].text, self.new_tokens[0].text
        return None


def has_encoding_damage(text: str) -> bool:
    """Tell whether a text shows signs of a broken encoding.

    Args:
        text: Text to inspect.

    Returns:
        Whether the text holds U+FFFD, a private use character, or UTF-8
        bytes decoded as cp1252.
    """
    if _REPLACEMENT_CHARACTER in text or _PRIVATE_USE.search(text):
        return True
    for match in _MOJIBAKE.finditer(text):
        try:
            repaired = match.group().encode("cp1252").decode("utf-8")
        except UnicodeError:
            continue
        if repaired != match.group():
            return True
    return False


def _encoding(view: _View, _: HunkContext) -> Classification | None:
    if has_encoding_damage(view.new) and not has_encoding_damage(view.ref):
        return Classification("text.encoding_issue", Severity.CRITICAL)
    return None


def _numbers(view: _View, ctx: HunkContext) -> Classification | None:
    ref_numbers = [
        t.text for t in view.ref_tokens if t.kind is TokenKind.NUMBER
    ]
    new_numbers = [
        t.text for t in view.new_tokens if t.kind is TokenKind.NUMBER
    ]
    if not ref_numbers and not new_numbers:
        return None
    ref_values = sorted(canonical_number(n, ctx.ref_lang) for n in ref_numbers)
    new_values = sorted(canonical_number(n, ctx.new_lang) for n in new_numbers)
    if ref_values != new_values:
        return Classification("text.number_changed", Severity.CRITICAL)
    ref_others = [
        t.text for t in view.ref_tokens if t.kind is not TokenKind.NUMBER
    ]
    new_others = [
        t.text for t in view.new_tokens if t.kind is not TokenKind.NUMBER
    ]
    if ref_numbers != new_numbers and ref_others == new_others:
        return Classification("text.number_format", Severity.INFO)
    return None


def _negation(view: _View, _: HunkContext) -> Classification | None:
    ref_count = negation_count(view.words(view.ref_tokens))
    if ref_count != negation_count(view.words(view.new_tokens)):
        return Classification("text.negation_changed", Severity.CRITICAL)
    return None


def _ligature(view: _View, _: HunkContext) -> Classification | None:
    if view.op != "replace" or view.ref == view.new:
        return None
    for ligature in _LIGATURES:
        for gap in _LIGATURE_GAPS:
            if view.ref.replace(ligature, gap) == view.new:
                return Classification("text.ligature_lost", Severity.MAJOR)
    return None


def _fold(text: str) -> str:
    return strip_accents(text).casefold()


def _accents(view: _View, _: HunkContext) -> Classification | None:
    ref = without_letter_hyphens(view.ref)
    new = without_letter_hyphens(view.new)
    if view.op != "replace" or ref == new or _fold(ref) != _fold(new):
        return None
    if count_marks(new) < count_marks(ref):
        return Classification("text.accent_lost", Severity.MAJOR)
    if strip_accents(ref) == strip_accents(new):
        return Classification("text.accent_changed", Severity.MINOR)
    return None


def _case(view: _View, _: HunkContext) -> Classification | None:
    ref = without_letter_hyphens(view.ref)
    new = without_letter_hyphens(view.new)
    if ref != new and ref.casefold() == new.casefold():
        return Classification("text.case_changed", Severity.MINOR)
    return None


def _punctuation(view: _View, _: HunkContext) -> Classification | None:
    if view.words(view.ref_tokens) == view.words(view.new_tokens):
        return Classification("text.punctuation_changed", Severity.INFO)
    return None


def _modality(view: _View, _: HunkContext) -> Classification | None:
    words = view.single_words()
    if words is None:
        return None
    ref_word, new_word = words
    if (
        is_modal(ref_word)
        and is_modal(new_word)
        and ref_word.casefold() != new_word.casefold()
    ):
        return Classification("text.modality_changed", Severity.MAJOR)
    return None


def _typo(view: _View, ctx: HunkContext) -> Classification | None:
    """Classify a word replaced by a close one.

    The words are close when their Damerau-Levenshtein distance, without
    case, is at most 1, or `TYPO_MAX_DISTANCE` past `SHORT_WORD_LEN`
    letters. The dictionaries judge them, as `_confirmed` does; when they
    cannot, the new word is a word replaced if the reference uses it, a
    typo otherwise.

    Args:
        view: The two sides of the difference.
        ctx: The compared blocks.

    Returns:
        The classification, or None if the difference is not one word
        replaced by a close one.
    """
    pair = view.single_words()
    if pair is None:
        return None
    ref_word, new_word = (word.casefold() for word in pair)
    limit = 1 if len(ref_word) <= SHORT_WORD_LEN else TYPO_MAX_DISTANCE
    distance = DamerauLevenshtein.distance(ref_word, new_word)
    if not 0 < distance <= limit:
        return None
    confirmed = _confirmed(view, ctx)
    if confirmed is not None:
        return confirmed
    if new_word in ctx.ref_vocabulary:
        return Classification("text.word_replaced", Severity.MAJOR)
    return Classification("text.typo", Severity.MAJOR)


def _confirmed(view: _View, ctx: HunkContext) -> Classification | None:
    """Classify two close words by the dictionaries, if they can tell.

    They can when the context has a speller, both languages have a
    dictionary and both words are `_checkable`. A new word that the
    reference uses, a personal word or a word of the dictionaries is known:
    if the old word is in lowercase and unknown, the new one corrects it;
    otherwise it replaces it. An unknown new word is a typo. A dictionary
    that cannot be loaded raises the `DictionaryError` of the speller.

    Args:
        view: The two sides of a one word replacement.
        ctx: The compared blocks, with the speller.

    Returns:
        The classification, or None if the dictionaries cannot tell, or if
        the speller could not truly look a word up, as `_vouched` tells.
    """
    speller = ctx.speller
    ref_lang, new_lang = ctx.ref_lang, ctx.new_lang
    ref_token, new_token = view.ref_tokens[0], view.new_tokens[0]
    if (
        speller is None
        or ref_lang not in DICTIONARY_LANGUAGES
        or new_lang not in DICTIONARY_LANGUAGES
        or not _checkable(ctx.ref_text, ref_token, ctx.code_words)
        or not _checkable(ctx.new_text, new_token, ctx.code_words)
    ):
        return None
    ref_word, new_word = ref_token.text, new_token.text
    new_known = new_word.casefold() in ctx.ref_vocabulary or _vouched(
        speller,
        new_word,
        new_lang,
    )
    if new_known is None:
        return None
    if not new_known:
        return Classification("text.typo", Severity.MAJOR)
    if ref_word.islower():
        ref_known = _vouched(speller, ref_word, ref_lang)
        if ref_known is None:
            return None
        if not ref_known:
            return Classification("text.typo_fixed", Severity.INFO)
    return Classification("text.word_replaced", Severity.MAJOR)


def _checkable(text: str, token: Token, code_words: frozenset[str]) -> bool:
    """Return whether the speller may judge a word of a difference.

    Args:
        text: The text of the side of the word.
        token: The word.
        code_words: The casefolded words of inline code.

    Returns:
        Whether the word has no `skip_reason`, is no word of inline code
        and is `_spelled` in its text.
    """
    return (
        skip_reason(token.text) is None
        and token.text.casefold() not in code_words
        and _spelled(text, token)
    )


def _spelled(text: str, token: Token) -> bool:
    """Return whether a token is a word that the speller reads in its text.

    The token must lie inside one of the `words` of its run of non-blank
    characters, as an elided word lies inside ``d'accueil``: a part of a
    URL, an e-mail address, a path or an option does not. The run is read
    at most `RUN_SCAN_MAX` characters around the token, and a longer run
    gives no word.

    Args:
        text: The text of the token.
        token: The word token.

    Returns:
        Whether the token is a word that the speller reads.
    """
    start, end = token.start, token.end
    low = max(0, start - RUN_SCAN_MAX)
    high = min(len(text), end + RUN_SCAN_MAX)
    while start > low and not text[start - 1].isspace():
        start -= 1
    while end < high and not text[end].isspace():
        end += 1
    if (start > 0 and not text[start - 1].isspace()) or (
        end < len(text) and not text[end].isspace()
    ):
        return False
    first, last = token.start - start, token.end - start
    return any(
        word.start <= first and last <= word.end
        for word in words(text[start:end])
    )


def _vouched(speller: Speller, word: str, lang: str) -> bool | None:
    """Return whether the speller knows a word, None if it did not look.

    Args:
        speller: The speller.
        word: The word.
        lang: Its language, one with a dictionary.

    Returns:
        Whether the word is known, or None if the speller had spent its
        budget before it could look the word up, or took it as known
        because spylls failed on it.
    """
    known = speller.known(word, lang)
    if known and (
        speller.doubtful(word, lang)
        or not speller.known(word, lang, spent=False)
    ):
        return None
    return known


def _size(word_count: int) -> Severity:
    return Severity.MAJOR if word_count >= MANY_WORDS else Severity.MINOR


def _missing(view: _View, _: HunkContext) -> Classification | None:
    if view.op == "delete":
        count = len(view.words(view.ref_tokens))
        return Classification("text.words_missing", _size(count))
    return None


def _added(view: _View, _: HunkContext) -> Classification | None:
    if view.op == "insert":
        return Classification("text.words_added", Severity.MINOR)
    return None


_RULES: Final[
    tuple[Callable[[_View, HunkContext], Classification | None], ...]
] = (
    _encoding,
    _numbers,
    _negation,
    _ligature,
    _accents,
    _case,
    _punctuation,
    _modality,
    _typo,
    _missing,
    _added,
)


def classify_hunk(hunk: Hunk, ctx: HunkContext) -> Classification:
    """Return the rule and severity of a difference.

    The rules are tried in order and the first that applies wins; a
    difference no rule recognizes is a plain modification. Only a word
    replaced by a close one, that no rule before claims, is looked up in
    the dictionaries; one that cannot be loaded raises the
    `DictionaryError` of the speller.

    Args:
        hunk: The difference.
        ctx: The compared blocks.

    Returns:
        The classification.
    """
    view = _View(
        hunk.op,
        span_text(ctx.ref_text, ctx.ref_tokens, hunk.ref_start, hunk.ref_end),
        span_text(ctx.new_text, ctx.new_tokens, hunk.new_start, hunk.new_end),
        ctx.ref_tokens[hunk.ref_start : hunk.ref_end],
        ctx.new_tokens[hunk.new_start : hunk.new_end],
    )
    for rule in _RULES:
        result = rule(view, ctx)
        if result is not None:
            return result
    changed = max(
        len(view.words(view.ref_tokens)),
        len(view.words(view.new_tokens)),
    )
    return Classification("text.modified", _size(changed))
