# Copyright 2026 Christophe Le Douarec
"""Check the spelling of the prose of a document.

Each word of a block of French or English prose that `skip_reason` lets
through is looked up in the language of the block. A word that the speller
does not know, nor the document sets as code, is an unknown word. The
unknown words are counted without case over the whole document: one seen at
least `JARGON_MIN` times is jargon, the vocabulary of the document; else one
whose first letter is a capital may be a name; else it is an unknown word,
maybe a slip.

A hyphenated word is known when each of its parts is skipped, code or
known: ``pgf-system`` with the code ``pgf``, ``config-based``, ``x-axis``.

A block of at least `BLOCK_GUARD_MIN_WORDS` spelled words, more than
`BLOCK_UNKNOWN_SHARE_MAX` of them unknown, is not in the language it is
checked in, as placeholder text (lorem ipsum) or a quotation in a third
language: its words are counted as spelled, and its unknown words count
towards the jargon of the document, but none of them is reported or
listed.

A word is matched with the code of the document, counted, and its first
letter is read, once a French elided word of `ELISIONS` is taken from its
start and a possessive from its end, as `bare` does: ``tlmgr's`` is the code
``tlmgr``; ``frobnicate's`` counts with ``frobnicate``; ``d'Emacs`` is a
name, but ``L'orblaxe`` at the start of a sentence is not, since its word
is in lower case.
"""

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from doc_checker.audit.sides import quote
from doc_checker.compare.drafts import Draft, marked_span
from doc_checker.spell.dictionary import DICTIONARY_LANGUAGES
from doc_checker.spell.words import (
    APOSTROPHE,
    ELISIONS,
    HYPHEN,
    MAX_WORD_CHARS,
    MEMO_MAX,
    fold,
    skip_reason,
    without_possessive,
    words,
)
from doc_checker.text.lang import LANGUAGE_NAMES

if TYPE_CHECKING:
    from collections.abc import Sequence

    from doc_checker.audit.prose import Prose
    from doc_checker.spell.speller import Speller
    from doc_checker.spell.words import Word

# An unknown word seen at least this many times in a document, whatever its
# case, with its possessives and elided forms, is jargon.
JARGON_MIN: Final = 3
# The most drafts of each rule that a reading makes, unless told otherwise:
# the first ones, in the order of the document. A document of countless
# unknown words does not hold a draft for each.
TEXT_DRAFTS_PER_RULE: Final = 10_000
# A block of at least this many spelled words, more than this share of them
# unknown, is guarded: its unknown words are not reported (tcolorbox has 120
# blocks of lorem ipsum; real prose stays far under, see the README).
BLOCK_GUARD_MIN_WORDS: Final = 20
BLOCK_UNKNOWN_SHARE_MAX: Final = 0.5

# The most distinct words whose verdict one reading remembers, as many as
# `words` remembers runs: a frequent word is judged once, and a document of
# countless distinct words does not grow the memory without bound. A word
# that is not spelled is not remembered.
_VERDICTS_MAX: Final = MEMO_MAX
_UNKNOWN: Final = "text.unknown_word"
_NAME: Final = "text.unknown_name"
_JARGON: Final = "text.jargon"
_RULES: Final = (_UNKNOWN, _NAME, _JARGON)


@dataclass(frozen=True, slots=True)
class Spelling:
    """The spelling of the prose of a document.

    Attributes:
        drafts: A draft for each occurrence of an unknown word, in the
            order of the document.
        words: The unknown words, each once, as first written, sorted by
            their form without case, then as written.
        checked: The occurrences of words that were spelled: those that
            `skip_reason` lets through, in a block of French or English,
            but those that a spent budget left without a verdict.
        unknown: The occurrences of unknown words among them.
        cut: The rules that had more occurrences than the limit of drafts,
            in the order ``text.unknown_word``, ``text.unknown_name``,
            ``text.jargon``.
        guarded: The blocks of mostly unknown words, whose words are
            neither reported nor listed.
    """

    drafts: list[Draft]
    words: tuple[str, ...]
    checked: int
    unknown: int
    cut: tuple[str, ...] = ()
    guarded: int = 0


def spelling(
    proses: Sequence[Prose],
    speller: Speller,
    *,
    code_words: frozenset[str] = frozenset(),
    limit: int | None = TEXT_DRAFTS_PER_RULE,
) -> Spelling:
    """Return the spelling of the prose of a document.

    Each word of a block of a language with a dictionary, as `words` reads
    its text without its excluded parts, that `skip_reason` lets through,
    is spelled. It is unknown if the speller does not know it in the
    language of the block, and it is not one of ``code_words``, whatever
    its case and once its elided start and possessive end are taken off, as
    `bare` does; a hyphenated word is not unknown if each of its parts is
    skipped, code or known. A dictionary that cannot be loaded raises the
    `DictionaryError` of the speller.

    The rule of an occurrence depends on how many times the whole document
    has its word, so the drafts are made once the document is read, at
    most ``limit`` of each rule. A block of at least
    `BLOCK_GUARD_MIN_WORDS` spelled words, more than
    `BLOCK_UNKNOWN_SHARE_MAX` of them unknown, gives no draft and no word.

    Args:
        proses: The prose of the document, in order.
        speller: The speller.
        code_words: The words that the document sets as code, such as the
            words of a PDF set in a monospaced font: they are no unknown
            words, and are not looked up. The same word in prose may start
            a sentence, or end with a possessive.
        limit: The most drafts of each rule, the first ones in the order of
            the document; None for a draft of each occurrence.

    Returns:
        The drafts, the unknown words and the counts. The draft of an
        unknown word seen at least `JARGON_MIN` times in the document,
        whatever its case and counted by its `bare` form, is
        ``text.jargon``; else, if the first letter of its `bare` form is a
        capital, ``text.unknown_name``; else ``text.unknown_word``.
    """
    tally = _Tally(speller, frozenset(map(fold, code_words)), limit)
    for prose in proses:
        if prose.lang in DICTIONARY_LANGUAGES:
            tally.read(prose, prose.lang)
    return tally.result()


def bare(word: str) -> str:
    """Return a word without its elided start and its possessive end.

    Args:
        word: A word, as `words` returns it.

    Returns:
        The word without its possessive, as `without_possessive` tells, and
        without what ends at its first apostrophe, if that is one of
        `ELISIONS` and more follows; its other apostrophes read ``'``.
    """
    parts = APOSTROPHE.split(without_possessive(word))
    if len(parts) > 1 and parts[0].casefold() in ELISIONS:
        del parts[0]
    return "'".join(parts)


class _Tally:
    """The unknown words of a document, as its prose is read.

    The unknown words are counted by their stem, their `bare` form without
    case, those of the guarded blocks too. An occurrence of a block that is
    not guarded is kept until the drafts are made, unless its stem was seen
    `JARGON_MIN` times already and ``limit`` such occurrences came before
    it: it cannot be among the first ``limit`` jargon. The others are fewer
    than `JARGON_MIN` for each stem of a word that the speller refuses, and
    it looks up a bounded number of words: a document of countless unknown
    words holds neither a draft nor an occurrence for each.
    """

    __slots__ = (
        "checked",
        "code",
        "first",
        "found",
        "guarded",
        "jargon",
        "limit",
        "shown",
        "speller",
        "stems",
        "unknown",
        "verdicts",
    )

    def __init__(
        self,
        speller: Speller,
        code: frozenset[str],
        limit: int | None,
    ) -> None:
        """Start a reading.

        Args:
            speller: The speller.
            code: The code words of the document, as `fold` returns them.
            limit: The most drafts of each rule, or None.
        """
        self.speller = speller
        self.code = code
        self.limit = limit
        self.verdicts: dict[tuple[str, str], bool] = {}
        self.found: list[tuple[Prose, Word, str, str]] = []
        self.stems: Counter[str] = Counter()
        self.shown: Counter[str] = Counter()
        self.first: dict[str, str] = {}
        self.checked = self.unknown = self.jargon = self.guarded = 0

    def read(self, prose: Prose, lang: str) -> None:
        """Read the words of a block of prose.

        The words are counted as spelled, and the unknown ones by their
        stem; they are kept, unless the block is of mostly unknown words.

        Args:
            prose: The block.
            lang: Its language, one with a dictionary.
        """
        spelled = 0
        unknown: list[Word] = []
        for word in words(prose.block.text, prose.excluded):
            known = self._verdict(word.text, lang)
            if known is None:
                continue
            spelled += 1
            if not known:
                unknown.append(word)
        self.checked += spelled
        self.unknown += len(unknown)
        guarded = (
            spelled >= BLOCK_GUARD_MIN_WORDS
            and len(unknown) > BLOCK_UNKNOWN_SHARE_MAX * spelled
        )
        if guarded:
            self.guarded += 1
        for word in unknown:
            stem = fold(bare(word.text))
            self.stems[stem] += 1
            if guarded:
                continue
            key = fold(word.text)
            self.first.setdefault(key, word.text)
            self.shown[stem] += 1
            if self.stems[stem] >= JARGON_MIN:
                self.jargon += 1
                if self.limit is not None and self.jargon > self.limit:
                    continue
            self.found.append((prose, word, key, stem))

    def _verdict(self, text: str, lang: str) -> bool | None:
        """Return whether a word is known, if it is spelled.

        Each of the first `_VERDICTS_MAX` distinct words that are judged
        is judged once; a word longer than `MAX_WORD_CHARS` is not spelled,
        whatever else it is.

        Args:
            text: The word.
            lang: The language of its block.

        Returns:
            None if `skip_reason` gives a reason, or if the budget of the
            speller was spent before the word was judged; else whether the
            word is known, as `_known` tells.
        """
        if len(text) > MAX_WORD_CHARS:
            return None
        known = self.verdicts.get((lang, text))
        if known is not None:
            return known
        if skip_reason(text) is not None:
            return None
        known = self._known(text, lang)
        if known is not None and len(self.verdicts) < _VERDICTS_MAX:
            self.verdicts[lang, text] = known
        return known

    def _known(self, text: str, lang: str) -> bool | None:
        """Return whether a word that is spelled is known, if it can be told.

        Args:
            text: The word.
            lang: The language of its block.

        Returns:
            Whether the word is code, as `_is_code` tells, or known to the
            speller, or has hyphens and each of its parts is skipped, code
            or known: Hunspell refuses a compound of which a part is not in
            its dictionary, whatever else knows that part. None when the
            speller has no `Speller.verdict` for the word, or for a part
            and no other part is unknown. The lookups of the parts are
            charged to the speller, up to the first unknown part.
        """
        code = self.code
        speller = self.speller
        if _is_code(text, code):
            return True
        whole = speller.verdict(text, lang)
        if whole is None or whole:
            return whole
        parts = HYPHEN.split(text)
        if len(parts) == 1:
            return False
        judged = True
        for part in parts:
            if skip_reason(part) is not None or _is_code(part, code):
                continue
            known = speller.verdict(part, lang)
            if known is None:
                judged = False
            elif not known:
                return False
        return True if judged else None

    def result(self) -> Spelling:
        """Return the spelling of what was read, its drafts made."""
        limit = self.limit
        drafts: list[Draft] = []
        totals: Counter[str] = Counter()
        for prose, word, key, stem in self.found:
            count = self.stems[stem]
            rule = _rule(word.text, count)
            totals[rule] += 1
            if limit is None or totals[rule] <= limit:
                drafts.append(_draft(prose, word, rule, key, count))
        totals[_JARGON] = sum(
            count
            for stem, count in self.shown.items()
            if self.stems[stem] >= JARGON_MIN
        )
        cut = tuple(
            rule
            for rule in _RULES
            if limit is not None and totals[rule] > limit
        )
        listed = sorted(
            self.first.values(),
            key=lambda text: (fold(text), text),
        )
        return Spelling(
            drafts,
            tuple(listed),
            self.checked,
            self.unknown,
            cut,
            self.guarded,
        )


def _is_code(text: str, code: frozenset[str]) -> bool:
    """Return whether a word is code, whatever its case and ending.

    Args:
        text: The word.
        code: The code words of the document, as `fold` returns them.

    Returns:
        Whether the word, or its `bare` form, is one of them, without case.
    """
    return bool(code) and (fold(text) in code or fold(bare(text)) in code)


def _rule(text: str, count: int) -> str:
    """Return the rule of an occurrence of an unknown word.

    Args:
        text: The occurrence, as written.
        count: How many times the document has its stem.

    Returns:
        ``text.jargon`` from `JARGON_MIN` times; else ``text.unknown_name``
        if the first letter of its `bare` form is a capital (``d'Emacs``,
        not ``L'orblaxe``); else ``text.unknown_word``.
    """
    if count >= JARGON_MIN:
        return _JARGON
    if bare(text)[0].isupper():
        return _NAME
    return _UNKNOWN


def _draft(
    prose: Prose,
    word: Word,
    rule: str,
    key: str,
    count: int,
) -> Draft:
    """Return the draft of an occurrence of an unknown word.

    Args:
        prose: The block of the occurrence, of a language of
            `LANGUAGE_NAMES`.
        word: The occurrence.
        rule: Its rule, as `_rule` tells.
        key: The word without case, as `fold` returns it.
        count: How many times the document has its stem.

    Returns:
        The draft, its side marking the occurrence.
    """
    block = prose.block
    detail = (
        f'"{quote(word.text)}" is not in the '
        f"{LANGUAGE_NAMES[prose.lang]} dictionary"
    )
    if rule == _JARGON:
        detail += f"; it occurs {count} times"
    elif rule == _NAME:
        detail += "; it may be a name"
    side = marked_span(
        block.source,
        block.text,
        (word.start, word.end),
        kind=block.kind,
    )
    return Draft(rule, None, side, None, detail, quote(key))
