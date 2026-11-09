# Copyright 2026 Christophe Le Douarec
"""Check the text of a document: its spelling, typography and style.

The family `TEXT` reads the prose of a document, as `prose_blocks` of
`doc_checker.audit.prose` returns it, in the language of its text, as
`document_language` tells. It spells its words, by `spelling` of
`doc_checker.audit.spelling`; checks the typography of each block, by
`typography_marks` of `doc_checker.audit.typography`; then finds its words
written twice, its long sentences and its blocks in another language, by
`repeated_words`, `long_sentences` and `foreign_block` of
`doc_checker.audit.style`.

A document whose words the dictionaries mostly do not know is not in the
language it is checked in: once `GUARD_MIN_WORDS` words are spelled, if
more than `UNKNOWN_SHARE_MAX` of them are unknown, the findings of the
rules that depend on the language, `LANGUAGE_RULES`, are left out; a block
of mostly unknown words loses its spelling alone, as `spelling` tells, with
one warning for the document. Each rule gives at most
`TEXT_DRAFTS_PER_RULE` drafts, the first ones. What the checks could not do
becomes a warning, never an error.
"""

from collections import Counter
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Final

from doc_checker.audit.prose import (
    SENTENCE_KINDS,
    document_language,
    prose_blocks,
)
from doc_checker.audit.sides import block_side
from doc_checker.audit.spelling import (
    BLOCK_GUARD_MIN_WORDS,
    TEXT_DRAFTS_PER_RULE,
    Spelling,
    spelling,
)
from doc_checker.audit.style import (
    foreign_block,
    long_sentences,
    repeated_words,
)
from doc_checker.audit.typography import code_tokens, typography_marks
from doc_checker.compare.drafts import Draft, marked_span
from doc_checker.config import rule_disabled
from doc_checker.errors import DictionaryError
from doc_checker.spell.dictionary import DICTIONARY_LANGUAGES
from doc_checker.text.normalize import printable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from doc_checker.audit.prose import Mark, Prose
    from doc_checker.config import Config
    from doc_checker.model.document import Document
    from doc_checker.spell.speller import Speller

TEXT: Final = "text"
# The language guard: once this many words are spelled, a document of which
# more than this share of words is unknown is not in the language it is
# checked in (measured: Spanish 59 %, German 75 %, real documentation 2.2 %
# in the median, 10.2 % at the 90th percentile).
UNKNOWN_SHARE_MAX: Final = 0.25
GUARD_MIN_WORDS: Final = 200
# The rules whose findings depend on the language of the text.
LANGUAGE_RULES: Final = frozenset(
    {
        "text.unknown_word",
        "text.unknown_name",
        "text.jargon",
        "text.space_missing",
        "text.space_breakable",
        "text.space_before_punctuation",
        "text.quotes",
        "text.ordinal",
        "text.repeated_word",
        "text.block_language",
    },
)

_PREFIX: Final = "text checks: "
_FOREIGN: Final = "text.block_language"
_HTML: Final = "html"


@dataclass(frozen=True, slots=True)
class TextOutcome:
    """The checks of the text of a document.

    Attributes:
        drafts: The drafts of the spelling, then of the typography, of the
            repeated words, of the long sentences and of the blocks in
            another language, each in the order of the document.
        words: The unknown words, as `Spelling` lists them; none when the
            language guard left the spelling out.
        checked: The words spelled, as `Spelling.checked` counts them.
        warnings: What the checks could not do, each on one line.
    """

    drafts: list[Draft]
    words: tuple[str, ...]
    checked: int
    warnings: tuple[str, ...]


class _Kept:
    """The drafts of the checks of the text, the first ones of each rule.

    Attributes:
        drafts: The drafts kept, in the order they came.
        counts: The drafts that came for each rule, kept or not.
        cut: The rules that came with more than `TEXT_DRAFTS_PER_RULE`
            drafts, in the order they were cut.
    """

    __slots__ = ("counts", "cut", "drafts")

    def __init__(self) -> None:
        """Start with no draft."""
        self.drafts: list[Draft] = []
        self.counts: Counter[str] = Counter()
        self.cut: list[str] = []

    def add(self, prose: Prose, mark: Mark) -> None:
        """Keep the draft of a mark, unless its rule has enough drafts.

        Args:
            prose: The block of the mark.
            mark: The mark: its side marks its place, but the side of a
                block in another language shows the whole block.
        """
        rule = mark.rule
        self.counts[rule] += 1
        count = self.counts[rule]
        if count > TEXT_DRAFTS_PER_RULE:
            if count == TEXT_DRAFTS_PER_RULE + 1:
                self.cut.append(rule)
            return
        block = prose.block
        if rule == _FOREIGN:
            side = block_side(block)
        else:
            side = marked_span(
                block.source,
                block.text,
                mark.span,
                kind=block.kind,
            )
        self.drafts.append(
            Draft(rule, mark.severity, side, None, mark.detail, mark.pattern),
        )


class _Vouch:
    """Tell the words that a dictionary vouches for, failures noted.

    Attributes:
        speller: The speller.
        broken: Why a dictionary could not be loaded, each reason once.
    """

    __slots__ = ("broken", "speller")

    def __init__(self, speller: Speller, broken: dict[str, None]) -> None:
        """Remember the speller and where to note the failures.

        Args:
            speller: The speller.
            broken: The reasons noted so far, in order.
        """
        self.speller = speller
        self.broken = broken

    def known(self, word: str, lang: str) -> bool:
        """Return whether a dictionary or the personal words know a word.

        Args:
            word: The word.
            lang: The language of its block, one with a dictionary.

        Returns:
            Whether the speller knows the word: a word that it did not look
            up before its budget was spent, or whose dictionary cannot be
            loaded, is not vouched for.
        """
        try:
            return self.speller.known(word, lang, spent=False)
        except DictionaryError as exc:
            self.broken[str(exc)] = None
            return False


def text_drafts(doc: Document, cfg: Config, speller: Speller) -> TextOutcome:
    """Check the text of a document.

    Args:
        doc: The document.
        cfg: Run configuration: its language and its thresholds of
            sentences.
        speller: The speller, with the personal words.

    Returns:
        The drafts, at most `TEXT_DRAFTS_PER_RULE` of each rule, the
        unknown words, the words spelled, and the warnings: the language
        guard, a language without dictionary or unknown, the blocks of
        mostly unknown words, a dictionary that cannot be loaded, a spent
        budget, failed lookups, then each rule cut that the configuration
        does not disable, in that order.
    """
    lang = document_language(doc, cfg.lang)
    proses = prose_blocks(doc, lang)
    broken: dict[str, None] = {}
    try:
        spelled = spelling(proses, speller, code_words=doc.code_words)
    except DictionaryError as exc:
        broken[str(exc)] = None
        spelled = Spelling([], (), 0, 0)
    # The repeated words look up words of their own after the spelling: only
    # a budget that the spelling spent stopped it.
    stopped = speller.lookups if speller.exhausted else None
    guarded = (
        spelled.checked >= GUARD_MIN_WORDS
        and spelled.unknown / spelled.checked > UNKNOWN_SHARE_MAX
    )
    kept = _Kept()
    if not guarded:
        kept.drafts.extend(spelled.drafts)
        kept.cut.extend(spelled.cut)
    _typography(
        proses,
        kept,
        breakable=doc.format == _HTML,
        guarded=guarded,
        code=code_tokens(doc.code_words),
    )
    if not guarded:
        _repeated(proses, _Vouch(speller, broken), kept)
    _long(proses, cfg, kept)
    if not guarded:
        _foreign(proses, lang, kept)
    notes = _notes(spelled, lang, guarded=guarded, prose=bool(proses))
    notes.extend(
        f"{reason}; the checks that need it were left out" for reason in broken
    )
    notes.extend(_speller_notes(stopped, speller.failed))
    notes.extend(
        f"{rule}: only the first {TEXT_DRAFTS_PER_RULE} findings are reported"
        for rule in kept.cut
        if not rule_disabled(cfg, rule)
    )
    return TextOutcome(
        kept.drafts,
        () if guarded else spelled.words,
        spelled.checked,
        tuple(printable(_PREFIX + note) for note in notes),
    )


def _typography(
    proses: Sequence[Prose],
    kept: _Kept,
    *,
    breakable: bool,
    guarded: bool,
    code: frozenset[str],
) -> None:
    """Keep the marks of the typography of each block of prose.

    Args:
        proses: The prose of the document, in order.
        kept: The drafts kept.
        breakable: Whether the document tells a breakable space from a
            non-breaking one, as HTML does.
        guarded: Whether the language guard leaves out the rules of
            `LANGUAGE_RULES`.
        code: The tokens of the code words of the document, as
            `code_tokens` returns them.
    """
    for prose in proses:
        for mark in typography_marks(
            prose.block.text,
            prose.lang,
            prose.excluded,
            breakable=breakable,
            code=code,
        ):
            if not guarded or mark.rule not in LANGUAGE_RULES:
                kept.add(prose, mark)


def _repeated(proses: Sequence[Prose], vouch: _Vouch, kept: _Kept) -> None:
    """Keep the words written twice in a row in the blocks of a dictionary.

    Args:
        proses: The prose of the document, in order.
        vouch: Tells the words that a dictionary vouches for.
        kept: The drafts kept.
    """
    for prose in proses:
        if prose.lang not in DICTIONARY_LANGUAGES:
            continue
        for mark in repeated_words(
            prose.block.text,
            prose.lang,
            prose.excluded,
            partial(vouch.known, lang=prose.lang),
        ):
            kept.add(prose, mark)


def _long(proses: Sequence[Prose], cfg: Config, kept: _Kept) -> None:
    """Keep the long sentences of the blocks of sentences.

    Args:
        proses: The prose of the document, in order.
        cfg: Run configuration, with the thresholds of sentences.
        kept: The drafts kept.
    """
    minor = cfg.audit.long_sentence_minor
    major = cfg.audit.long_sentence_major
    for prose in proses:
        if prose.block.kind not in SENTENCE_KINDS:
            continue
        for mark in long_sentences(
            prose.block.text,
            prose.excluded,
            minor=minor,
            major=major,
        ):
            kept.add(prose, mark)


def _foreign(proses: Sequence[Prose], lang: str | None, kept: _Kept) -> None:
    """Keep the blocks that plainly read as another language.

    Args:
        proses: The prose of the document, in order.
        lang: The language of its text, if known.
        kept: The drafts kept.
    """
    for prose in proses:
        mark = foreign_block(prose, lang)
        if mark is not None:
            kept.add(prose, mark)


def _notes(
    spelled: Spelling,
    lang: str | None,
    *,
    guarded: bool,
    prose: bool,
) -> list[str]:
    """Return what the language left unchecked, without the prefix.

    Args:
        spelled: The spelling of the document.
        lang: The language of its text, if known.
        guarded: Whether the language guard left the checks out.
        prose: Whether the document has prose: without any, nothing was
            left unchecked.

    Returns:
        The note of the language guard, then that of a language without
        dictionary or unknown, then that of the blocks of mostly unknown
        words, unless the guard left all the spelling out.
    """
    notes: list[str] = []
    if guarded:
        share = spelled.unknown / spelled.checked
        notes.append(
            f"{share:.0%} of the {spelled.checked} words checked are unknown "
            "to the dictionaries; spelling and the checks that depend on the "
            "language were left out (is the language right? see --lang; or "
            "the document is in a language without dictionary: only French "
            "and English are spelled)",
        )
    if not prose:
        return notes
    only = (
        "the checks that depend on the language read only the blocks that "
        "plainly read as French or English"
    )
    if lang is None:
        notes.append(
            f"the language of the document is unknown; {only} (see --lang)",
        )
    elif lang not in DICTIONARY_LANGUAGES:
        notes.append(f"no dictionary for the language {lang!r}; {only}")
    if spelled.guarded and not guarded:
        blocks = (
            "1 block" if spelled.guarded == 1 else f"{spelled.guarded} blocks"
        )
        notes.append(
            f"{blocks} of {BLOCK_GUARD_MIN_WORDS} spelled words or more, more "
            "than half of them unknown, had their spelling left out: they may "
            "be in another language, or placeholder text",
        )
    return notes


def _speller_notes(stopped: int | None, failed: int) -> list[str]:
    """Return what the speller could not do, without the prefix.

    Args:
        stopped: The distinct words looked up when the budget stopped the
            spelling, or None if it did not.
        failed: The words that spylls failed on.

    Returns:
        The note of the budget, then that of the failed lookups.
    """
    notes: list[str] = []
    if stopped is not None:
        notes.append(
            f"spelling stopped after {stopped} distinct words; the rest was "
            "not spelled, nor checked for words written twice",
        )
    if failed:
        notes.append(
            f"{failed} word(s) could not be looked up and were taken as known",
        )
    return notes
