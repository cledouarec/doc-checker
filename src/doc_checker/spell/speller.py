# Copyright 2026 Christophe Le Douarec
"""The speller: tell the words that the dictionaries know.

A word is known if it is a personal word or the possessive of one,
whatever its case, or if a dictionary knows it: the dictionary of its
language, then the one of the other language, as technical French quotes
English and English quotes French. In French, a word after an elided word
(``l'endpoint``, ``d'upload``) is a personal word, or known in English,
when the word without its elided start is. In English, a possessive, a British
spelling and a word after a common prefix are known when the word they
leave is. In both languages, and in the other language too, a word cut by
a hyphen at the end of a line is known when the word without its hyphens
is.

Documents are untrusted, and a Hunspell lookup may be slow: the speller
looks each word up once, and stops after a work budget, where the words not
looked up yet are taken as known. A word that spylls fails on is taken as
known and counted.
"""

import re
from itertools import chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from doc_checker.errors import DictionaryError
from doc_checker.spell.dictionary import DICTIONARY_LANGUAGES, load_dictionary
from doc_checker.spell.words import (
    ELISIONS,
    LATIN_WORD,
    MAX_WORD_CHARS,
    MAX_WORD_PARTS,
    fold,
    without_possessive,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from spylls.hunspell import Dictionary

# The work budget of a speller. Each new lookup charges the characters of
# its word, at least `LOOKUP_MIN_COST`, times one plus twice its hyphens,
# times one plus its apostrophes: Hunspell tries every way of breaking a
# word at those. 400,000 is 2.5 times what the git manual needs (157,000,
# measured); a hostile document of distinct words spends it in at most
# about 25 s, whatever the shape of its words (measured).
SPELL_WORK: Final = 400_000
LOOKUP_MIN_COST: Final = 8
# The English prefixes after which a known word of three letters or more is
# a known word: "subcommand", "unbuffered".
EN_PREFIXES: Final = (
    "anti",
    "auto",
    "co",
    "de",
    "inter",
    "meta",
    "micro",
    "mis",
    "multi",
    "non",
    "out",
    "over",
    "post",
    "pre",
    "re",
    "semi",
    "sub",
    "super",
    "un",
    "under",
)
# The British endings and their American spelling, which the en_US
# dictionary knows.
BRITISH_ENDINGS: Final = (
    (re.compile(r"our(s?)$"), r"or\1"),
    (re.compile(r"our(ed|ing|ite|ites|able|er|ers)$"), r"or\1"),
    (re.compile(r"is(e|es|ed|ing|ation|ations|er|ers|able)$"), r"iz\1"),
    (re.compile(r"ys(e|es|ed|ing)$"), r"yz\1"),
    (re.compile(r"tre(s?)$"), r"ter\1"),
    (re.compile(r"ogue(s?)$"), r"og\1"),
    (re.compile(r"(def|off|lic|pret)ence(s?)$"), r"\1ense\2"),
    (re.compile(r"ll(ed|ing|er|ers)$"), r"l\1"),
    (re.compile(r"dgement(s?)$"), r"dgment\1"),
)

_OTHER_LANGUAGE: Final = MappingProxyType({"fr": "en", "en": "fr"})
_ENGLISH: Final = "en"
_FRENCH: Final = "fr"
_EN_PREFIX_REST_MIN: Final = 3
# The form a word is looked up in: straight apostrophes and hyphens.
_LOOKUP_FORM: Final = str.maketrans(
    {
        "\U00002019": "'",
        "\U000002bc": "'",
        "\U00002010": "-",
        "\U00002011": "-",
    },
)


def other_language(lang: str) -> str | None:
    """Return the other language with a dictionary.

    Args:
        lang: A language code.

    Returns:
        ``"en"`` for ``"fr"``, ``"fr"`` for ``"en"``, None otherwise.
    """
    return _OTHER_LANGUAGE.get(lang)


class Speller:
    """Tell the words that the dictionaries know, within a work budget.

    The dictionaries are loaded when a word is first looked up in them.
    """

    def __init__(
        self,
        words: Iterable[str] = (),
        *,
        work: int = SPELL_WORK,
    ) -> None:
        """Remember the personal words and the work budget.

        Args:
            words: The personal words, known whatever their case, and so
                are their possessives (``'s`` or a final apostrophe).
            work: The work budget, in characters of distinct words.
        """
        self._personal = frozenset(
            fold(word).translate(_LOOKUP_FORM) for word in words
        )
        self._work = work
        self._known: dict[tuple[str, str], bool] = {}
        self._dictionaries: dict[str, Dictionary] = {}
        self._broken: dict[str, str] = {}
        self._exhausted = False
        self._failed = 0

    @property
    def lookups(self) -> int:
        """The words looked up, each counted once per language."""
        return len(self._known)

    @property
    def exhausted(self) -> bool:
        """Whether the budget stopped the lookups."""
        return self._exhausted

    @property
    def failed(self) -> int:
        """The words that spylls failed on, taken as known."""
        return self._failed

    def known(self, word: str, lang: str) -> bool:
        """Return whether a word is known in a language.

        Typographic apostrophes and hyphens are read straight. A personal
        word or its possessive is known, and costs no lookup, and so is a
        French word whose `elided_rest` is one. A word that
        is not one, and is longer than `MAX_WORD_CHARS`, has
        more than `MAX_WORD_PARTS` parts or is not a `LATIN_WORD`, is
        unknown and never looked up: Hunspell may take seconds on it. A word
        looked up before gives the same answer. A new word is looked up if
        its charge fits in the work left: ``max(len(word),
        LOOKUP_MIN_COST)``, times ``1 + 2 * hyphens``, times ``1 +
        apostrophes``. Otherwise the budget is spent, and from then on every
        new word is taken as known. A dictionary that cannot be loaded
        raises `DictionaryError`, which names it.

        Args:
            word: The word, as the document writes it.
            lang: ``"fr"`` or ``"en"``.

        Returns:
            Whether the word is known.

        Raises:
            ValueError: If the language has no dictionary.
        """
        if lang not in DICTIONARY_LANGUAGES:
            msg = f"no dictionary for {lang!r}"
            raise ValueError(msg)
        word = word.translate(_LOOKUP_FORM)
        rest = elided_rest(word) if lang == _FRENCH else None
        if self._is_personal(fold(word)) or (
            rest is not None and self._is_personal(fold(rest))
        ):
            return True
        if (
            len(word) > MAX_WORD_CHARS
            or word.count("-") >= MAX_WORD_PARTS
            or LATIN_WORD.fullmatch(word) is None
        ):
            return False
        key = (lang, word)
        answer = self._known.get(key)
        if answer is not None:
            return answer
        if self._exhausted:
            return True
        charge = (
            max(len(word), LOOKUP_MIN_COST)
            * (1 + 2 * word.count("-"))
            * (1 + word.count("'"))
        )
        if charge > self._work:
            self._exhausted = True
            return True
        answer = self._search(word, lang)
        self._work -= charge
        self._known[key] = answer
        return answer

    def _is_personal(self, folded: str) -> bool:
        """Return whether a word is a personal word or its possessive.

        Args:
            folded: The word, as `fold` returns it.

        Returns:
            Whether the word, or the word without its ``'s`` or final
            apostrophe, is a personal word. No French word ends so: the
            possessive is read in both languages.
        """
        personal = self._personal
        return folded in personal or without_possessive(folded) in personal

    def _search(self, word: str, lang: str) -> bool:
        """Return whether a dictionary knows a form of a word.

        Args:
            word: The word, its apostrophes straight.
            lang: The language of the word.

        Returns:
            Whether a form of the word is known in its language, or the
            word in the other language; in French, the word without its
            elided start too, as `elided_rest` tells.
        """
        if any(
            self._found(form, lang) for form in _forms(word, lang, own=True)
        ):
            return True
        other = other_language(lang)
        if other is None:
            return False
        forms = _forms(word, other, own=False)
        rest = elided_rest(word) if lang == _FRENCH else None
        if rest is not None:
            forms = chain(forms, _forms(rest, other, own=False))
        return any(self._found(form, other) for form in forms)

    def _found(self, form: str, lang: str) -> bool:
        """Return whether the dictionary of a language knows a form.

        Args:
            form: The form looked up.
            lang: The language.

        Returns:
            Whether the form is known; True if spylls fails on it.
        """
        dictionary = self._dictionary(lang)
        try:
            return dictionary.lookup(form)
        except Exception:  # ruff: ignore[blind-except] - spylls may fail in any way on a word of an untrusted document; the word is then taken as known, and counted
            self._failed += 1
            return True

    def _dictionary(self, lang: str) -> Dictionary:
        """Return the dictionary of a language, loaded on first use.

        Args:
            lang: The language.

        Returns:
            The dictionary.

        Raises:
            DictionaryError: If the dictionary cannot be loaded; a failed
                load is not tried again.
        """
        dictionary = self._dictionaries.get(lang)
        if dictionary is not None:
            return dictionary
        reason = self._broken.get(lang)
        if reason is not None:
            raise DictionaryError(reason)
        # spylls may fail in any way on its data: every failure becomes the
        # one error that the caller turns into a warning.
        try:
            dictionary = load_dictionary(lang)
        except Exception as exc:
            reason = f"cannot load the {lang!r} dictionary: {_describe(exc)}"
            self._broken[lang] = reason
            raise DictionaryError(reason) from exc
        self._dictionaries[lang] = dictionary
        return dictionary


def elided_rest(word: str) -> str | None:
    """Return what follows a French elided word at the start of a word.

    Args:
        word: The word, its apostrophes straight.

    Returns:
        What follows the first apostrophe, if what comes before it is one
        of `ELISIONS`, whatever its case, and more follows: ``endpoint``
        for ``l'endpoint``; else None.
    """
    head, apostrophe, rest = word.partition("'")
    if apostrophe and rest and head.casefold() in ELISIONS:
        return rest
    return None


def _describe(exc: Exception) -> str:
    """Return the kind of an exception and its message."""
    message = str(exc)
    name = type(exc).__name__
    return f"{name}: {message}" if message else name


def _forms(word: str, lang: str, *, own: bool) -> Iterator[str]:
    """Yield the forms of a word to look up in a language, in order.

    Args:
        word: The word, its apostrophes straight.
        lang: The language of the dictionary.
        own: Whether it is the language of the word: only then are the
            `_english_forms` of an English word looked up.

    Yields:
        The word, then in its own English its `_english_forms`, then the
        word without its hyphens: a word cut at the end of a line may be a
        word of the other language.
    """
    yield word
    if lang == _ENGLISH and own:
        yield from _english_forms(word)
    if "-" in word:
        yield word.replace("-", "")


def _english_forms(word: str) -> Iterator[str]:
    """Yield the English forms of a word, in order.

    Args:
        word: The word, its apostrophes straight.

    Yields:
        The word without its possessive, its American spellings, and what
        follows each prefix of `EN_PREFIXES` that starts it, when that has
        three letters or more (a hyphen after the prefix is dropped).
    """
    base = without_possessive(word)
    if base != word:
        yield base
    for ending, american in BRITISH_ENDINGS:
        form = ending.sub(american, base, count=1)
        if form != base:
            yield form
    for prefix in EN_PREFIXES:
        if base[: len(prefix)].lower() == prefix:
            rest = base[len(prefix) :].removeprefix("-")
            if sum(char.isalpha() for char in rest) >= _EN_PREFIX_REST_MIN:
                yield rest
