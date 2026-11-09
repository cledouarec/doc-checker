# Copyright 2026 Christophe Le Douarec
"""Tests for the speller: dictionaries, personal words and bounds."""

import time
from typing import NoReturn

import pytest

from doc_checker.errors import DictionaryError
from doc_checker.spell import speller as speller_module
from doc_checker.spell.speller import Speller, other_language
from doc_checker.spell.words import (
    MAX_WORD_CHARS,
    MAX_WORD_PARTS,
    skip_reason,
    words,
)

APOSTROPHE = "\U00002019"
HYPHEN = "\U00002010"
NB_HYPHEN = "\U00002011"


class _FailingDictionary:
    """A dictionary whose every lookup fails."""

    @staticmethod
    def lookup(word: str) -> bool:
        """Fail, as spylls might on a hostile word."""
        raise IndexError(word)


def test_known_words_in_both_languages() -> None:
    """Each dictionary knows its words and not a misspelling."""
    speller = Speller()
    assert speller.known("répertoire", "fr")
    assert speller.known("receive", "en")
    assert not speller.known("répertoir", "fr")
    assert speller.known("l\U00002019arbre", "fr")
    assert speller.known("aujourd'hui", "fr")
    assert speller.known("c'est-\xe0-dire", "fr")
    assert speller.known("don\U000002bct", "en")


def test_personal_words_are_known_without_case() -> None:
    """A personal word is known whatever its case."""
    speller = Speller(["Kubectl"])
    assert speller.known("kubectl", "fr")
    assert speller.known("KUBECTL", "fr")
    assert speller.lookups == 0


def test_personal_words_cover_their_possessive() -> None:
    """A personal word also covers its possessive, in both languages."""
    speller = Speller(["Qwzzy"])
    assert speller.known("qwzzy's", "en")
    assert speller.known(f"Qwzzy{APOSTROPHE}s", "en")
    assert speller.known("qwzzy\U000002bcs", "fr")
    assert speller.known("qwzzy'", "en")
    assert speller.lookups == 0
    assert not speller.known("qwzzys", "en")
    assert not speller.known("qwzzyx's", "en")


def test_personal_words_read_typographic_apostrophes() -> None:
    """A personal word with a typographic apostrophe matches a straight one."""
    assert Speller([f"o{APOSTROPHE}xyzzy"]).known("O'xyzzy", "en")


def test_the_other_language_is_accepted() -> None:
    """An English word in French, a French word in English, are known."""
    speller = Speller()
    assert speller.known("authentication", "fr")
    assert speller.known("résumé", "en")
    assert other_language("fr") == "en"
    assert other_language("en") == "fr"
    assert other_language("de") is None


def test_french_elided_words_reach_personal_and_english_words() -> None:
    """In French, l'orbitctl is a personal word and l'upload English."""
    speller = Speller(["orbitctl"])
    for word in (
        "l'orbitctl",
        f"d{APOSTROPHE}orbitctl",
        "L'Orbitctl",
        "qu'orbitctl",
        "l'endpoint",
        "l'upload",
        f"d{APOSTROPHE}authentication",
        "l'override",
    ):
        assert speller.known(word, "fr"), word
    assert speller.lookups == 4
    for word in ("l'qwzzyx", "x'upload", "l'uplaod"):
        assert not speller.known(word, "fr"), word
    assert not speller.known("l'orbitctl", "en")
    assert not speller.known("l'upload", "en")
    assert not Speller().known("l'orbitctl", "fr")


def test_english_variants() -> None:
    """British spellings, possessives and prefixed words are English."""
    speller = Speller()
    for word in (
        "colour",
        "behaviour",
        "organise",
        "centre",
        "catalogue",
        "travelled",
        "user's",
        f"user{APOSTROPHE}s",
        "users'",
        "subcommand",
        "unbuffered",
        "preprocessing",
    ):
        assert speller.known(word, "en"), word
    for word in ("recieve", "teh"):
        assert not speller.known(word, "en"), word


def test_british_dgement_and_isable() -> None:
    """Acknowledgement, judgement and recognisable are British spellings."""
    speller = Speller()
    for word in (
        "acknowledgement",
        "acknowledgements",
        "judgement",
        "judgements",
        "abridgement",
        "recognisable",
        "realisable",
    ):
        assert speller.known(word, "en"), word
    for word in ("judgemint", "fledgement", "recognisible", "qwzzisable"):
        assert not speller.known(word, "en"), word


def test_only_four_british_ence_words_are_american_ense() -> None:
    """Defence, offence, licence, pretence; not expence or nonsence."""
    speller = Speller()
    for word in ("defence", "offences", "licence", "pretence"):
        assert speller.known(word, "en"), word
    for word in ("expence", "nonsence", "suspence", "intence"):
        assert not speller.known(word, "en"), word


def test_typographic_hyphens_are_hyphens() -> None:
    """U+2010 and U+2011 are looked up as hyphens, and share the cache."""
    speller = Speller()
    assert speller.known("peut-être", "fr")
    assert speller.known(f"peut{HYPHEN}être", "fr")
    assert speller.known(f"peut{NB_HYPHEN}être", "fr")
    assert speller.lookups == 1
    assert not speller.known(NB_HYPHEN.join(["ab"] * 6), "fr")
    assert speller.lookups == 1


def test_line_break_hyphens_and_prefixes_are_accepted() -> None:
    """A word cut by a hyphen at a line end is known without it."""
    speller = Speller()
    assert speller.known("connec-tion", "en")
    assert speller.known("doc-uments", "en")
    assert speller.known("pro-priétaire", "fr")
    assert not speller.known("sub-xqz", "en")
    assert speller.known("boun-ding", "fr")
    assert speller.known(f"wrap{HYPHEN}pers", "fr")


def test_hyphen_chains_and_long_words_are_not_spelled() -> None:
    """Too many parts or characters, and the chains cost no lookup."""
    assert skip_reason("-".join(["mot"] * 30)) == "long"
    assert skip_reason("-".join(["ab"] * 6)) == "parts"
    text = " ".join(["-".join(["mot"] * 30)] * 1_000)
    start = time.perf_counter()
    reasons = [skip_reason(word.text) for word in words(text)]
    assert time.perf_counter() - start < 1
    assert reasons == ["long"] * 1_000


def test_non_latin_words_are_not_spelled() -> None:
    """Chinese and Greek words are never looked up."""
    chinese = "\U00004e2d\U00006587"
    alpha, beta, gamma = "\U000003b1", "\U000003b2", "\U000003b3"
    delta = "\U000003b4\U000003ad\U000003bb\U000003c4\U000003b1"
    stop = "\U00003002"
    text = f"{chinese}{chinese} {chinese}{stop} {alpha}{beta}{gamma} {delta}."
    found = words(text)
    assert len(found) == 4
    assert all(skip_reason(word.text) == "symbol" for word in found)


def test_words_out_of_bounds_are_never_looked_up() -> None:
    """A word too long or with too many parts is unknown, and costs nothing."""
    speller = Speller(["-".join(["ab"] * 6)])
    start = time.perf_counter()
    assert not speller.known("a" * 200_000, "en")
    assert not speller.known("a" * (MAX_WORD_CHARS + 1), "fr")
    assert not speller.known("-".join(["mot"] * MAX_WORD_PARTS) + "-a", "fr")
    assert time.perf_counter() - start < 1
    assert speller.lookups == 0
    assert not speller.exhausted
    assert speller.known("-".join(["AB"] * 6), "en")
    assert not speller.known("a" * MAX_WORD_CHARS, "en")
    assert speller.lookups == 1


def test_spent_budget_takes_words_as_known() -> None:
    """Past the work budget, a new word is taken as known."""
    speller = Speller(work=20)
    assert not speller.known("abcdefghijkl", "en")
    assert not speller.exhausted
    assert speller.known("mnopqrstuvwx", "en")
    assert speller.exhausted
    assert speller.lookups == 1


def test_the_verdict_tells_a_word_not_looked_up() -> None:
    """Past the budget, a new word has no verdict; one looked up keeps it."""
    speller = Speller(["Orbitctl"], work=16)
    assert speller.verdict("maison", "fr") is True
    assert speller.verdict("qwzzy", "fr") is False
    assert not speller.exhausted
    assert speller.verdict("zorblax", "fr") is None
    assert speller.exhausted
    assert speller.verdict("maison", "fr") is True
    assert speller.verdict("qwzzy", "fr") is False
    assert speller.verdict("orbitctl", "fr") is True
    assert speller.verdict("a" * 200, "fr") is False
    assert speller.known("zorblax", "fr")
    assert not speller.known("zorblax", "fr", spent=False)
    assert speller.lookups == 2


def test_non_latin_shapes_are_never_looked_up() -> None:
    """A word that is not a Latin word is unknown, and costs nothing."""
    speller = Speller()
    start = time.perf_counter()
    for word in (
        "ab-'cd",
        "snake_case",
        "x86",
        f"'l'{APOSTROPHE}alle-'e-{APOSTROPHE}'{APOSTROPHE}eeee-lealee-'la",
        "\U000003b1\U000003b2\U000003b3",
    ):
        assert not speller.known(word, "fr"), word
    assert time.perf_counter() - start < 1
    assert speller.lookups == 0
    assert not speller.exhausted


def test_hyphenated_words_cost_more() -> None:
    """Each hyphen adds twice the length of the word to its charge."""
    speller = Speller(work=100)
    assert not speller.known("abcdefgh", "en")
    assert speller.known("ab-cd-ef-gh-ij", "en")
    assert speller.exhausted
    assert speller.lookups == 1


def test_apostrophes_cost_more() -> None:
    """Each apostrophe adds the charge of the word without apostrophes."""
    speller = Speller(work=60)
    assert not speller.known("abcdefgh", "en")
    assert speller.known("l'a-l'is'", "en")
    assert speller.exhausted
    assert speller.lookups == 1
    word = f"l{APOSTROPHE}a-l{APOSTROPHE}is{APOSTROPHE}"
    assert Speller(work=107).known(word, "fr")
    exact = Speller(work=108)
    exact.known(word, "fr")
    assert exact.lookups == 1
    assert not exact.exhausted


def test_spent_budget_stops_the_lookups_and_caches_nothing() -> None:
    """Once spent, the budget refuses even a word that would fit."""
    speller = Speller(work=12)
    assert not speller.known("xqzw", "en")
    assert speller.known("xqzwvbnmkj", "en")
    assert speller.exhausted
    assert speller.known("qzxw", "en")
    assert not speller.known("xqzw", "en")
    assert speller.lookups == 1


def test_a_spent_budget_may_refuse_the_words_not_looked_up() -> None:
    """With ``spent=False``, only words looked up in time are known."""
    speller = Speller(["mine"], work=20)
    assert speller.known("configuration", "en", spent=False)
    assert not speller.known("abcdefgh", "en", spent=False)
    assert speller.exhausted
    assert speller.known("configuration", "en", spent=False)
    assert not speller.known("library", "en", spent=False)
    assert speller.known("library", "en")
    assert speller.known("mine", "en", spent=False)
    assert speller.lookups == 1


def test_a_failing_lookup_takes_the_word_as_known(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A spylls failure on a word takes it as known and is counted."""
    monkeypatch.setattr(
        speller_module,
        "load_dictionary",
        lambda _: _FailingDictionary(),
    )
    speller = Speller()
    assert speller.known("anything", "fr")
    assert speller.known("anything", "fr")
    assert speller.failed == 1
    assert speller.lookups == 1


def test_a_word_taken_as_known_after_a_failure_is_doubtful(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The words that spylls failed on stay doubtful, and no other word."""
    monkeypatch.setattr(
        speller_module,
        "load_dictionary",
        lambda _: _FailingDictionary(),
    )
    speller = Speller(["mine"])
    assert not speller.doubtful("anything", "fr")
    assert speller.known("anything", "fr")
    assert speller.doubtful("anything", "fr")
    assert speller.known("anything", "fr")
    assert speller.doubtful("anything", "fr")
    assert not speller.doubtful("anything", "en")
    assert speller.known("mine", "fr")
    assert not speller.doubtful("mine", "fr")


def test_a_word_found_is_not_doubtful() -> None:
    """A word that the dictionary answered for is not doubtful."""
    speller = Speller()
    assert speller.known(f"don{APOSTROPHE}t", "en")
    assert not speller.known("zorblaxed", "en")
    assert not speller.doubtful("don't", "en")
    assert not speller.doubtful("zorblaxed", "en")


def test_a_dictionary_that_cannot_be_loaded_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A load failure is a DictionaryError, and is not tried again."""
    calls: list[str] = []

    def broken(lang: str) -> NoReturn:
        calls.append(lang)
        encoding, message = "utf-8", "truncated affix file"
        raise UnicodeDecodeError(encoding, b"\xff", 0, 1, message)

    monkeypatch.setattr(speller_module, "load_dictionary", broken)
    speller = Speller()
    with pytest.raises(DictionaryError, match="cannot load the 'en' dict"):
        speller.known("word", "en")
    with pytest.raises(DictionaryError, match="truncated affix file"):
        speller.known("other", "en")
    assert calls == ["en"]
    assert speller.lookups == 0


def test_dictionaries_are_loaded_only_when_a_word_is_looked_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Personal words, cached words and a spent budget load nothing."""
    calls: list[str] = []

    def counted(lang: str) -> _FailingDictionary:
        calls.append(lang)
        return _FailingDictionary()

    monkeypatch.setattr(speller_module, "load_dictionary", counted)
    speller = Speller(["mine"], work=8)
    assert speller.known("mine", "en")
    assert speller.known("longerthanthebudget", "en")
    assert calls == []


def test_lookups_are_cached() -> None:
    """The same word is looked up once."""
    speller = Speller()
    assert speller.known("configuration", "en")
    assert speller.known("configuration", "en")
    assert speller.lookups == 1
    assert speller.known(f"don{APOSTROPHE}t", "en")
    assert speller.known("don't", "en")
    assert speller.lookups == 2


def test_only_french_and_english() -> None:
    """A language without dictionary is an error, even for personal words."""
    with pytest.raises(ValueError, match="no dictionary for 'de'"):
        Speller().known("x", "de")
    with pytest.raises(ValueError, match="no dictionary for 'de'"):
        Speller(["x"]).known("x", "de")
