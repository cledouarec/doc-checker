# Copyright 2026 Christophe Le Douarec
"""Tests for the spelling of the audit: unknown words, names and jargon."""

import random
import string
import time
import tracemalloc
from typing import cast

from doc_checker.audit.prose import Prose, excluded_spans
from doc_checker.audit.spelling import (
    BLOCK_GUARD_MIN_WORDS,
    BLOCK_UNKNOWN_SHARE_MAX,
    ELISIONS,
    JARGON_MIN,
    TEXT_DRAFTS_PER_RULE,
    bare,
    spelling,
)
from doc_checker.model.findings import DiffSegment
from doc_checker.spell.speller import Speller
from tests.builders import make_block

APOSTROPHE = "\U00002019"


def _english(*texts: str) -> list[Prose]:
    """Return blocks of English prose, their excluded parts found."""
    blocks = [make_block(text) for text in texts]
    return [
        Prose(block, "en", None, excluded_spans(block)) for block in blocks
    ]


def test_unknown_words_names_and_jargon() -> None:
    """A rare word is minor, a capitalized or frequent one is info."""
    blocks = [
        make_block("The installer donwloads the files."),
        make_block("Zorblax runs frobnicate, then frobnicate."),
        make_block("Frobnicate again before you start."),
    ]
    proses = [Prose(b, "en", None, ()) for b in blocks]
    result = spelling(proses, Speller())
    assert [(d.rule, d.pattern) for d in result.drafts] == [
        ("text.unknown_word", "donwloads"),
        ("text.unknown_name", "zorblax"),
        ("text.jargon", "frobnicate"),
        ("text.jargon", "frobnicate"),
        ("text.jargon", "frobnicate"),
    ]
    assert result.words == ("donwloads", "frobnicate", "Zorblax")
    assert result.drafts[0].ref.segments[1] == DiffSegment("mark", "donwloads")


def test_details_name_the_language_and_the_count() -> None:
    """Each detail quotes the word as written, in the block's language."""
    result = spelling(
        _english(
            "Zorblax will recieve it.",
            "Frobnicate, frobnicate and frobnicate.",
        ),
        Speller(),
    )
    assert [d.detail for d in result.drafts] == [
        '"Zorblax" is not in the English dictionary; it may be a name',
        '"recieve" is not in the English dictionary',
        '"Frobnicate" is not in the English dictionary; it occurs 3 times',
        '"frobnicate" is not in the English dictionary; it occurs 3 times',
        '"frobnicate" is not in the English dictionary; it occurs 3 times',
    ]
    assert all(d.severity is None and d.new is None for d in result.drafts)
    assert result.checked == 8
    assert result.unknown == 5
    french = make_block("Le fichier est recieved.")
    (draft,) = spelling([Prose(french, "fr", "fr", ())], Speller()).drafts
    assert draft.detail == '"recieved" is not in the French dictionary'


def test_jargon_starts_at_its_minimum() -> None:
    """A word seen fewer than `JARGON_MIN` times keeps its own rule."""
    assert JARGON_MIN == 3
    twice = spelling(_english("Recieve it, then recieve it."), Speller())
    assert [(d.rule, d.pattern) for d in twice.drafts] == [
        ("text.unknown_name", "recieve"),
        ("text.unknown_word", "recieve"),
    ]
    thrice = spelling(
        _english("Recieve it, then recieve it.", "Do not recieve it."),
        Speller(),
    )
    assert {d.rule for d in thrice.drafts} == {"text.jargon"}
    assert len(thrice.drafts) == JARGON_MIN


def test_each_block_is_spelled_in_its_language() -> None:
    """A block is looked up in the language of its prose."""
    french = Prose(make_block("Le répertoire est là."), "fr", "fr", ())
    english = Prose(make_block("The répertoir is here."), "en", None, ())
    assert spelling([french], Speller()).drafts == []
    (draft,) = spelling([english], Speller()).drafts
    assert (draft.rule, draft.pattern) == ("text.unknown_word", "répertoir")


def test_personal_words_are_not_reported() -> None:
    """A personal word is known, so it is no finding and not listed."""
    result = spelling(
        _english("The installer donwloads the files."),
        Speller(["donwloads"]),
    )
    assert result.drafts == []
    assert result.words == ()
    assert result.unknown == 0
    assert result.checked == 5
    possessive = spelling(
        _english(f"Run qwzzy's job and Qwzzy{APOSTROPHE}s log."),
        Speller(["qwzzy"]),
    )
    assert possessive.drafts == []
    assert possessive.words == ()


def test_code_and_skipped_words_are_not_checked() -> None:
    """Excluded parts and skipped words are neither reported nor counted."""
    block = make_block("Run donwloads with GitHub on x86_64 via the API.")
    start = block.text.index("donwloads")
    prose = Prose(block, "en", None, ((start, start + len("donwloads")),))
    result = spelling([prose], Speller())
    assert result.drafts == []
    assert result.checked == 5
    assert result.unknown == 0
    marked = _english("Call `frobnicate` or https://zorblax.example/qwz now.")
    assert spelling(marked, Speller()).drafts == []


def test_code_words_are_exempt_whatever_their_case_and_ending() -> None:
    """A word that the document sets as code is no unknown word."""
    proses = _english(
        f"Run tlmgr first; Tlmgr, tlmgr{APOSTROPHE}s log and pgf's docs.",
        "Then tlmgrx and Qwzzy run.",
    )
    code = frozenset({"tlmgr", "Pgf", "qwzzy"})
    result = spelling(proses, Speller(), code_words=code)
    assert [(d.rule, d.pattern) for d in result.drafts] == [
        ("text.unknown_word", "tlmgrx"),
    ]
    assert result.words == ("tlmgrx",)
    assert result.checked == 14
    assert result.unknown == 1
    french = make_block(f"Le chargement d{APOSTROPHE}epstopdf et l'epstopdfx.")
    prose = Prose(french, "fr", "fr", ())
    elided = spelling([prose], Speller(), code_words=frozenset({"epstopdf"}))
    assert [d.pattern for d in elided.drafts] == ["l'epstopdfx"]


def test_words_not_looked_up_are_not_counted() -> None:
    """Past the budget, a word that was not looked up was not spelled."""
    block = make_block("maison qwzzy zorblax maison qwzzy-maison zorblax.")
    prose = Prose(block, "fr", "fr", ())
    result = spelling([prose], Speller(work=16))
    assert (result.checked, result.unknown) == (3, 1)
    assert [d.pattern for d in result.drafts] == ["qwzzy"]
    full = spelling([prose], Speller())
    assert (full.checked, full.unknown) == (6, 4)
    compound = Prose(make_block("qwzzy-zorblax."), "fr", "fr", ())
    unread_part = spelling([compound], Speller(work=44))
    assert (unread_part.checked, unread_part.drafts) == (0, [])
    unknown_part = spelling([compound], Speller(work=47))
    assert (unknown_part.checked, unknown_part.unknown) == (1, 1)


def _mixed(known: int, unknown: int, start: int = 0) -> str:
    """Return a text of known English words, then distinct unknown ones."""
    words = ["file", "agent", "folder", "reads", "keeps", "tool"] * 10
    odd = [f"qwz{a}{b}" for a in "bcdfghjklm" for b in "npqrstvxyz"]
    return " ".join(words[:known] + odd[start : start + unknown]) + "."


def test_a_block_of_mostly_unknown_words_is_not_reported() -> None:
    """Past half unknown words in 20 or more, a block is no prose to spell."""
    assert (BLOCK_GUARD_MIN_WORDS, BLOCK_UNKNOWN_SHARE_MAX) == (20, 0.5)
    lorem = _mixed(9, 11, start=50)
    proses = _english(lorem, _mixed(10, 10), _mixed(0, 19))
    result = spelling(proses, Speller())
    assert result.guarded == 1
    assert (result.checked, result.unknown) == (59, 40)
    assert {d.ref.text for d in result.drafts} == {
        _mixed(10, 10),
        _mixed(0, 19),
    }
    assert len(result.drafts) == 29
    assert "qwzhn" in lorem
    assert "qwzhn" not in result.words
    assert "qwzbn" in result.words


def test_guarded_blocks_count_towards_the_jargon() -> None:
    """A word of a guarded block is counted, though neither shown nor listed.

    Placeholder words seen in a short block too are the vocabulary of the
    document, as package names listed in a table are.
    """
    lorem = _mixed(0, 25)
    proses = _english(lorem, lorem, "The qwzbn tool.", "The qwzzy tool.")
    result = spelling(proses, Speller(), limit=1)
    assert result.guarded == 2
    assert [(d.rule, d.pattern) for d in result.drafts] == [
        ("text.jargon", "qwzbn"),
        ("text.unknown_word", "qwzzy"),
    ]
    assert result.drafts[0].detail.endswith("it occurs 3 times")
    assert result.words == ("qwzbn", "qwzzy")
    assert result.cut == ()


def test_bare_takes_off_elisions_and_possessives() -> None:
    """Only a known elided word and a possessive leave a word."""
    assert {"l", "d", "qu", "jusqu"} <= ELISIONS
    cases = {
        f"d{APOSTROPHE}Emacs": "Emacs",
        "Qu'il": "il",
        f"tlmgr{APOSTROPHE}s": "tlmgr",
        "users'": "users",
        "l'agent's": "agent",
        "qu'": "qu",
        "s'": "s",
        f"aujourd{APOSTROPHE}hui": "aujourd'hui",
        "O'Reilly": "O'Reilly",
        "rock'n'roll": "rock'n'roll",
        "zorblax": "zorblax",
    }
    assert {word: bare(word) for word in cases} == cases


def test_an_elided_name_is_a_name() -> None:
    """The capital of a name after an elided word makes it a name."""
    block = make_block(
        f"Le module d{APOSTROPHE}Orblax lit l{APOSTROPHE}orblaxe.",
    )
    result = spelling([Prose(block, "fr", "fr", ())], Speller())
    assert [(d.rule, d.pattern) for d in result.drafts] == [
        ("text.unknown_name", "d'orblax"),
        ("text.unknown_word", "l'orblaxe"),
    ]
    assert result.drafts[0].detail == (
        f'"d{APOSTROPHE}Orblax" is not in the French dictionary; '
        "it may be a name"
    )
    english = spelling(_english("Run qwzzy's job and Qwzzy's job."), Speller())
    assert [d.rule for d in english.drafts] == [
        "text.unknown_word",
        "text.unknown_name",
    ]


def test_elided_personal_and_english_words_are_known_in_french() -> None:
    """L'orbitctl, a personal word, and l'upload, English, are no slips."""
    block = make_block(
        f"Ouvrez l'orbitctl, puis lancez l{APOSTROPHE}upload de "
        "l'endpoint et d'orblaxe.",
    )
    prose = Prose(block, "fr", "fr", ())
    result = spelling([prose], Speller(["orbitctl"]))
    assert [d.pattern for d in result.drafts] == ["d'orblaxe"]
    assert result.words == ("d'orblaxe",)
    assert [d.pattern for d in spelling([prose], Speller()).drafts] == [
        "l'orbitctl",
        "d'orblaxe",
    ]


def test_a_capital_on_an_elided_word_makes_no_name() -> None:
    """``L'orblaxe`` starts a sentence; its word is in lower case."""
    block = make_block(f"L{APOSTROPHE}orblaxe lit le module.")
    (draft,) = spelling([Prose(block, "fr", "fr", ())], Speller()).drafts
    assert (draft.rule, draft.pattern) == ("text.unknown_word", "l'orblaxe")


def test_a_compound_of_code_and_known_words_is_known() -> None:
    """A hyphenated word is known when each of its parts is."""
    proses = _english(
        "The pgf-system, the svg-format and a qwzzy-based setup.",
        "A config-based remote, an x-axis, an API-level call.",
        "But pgf-sytem and confg-based are slips.",
    )
    code = frozenset({"pgf", "svg"})
    result = spelling(proses, Speller(["qwzzy"]), code_words=code)
    assert [(d.rule, d.pattern) for d in result.drafts] == [
        ("text.unknown_word", "pgf-sytem"),
        ("text.unknown_word", "confg-based"),
    ]
    assert result.words == ("confg-based", "pgf-sytem")


def test_a_possessive_counts_with_its_word() -> None:
    """A possessive of an unknown word adds to the count of its word."""
    proses = _english(
        f"Frobnicate, frobnicate, frobnicate and frobnicate{APOSTROPHE}s.",
    )
    result = spelling(proses, Speller())
    assert [d.rule for d in result.drafts] == ["text.jargon"] * 4
    assert result.drafts[-1].pattern == "frobnicate's"
    assert result.drafts[-1].detail == (
        f'"frobnicate{APOSTROPHE}s" is not in the English dictionary; '
        "it occurs 4 times"
    )
    assert result.words == ("Frobnicate", f"frobnicate{APOSTROPHE}s")
    french = make_block(
        f"L{APOSTROPHE}orblaxe, d{APOSTROPHE}orblaxe et orblaxe.",
    )
    elided = spelling([Prose(french, "fr", "fr", ())], Speller())
    assert {d.rule for d in elided.drafts} == {"text.jargon"}


def test_a_word_is_judged_in_the_language_of_each_block() -> None:
    """The same word may be known in one block and unknown in another."""
    french = Prose(make_block("Les colours du thème."), "fr", "fr", ())
    english = Prose(make_block("The colours of the theme."), "en", None, ())
    for proses in ([french, english], [english, french]):
        (draft,) = spelling(proses, Speller()).drafts
        assert draft.ref.segments[0] == DiffSegment("equal", "Les ")
        assert draft.pattern == "colours"


def test_drafts_are_limited_by_default() -> None:
    """Without a limit given, each rule keeps the first drafts only."""
    assert TEXT_DRAFTS_PER_RULE == 10_000
    # Blocks of a few words: a block of mostly unknown words gives none.
    proses = _english(*["qwzzy " * 10] * (TEXT_DRAFTS_PER_RULE // 10), "qwzzy")
    result = spelling(proses, Speller())
    assert len(result.drafts) == TEXT_DRAFTS_PER_RULE
    assert result.cut == ("text.jargon",)
    assert len(spelling(proses, Speller(), limit=None).drafts) == (
        TEXT_DRAFTS_PER_RULE + 1
    )


def test_the_verdict_memo_is_bounded() -> None:
    """Skipped words are not remembered, nor countless distinct words."""
    long_words = [
        " ".join(
            f"{'x' * 1_990}{block:02d}{index:04d}" for index in range(100)
        )
        for block in range(20)
    ]
    letters = string.ascii_lowercase
    distinct = [
        " ".join(
            "".join(letters[(n // 26**k) % 26] for k in range(6)) * 5
            for n in range(block * 4_000, (block + 1) * 4_000)
        )
        for block in range(20)
    ]
    proses = _english(*long_words, *distinct)

    class Knowing:
        """A speller that knows every word without looking it up."""

        @staticmethod
        def verdict(word: str, lang: str) -> bool:
            """Know the word."""
            del word, lang
            return True

    tracemalloc.start()
    try:
        result = spelling(proses, cast("Speller", Knowing()))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result.checked == 80_000
    assert peak < 5_000_000


def test_blocks_without_language_are_not_spelled() -> None:
    """A block in no language, or one without dictionary, is not spelled."""
    block = make_block("The installer donwloads the files.")
    for lang in (None, "de"):
        result = spelling([Prose(block, lang, None, ())], Speller())
        assert result.drafts == []
        assert result.checked == 0
        assert result.words == ()


def test_the_word_list_keeps_the_first_spelling() -> None:
    """The word list has each word once, as it was first written."""
    result = spelling(_english("Recieve it.", "Then recieve it."), Speller())
    assert result.words == ("Recieve",)
    mixed = spelling(
        _english("Zorblax and qwzzy.", "Then zorblax and Qwzzy."),
        Speller(),
    )
    assert mixed.words == ("qwzzy", "Zorblax")


def test_marks_point_at_each_occurrence() -> None:
    """Each draft marks its own occurrence, with the text around it."""
    result = spelling(_english("One qwzzy, two qwzzy."), Speller())
    first, second = (draft.ref.segments for draft in result.drafts)
    assert first == (
        DiffSegment("equal", "One "),
        DiffSegment("mark", "qwzzy"),
        DiffSegment("equal", ", two qwzzy."),
    )
    assert second == (
        DiffSegment("equal", "One qwzzy, two "),
        DiffSegment("mark", "qwzzy"),
        DiffSegment("equal", "."),
    )


def test_a_limit_keeps_the_first_drafts_of_each_rule() -> None:
    """Past the limit a rule gives no more drafts, but every word counts."""
    proses = _english(
        "Zorblax, Qwzzy and Plorf; brrk, drrk, frrk.",
        "Glorp glorp glorp glorp.",
    )
    result = spelling(proses, Speller(), limit=2)
    assert [(d.rule, d.pattern) for d in result.drafts] == [
        ("text.unknown_name", "zorblax"),
        ("text.unknown_name", "qwzzy"),
        ("text.unknown_word", "brrk"),
        ("text.unknown_word", "drrk"),
        ("text.jargon", "glorp"),
        ("text.jargon", "glorp"),
    ]
    assert result.cut == (
        "text.unknown_word",
        "text.unknown_name",
        "text.jargon",
    )
    assert result.unknown == 10
    assert result.words == (
        "brrk",
        "drrk",
        "frrk",
        "Glorp",
        "Plorf",
        "Qwzzy",
        "Zorblax",
    )
    assert spelling(proses, Speller()).cut == ()


def test_a_limit_keeps_the_first_jargon_whatever_comes_later() -> None:
    """Jargon past the limit is dropped, yet counted in each detail."""
    proses = _english("Glorp glorp glorp glorp, plorf plorf plorf; brrk.")
    result = spelling(proses, Speller(), limit=1)
    assert [(d.rule, d.pattern) for d in result.drafts] == [
        ("text.jargon", "glorp"),
        ("text.unknown_word", "brrk"),
    ]
    assert result.drafts[0].ref.segments[0] == DiffSegment("mark", "Glorp")
    assert result.drafts[0].detail.endswith("it occurs 4 times")
    assert result.cut == ("text.jargon",)
    assert result.unknown == 8
    assert result.words == ("brrk", "Glorp", "plorf")


def test_spelling_stays_bounded_on_hostile_text() -> None:
    """More distinct words than the memo holds stay within the budget."""
    rng = random.Random(0)  # ruff: ignore[suspicious-non-cryptographic-random-usage] - test words, not security
    distinct = " ".join(
        "".join(rng.choices(string.ascii_lowercase, k=12))
        for _ in range(80_000)
    )
    # Half the words of each block are known, so that no block is one of
    # mostly unknown words, which gives no draft.
    repeated = "qwzzy the " * 100_000
    proses = _english(repeated, distinct.replace(" ", " the "), repeated)
    speller = Speller(work=10_000)
    start = time.perf_counter()
    result = spelling(proses, speller, limit=1_000)
    assert time.perf_counter() - start < 30
    assert speller.exhausted
    assert result.guarded == 0
    # The words left without a verdict by the spent budget are not counted.
    looked_up = speller.lookups - 2
    assert result.checked == 400_000 + 79_999 + looked_up
    assert result.unknown == 200_000 + looked_up
    assert len(result.drafts) == 1_000 + looked_up
    assert result.drafts[0].pattern == "qwzzy"
    assert result.drafts[0].detail.endswith("it occurs 200000 times")
