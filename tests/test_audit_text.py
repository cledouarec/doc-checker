# Copyright 2026 Christophe Le Douarec
"""Tests for the text family of the audit: drafts, language guard, warnings."""

import random
import string
import time
from dataclasses import replace
from typing import TYPE_CHECKING, NoReturn

from doc_checker.audit.spelling import TEXT_DRAFTS_PER_RULE
from doc_checker.audit.text import (
    GUARD_MIN_WORDS,
    LANGUAGE_RULES,
    TEXT,
    UNKNOWN_SHARE_MAX,
    text_drafts,
)
from doc_checker.config import Config
from doc_checker.model.document import BlockKind
from doc_checker.model.findings import Severity
from doc_checker.spell import speller as speller_module
from doc_checker.spell.dictionary import load_dictionary
from doc_checker.spell.speller import Speller
from tests.builders import make_audit_document, make_block, sentence

if TYPE_CHECKING:
    import pytest

    from doc_checker.audit.text import TextOutcome
    from doc_checker.model.document import Block, Document

CONFIG = Config()
SPANISH = (
    "La instalación del paquete requiere espacio libre y una conexión "
    "estable para descargar los archivos."
)
# A Spanish sentence with a mark of each rule of the typography and a
# repeated word, as French reads it.
SPANISH_MARKED = (
    'Necesita 4GB : la la conexión "rápida" es 2ème opción,, sin duda.'
)
GUARD_NOTE = "unknown to the dictionaries"
# Fifteen regular French verbs and twenty of their forms: 300 known words.
FRENCH_STEMS = (
    "parl",
    "donn",
    "cherch",
    "trouv",
    "montr",
    "gard",
    "port",
    "mont",
    "tourn",
    "regard",
    "ajout",
    "ferm",
    "entr",
    "demand",
    "pass",
)
FRENCH_ENDINGS = (
    "er",
    "e",
    "es",
    "ons",
    "ez",
    "ent",
    "ais",
    "ait",
    "ions",
    "iez",
    "aient",
    "erai",
    "eras",
    "era",
    "erons",
    "erez",
    "eront",
    "é",
    "ée",
    "és",
)


def _french(*blocks: Block, fmt: str = "html") -> TextOutcome:
    """Return the text checks of a French document of the given blocks."""
    doc = make_audit_document(*blocks, fmt=fmt, lang="fr", declared_lang="fr")
    return text_drafts(doc, CONFIG, Speller())


def test_the_family_and_its_guard() -> None:
    """The family is named ``text``; its guard has the measured bounds."""
    assert TEXT == "text"
    assert (GUARD_MIN_WORDS, UNKNOWN_SHARE_MAX) == (200, 0.25)
    assert "text.unit_space" not in LANGUAGE_RULES
    assert "text.long_sentence" not in LANGUAGE_RULES
    assert len(LANGUAGE_RULES) == 10


def test_text_drafts_in_rule_order() -> None:
    """Spelling, typography, repetitions, then long sentences, each marked."""
    doc = make_audit_document(
        make_block("The installer donwloads the files."),
        make_block(
            "Conflicts are rare : when the agent reads the the file, it "
            "needs 4GB.",
        ),
        make_block(sentence(45)),
    )
    outcome = text_drafts(doc, CONFIG, Speller())
    assert [d.rule for d in outcome.drafts] == [
        "text.unknown_word",
        "text.space_before_punctuation",
        "text.unit_space",
        "text.repeated_word",
        "text.long_sentence",
    ]
    for draft in outcome.drafts:
        assert draft.new is None
        assert [s.op for s in draft.ref.segments].count("mark") == 1
    marked = [
        next(s.text for s in d.ref.segments if s.op == "mark")
        for d in outcome.drafts
    ]
    assert marked[:4] == ["donwloads", " :", "4GB", "the the"]
    assert outcome.drafts[4].severity is not None
    assert outcome.words == ("donwloads",)
    assert outcome.checked == 62
    assert outcome.warnings == ()


def test_breakable_spaces_only_in_html() -> None:
    """A breakable space before a colon is seen in HTML, not in a PDF."""
    for fmt, expected in (("html", 1), ("pdf", 0)):
        outcome = _french(make_block("Rares : lorsque"), fmt=fmt)
        rules = [d.rule for d in outcome.drafts]
        assert rules.count("text.space_breakable") == expected, fmt


def test_unknown_words_beyond_the_share_turn_the_language_checks_off() -> None:
    """A French document of Spanish text keeps only language-free checks."""
    outcome = _french(
        make_block(" ".join([SPANISH] * 20 + [SPANISH_MARKED])),
        make_block(sentence(70)),
    )
    assert outcome.checked == 359
    assert outcome.checked >= GUARD_MIN_WORDS
    assert not [d for d in outcome.drafts if d.rule in LANGUAGE_RULES]
    assert [d.rule for d in outcome.drafts] == [
        "text.unit_space",
        "text.double_punctuation",
        "text.long_sentence",
    ]
    assert outcome.words == ()
    (warning,) = outcome.warnings
    assert warning.startswith("text checks: ")
    assert (
        "of the 359 words checked are unknown to the dictionaries" in warning
    )
    assert warning.endswith(
        "; spelling and the checks that depend on the language were left "
        "out (is the language right? see --lang; or the document is in a "
        "language without dictionary: only French and English are spelled)",
    )


def test_the_guard_only_removes_what_it_would_find() -> None:
    """Without the guard, the marked sentence gives every rule it holds."""
    rules = {d.rule for d in _french(make_block(SPANISH_MARKED)).drafts}
    assert rules - {"text.unknown_word", "text.unknown_name"} == {
        "text.space_breakable",
        "text.quotes",
        "text.ordinal",
        "text.repeated_word",
        "text.unit_space",
        "text.double_punctuation",
    }
    assert "text.unknown_word" in rules


def test_the_guard_cuts_no_rule_it_removes() -> None:
    """A rule that the guard removes is not said to be cut."""
    over = TEXT_DRAFTS_PER_RULE + 5
    outcome = _french(
        make_block("a: " * over),
        make_block(" ".join(["qwzzy"] * over)),
    )
    assert outcome.drafts == []
    (warning,) = outcome.warnings
    assert GUARD_NOTE in warning


def test_the_bounds_of_the_guard() -> None:
    """From 200 words checked, more than a quarter unknown turns it on."""

    def guarded(known: int, unknown: int) -> tuple[int, bool]:
        """Return the words checked and whether the guard turned on.

        The words come in blocks of ten, too few for the guard of a block.
        """
        words = ["maison."] * known + ["qwzzy."] * unknown
        outcome = _french(
            *(
                make_block(" ".join(words[start : start + 10]))
                for start in range(0, len(words), 10)
            ),
        )
        on = any(GUARD_NOTE in warning for warning in outcome.warnings)
        assert on == (outcome.words == ())
        return outcome.checked, on

    assert guarded(150, 50) == (200, False)
    assert guarded(149, 51) == (200, True)
    assert guarded(0, 199) == (199, False)


def test_code_words_of_a_pdf_are_not_spelled() -> None:
    """A word that the PDF sets in a monospaced font is no unknown word."""
    doc = make_audit_document(
        make_block("Run the tlmgrx tool now."),
        fmt="pdf",
    )

    def found(doc: Document) -> list[tuple[str, str | None]]:
        """Return the rules and patterns of the text checks of a document."""
        drafts = text_drafts(doc, CONFIG, Speller()).drafts
        return [(d.rule, d.pattern) for d in drafts]

    assert found(doc) == [("text.unknown_word", "tlmgrx")]
    assert found(replace(doc, code_words=frozenset({"tlmgrx"}))) == []


def test_typography_of_a_pdf_spares_its_code_words() -> None:
    """A unit or a punctuation mark of PDF code is no typography finding.

    A repeated word is still reported: common words such as ``the`` are
    set in monospace somewhere in most manuals.
    """
    doc = make_audit_document(
        make_block(
            "Set it to 1cm with the the key xshift , then use 4GB of memory.",
        ),
        fmt="pdf",
    )

    def found(doc: Document) -> list[tuple[str, str | None]]:
        """Return the rules and patterns of the text checks of a document."""
        drafts = text_drafts(doc, CONFIG, Speller()).drafts
        return [(d.rule, d.pattern) for d in drafts]

    assert found(doc) == [
        ("text.unknown_word", "xshift"),
        ("text.unit_space", "cm"),
        ("text.space_before_punctuation", ","),
        ("text.unit_space", "GB"),
        ("text.repeated_word", "the"),
    ]
    code = frozenset({"\\pgfkeys{xshift=1cm}", "the"})
    assert found(replace(doc, code_words=code)) == [
        ("text.unknown_word", "xshift"),
        ("text.unit_space", "GB"),
        ("text.repeated_word", "the"),
    ]


def test_long_sentences_follow_the_thresholds_and_skip_tables() -> None:
    """The thresholds come from the configuration; tables have none."""
    audit = replace(
        CONFIG.audit,
        long_sentence_minor=5,
        long_sentence_major=10,
    )
    doc = make_audit_document(
        make_block(sentence(8)),
        make_block(sentence(12)),
        make_block(sentence(45), BlockKind.TABLE),
        make_block(sentence(45), BlockKind.HEADING),
    )
    outcome = text_drafts(doc, replace(CONFIG, audit=audit), Speller())
    assert [(d.rule, d.severity, d.pattern) for d in outcome.drafts] == [
        ("text.long_sentence", Severity.MINOR, "over 5 words"),
        ("text.long_sentence", Severity.MAJOR, "over 10 words"),
    ]


def test_blocks_of_mostly_unknown_words_are_one_warning() -> None:
    """Lorem ipsum blocks give no spelling finding, and one warning."""
    lorem = (
        "Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Ut purus "
        "elit, vestibulum ut, placerat ac, adipiscing vitae, felis. "
        "Curabitur dictum gravida mauris."
    )
    doc = make_audit_document(
        make_block(lorem),
        make_block(sentence(40)),
        make_block(lorem),
        make_block("The installer donwloads the files."),
    )
    outcome = text_drafts(doc, CONFIG, Speller())
    assert [(d.rule, d.pattern) for d in outcome.drafts] == [
        ("text.unknown_word", "donwloads"),
    ]
    assert outcome.words == ("donwloads",)
    assert outcome.warnings == (
        (
            "text checks: 2 blocks of 20 spelled words or more, more than "
            "half of them unknown, had their spelling left out: they may be "
            "in another language, or placeholder text"
        ),
    )
    guarded = make_audit_document(
        *([make_block(lorem)] * 12),
        lang="fr",
        declared_lang="fr",
    )
    (warning,) = text_drafts(guarded, CONFIG, Speller()).warnings
    assert GUARD_NOTE in warning


def test_a_few_words_are_not_enough_for_the_guard() -> None:
    """Under `GUARD_MIN_WORDS` words, unknown words are reported."""
    outcome = _french(make_block(SPANISH))
    assert outcome.checked < GUARD_MIN_WORDS
    assert "text.unknown_word" in {d.rule for d in outcome.drafts}
    assert outcome.warnings == ()


def test_a_language_without_dictionary_is_a_warning() -> None:
    """A German document has only its plainly English blocks checked."""
    doc = make_audit_document(
        make_block(
            "Die Installation des Pakets braucht freien Speicher und und "
            "eine stabile Verbindung.",
        ),
        make_block(
            "The installer donwloads the files and then it checks them for "
            "you.",
        ),
        lang="de",
        declared_lang="de",
    )
    outcome = text_drafts(doc, CONFIG, Speller())
    assert outcome.warnings == (
        (
            "text checks: no dictionary for the language 'de'; the checks "
            "that depend on the language read only the blocks that plainly "
            "read as French or English"
        ),
    )
    assert [(d.rule, d.pattern) for d in outcome.drafts] == [
        ("text.unknown_word", "donwloads"),
    ]
    assert outcome.checked == 12


def test_a_language_without_dictionary_keeps_the_other_checks() -> None:
    """Units, doubled marks and long sentences are judged in any language."""
    german = (
        "Die Installation braucht 4GB freien Speicher und und eine stabile "
        "Verbindung,, sagt der Autor in seinem langen Satz "
        + " ".join(["Wort"] * 40)
        + "."
    )
    doc = make_audit_document(
        make_block(german),
        lang="de",
        declared_lang="de",
    )
    outcome = text_drafts(doc, CONFIG, Speller())
    assert [d.rule for d in outcome.drafts] == [
        "text.unit_space",
        "text.double_punctuation",
        "text.long_sentence",
    ]
    assert outcome.checked == 0


def test_an_unknown_language_is_a_warning() -> None:
    """A document of no known language says so, and points at --lang."""
    doc = make_audit_document(
        make_block("Qwzzy plorf brrk."),
        lang=None,
        declared_lang=None,
    )
    outcome = text_drafts(doc, CONFIG, Speller())
    assert outcome.warnings == (
        (
            "text checks: the language of the document is unknown; the "
            "checks that depend on the language read only the blocks that "
            "plainly read as French or English (see --lang)"
        ),
    )
    assert outcome.drafts == []
    assert outcome.checked == 0


def test_a_document_without_prose_has_no_language_warning() -> None:
    """Without prose, nothing was left unchecked: no warning."""
    doc = make_audit_document(
        make_block("ls -l", BlockKind.CODE),
        lang=None,
        declared_lang=None,
    )
    outcome = text_drafts(doc, CONFIG, Speller())
    assert (outcome.drafts, outcome.checked, outcome.warnings) == ([], 0, ())


def test_a_misdeclared_document_is_checked_in_its_text_language() -> None:
    """French text declared English gets the French typography."""
    doc = make_audit_document(
        make_block(
            "Le fichier de configuration est lu au démarrage et il est "
            "ensuite gardé dans la mémoire pour que les agents puissent le "
            "lire sans attendre. Les options de la ligne de commande sont "
            "prioritaires sur celles du fichier, et elles sont décrites dans "
            "la section suivante: lisez-la avant de commencer.",
        ),
    )
    assert (doc.lang, doc.declared_lang) == ("en", "en")
    outcome = text_drafts(doc, CONFIG, Speller())
    assert [(d.rule, d.pattern) for d in outcome.drafts] == [
        ("text.space_missing", ":"),
    ]


def test_code_like_blocks_are_not_checked() -> None:
    """Code and blocks that look like code are neither spelled nor read."""
    doc = make_audit_document(
        make_block("if (x == y) { donwload = z; } else { qwzzy = w; }"),
        make_block("donwload the the files ,, 4GB", BlockKind.CODE),
    )
    outcome = text_drafts(doc, CONFIG, Speller())
    assert outcome.drafts == []
    assert outcome.checked == 0
    assert outcome.warnings == ()


def test_a_block_in_the_other_language_is_marked_whole() -> None:
    """A French paragraph of an English document is shown unmarked."""
    french = (
        "Le fichier est lu par les agents et il est ensuite gardé dans la "
        "mémoire pour que les outils le lisent."
    )
    doc = make_audit_document(make_block(sentence(30)), make_block(french))
    outcome = text_drafts(doc, CONFIG, Speller())
    (draft,) = outcome.drafts
    assert (draft.rule, draft.pattern) == ("text.block_language", "fr")
    assert [(s.op, s.text) for s in draft.ref.segments] == [
        ("equal", french),
    ]


def test_repeated_words_count_only_the_words_looked_up() -> None:
    """Past the budget, a word not looked up is not vouched for."""
    doc = make_audit_document(
        make_block("the the files."),
        make_block("files files."),
    )
    speller = Speller(work=8)
    outcome = text_drafts(doc, CONFIG, speller)
    assert [(d.rule, d.pattern) for d in outcome.drafts] == [
        ("text.repeated_word", "the"),
    ]
    assert outcome.warnings == (
        (
            "text checks: spelling stopped after 1 distinct words; the rest "
            "was not spelled, nor checked for words written twice"
        ),
    )


def test_the_guard_counts_only_the_words_looked_up() -> None:
    """Words left unread by a spent budget weigh nothing in the guard."""
    others = " ".join(
        f"Mot{a}{b}{c}."
        for a in "abcdefghij"
        for b in "klmnopqrst"
        for c in "uv"
    )
    doc = make_audit_document(
        make_block("Maison. " * 140 + "Qwzzy. " * 60 + others),
        lang="fr",
        declared_lang="fr",
    )
    outcome = text_drafts(doc, CONFIG, Speller(work=16))
    assert outcome.checked == 200
    assert outcome.drafts == []
    assert outcome.warnings == (
        (
            "text checks: 30% of the 200 words checked are unknown to the "
            "dictionaries; spelling and the checks that depend on the "
            "language were left out (is the language right? see --lang; or "
            "the document is in a language without dictionary: only French "
            "and English are spelled)"
        ),
        (
            "text checks: spelling stopped after 2 distinct words; the rest "
            "was not spelled, nor checked for words written twice"
        ),
    )


def test_repeated_word_lookups_do_not_stop_the_spelling() -> None:
    """Only the spelling's budget is told: every word here was spelled."""
    doc = make_audit_document(make_block("Hello Hello."))
    outcome = text_drafts(doc, CONFIG, Speller(work=12))
    assert (outcome.checked, outcome.warnings) == (2, ())
    assert outcome.drafts == []


def test_the_warnings_come_in_order() -> None:
    """The language, then the budget, then the cut rules."""
    over = TEXT_DRAFTS_PER_RULE + 5
    doc = make_audit_document(
        make_block(
            "The installer donwloads the files and then it checks them for "
            "you.",
        ),
        make_block("1km " * over),
        lang="de",
        declared_lang="de",
    )
    outcome = text_drafts(doc, CONFIG, Speller(work=8))
    assert outcome.warnings == (
        (
            "text checks: no dictionary for the language 'de'; the checks "
            "that depend on the language read only the blocks that plainly "
            "read as French or English"
        ),
        (
            "text checks: spelling stopped after 1 distinct words; the rest "
            "was not spelled, nor checked for words written twice"
        ),
        (
            "text checks: text.unit_space: only the first "
            f"{TEXT_DRAFTS_PER_RULE} findings are reported"
        ),
    )


def test_a_broken_dictionary_comes_before_failed_lookups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dictionary that cannot be loaded is told before failed lookups."""

    class English:
        """An English dictionary that fails on a word, and knows none."""

        @staticmethod
        def lookup(word: str) -> bool:
            """Fail on a word that starts with X, else know nothing."""
            if word.startswith("X"):
                raise IndexError(word)
            return False

    def load(lang: str) -> English:
        if lang == "en":
            return English()
        raise OSError(lang)

    monkeypatch.setattr(speller_module, "load_dictionary", load)
    doc = make_audit_document(
        make_block("Xqzw is the name of the file and the tool for you."),
    )
    outcome = text_drafts(doc, CONFIG, Speller())
    assert outcome.warnings == (
        (
            "text checks: cannot load the 'fr' dictionary: OSError: fr; the "
            "checks that need it were left out"
        ),
        (
            "text checks: 1 word(s) could not be looked up and were taken as "
            "known"
        ),
    )


def test_failed_lookups_are_a_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Words that spylls fails on are counted in a warning."""

    class Failing:
        """A dictionary whose every lookup fails."""

        @staticmethod
        def lookup(word: str) -> bool:
            """Fail, as spylls might on a hostile word."""
            raise IndexError(word)

    monkeypatch.setattr(speller_module, "load_dictionary", lambda _: Failing())
    doc = make_audit_document(make_block("Qwzzy plorf brrk."))
    outcome = text_drafts(doc, CONFIG, Speller())
    assert outcome.drafts == []
    assert outcome.warnings == (
        (
            "text checks: 3 word(s) could not be looked up and were taken as "
            "known"
        ),
    )


def test_a_dictionary_that_cannot_be_loaded_is_a_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dictionary that fails to load leaves the spelling out, no trace."""

    def broken(lang: str) -> NoReturn:
        message = f"{lang}\n\x1b[2Jtruncated"
        raise OSError(message)

    monkeypatch.setattr(speller_module, "load_dictionary", broken)
    doc = make_audit_document(
        make_block("The installer donwloads the files,, then the the end."),
    )
    outcome = text_drafts(doc, CONFIG, Speller())
    assert [d.rule for d in outcome.drafts] == ["text.double_punctuation"]
    assert outcome.words == ()
    assert outcome.checked == 0
    assert outcome.warnings == (
        (
            "text checks: cannot load the 'en' dictionary: OSError: en "
            "[2Jtruncated; "
            "the checks that need it were left out"
        ),
    )


def test_each_rule_keeps_its_first_drafts() -> None:
    """Past `TEXT_DRAFTS_PER_RULE`, a rule is cut, and a warning says so."""
    over = TEXT_DRAFTS_PER_RULE + 5
    outcome = _french(make_block("a: " * over))
    assert [d.rule for d in outcome.drafts] == (
        ["text.space_missing"] * TEXT_DRAFTS_PER_RULE
    )
    assert outcome.warnings == (
        (
            "text checks: text.space_missing: only the first "
            f"{TEXT_DRAFTS_PER_RULE} findings are reported"
        ),
    )


def test_a_disabled_rule_is_cut_without_a_warning() -> None:
    """The configuration that disables a rule hears nothing of its cut."""
    over = TEXT_DRAFTS_PER_RULE + 5
    doc = make_audit_document(make_block("1km x,, " * over))

    def warned(*rules: str) -> tuple[str, ...]:
        """Return the warnings of the text checks with rules disabled."""
        ignore = replace(CONFIG.ignore, rules=rules)
        cfg = replace(CONFIG, ignore=ignore)
        return text_drafts(doc, cfg, Speller()).warnings

    def cut(rule: str) -> str:
        """Return the warning of a rule cut at `TEXT_DRAFTS_PER_RULE`."""
        return (
            f"text checks: {rule}: only the first {TEXT_DRAFTS_PER_RULE} "
            "findings are reported"
        )

    unit, double = cut("text.unit_space"), cut("text.double_punctuation")
    assert warned() == (unit, double)
    assert warned("text.unit_space") == (double,)
    assert warned("text.d*", "link.*") == (unit,)
    assert warned("text.*") == ()


def test_hostile_text_stays_bounded() -> None:
    """Hostile paragraphs are read in bounded time, with their warnings."""
    rng = random.Random(0)  # ruff: ignore[suspicious-non-cryptographic-random-usage] - test words, not security
    chains = " ".join("-".join(["ab"] * 30) for _ in range(2_000))
    distinct = " ".join(
        "".join(rng.choices(string.ascii_lowercase, k=40))
        for _ in range(20_000)
    )
    absent = tuple(f"absent{index}()" for index in range(5_000))
    hostile = replace(make_block(f"{chains} {distinct}"), code_spans=absent)
    runs = [
        "".join(rng.choices(string.ascii_letters, k=200_000)),
        "." * 200_000 + "a",
        "?" * 200_000 + "a",
        "a" + " " * 200_000 + ",a",
        "1" * 200_000 + "km",
        "X" * 200_000 + "ème",
        "a@" * 100_000,
        "a: 1km " * 20_000,
    ]
    doc = make_audit_document(
        hostile,
        *map(make_block, runs),
        lang="fr",
        declared_lang="fr",
    )
    start = time.perf_counter()
    outcome = text_drafts(doc, CONFIG, Speller(work=10_000))
    assert time.perf_counter() - start < 30
    # Only the words looked up count: they are all unknown, so the guard
    # leaves the French rules out, text.space_missing among them.
    assert outcome.warnings[0].startswith(
        f"text checks: 100% of the {outcome.checked} words checked are",
    )
    assert outcome.warnings[1].startswith(
        "text checks: spelling stopped after",
    )
    assert outcome.warnings[2:] == (
        (
            "text checks: text.unit_space: only the first "
            f"{TEXT_DRAFTS_PER_RULE} findings are reported"
        ),
    )


def test_large_document_is_checked_quickly() -> None:
    """5,000 French paragraphs of 30 words are checked in a few seconds.

    Each paragraph has 27 words of 300 known ones and 3 of 2,000 invented
    ones: a tenth of unknown words, under the language guard, each invented
    word seen about 7 times, as jargon.
    """
    rng = random.Random(0)  # ruff: ignore[suspicious-non-cryptographic-random-usage] - test words, not security
    known = [stem + end for stem in FRENCH_STEMS for end in FRENCH_ENDINGS]
    invented = [
        "".join(rng.choices(string.ascii_lowercase, k=8)) for _ in range(2_000)
    ]
    paragraphs = []
    for _ in range(5_000):
        chosen = rng.choices(known, k=27) + rng.choices(invented, k=3)
        rng.shuffle(chosen)
        paragraphs.append(make_block(" ".join(chosen) + "."))
    doc = make_audit_document(*paragraphs, lang="fr", declared_lang="fr")
    load_dictionary("fr")
    load_dictionary("en")
    start = time.perf_counter()
    outcome = text_drafts(doc, CONFIG, Speller())
    # Measured: 2.1 s on the implementer's machine, two thirds of it in the
    # lookups of the 2,300 distinct words; the bound is ten times that.
    assert time.perf_counter() - start < 21
    assert outcome.checked == 150_000
    assert outcome.warnings == (
        (
            "text checks: text.jargon: only the first "
            f"{TEXT_DRAFTS_PER_RULE} findings are reported"
        ),
    )
