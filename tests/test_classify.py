# Copyright 2026 Christophe Le Douarec
"""Tests for word level differences and their classification."""

import pytest

from doc_checker.compare.classify import (
    RUN_SCAN_MAX,
    HunkContext,
    classify_hunk,
    has_encoding_damage,
)
from doc_checker.compare.textdiff import diff_tokens
from doc_checker.model.findings import Severity
from doc_checker.spell import speller as speller_module
from doc_checker.spell.speller import Speller
from doc_checker.text.tokenize import TokenKind, tokenize

LOAD_DICTIONARY = speller_module.load_dictionary
EM_DASH = "\U00002014"
C = Severity.CRITICAL
MA = Severity.MAJOR
MI = Severity.MINOR
IN = Severity.INFO


def _classify(
    ref: str,
    new: str,
    lang: str | None = "en",
    *,
    speller: Speller | None = None,
) -> list[tuple[str, Severity]]:
    ref_tokens, new_tokens = tokenize(ref), tokenize(new)
    vocabulary = frozenset(
        t.text.casefold() for t in ref_tokens if t.kind is TokenKind.WORD
    )
    ctx = HunkContext(
        ref,
        new,
        ref_tokens,
        new_tokens,
        vocabulary,
        lang,
        lang,
        speller=speller,
    )
    return [
        (c.rule, c.severity)
        for c in (
            classify_hunk(h, ctx) for h in diff_tokens(ref_tokens, new_tokens)
        )
    ]


@pytest.mark.parametrize(
    ("ref", "new", "lang", "expected"),
    [
        (
            "Read the état page.",
            "Read the \xc3\xa9tat page.",
            "fr",
            ("text.encoding_issue", C),
        ),
        (
            "Accepts 100 requests.",
            "Accepts 10 requests.",
            "en",
            ("text.number_changed", C),
        ),
        (
            "Uses port 8080 by default.",
            "Uses port by default.",
            "en",
            ("text.number_changed", C),
        ),
        (
            "Processes 1,000 jobs.",
            "Processes 1000 jobs.",
            "en",
            ("text.number_format", IN),
        ),
        ("Prend 1,5 Go.", "Prend 1.5 Go.", "fr", ("text.number_format", IN)),
        (
            "Do not delete the cache.",
            "Do delete the cache.",
            "en",
            ("text.negation_changed", C),
        ),
        (
            "Ne supprimez pas le cache.",
            "Supprimez le cache.",
            "fr",
            ("text.negation_changed", C),
        ),
        (
            "The efficient scheduler.",
            "The ecient scheduler.",
            "en",
            ("text.ligature_lost", MA),
        ),
        (
            "Edit the configuration file.",
            "Edit the con guration file.",
            "en",
            ("text.ligature_lost", MA),
        ),
        ("Pendant l'été.", "Pendant l'ete.", "fr", ("text.accent_lost", MA)),
        (
            "Pendant l'ete.",
            "Pendant l'été.",
            "fr",
            ("text.accent_changed", MI),
        ),
        (
            "Install Python now.",
            "Install python now.",
            "en",
            ("text.case_changed", MI),
        ),
        (
            "Install it, then run.",
            "Install it then run.",
            "en",
            ("text.punctuation_changed", IN),
        ),
        (
            "You must restart.",
            "You may restart.",
            "en",
            ("text.modality_changed", MA),
        ),
        (
            "The installer downloads files.",
            "The installer donwloads files.",
            "en",
            ("text.typo", MA),
        ),
        (
            "Copy from the form.",
            "Copy form the form.",
            "en",
            ("text.word_replaced", MA),
        ),
        (
            "Install the package with the default settings.",
            "Install the package.",
            "en",
            ("text.words_missing", MA),
        ),
        (
            "Install the package now.",
            "Install the package.",
            "en",
            ("text.words_missing", MI),
        ),
        (
            "Install the package.",
            "Install the package quickly.",
            "en",
            ("text.words_added", MI),
        ),
        (
            "The service restarts automatically after updates.",
            "A different sentence replaces everything here completely.",
            "en",
            ("text.modified", MA),
        ),
    ],
)
def test_classify_hunk(
    ref: str,
    new: str,
    lang: str,
    expected: tuple[str, Severity],
) -> None:
    """Each kind of difference gets its rule and severity."""
    assert _classify(ref, new, lang) == [expected]


@pytest.mark.parametrize(
    ("ref", "new", "words", "expected"),
    [
        (
            "The installer downloads files.",
            "The installer donwloads files.",
            (),
            ("text.typo", MA),
        ),
        (
            "Open the files now.",
            "Open the files new.",
            (),
            ("text.word_replaced", MA),
        ),
        (
            "Copy from the form.",
            "Copy form the form.",
            (),
            ("text.word_replaced", MA),
        ),
        (
            "Please recieve the file.",
            "Please receive the file.",
            (),
            ("text.typo_fixed", IN),
        ),
        (
            "Recieve the file.",
            "Receive the file.",
            (),
            ("text.word_replaced", MA),
        ),
        ("Kubernetes runs.", "Kubernets runs.", (), ("text.typo", MA)),
        (
            "Order part XJ-120-4 today.",
            "Order part XJ-1204 today.",
            (),
            ("text.typo", MA),
        ),
        (
            "The installer downloads files.",
            "The installer donwloads files.",
            ("donwloads",),
            ("text.word_replaced", MA),
        ),
    ],
)
def test_classify_hunk_with_the_dictionary(
    ref: str,
    new: str,
    words: tuple[str, ...],
    expected: tuple[str, Severity],
) -> None:
    """The dictionary confirms a typo, a valid word or a correction."""
    assert _classify(ref, new, speller=Speller(words)) == [expected]


def test_without_dictionary_language_the_vocabulary_decides() -> None:
    """A language without dictionary keeps the classification of phase 1."""
    assert _classify(
        "Open the files now.",
        "Open the files new.",
        "de",
        speller=Speller(),
    ) == [("text.typo", MA)]


def test_a_negation_stays_critical_with_the_dictionary() -> None:
    """A word that changes a negation is never a mere word replacement."""
    assert _classify(
        "Copy the files now.",
        "Copy the files not.",
        speller=Speller(),
    ) == [("text.negation_changed", C)]


def test_a_word_not_looked_up_keeps_the_classification_of_phase_1() -> None:
    """Once the budget is spent, the vocabulary decides, as in phase 1."""
    speller = Speller(work=0)
    assert _classify(
        "Please recieve the file.",
        "Please receive the file.",
        speller=speller,
    ) == [("text.typo", MA)]
    assert _classify(
        "Copy from the form.",
        "Copy form the form.",
        speller=speller,
    ) == [("text.word_replaced", MA)]
    assert speller.exhausted
    assert speller.lookups == 0


@pytest.mark.parametrize(
    ("ref", "new"),
    [
        (
            "See https://example.com/docs/recieve for it.",
            "See https://example.com/docs/receive for it.",
        ),
        (
            "Edit the /etc/recieve.conf file now.",
            "Edit the /etc/receive.conf file now.",
        ),
        (
            "Set tbl/pcell/begin in the hook.",
            "Set tbl/cell/begin in the hook.",
        ),
        ("Pass --recieve to the tool.", "Pass --receive to the tool."),
        (
            "Write to recieve@example.com today.",
            "Write to receive@example.com today.",
        ),
    ],
)
def test_a_word_inside_code_keeps_the_classification_of_phase_1(
    ref: str,
    new: str,
) -> None:
    """A URL, a path, an option or an address is no word to spell."""
    assert _classify(ref, new, speller=Speller()) == [("text.typo", MA)]


@pytest.mark.parametrize(
    ("glued", "expected"),
    [
        (RUN_SCAN_MAX - 1, ("text.typo_fixed", IN)),
        (RUN_SCAN_MAX, ("text.typo", MA)),
    ],
)
def test_the_run_of_a_word_is_read_within_a_bound(
    glued: int,
    expected: tuple[str, Severity],
) -> None:
    """A word in a run longer than the bound is no word to spell."""
    ref = f"Read {'a' * glued}{EM_DASH}recieve now."
    new = ref.replace("recieve", "receive")
    assert _classify(ref, new, speller=Speller()) == [expected]


@pytest.mark.parametrize(
    ("ref", "new", "lang", "expected"),
    [
        (
            "La page d'acceuil est vide.",
            "La page d'accueil est vide.",
            "fr",
            ("text.typo_fixed", IN),
        ),
        (
            "Its raison is clear.",
            "Its reason is clear.",
            "en",
            ("text.word_replaced", MA),
        ),
    ],
)
def test_each_language_has_its_dictionary(
    ref: str,
    new: str,
    lang: str,
    expected: tuple[str, Severity],
) -> None:
    """A French correction is info; a French word in English is a word."""
    assert _classify(ref, new, lang, speller=Speller()) == [expected]


class _FailsOn:
    """A dictionary that fails on one word and knows the others it knows."""

    def __init__(self, lang: str, word: str) -> None:
        """Wrap the dictionary of a language."""
        self.dictionary = LOAD_DICTIONARY(lang)
        self.word = word

    def lookup(self, form: str) -> bool:
        """Fail on the word, as spylls might on a hostile one."""
        if form == self.word:
            raise IndexError(form)
        return self.dictionary.lookup(form)


def test_a_failed_lookup_of_the_new_word_is_never_a_correction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new word that spylls fails on keeps the classification of phase 1."""
    monkeypatch.setattr(
        speller_module,
        "load_dictionary",
        lambda lang: _FailsOn(lang, "zorblaxed"),
    )
    speller = Speller()
    for _ in range(2):
        assert _classify(
            "Run the zorblax step.",
            "Run the zorblaxed step.",
            speller=speller,
        ) == [("text.typo", MA)]
    assert speller.failed == 1


def test_identical_text_has_no_hunk() -> None:
    """Identical texts have no difference."""
    tokens = tokenize("Same text here.")
    assert diff_tokens(tokens, tokens) == []


def test_intra_word_hyphens_are_not_differences() -> None:
    """A hyphen inside a word is no difference, other changes still are."""
    assert (
        _classify("Use the non-blocking mode.", "Use the nonblocking mode.")
        == []
    )
    assert _classify(
        "The scheduler processes 1,000 jobs.",
        "The scheduler pro-cesses 1000 jobs.",
    ) == [("text.number_format", IN)]


@pytest.mark.parametrize(
    ("ref", "new"),
    [
        ("Use the nonblocking mode.", "Use the non-blocking mode."),
        ("Send an e-mail today.", "Send an email today."),
        ("The pro-cesses run well.", "The proc-esses run well."),
        ("A well-known check-in.", "A wellknown checkin."),
    ],
)
def test_hyphens_may_be_gained_lost_or_moved(ref: str, new: str) -> None:
    """Hyphens may appear, disappear or move inside words."""
    assert _classify(ref, new) == []


def test_hyphenated_words_are_still_compared_by_their_letters() -> None:
    """Dropping the hyphens does not hide a change of letters."""
    assert _classify(
        "Use the non-blocking mode.",
        "Use the non-blocing mode.",
    ) == [("text.typo", MA)]


@pytest.mark.parametrize(
    ("ref", "new"),
    [
        ("Order part XJ-120-4 today.", "Order part XJ-1204 today."),
        ("Build for x86-64 only.", "Build for x8664 only."),
    ],
)
def test_hyphens_next_to_digits_are_compared(ref: str, new: str) -> None:
    """A hyphen next to a digit belongs to a code: dropping it is a change."""
    assert _classify(ref, new) == [("text.typo", MA)]


@pytest.mark.parametrize(
    ("text", "damaged"),
    [
        ("état", False),
        ("Ça va", False),
        ("\xc3\xa9tat", True),
        ("a\U0000fffdb", True),
        ("l\xe2\U000020ac\U00002122app", True),
        ("x\U0000e000y", True),
    ],
)
def test_has_encoding_damage(text: str, damaged: bool) -> None:  # ruff: ignore[boolean-type-hint-positional-argument] - pytest parameter
    """Replacement, private use and mojibake characters are detected."""
    assert has_encoding_damage(text) is damaged


def test_case_change_across_a_hyphen() -> None:
    """A case change is no typo because a hyphen came or went."""
    assert _classify(
        "Use the Non-blocking mode.",
        "Use the nonblocking mode.",
    ) == [("text.case_changed", MI)]
