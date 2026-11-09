# Copyright 2026 Christophe Le Douarec
"""Tests for the words that the speller checks."""

import time
from typing import TYPE_CHECKING

import pytest

from doc_checker.errors import ConfigError
from doc_checker.spell import words as words_module
from doc_checker.spell.words import (
    BLANKS,
    CHUNK,
    CODE_CHUNK,
    DOTTED,
    ELISIONS,
    EMAIL,
    MEMO_MAX,
    URL,
    Word,
    fold,
    is_word_char,
    read_word_file,
    skip_reason,
    touches,
    without_possessive,
    words,
)

if TYPE_CHECKING:
    from pathlib import Path

APOSTROPHE = "\U00002019"
EN_DASH = "\U00002013"
EM_DASH = "\U00002014"
HYPHEN = "\U00002010"
NB_HYPHEN = "\U00002011"
# Long runs of one shape, on which a pattern that restarts at every place of
# the run would be quadratic.
LONG_RUNS = (
    "a" * 200_000,
    "1" * 200_000 + "x",
    "a." * 100_000,
    "a@" * 100_000,
    "a-" * 100_000,
)


@pytest.mark.parametrize(
    ("word", "reason"),
    [
        ("configuration", None),
        ("l'arbre", None),
        ("c'est-à-dire", None),
        ("Kubernetes", None),
        ("users'", None),
        (f"aujourd{APOSTROPHE}hui", None),
        ("Jean-Pierre", None),
        (f"peut{HYPHEN}être", None),
        (f"c{NB_HYPHEN}est", None),
        (f"Peut{NB_HYPHEN}\xcatre", None),
        (NB_HYPHEN.join(["ab"] * 6), "parts"),
        (f"aB{HYPHEN}cd", "mixed"),
        ("a", "short"),
        ("l'", "short"),
        ("x" * 41, "long"),
        ("-".join(["mot"] * 6), "parts"),
        ("x86", "digit"),
        ("2ème", "digit"),
        ("10km", "digit"),
        ("API", "acronym"),
        ("API's", "acronym"),
        (f"API{APOSTROPHE}s", "acronym"),
        ("GitHub", "mixed"),
        ("camelCase", "mixed"),
        ("snake_case", "symbol"),
        ("ab-'cd", "symbol"),
        ("ab''cd", "symbol"),
        ("ab-", "symbol"),
        ("ab--cd", "symbol"),
        ("caf\U000000e9\U000000d7", "symbol"),
        ("\U00004e2d\U00006587", "symbol"),
        ("\U000003b1\U000003b2", "symbol"),
        ("caf\U0000fffd", "symbol"),
    ],
)
def test_skip_reasons(word: str, reason: str | None) -> None:
    """Identifiers, numbers, acronyms and long words are not spelled."""
    assert skip_reason(word) == reason


def test_words_skip_code_and_urls() -> None:
    """Paths, options, dotted names and URLs give no word; edges go."""
    text = (
        "Run \xaborbit\xbb with --verbose, see https: //tug.org/x or a.b.c, "
        "the file /etc/orbit.toml and (version) mailto:a@b.org now."
    )
    found = [w.text for w in words(text)]
    assert found == [
        "Run",
        "orbit",
        "with",
        "see",
        "or",
        "the",
        "file",
        "and",
        "version",
        "now",
    ]
    assert text[words(text)[1].start : words(text)[1].end] == "orbit"


def test_words_skip_excluded_spans() -> None:
    """A chunk that touches an excluded span is left out."""
    text = "Use orbit start now."
    assert [w.text for w in words(text, [(4, 15)])] == ["Use", "now"]


def test_words_touching_an_excluded_span_are_left_out() -> None:
    """A place that ends where a chunk starts touches it; a blank does not."""
    text = "ab cd ef"
    assert [w.text for w in words(text, [(0, 3)])] == ["ef"]
    assert [w.text for w in words(text, [(5, 6)])] == ["ab"]
    assert [w.text for w in words("ab  cd ef", [(2, 3)])] == ["cd", "ef"]
    assert [w.text for w in words(text, [(6, 6), (0, 1)])] == ["cd"]


def test_words_are_cut_at_dashes_and_keep_their_places() -> None:
    """En and em dashes cut words; each word knows its place in the text."""
    text = f"Le mot{EM_DASH}clé, puis 10{EN_DASH}20 et ({EN_DASH}fin) mot."
    found = words(text)
    assert [w.text for w in found] == [
        "Le",
        "mot",
        "clé",
        "puis",
        "10",
        "20",
        "et",
        "fin",
        "mot",
    ]
    assert all(text[w.start : w.end] == w.text for w in found)
    assert found[-1] == Word("mot", len(text) - 4, len(text) - 1)


def test_e_mail_addresses_and_schemes_are_not_words() -> None:
    """An e-mail address, a URL and a scheme name with its colon go."""
    text = "Write to jo.doe+x@mail.example.org or see www.example.org, file: x"
    assert [w.text for w in words(text)] == ["Write", "to", "or", "see", "x"]
    assert EMAIL.search("jo.doe+x@mail.example.org").group() == (
        "jo.doe+x@mail.example.org"
    )
    assert URL.search("HTTPS://example.org").group() == "HTTPS://example.org"


@pytest.mark.parametrize(
    ("chunk", "code"),
    [
        ("src/x.py", True),
        ("a.b", True),
        ("--force", True),
        ("x::y", True),
        ("$HOME", True),
        ("2.5GB", False),
        ("3.12", False),
        ("l'arbre", False),
    ],
)
def test_code_chunks(chunk: str, *, code: bool) -> None:
    """Code marks, dotted names and options; a decimal number is not code."""
    assert (CODE_CHUNK.search(chunk) is not None) is code


def test_dotted_names_are_code() -> None:
    """A chunk that starts with a dot and a letter is a file, not a word."""
    text = (
        "Edit .gitignore and .txt files (.dvi-files, \U0000201c.progname"
        "\U0000201d), not ...and nor .5 mm."
    )
    assert [w.text for w in words(text)] == [
        "Edit",
        "and",
        "files",
        "not",
        "and",
        "nor",
        "5",
        "mm",
    ]
    assert DOTTED.match("\xab.txt") is not None
    assert DOTTED.match("...and") is None
    assert DOTTED.match("a.txt") is None


def test_patterns_are_linear_on_long_runs() -> None:
    """Every pattern reads a long run of one shape in linear time."""
    start = time.perf_counter()
    for run in LONG_RUNS:
        for pattern in (URL, EMAIL, CODE_CHUNK, DOTTED):
            sum(1 for _ in pattern.finditer(run))
        words(run)
    assert time.perf_counter() - start < 5


def test_typographic_hyphens_stay_inside_words() -> None:
    """U+2010 and U+2011 join the parts of a word, like a hyphen."""
    text = f"C{NB_HYPHEN}est peut{HYPHEN}être vrai."
    assert [w.text for w in words(text)] == [
        f"C{NB_HYPHEN}est",
        f"peut{HYPHEN}être",
        "vrai",
    ]


def test_the_memo_of_runs_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Past its cap, the memo stops growing and the words stay the same."""
    assert MEMO_MAX == 4_096
    text = "aa bb (cc) aa dd bb ee, aa"
    expected = [(w.text, w.start, w.end) for w in words(text)]
    monkeypatch.setattr(words_module, "MEMO_MAX", 2)
    assert [(w.text, w.start, w.end) for w in words(text)] == expected
    assert [text for text, *_ in expected] == [
        "aa",
        "bb",
        "cc",
        "aa",
        "dd",
        "bb",
        "ee",
        "aa",
    ]


def test_touches() -> None:
    """A place overlaps or touches one of the sorted places, or not."""
    spans = [(2, 4), (10, 12), (20, 30)]
    assert touches((0, 2), spans)
    assert touches((4, 6), spans)
    assert touches((12, 14), spans)
    assert touches((22, 23), spans)
    assert touches((0, 40), spans)
    assert not touches((5, 9), spans)
    assert not touches((0, 1), spans)
    assert not touches((31, 40), spans)
    assert not touches((0, 1), [])


def test_fold() -> None:
    """Case and the typographic apostrophes are folded."""
    assert fold(f"L{APOSTROPHE}Arbre") == "l'arbre"
    assert fold("L\U000002bcarbre") == "l'arbre"
    assert fold("KUBECTL") == "kubectl"


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("user's", "user"),
        (f"user{APOSTROPHE}s", "user"),
        ("user\U000002bcs", "user"),
        ("users'", "users"),
        (f"users{APOSTROPHE}", "users"),
        ("API'S", "API'S"),
        ("'s", "'s"),
        ("'", "'"),
        ("l'agent", "l'agent"),
        ("users", "users"),
    ],
)
def test_without_possessive(word: str, expected: str) -> None:
    """A final 's or apostrophe after more of the word is left out."""
    assert without_possessive(word) == expected


def test_shared_pieces_of_words() -> None:
    """The runs, apostrophes, hyphens, elisions and word characters."""
    text = "  l\U00002019agent  x-y\U00002010z\U00002011w "
    assert [m.group() for m in CHUNK.finditer(text)] == [
        "l\U00002019agent",
        "x-y\U00002010z\U00002011w",
    ]
    assert BLANKS.match(text).end() == 2
    assert BLANKS.match(text, 3).end() == 3
    apostrophes = "a'b\U00002019c\U000002bcd"
    assert words_module.APOSTROPHE.split(apostrophes) == list("abcd")
    hyphens = "x-y\U00002010z\U00002011w"
    assert words_module.HYPHEN.split(hyphens) == list("xyzw")
    assert {"l", "d", "qu", "jusqu"} <= ELISIONS
    assert [is_word_char(c) for c in "a\xe91_ -'"] == [
        True,
        True,
        True,
        True,
        False,
        False,
        False,
    ]


def test_word_file(tmp_path: Path) -> None:
    """One word per line, comments and blank lines ignored, BOM read."""
    path = tmp_path / "words.txt"
    path.write_text(
        "\U0000feffKubernetes\n# jargon\n\n  kubectl  \n",
        encoding="utf-8",
    )
    assert read_word_file(path) == ("Kubernetes", "kubectl")
    path.write_text("two words\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=":1: expected one word per line"):
        read_word_file(path)


def test_word_file_errors_name_the_file(tmp_path: Path) -> None:
    """A missing or undecodable file is a configuration error."""
    missing = tmp_path / "missing.txt"
    with pytest.raises(ConfigError, match=r"missing\.txt: cannot read file: "):
        read_word_file(missing)
    latin = tmp_path / "latin.txt"
    latin.write_bytes(b"caf\xe9\n")
    with pytest.raises(
        ConfigError,
        match=r"latin\.txt: cannot read file: not UTF-8",
    ):
        read_word_file(latin)
    lines = tmp_path / "lines.txt"
    lines.write_bytes(b"ok\r\n\r\nbad word\r\n")
    with pytest.raises(ConfigError, match=":3: expected one word per line"):
        read_word_file(lines)
