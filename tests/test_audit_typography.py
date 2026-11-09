# Copyright 2026 Christophe Le Douarec
"""Tests for the typography of the audit: spaces, quotes, ordinals, units."""

import time

import pytest

from doc_checker.audit.prose import Mark, excluded_spans
from doc_checker.audit.typography import (
    FRENCH_UNITS,
    NON_BREAKING,
    UNITS,
    code_tokens,
    typography_marks,
)
from doc_checker.compare.aggregate import PATTERN_MAX_CHARS
from doc_checker.compare.drafts import ARROW
from doc_checker.text.normalize import ELLIPSIS
from tests.builders import make_block

# Long runs of one shape: each pattern must read them in linear time.
HOSTILE_RUNS = (
    "a" * 200_000,
    " " * 200_000,
    "\xa0" * 200_000,
    "a" + " " * 200_000 + ",a",
    "a" + "\xa0" * 200_000 + ";a",
    "a" + " \xa0" * 100_000 + "?",
    "." * 200_000 + "a",
    "?" * 200_000 + "a",
    "!" * 200_000,
    ":" * 200_000 + "a",
    ";." * 100_000,
    "a:" * 100_000,
    "a ;" * 70_000,
    "1" * 200_000,
    "1" * 200_000 + "km",
    "1" * 200_000 + "x",
    "1" * 200_000 + "th",
    "1" * 200_000 + "ème",
    "1." * 100_000 + "km",
    "1," * 100_000 + "GB",
    "X" * 200_000 + "ème",
    "X" * 200_000 + "a",
    "XIX" * 70_000 + "ères",
    "\xab" * 200_000,
    "\xbb" * 200_000,
    "\xab " * 100_000,
    " \xbb" * 100_000,
    "\xab" + "a" * 200_000,
    '"' * 200_000,
    '"' + "a" * 200_000,
    '"a' * 100_000,
    '" ' * 100_000,
    "\U0000201c" * 200_000,
    "\U0000201c" + " " * 200_000 + "\U0000201d",
    "a.." * 70_000,
    "a" * 200_000 + "!",
    "a" * 200_000 + " ,",
    "\xe9" * 200_000 + ".",
    "a " * 100_000,
    "a@" * 100_000,
    "I" * 200_000 + "ère",
    "XL" * 100_000 + "ème",
    "XXXX" * 50_000 + "ème",
    "\xab\U0000201ca\U0000201d" * 60_000,
    "\U0000201ca\U0000201d\xbb" * 60_000,
    "\U000003bc" * 200_000 + "s",
    "1" * 200_000 + "\U000003bcs",
    "\U00000661" * 200_000 + "km",
    "\U00000661" * 200_000 + "th",
    "a ? b" * 40_000,
    "a ;" * 70_000 + "!",
)


def _rules(
    text: str,
    lang: str | None,
    *,
    breakable: bool = True,
) -> list[tuple[str, str]]:
    """Return the rule and the marked text of each mark of a block's text."""
    block = make_block(text)
    marks = typography_marks(
        block.text,
        lang,
        excluded_spans(block),
        breakable=breakable,
    )
    return [(m.rule, block.text[m.span[0] : m.span[1]]) for m in marks]


def _marks(text: str, lang: str | None) -> list[Mark]:
    """Return the marks of a block's text, its spaces judged."""
    block = make_block(text)
    return typography_marks(
        block.text,
        lang,
        excluded_spans(block),
        breakable=True,
    )


def test_spaces_and_units() -> None:
    """The non-breaking spaces and the units are those of the Rulings."""
    assert NON_BREAKING == "\xa0\U0000202f\U00002007"
    assert UNITS == (
        "km",
        "cm",
        "mm",
        "\xb5m",
        "nm",
        "kg",
        "mg",
        "ms",
        "\xb5s",
        "ns",
        "min",
        "Hz",
        "kHz",
        "MHz",
        "GHz",
        "kB",
        "KB",
        "MB",
        "GB",
        "TB",
        "PB",
        "KiB",
        "MiB",
        "GiB",
        "TiB",
        "kW",
        "kWh",
        "mAh",
        "dB",
    )
    assert FRENCH_UNITS == ("Ko", "Mo", "Go", "To", "Kio", "Mio", "Gio", "Tio")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Les conflits sont rares: lorsque", [("text.space_missing", ":")]),
        ("Une base; une autre", [("text.space_missing", ";")]),
        (
            "Vraiment? Oui!",
            [("text.space_missing", "?"), ("text.space_missing", "!")],
        ),
        (
            "\xabbonjour\xbb",
            [("text.space_missing", "\xab"), ("text.space_missing", "\xbb")],
        ),
        ("Rares\xa0: lorsque", []),
        ("Vraiment\U0000202f? Oui\U0000202f!", []),
        ("Rares\U00002007; lorsque", []),
        ("\xab\xa0bonjour\xa0\xbb", []),
        ("Voir https://x.org/a?b=1 et ftp: ou https:, puis 10:30 et a:b", []),
        ("Quoi?! Non?? Fin.", []),
        (
            "Il dit\xa0: \xab\xa0oui\xa0\xbb: puis",
            [("text.space_missing", ":")],
        ),
        ("(voir plus bas): puis", [("text.space_missing", ":")]),
        ("Rares : lorsque", [("text.space_breakable", " :")]),
        (
            "\xab bonjour \xbb",
            [
                ("text.space_breakable", "\xab "),
                ("text.space_breakable", " \xbb"),
            ],
        ),
        (
            "Rares\U00002009: lorsque",
            [("text.space_breakable", "\U00002009:")],
        ),
    ],
)
def test_french_spacing(text: str, expected: list[tuple[str, str]]) -> None:
    """Glued punctuation is missing a space; a plain space can break."""
    assert _rules(text, "fr") == expected


@pytest.mark.parametrize(
    ("text", "marked"),
    [
        ("a:b", []),
        ("C:\\x", []),
        ("x:/y", []),
        ("k:=v", []),
        ("10:30 et 3: puis", []),
        ("a?b", []),
        ("a?1", []),
        ("a!=b", []),
        ("a?&b", []),
        ("a;b", []),
        ("a;&b", []),
        ("a?! b!? c??", []),
        ("a;1", [";"]),
        ("a!! b", ["!"]),
        ("a:) b", [":"]),
        ("a?", ["?"]),
        ("l\U00002019a\U00002019: b", [":"]),
        ('"a": b', [":"]),
        ("\U0000201ca\U0000201d; b", [";"]),
    ],
)
def test_french_glued_punctuation(text: str, marked: list[str]) -> None:
    """A mark glued to a word, but not one inside a name or a number."""
    marks = typography_marks(text, "fr", (), breakable=True)
    assert [
        text[m.span[0] : m.span[1]]
        for m in marks
        if m.rule == "text.space_missing"
    ] == marked


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("pour \xab, le guillemet gauche", []),
        ("les guillemets \xab\xbb et (\xbb)", []),
        (
            "en fran\xe7ais\xa0: \xab, \xbb",
            [("text.space_breakable", " \xbb")],
        ),
        (
            "un \xab ; \xbb final",
            [
                ("text.space_breakable", "\xab "),
                ("text.space_breakable", " \xbb"),
            ],
        ),
        (
            "le \xab(voir)\xbb",
            [
                ("text.space_missing", "\xab"),
                ("text.space_missing", "\xbb"),
            ],
        ),
    ],
)
def test_guillemets_named_as_characters(
    text: str,
    expected: list[tuple[str, str]],
) -> None:
    """A guillemet next to a mark is the character, not a quotation."""
    assert _rules(text, "fr") == expected


def test_breakable_spaces_are_judged_only_when_asked() -> None:
    """In a PDF a plain space may be a lost non-breaking one."""
    assert (
        _rules("Rares : lorsque \xab bonjour \xbb", "fr", breakable=False)
        == []
    )
    assert _rules("Rares: lorsque", "fr", breakable=False) == [
        ("text.space_missing", ":"),
    ]


def test_a_pdf_judges_the_colon_and_the_guillemets_only() -> None:
    """A PDF drops the thin space that LaTeX sets before ; ! and ?."""
    text = "Oui; non! Vrai? Rares: x \xabbof\xbb"
    assert _rules(text, "fr", breakable=False) == [
        ("text.space_missing", ":"),
        ("text.space_missing", "\xab"),
        ("text.space_missing", "\xbb"),
    ]
    assert [part for _, part in _rules(text, "fr")] == [
        ";",
        "!",
        "?",
        ":",
        "\xab",
        "\xbb",
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Conflicts are rare : when", [" :"]),
        ("Wait , then", [" ,"]),
        ("Is it rare ?", [" ?"]),
        ("Item 3 ; next", [" ;"]),
        ("Rare\xa0: when", ["\xa0:"]),
        ("before the four characters ; ! ? and :. The", []),
        ("the marks , ; and more", []),
        ("a .gitignore file", []),
        ("Rare: when", []),
        ("Run it ,then", []),
        ("marked with a ? in the output", []),
        ("An ! starts a negation", []),
        ("the ; and , characters", []),
        ("either ! or ?", []),
        ("A suffix : followed by a path", [" :"]),
        ("If you combine ! and ^, it fails", []),
        ("Use ? to mean that", []),
        ("a literal ! character", []),
        ("Is it rare ? Yes", [" ?"]),
        ("Use ?\nto mean that", []),
        ("rare : when", [" :"]),
    ],
)
def test_english_space_before_punctuation(
    text: str,
    expected: list[str],
) -> None:
    """A blank before a mark is a mistake, unless marks are listed."""
    assert _rules(text, "en") == [
        ("text.space_before_punctuation", part) for part in expected
    ]


def test_quotes() -> None:
    """French is quoted with guillemets, English with quotation marks."""
    assert _rules('le mode "strict" refuse', "fr") == [
        ("text.quotes", '"strict"'),
    ]
    curly = "le mode \U0000201cstrict\U0000201d refuse"
    assert _rules(curly, "fr") == [
        ("text.quotes", "\U0000201cstrict\U0000201d"),
    ]
    assert _rules("the \xabstrict\xbb mode", "en") == [
        ("text.quotes", "\xabstrict\xbb"),
    ]
    assert _rules('the "strict" mode', "en") == []
    assert _rules("le mode \xab\xa0strict\xa0\xbb refuse", "fr") == []
    assert _rules('un \xe9cran de 15" ou 17"', "fr") == []
    assert _rules('le mode "' + "a" * 81 + '" refuse', "fr") == []
    assert _rules('le mode " strict " refuse', "fr") == [
        ("text.quotes", '" strict "'),
    ]


def test_quotations_inside_guillemets() -> None:
    """French quotes a quotation inside guillemets with quotation marks."""
    nested = "\xab\xa0Viens \U0000201cvite\U0000201d, dit-il\xa0\xbb"
    assert _rules(nested, "fr") == []
    after = "\xab\xa0oui\xa0\xbb puis \U0000201cvite\U0000201d"
    assert _rules(after, "fr") == [
        ("text.quotes", "\U0000201cvite\U0000201d"),
    ]
    unclosed = "\xab\xa0Viens \U0000201cvite\U0000201d"
    assert _rules(unclosed, "fr") == [
        ("text.quotes", "\U0000201cvite\U0000201d"),
    ]
    straight = '\xab\xa0Viens "vite"\xa0\xbb'
    assert _rules(straight, "fr") == [("text.quotes", '"vite"')]


def test_quoted_marks_are_no_quotation() -> None:
    """Quotation marks around a mark of punctuation name characters."""
    assert _rules("the opening (\xab) or closing (\xbb) one", "en") == []
    assert _rules("type \U00002018\xab\U00002019 then \xbb", "en") == []
    assert _rules('le motif "(x)" ou "." ici', "fr") == []


@pytest.mark.parametrize(
    ("text", "pattern"),
    [
        ("le 2ème jour", f"ème{ARROW}e"),
        ("le 2eme jour", f"eme{ARROW}e"),
        ("le 3ième jour", f"ième{ARROW}e"),
        ("le 4ieme jour", f"ieme{ARROW}e"),
        ("le 5éme jour", f"éme{ARROW}e"),
        ("le XIXème siècle", f"ème{ARROW}e"),
        ("la 1ère fois", f"ère{ARROW}re"),
        ("la 1ere fois", f"ere{ARROW}re"),
        ("la 1ière fois", f"ière{ARROW}re"),
        ("les 2èmes places", f"èmes{ARROW}es"),
        ("les 1ères places", f"ères{ARROW}res"),
        ("le 1ème jour", f"ème{ARROW}er"),
        ("le Ième siècle", f"ème{ARROW}er"),
        ("la Ière partie", f"ère{ARROW}re"),
        ("le IVème chapitre", f"ème{ARROW}e"),
        ("le XXXIXème", f"ème{ARROW}e"),
    ],
)
def test_french_ordinals(text: str, pattern: str) -> None:
    """A French ordinal is written with its last letters only."""
    marks = _marks(text, "fr")
    assert [(m.rule, m.pattern) for m in marks] == [("text.ordinal", pattern)]


@pytest.mark.parametrize(
    ("text", "detail"),
    [
        ("le 1ème jour", '"1ème" is written "1er" or "1re"'),
        ("la 1ere fois", '"1ere" is written "1er" or "1re"'),
        ("la 1ère fois", '"1ère" is written "1er" or "1re"'),
        ("le Ième siècle", '"Ième" is written "Ier" or "Ire"'),
        ("les 1èmes", '"1èmes" is written "1ers" or "1res"'),
        ("le 2ème jour", '"2ème" is written "2e"'),
    ],
)
def test_the_first_names_both_genders(text: str, detail: str) -> None:
    """The gender of the first cannot be read: both forms are named."""
    assert [m.detail for m in _marks(text, "fr")] == [detail]


@pytest.mark.parametrize(
    ("text", "pattern"),
    [
        ("the 21th time", f"th{ARROW}st"),
        ("the 11st time", f"st{ARROW}th"),
        ("the 3th time", f"th{ARROW}rd"),
        ("the 2rd time", f"rd{ARROW}nd"),
        ("the 112nd time", f"nd{ARROW}th"),
    ],
)
def test_english_ordinals(text: str, pattern: str) -> None:
    """An English ordinal ends as its last two digits say."""
    marks = _marks(text, "en")
    assert [(m.rule, m.pattern) for m in marks] == [("text.ordinal", pattern)]


@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("le 2e jour, le 1er et la 1re, le XIXe", "fr"),
        ("la 2ère fois, un Mème, x2ème, 1.2ème, la XIère", "fr"),
        ("the 1st, 12th, 22nd, 13th, 111th and 0th", "en"),
        ("the 2ème time", "en"),
        ("le 21th jour", "fr"),
        ("Leme, Lème, Ceme, Dème, XLème, IIIIème, VVème, xixème", "fr"),
        ("the \U00000662\U00000661st time", "en"),
        ("le \U00000662\U00000661ème jour", "fr"),
    ],
)
def test_right_ordinals(text: str, lang: str) -> None:
    """A right ordinal, or one of the other language, is no mistake."""
    assert _rules(text, lang) == []


@pytest.mark.parametrize("lang", ["fr", "en", None])
@pytest.mark.parametrize(
    ("text", "unit"),
    [
        ("10km", "km"),
        ("2.5GB", "GB"),
        ("2,5GB", "GB"),
        ("100ms", "ms"),
        ("32KiB", "KiB"),
        ("5kWh", "kWh"),
        ("3\xb5s", "\xb5s"),
        ("3\U000003bcs", "\U000003bcs"),
        ("2\U000003bcm", "\U000003bcm"),
    ],
)
def test_units(text: str, unit: str, lang: str | None) -> None:
    """A number glued to a unit of `UNITS` is missing a space."""
    marks = _marks(f"Il faut {text} ici", lang)
    assert [(m.rule, m.pattern) for m in marks] == [("text.unit_space", unit)]
    assert marks[0].span == (8, 8 + len(text))


@pytest.mark.parametrize(
    ("text", "unit"),
    [("4Go", "Go"), ("2,5Go", "Go"), ("8Tio", "Tio"), ("512Ko", "Ko")],
)
def test_french_units_in_french(text: str, unit: str) -> None:
    """The French storage units are units in French text only."""
    marks = _marks(f"Il faut {text} ici", "fr")
    assert [(m.rule, m.pattern) for m in marks] == [("text.unit_space", unit)]
    for lang in ("en", None):
        assert _rules(f"Il faut {text} ici", lang) == []


def test_a_footnote_number_before_to_is_no_unit_in_english() -> None:
    """A footnote number glued to "To" starts an English sentence."""
    text = "as shown.10To be more precise, see 78To the left."
    assert _rules(text, "en") == []
    assert _rules(text, None) == []


@pytest.mark.parametrize("lang", ["fr", "en", None])
@pytest.mark.parametrize(
    "text",
    [
        "10 km",
        "16px",
        "4G",
        "3D",
        "x86",
        "v2.0",
        "12pt",
        "10mins",
        "1.2.3ms",
        "\U00000661\U00000660km",
    ],
)
def test_no_unit_space(text: str, lang: str | None) -> None:
    """A unit outside `UNITS`, or a number inside a word, is no unit."""
    assert _rules(f"Il faut {text} ici", lang) == []


@pytest.mark.parametrize(
    ("text", "marked"),
    [
        ("the graph..", ".."),
        ("mot,, puis", ",,"),
        ("fin;. Ensuite", ";."),
        ("wow!! now", "!!"),
        ("le point:. puis", ":."),
        ("le mot,. puis", ",."),
        ("ou;; puis", ";;"),
        ('le "mot.." puis', ".."),
        ("(le mot..) puis", ".."),
    ],
)
def test_doubled_punctuation(text: str, marked: str) -> None:
    """Two marks where one is meant, after a word."""
    assert _rules(text, None) == [("text.double_punctuation", marked)]


@pytest.mark.parametrize(
    "text",
    [
        "wait...",
        "a..b",
        "Quoi?!",
        "Non??",
        "std::vector",
        "e.g., this",
        "the range 1..5 ends",
        "and :. The",
        "the range a..b. Then",
        "the object (4b9458b..) here",
        "see x86.. now",
    ],
)
def test_no_doubled_punctuation(text: str) -> None:
    """An ellipsis, a range or a pair of marks is meant."""
    assert _rules(text, None) == []


def test_marks_touching_code_are_dropped() -> None:
    """A mark that touches an excluded part is about code, not prose."""
    assert _rules("Voir orbit.toml: puis", "fr") == []
    text = "Lisez a: puis b: fin"
    assert typography_marks(text, "fr", [(6, 7)], breakable=True) == [
        Mark(
            "text.space_missing",
            (15, 16),
            ":",
            'no space before ":"',
        ),
    ]


def test_code_tokens_are_the_runs_of_code_words() -> None:
    """A code word gives its runs of word characters and dots, no case."""
    words = {"\\usepackage{listings}", "xshift=0.5cm", "...", "Foo.Bar."}
    assert code_tokens(words) == {
        "usepackage",
        "listings",
        "xshift",
        "0.5cm",
        "foo.bar",
    }
    assert code_tokens(()) == frozenset()


def _coded(text: str, lang: str, *code_words: str) -> list[tuple[str, str]]:
    """Return the rules and marked texts of a PDF text with code words."""
    marks = typography_marks(
        text,
        lang,
        (),
        breakable=False,
        code=code_tokens(code_words),
    )
    return [(m.rule, text[m.span[0] : m.span[1]]) for m in marks]


def test_a_unit_set_as_code_is_no_finding() -> None:
    """A number and unit that the document sets as code is not prose."""
    text = "Shift it by 1cm or 0.5mm, with 4GB free."
    assert _coded(text, "en", "XSHIFT=1CM", "0.5mm") == [
        ("text.unit_space", "4GB"),
    ]
    assert len(_coded(text, "en")) == 3


def test_punctuation_after_a_code_word_is_no_finding() -> None:
    """A blank or no space before a mark after a code word is not prose."""
    english = "Load skins , breakable , or hooks , now."
    assert _coded(english, "en", "skins", "\\tcbuselibrary{breakable}") == [
        ("text.space_before_punctuation", " ,"),
    ]
    french = "Des outils comme tar: le reste, voici: rien."
    assert _coded(french, "fr", "tar") == [("text.space_missing", ":")]
    assert _coded(french, "fr", "tar")[0] == _coded(french, "fr")[1]
    assert _coded("Le mode (tar): rien.", "fr", "tar") == [
        ("text.space_missing", ":"),
    ]


def test_code_tokens_keep_the_checks_linear() -> None:
    """The word before each mark is read once, however long the runs."""
    code = code_tokens({"zz"})
    runs = (
        "a" * 200_000 + ": x",
        "ab: " * 50_000,
        "1" * 200_000 + "km",
        "a" * 200_000 + " , x",
        "a , " * 50_000,
    )
    start = time.perf_counter()
    for text in runs:
        for lang in ("fr", "en"):
            typography_marks(text, lang, (), breakable=False, code=code)
    # A word read again from each mark would take minutes.
    assert time.perf_counter() - start < 10


def test_code_words_spare_no_other_mark() -> None:
    """Quotes, ordinals and doubled marks around code words stay findings."""
    text = 'le mode "tar" est 2\xe8me choix tar..'
    assert _coded(text, "fr", "tar", "2\xe8me", "mode") == _coded(text, "fr")
    assert len(_coded(text, "fr")) == 3


def test_no_language_checks_without_a_language() -> None:
    """Without a language only the units and the doubled marks are judged."""
    assert _rules("rares: 10km", None) == [("text.unit_space", "10km")]
    assert _rules("rares: 10km", "de") == [("text.unit_space", "10km")]
    assert _rules("rare : 2ème \xabx\xbb 10km", "es") == [
        ("text.unit_space", "10km"),
    ]


def test_patterns_and_details() -> None:
    """Each rule names what it found, and the right form when it knows it."""
    french = _marks(
        'Ici: \xabx\xbb, a ; "y", 2ème, 10km, fin..',
        "fr",
    )
    assert [(m.rule, m.pattern, m.detail) for m in french] == [
        ("text.space_missing", ":", 'no space before ":"'),
        ("text.space_missing", "\xab", 'no space after "\xab"'),
        ("text.space_missing", "\xbb", 'no space before "\xbb"'),
        (
            "text.space_breakable",
            ";",
            'the space before ";" can break the line',
        ),
        ("text.quotes", '"', f'quoted with "{ELLIPSIS}" in French text'),
        ("text.ordinal", f"ème{ARROW}e", '"2ème" is written "2e"'),
        ("text.unit_space", "km", "no space between 10 and km"),
        ("text.double_punctuation", "..", '".." doubles the punctuation'),
    ]
    assert all(m.severity is None for m in french)
    opening = _marks("\xab x\xbb", "fr")
    assert [(m.pattern, m.detail) for m in opening] == [
        ("\xab", 'the space after "\xab" can break the line'),
        ("\xbb", 'no space before "\xbb"'),
    ]
    english = _marks("Wait , the \xabx\xbb 21th", "en")
    assert [(m.rule, m.pattern, m.detail) for m in english] == [
        ("text.space_before_punctuation", ",", 'space before ","'),
        (
            "text.quotes",
            "\xab",
            f"quoted with \xab{ELLIPSIS}\xbb in English text",
        ),
        ("text.ordinal", f"th{ARROW}st", '"21th" is written "21st"'),
    ]


def test_marks_are_sorted_by_place_then_rule() -> None:
    """Marks are in the order of the text; at one place, by rule."""
    marks = _marks("Oh!! \xabbof\xbb", "fr")
    assert [(m.rule, m.span) for m in marks] == [
        ("text.space_missing", (2, 3)),
        ("text.double_punctuation", (2, 4)),
        ("text.space_missing", (5, 6)),
        ("text.space_missing", (9, 10)),
    ]


def test_document_text_in_marks_is_quoted() -> None:
    """A long number or ordinal is cut in the pattern and the detail."""
    digits = "1" * 200
    marks = _marks(f"{digits}km {digits}st", "en")
    assert [m.rule for m in marks] == ["text.unit_space", "text.ordinal"]
    unit, ordinal = marks
    assert unit.detail == (
        f"no space between {digits[:PATTERN_MAX_CHARS]}{ELLIPSIS} and km"
    )
    assert ordinal.detail == (
        f'"{digits[:PATTERN_MAX_CHARS]}{ELLIPSIS}" is written '
        f'"{digits[:PATTERN_MAX_CHARS]}{ELLIPSIS}"'
    )
    french = _marks(f"le {'2' * 200}ème", "fr")
    assert french[0].detail == (
        f'"{"2" * PATTERN_MAX_CHARS}{ELLIPSIS}" is written '
        f'"{"2" * PATTERN_MAX_CHARS}{ELLIPSIS}"'
    )
    long_quote = _marks(f'"{"a" * 80}"', "fr")
    assert [m.pattern for m in long_quote] == ['"']


def test_typography_stays_linear_on_hostile_runs() -> None:
    """Each pattern reads a long run of one shape in linear time."""
    blocks = [make_block(run) for run in HOSTILE_RUNS]
    start = time.perf_counter()
    for block in blocks:
        excluded = excluded_spans(block)
        for lang in ("fr", "en", None):
            typography_marks(block.text, lang, excluded, breakable=True)
    # About 6 s on a desktop, 12 s on a slower one; a pattern that
    # restarted at every place of a run would take many minutes.
    assert time.perf_counter() - start < 60
