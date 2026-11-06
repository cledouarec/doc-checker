# Copyright 2026 Christophe Le Douarec
"""Tests for lexicons and language detection."""

import pytest

from doc_checker.text.lang import (
    declared_language,
    detect_language,
    resolve_language,
)
from doc_checker.text.lexicon import (
    DEFAULT_ADMONITION_TITLES,
    is_admonition_title,
    is_modal,
    negation_count,
)

FR_SAMPLE = (
    "Le fichier est dans le dossier de la configuration "
    "et les options sont pour vous"
)
EN_SAMPLE = (
    "The file is in the folder and the options are for you with this tool"
)


def test_negation_count_fr_en() -> None:
    """French and English negation markers are counted."""
    assert negation_count(["Ne", "supprimez", "pas"]) == 2
    assert negation_count(["do", "not", "delete"]) == 1
    assert negation_count(["don't"]) == 1
    assert negation_count(["can\U00002019t"]) == 1
    assert negation_count(["delete"]) == 0


def test_modals() -> None:
    """Modal verbs are recognized regardless of case."""
    assert is_modal("Must")
    assert is_modal("doit")
    assert not is_modal("restart")


def test_detect_language() -> None:
    """French and English texts are told apart by their stopwords."""
    assert detect_language(FR_SAMPLE) == "fr"
    assert detect_language(EN_SAMPLE) == "en"
    assert detect_language("Hello") is None


def test_resolve_language() -> None:
    """An explicit language wins, then the declared one, then detection."""
    assert resolve_language("fr", "en-US", "") == "fr"
    assert resolve_language("auto", "en-US", "") == "en"
    assert resolve_language("auto", "DE", "") == "de"
    assert resolve_language("auto", None, FR_SAMPLE) == "fr"


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        ("en", "en"),
        ("fr-CA", "fr-CA"),
        ("zh-Hant-TW", "zh-Hant-TW"),
        (None, None),
        ("", None),
        (" en", None),
        ("en\n", None),
        ("en_US", None),
        ("en-", None),
        ("toolonglanguage", None),
        ("en\x07\U0000202e", None),
        ('en"><script>alert(1)</script>', None),
    ],
)
def test_declared_language(declared: str | None, expected: str | None) -> None:
    """Only a language tag is kept as the language a document declares."""
    assert declared_language(declared) == expected


def test_admonition_titles() -> None:
    """Default admonition titles are known, whatever their case or colon."""
    assert {"note", "warning", "tip", "see also"} <= DEFAULT_ADMONITION_TITLES
    assert is_admonition_title("Note:")
    assert is_admonition_title(" Voir aussi ")
    assert not is_admonition_title("Notes on the release")
