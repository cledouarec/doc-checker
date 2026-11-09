# Copyright 2026 Christophe Le Douarec
"""Tests for the bundled Hunspell dictionaries."""

import gc
import hashlib
import warnings
from importlib.resources import files

import pytest

from doc_checker.spell.dictionary import load_dictionary

APOSTROPHE = "\U00002019"


def test_french_dictionary_knows_elisions_compounds_and_capitals() -> None:
    """Elisions, hyphenated compounds, accented capitals and both spellings."""
    fr = load_dictionary("fr")
    for word in (
        "l'arbre",
        f"l{APOSTROPHE}arbre",
        f"aujourd{APOSTROPHE}hui",
        "c'est-à-dire",
        "peut-être",
        "Étienne",
        "ÉTÉ",
        "2e",
        "XIXe",
        "événement",
        "évènement",
    ):
        assert fr.lookup(word), word
    for word in ("configuraiton", "répertoir", "2ème", "versionning"):
        assert not fr.lookup(word), word


def test_english_dictionary_is_the_one_of_spylls() -> None:
    """The en_US dictionary that spylls ships knows contractions."""
    en = load_dictionary("en")
    assert en.lookup("configuration")
    assert en.lookup(f"don{APOSTROPHE}t")
    assert not en.lookup("recieve")


def test_dictionaries_are_loaded_once() -> None:
    """A second request gives the dictionary already loaded."""
    assert load_dictionary("fr") is load_dictionary("fr")


def test_loading_leaves_no_file_open() -> None:
    """Reading a dictionary leaves no file for the garbage collector."""
    load_dictionary.cache_clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_dictionary("en")
        load_dictionary("fr")
        gc.collect()
    assert not [w for w in caught if w.category is ResourceWarning]


def test_only_french_and_english_have_a_dictionary() -> None:
    """Another language has no dictionary."""
    with pytest.raises(ValueError, match="no dictionary for 'de'"):
        load_dictionary("de")


def test_french_dictionary_ships_with_its_licence_and_source() -> None:
    """The data files, their licence and their source are in the package."""
    data = files("doc_checker.spell") / "data" / "fr"
    names = {item.name for item in data.iterdir()}
    assert {
        "fr-toutesvariantes.aff",
        "fr-toutesvariantes.dic",
        "README_dict_fr.txt",
        "LICENSE",
        "SOURCE.txt",
    } <= names
    licence = (data / "LICENSE").read_text(encoding="utf-8")
    assert "Mozilla Public License Version 2.0" in licence
    head = (data / "fr-toutesvariantes.aff").read_text(encoding="utf-8")
    assert "v7.7" in head[:400]
    assert "hunspell-french-dictionaries-v7.7.zip" in (
        data / "SOURCE.txt"
    ).read_text(encoding="utf-8")


def test_shipped_files_match_their_checksums() -> None:
    """The data files are byte for byte those that SOURCE.txt hashes.

    A checkout that converts line endings, as Git on Windows may, would
    change them: `.gitattributes` marks the data as no text.
    """
    data = files("doc_checker.spell") / "data" / "fr"
    source = (data / "SOURCE.txt").read_text(encoding="utf-8")
    shipped = source.split("SHA-256 of the files shipped:")[1]
    sums = dict(
        reversed(line.split()) for line in shipped.splitlines() if "  " in line
    )
    assert sorted(sums) == [
        "README_dict_fr.txt",
        "fr-toutesvariantes.aff",
        "fr-toutesvariantes.dic",
    ]
    for name, digest in sums.items():
        assert hashlib.sha256((data / name).read_bytes()).hexdigest() == digest
