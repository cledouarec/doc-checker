# Copyright 2026 Christophe Le Douarec
"""Tests for the comparison engine."""

from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, NoReturn

import pytest

from doc_checker.compare import engine as engine_module
from doc_checker.compare.engine import (
    CompareResult,
    CompareStats,
    compare_documents,
)
from doc_checker.config import (
    Config,
    IgnoreConfig,
    PdfConfig,
    SpellingConfig,
)
from doc_checker.errors import InputError
from doc_checker.model.document import Block, BlockKind
from doc_checker.model.findings import DiffSegment, Severity
from doc_checker.pipeline import run_compare
from doc_checker.spell import speller as speller_module
from doc_checker.spell.speller import Speller
from tests.builders import (
    PARAGRAPHS,
    SAMPLE_EN,
    SPLIT_PARTS,
    SPLIT_WHOLE,
    build_mkdocs_like,
    make_block,
    make_document,
    parts_tables,
    random_blocks,
)

if TYPE_CHECKING:
    from pathlib import Path

P = PARAGRAPHS
ADDED = "An extra paragraph was added here."
CFG = Config()
DOWNLOADS = "The installer downloads files."
DOWNLOADS_TYPO = "The installer donwloads files."


class _FailingDictionary:
    """A dictionary whose every lookup fails."""

    @staticmethod
    def lookup(word: str) -> bool:
        """Fail, as spylls might on a hostile word."""
        raise IndexError(word)


def _rules(result: CompareResult) -> list[tuple[str, Severity]]:
    return [(f.rule, f.severity) for f in result.findings]


def _b(*texts: str) -> list[Block]:
    return [make_block(t) for t in texts]


def _heading(text: str, level: int) -> Block:
    return make_block(text, BlockKind.HEADING, level=level)


def test_identical_documents_have_no_findings() -> None:
    """Identical documents produce no finding."""
    ref, new = make_document(*_b(*P)), make_document(*_b(*P))
    assert compare_documents(ref, new, Config()).findings == []


def test_missing_block_is_critical() -> None:
    """A missing block is a critical finding with its reference text."""
    ref = make_document(*_b(*P))
    new = make_document(*_b(P[0], P[2], P[3]))
    result = compare_documents(ref, new, Config())
    assert _rules(result) == [("block.missing", Severity.CRITICAL)]
    assert result.findings[0].ref is not None
    assert result.findings[0].ref.text == P[1]
    assert result.findings[0].new is None


def test_missing_section_is_one_finding() -> None:
    """A section missing with all its content is a single finding."""
    ref = make_document(
        _heading("Intro chapter", 1),
        make_block(P[0]),
        _heading("Old section", 2),
        make_block(P[1]),
        make_block(P[2]),
        _heading("Kept section", 2),
        make_block(P[3]),
    )
    new = make_document(
        _heading("Intro chapter", 1),
        make_block(P[0]),
        _heading("Kept section", 2),
        make_block(P[3]),
    )
    result = compare_documents(ref, new, Config())
    assert _rules(result) == [("section.missing", Severity.CRITICAL)]
    assert result.findings[0].detail == "3 blocks missing in this section"


def test_added_and_moved_blocks() -> None:
    """Added blocks are minor and moved blocks are informational."""
    ref = make_document(*_b(*P))
    added = compare_documents(ref, make_document(*_b(*P, ADDED)), Config())
    assert _rules(added) == [("block.added", Severity.MINOR)]
    moved_new = make_document(*_b(*P[1:], P[0]))
    moved = compare_documents(ref, moved_new, Config())
    assert _rules(moved) == [("block.moved", Severity.INFO)]


def test_hunk_finding_has_diff_segments() -> None:
    """A word level finding shows the deleted and inserted words."""
    text = (
        "Install the package with the default configuration before you start."
    )
    ref = make_document(make_block(text))
    new = make_document(
        make_block(text.replace("configuration", "configuraiton")),
    )
    (finding,) = compare_documents(ref, new, Config()).findings
    assert (finding.rule, finding.severity) == ("text.typo", Severity.MAJOR)
    assert finding.ref is not None
    assert finding.new is not None
    assert DiffSegment("delete", "configuration") in finding.ref.segments
    assert DiffSegment("insert", "configuraiton") in finding.new.segments


def test_split_is_info_only() -> None:
    """A block split in two without text change is informational."""
    ref = make_document(*_b(P[0], SPLIT_WHOLE, P[1]))
    new = make_document(*_b(P[0], *SPLIT_PARTS, P[1]))
    assert _rules(compare_documents(ref, new, Config())) == [
        ("block.split_merged", Severity.INFO),
    ]


def test_punctuation_only_blocks_are_ignored() -> None:
    """Blocks without letters or digits are not compared."""
    ref = make_document(*_b(P[0], "\U00002014"))
    new = make_document(*_b(P[0]))
    assert compare_documents(ref, new, Config()).findings == []


def test_chrome_keys_are_removed_on_both_sides() -> None:
    """Chrome found in one document is ignored in both."""
    banner = make_block("Docs for version 2.0.")
    ref = make_document(banner, *_b(P[0]))
    new = make_document(
        *_b(P[0]),
        chrome_keys=frozenset({banner.key_strict}),
    )
    assert compare_documents(ref, new, Config()).findings == []


def test_severity_override_disable_and_min_severity() -> None:
    """Severity overrides, disabled rules and the minimum severity apply."""
    ref, new = make_document(*_b(*P)), make_document(*_b(*P, ADDED))
    override = Config(severity={"block.added": Severity.INFO})
    assert _rules(compare_documents(ref, new, override)) == [
        ("block.added", Severity.INFO),
    ]
    disabled = Config(ignore=IgnoreConfig(rules=("block.added",)))
    assert compare_documents(ref, new, disabled).findings == []
    minimum = Config(min_severity=Severity.MAJOR)
    assert compare_documents(ref, new, minimum).findings == []


def test_ignore_patterns_and_finding_ids() -> None:
    """Ignored patterns and ignored finding identifiers are honored."""
    ref = make_document(
        make_block("Version 1.2.3 is installed by default here."),
    )
    new = make_document(
        make_block("Version 1.2.4 is installed by default here."),
    )
    patterns = Config(ignore=IgnoreConfig(patterns=(r"\d+\.\d+\.\d+",)))
    assert compare_documents(ref, new, patterns).findings == []
    (finding,) = compare_documents(ref, new, Config()).findings
    ignored = Config(ignore=IgnoreConfig(findings=(finding.id,)))
    assert compare_documents(ref, new, ignored).findings == []


ACCENTED = [
    f"Le paragraphe {word} décrit le service."
    for word in ("un", "deux", "trois", "quatre", "cinq", "six")
]


def test_a_systematic_pattern_is_one_finding_and_can_be_ignored() -> None:
    """Six identical accent losses are one group, which can be ignored."""
    ref = make_document(*_b(*ACCENTED))
    new = make_document(*_b(*(t.replace("é", "e") for t in ACCENTED)))
    (group,) = compare_documents(ref, new, Config()).findings
    assert (group.rule, len(group.occurrences)) == ("text.accent_lost", 6)
    ignored = Config(ignore=IgnoreConfig(findings=(group.id,)))
    assert compare_documents(ref, new, ignored).findings == []


def test_an_occurrence_can_be_ignored_below_the_group_size() -> None:
    """An ignored occurrence leaves five findings, still a group of five."""
    ref = make_document(*_b(*ACCENTED))
    new = make_document(*_b(*(t.replace("é", "e") for t in ACCENTED)))
    (group,) = compare_documents(ref, new, Config()).findings
    first = group.occurrences[0].id
    one = Config(ignore=IgnoreConfig(findings=(first,)))
    (rest,) = compare_documents(ref, new, one).findings
    assert len(rest.occurrences) == 5
    two = Config(ignore=IgnoreConfig(findings=(first, rest.occurrences[0].id)))
    findings = compare_documents(ref, new, two).findings
    assert [len(f.occurrences) for f in findings] == [0, 0, 0, 0]


def test_an_ignored_group_stays_ignored_below_the_group_size() -> None:
    """The members of an ignored group stay ignored when four are left."""
    ref = make_document(*_b(*ACCENTED))
    new = make_document(*_b(*(t.replace("é", "e") for t in ACCENTED)))
    (group,) = compare_documents(ref, new, Config()).findings
    first, second = (item.id for item in group.occurrences[:2])
    ignored = Config(ignore=IgnoreConfig(findings=(group.id, first, second)))
    assert compare_documents(ref, new, ignored).findings == []


def test_toc_blocks_are_skipped_unless_configured() -> None:
    """Contents blocks are compared only with the compare_toc setting."""
    toc = make_block("Getting started . . . . 2", BlockKind.TOC)
    ref = make_document(toc, *_b(*P))
    new = make_document(*_b(*P))
    result = compare_documents(ref, new, Config())
    assert (result.findings, result.stats.ref_blocks) == ([], 4)
    with_toc = Config(pdf=PdfConfig(compare_toc=True))
    assert _rules(compare_documents(ref, new, with_toc)) == [
        ("block.missing", Severity.CRITICAL),
    ]


@pytest.mark.parametrize("kind", [BlockKind.TOC, BlockKind.INDEX])
def test_contents_and_index_of_the_new_document_are_skipped(
    kind: BlockKind,
) -> None:
    """Contents and index blocks of the new document are skipped as well."""
    contents = make_block("Getting started . . . . 2", kind)
    ref = make_document(*_b(*P))
    new = make_document(contents, *_b(*P))
    result = compare_documents(ref, new, Config())
    assert (result.findings, result.stats.new_blocks) == ([], 4)
    with_toc = Config(pdf=PdfConfig(compare_toc=True))
    assert _rules(compare_documents(ref, new, with_toc)) == [
        ("block.added", Severity.MINOR),
    ]


def test_renumbered_contents_are_not_a_difference() -> None:
    """Contents whose page numbers changed are compared only on request."""
    ref = make_document(
        make_block("Getting started . . . . 2", BlockKind.TOC),
        *_b(*P),
    )
    new = make_document(
        make_block("Getting started . . . . 3", BlockKind.TOC),
        *_b(*P),
    )
    assert compare_documents(ref, new, Config()).findings == []
    with_toc = Config(pdf=PdfConfig(compare_toc=True))
    assert _rules(compare_documents(ref, new, with_toc)) == [
        ("text.number_changed", Severity.CRITICAL),
    ]


def test_stats() -> None:
    """Statistics count blocks, matches and moves."""
    ref = make_document(*_b(*P))
    new = make_document(*_b(*P[1:], P[0]))
    stats = compare_documents(ref, new, Config()).stats
    assert stats == CompareStats(
        ref_blocks=4,
        new_blocks=4,
        matched=4,
        moved=1,
    )


def test_run_compare_rejects_mixed_formats(tmp_path: Path) -> None:
    """Comparing an HTML site with a PDF is an input error."""
    site = build_mkdocs_like(SAMPLE_EN, tmp_path / "site")
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.7")
    with pytest.raises(InputError, match="same format"):
        run_compare(site, pdf, Config())


def test_unmatched_admonition_titles_are_not_compared() -> None:
    """A default admonition title on one side only gives no finding."""
    ref = make_document(*_b(P[0], "Note", P[1]))
    new = make_document(*_b(P[0], P[1], "warning:"))
    assert compare_documents(ref, new, Config()).findings == []
    titled = make_document(make_block(P[0]), _heading("Summary", 2))
    result = compare_documents(titled, make_document(*_b(P[0])), Config())
    assert _rules(result) == [("section.missing", Severity.CRITICAL)]


def test_cell_separators_are_compared_between_two_tables() -> None:
    """A table that stays a table keeps its cell separators in the diff."""
    ref = make_document(
        make_block("Option | Default\ntimeout | 30", BlockKind.TABLE),
    )
    new = make_document(
        make_block("Option | Default\ntimeout 30", BlockKind.TABLE),
    )
    result = compare_documents(ref, new, Config())
    assert _rules(result) == [("text.punctuation_changed", Severity.INFO)]
    assert result.findings[0].ref is not None
    assert result.findings[0].ref.kind is BlockKind.TABLE


def _one_change(ref_text: str, new_text: str, cfg: Config) -> CompareResult:
    """Compare two documents of one paragraph each."""
    ref = make_document(make_block(ref_text))
    return compare_documents(ref, make_document(make_block(new_text)), cfg)


def test_dictionary_is_loaded_only_for_a_typo_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No dictionary is loaded until a close word substitution needs one."""
    calls = _counted_loads(monkeypatch)
    same = make_document(*_b(*P))
    assert compare_documents(same, same, CFG).findings == []
    number = _one_change("Accepts 100 requests.", "Accepts 10 requests.", CFG)
    assert _rules(number) == [("text.number_changed", Severity.CRITICAL)]
    assert calls == []
    typo = _one_change(DOWNLOADS, DOWNLOADS_TYPO, CFG)
    assert _rules(typo) == [("text.typo", Severity.MAJOR)]
    assert calls


def test_personal_words_are_known_words_in_a_comparison() -> None:
    """A personal word of the configuration is a word, not a typo."""
    cfg = Config(spelling=SpellingConfig(words=("donwloads",)))
    result = _one_change(DOWNLOADS, DOWNLOADS_TYPO, cfg)
    assert _rules(result) == [("text.word_replaced", Severity.MAJOR)]
    assert result.new.warnings == []


def test_a_dictionary_that_cannot_be_loaded_is_a_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without its dictionary, the reference vocabulary tells a typo."""

    def broken(lang: str) -> NoReturn:
        raise OSError(lang)

    monkeypatch.setattr(speller_module, "load_dictionary", broken)
    new = make_document(make_block("Please receive the file."))
    ref = make_document(make_block("Please recieve the file."))
    result = compare_documents(ref, new, Config())
    assert _rules(result) == [("text.typo", Severity.MAJOR)]
    assert result.new.warnings == [
        (
            "typo check: cannot load the 'en' dictionary: OSError: en; the "
            "typos that needed it were told by the reference vocabulary only"
        ),
    ]
    assert new.warnings == []
    assert result.ref.warnings == []


def test_a_spent_budget_is_a_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    """A word that the budget left unchecked is classified as in phase 1."""
    monkeypatch.setattr(engine_module, "Speller", partial(Speller, work=0))
    result = _one_change("Please recieve it.", "Please receive it.", CFG)
    assert _rules(result) == [("text.typo", Severity.MAJOR)]
    assert result.new.warnings == [
        (
            "typo check: the dictionaries stopped after 0 distinct words; "
            "the other typos were told by the reference vocabulary only"
        ),
    ]


def test_failed_lookups_are_a_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    """A new word that spylls fails on is judged as in phase 1, and counted."""
    monkeypatch.setattr(
        speller_module,
        "load_dictionary",
        lambda _: _FailingDictionary(),
    )
    result = _one_change("Please recieve it.", "Please receive it.", CFG)
    assert _rules(result) == [("text.typo", Severity.MAJOR)]
    assert result.new.warnings == [
        "typo check: 1 word(s) could not be looked up and were taken as known",
    ]


def _counted_loads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Count the dictionaries that the speller loads."""
    calls: list[str] = []
    load = speller_module.load_dictionary

    def counted(lang: str) -> object:
        calls.append(lang)
        return load(lang)

    monkeypatch.setattr(speller_module, "load_dictionary", counted)
    return calls


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (BlockKind.PARAGRAPH, ("text.typo_fixed", Severity.INFO)),
        (BlockKind.CODE, ("text.typo", Severity.MAJOR)),
        (BlockKind.FORMULA, ("text.typo", Severity.MAJOR)),
    ],
)
def test_only_prose_is_judged_by_the_dictionaries(
    monkeypatch: pytest.MonkeyPatch,
    kind: BlockKind,
    expected: tuple[str, Severity],
) -> None:
    """A code block keeps the classification of phase 1, with no lookup."""
    calls = _counted_loads(monkeypatch)
    ref = make_document(make_block("Please recieve the file.", kind))
    new = make_document(make_block("Please receive the file.", kind))
    result = compare_documents(ref, new, CFG)
    assert _rules(result) == [expected]
    assert bool(calls) is (kind is BlockKind.PARAGRAPH)


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ((), ("text.typo_fixed", Severity.INFO)),
        ((("pcell",), ("cell",)), ("text.typo", Severity.MAJOR)),
    ],
    ids=["prose", "inline-code"],
)
def test_inline_code_keeps_the_classification_of_phase_1(
    monkeypatch: pytest.MonkeyPatch,
    code: tuple[tuple[str, ...], ...],
    expected: tuple[str, Severity],
) -> None:
    """A word of the inline code of a block is not judged by a dictionary."""
    calls = _counted_loads(monkeypatch)
    ref_block = make_block("Set the pcell option.")
    new_block = make_block("Set the cell option.")
    if code:
        ref_block = replace(ref_block, code_spans=code[0])
        new_block = replace(new_block, code_spans=code[1])
    ref, new = make_document(ref_block), make_document(new_block)
    result = compare_documents(ref, new, CFG)
    assert _rules(result) == [expected]
    assert bool(calls) is not bool(code)


def test_cut_alignment_is_a_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An alignment cut by its budget is told in the warnings."""
    monkeypatch.setattr("doc_checker.compare.align.ALIGN_WORK_BASE", 10)
    monkeypatch.setattr("doc_checker.compare.align.ALIGN_WORK_PER_CHAR", 0)
    ref = make_document(*random_blocks(1, 20))
    new = make_document(*random_blocks(2, 20))
    result = compare_documents(ref, new, Config())
    note = (
        "20 block(s) were compared by identical text only: "
        "the alignment reached its work limit"
    )
    assert result.new.warnings == [note]


def test_a_long_table_in_both_documents_is_never_missing() -> None:
    """A long table changed here and there gives a finding per changed row."""
    ref_text, new_text, changed = parts_tables(6_000, 0.3)
    ref = make_document(make_block(ref_text, BlockKind.TABLE))
    new = make_document(make_block(new_text, BlockKind.TABLE))
    result = compare_documents(ref, new, Config())
    assert len(ref.blocks[0].key_loose) > 130_000
    assert {f.rule for f in result.findings} == {"text.number_changed"}
    occurrences = sum(len(f.occurrences) or 1 for f in result.findings)
    assert occurrences == changed
    assert result.new.warnings == []
