# Copyright 2026 Christophe Le Douarec
"""Tests for the changes of kind and of heading level between documents."""

from dataclasses import replace

import pytest

from doc_checker.compare.align import Pair, PairKind, align
from doc_checker.compare.drafts import ARROW
from doc_checker.compare.engine import compare_documents
from doc_checker.compare.structure import (
    expected_levels,
    kind_drafts,
    level_drafts,
)
from doc_checker.config import Config
from doc_checker.model.document import HEADING_SOURCE, Block, BlockKind
from doc_checker.model.findings import Finding, Severity
from tests.builders import PARAGRAPHS, make_block, make_document

MA, MI, IN = Severity.MAJOR, Severity.MINOR, Severity.INFO
P = PARAGRAPHS
TITLE = "Configuration options of the server"
OTHER = "Another section of the guide"


def _findings(
    ref: list[Block],
    new: list[Block],
    *,
    pdf: bool = False,
) -> list[Finding]:
    """Compare two documents made of blocks, as HTML or as PDF pages."""
    ref_doc, new_doc = make_document(*ref), make_document(*new)
    if pdf:
        ref_doc = replace(ref_doc, format="pdf")
        new_doc = replace(new_doc, format="pdf")
    return compare_documents(ref_doc, new_doc, Config()).findings


def _rules(
    ref: list[Block],
    new: list[Block],
    *,
    pdf: bool = False,
) -> list[tuple[str, Severity]]:
    """Return the rule and the severity of each finding."""
    return [(f.rule, f.severity) for f in _findings(ref, new, pdf=pdf)]


def _heading(text: str, level: int, source: str | None = None) -> Block:
    """Make a heading, from the outline or the styles of a PDF if given."""
    block = make_block(text, BlockKind.HEADING, level=level)
    return replace(block, attrs={HEADING_SOURCE: source}) if source else block


def _p(text: str) -> Block:
    """Make a paragraph."""
    return make_block(text)


def test_heading_that_became_a_paragraph() -> None:
    """A heading matched with a paragraph is a major finding."""
    (finding,) = _findings(
        [_heading(TITLE, 2), _p(P[0])],
        [_p(TITLE), _p(P[0])],
    )
    assert (finding.rule, finding.severity) == ("structure.heading_lost", MA)
    assert finding.detail == "heading became paragraph"
    assert finding.ref is not None
    assert finding.new is not None
    assert (finding.ref.kind, finding.new.kind) == (
        BlockKind.HEADING,
        BlockKind.PARAGRAPH,
    )


def test_table_that_became_paragraphs() -> None:
    """A table split into paragraphs is one finding, without punctuation."""
    table = make_block(
        "Option | Default\ntimeout | 30\nretries | 3",
        BlockKind.TABLE,
    )
    rows = [_p("Option Default"), _p("timeout 30"), _p("retries 3")]
    assert _rules(
        [_p(P[0]), table, _p(P[1])],
        [_p(P[0]), *rows, _p(P[1])],
    ) == [
        ("block.split_merged", IN),
        ("structure.table_lost", MA),
    ]


def test_code_that_became_a_paragraph() -> None:
    """Code shown as a paragraph is a major finding."""
    code = make_block("pip install example\nexample --version", BlockKind.CODE)
    flat = _p("pip install example example --version")
    assert _rules([_p(P[0]), code, _p(P[1])], [_p(P[0]), flat, _p(P[1])]) == [
        ("structure.code_lost", MA),
    ]


def test_a_lost_list_is_one_finding() -> None:
    """The items of one list that became paragraphs give one finding."""
    items = [
        make_block("Python 3.14 or newer is required.", BlockKind.LIST_ITEM),
        make_block(
            "A working network connection is needed.",
            BlockKind.LIST_ITEM,
        ),
    ]
    plain = [_p(item.text) for item in items]
    (finding,) = _findings(
        [_p(P[0]), *items, _p(P[1])],
        [_p(P[0]), *plain, _p(P[1])],
    )
    assert (finding.rule, finding.severity) == ("structure.list_lost", MI)
    assert finding.detail == "2 of 2 list items became paragraph"


def test_a_paragraph_that_became_a_heading_is_info() -> None:
    """Other changes of kind are only information."""
    assert _rules([_p(TITLE), _p(P[0])], [_heading(TITLE, 2), _p(P[0])]) == [
        ("structure.kind_changed", IN),
    ]


@pytest.mark.parametrize(
    ("ref_block", "new_block", "rule"),
    [
        (
            make_block("pip install example", BlockKind.CODE),
            make_block("pip install example"),
            "structure.code_lost",
        ),
        (
            make_block(
                "Python 3.14 or newer is required.",
                BlockKind.LIST_ITEM,
            ),
            make_block("Python 3.14 or newer is required."),
            "structure.list_lost",
        ),
        (
            replace(
                make_block(TITLE, BlockKind.HEADING, level=2),
                attrs={HEADING_SOURCE: "style"},
            ),
            make_block(TITLE),
            "structure.heading_lost",
        ),
        (
            make_block("Option | Default", BlockKind.TABLE),
            make_block("Option Default"),
            "structure.table_lost",
        ),
    ],
)
def test_pdf_kinds_inferred_from_the_layout_are_info(
    ref_block: Block,
    new_block: Block,
    rule: str,
) -> None:
    """Between PDF files, a kind told by fonts or bullets is information."""
    ref = [_p(P[0]), ref_block, _p(P[1])]
    new = [_p(P[0]), new_block, _p(P[1])]
    assert _rules(ref, new, pdf=True) == [(rule, IN)]


def test_outline_heading_lost_where_the_new_outline_goes_as_deep() -> None:
    """An outline heading is major if the other outline reaches its level."""
    chapter = _heading("Chapter one of the guide", 1, "outline")
    ref = [
        chapter,
        _heading(TITLE, 2, "outline"),
        _p(P[0]),
        _heading(OTHER, 2, "outline"),
        _p(P[1]),
    ]
    deep = [
        chapter,
        _p(TITLE),
        _p(P[0]),
        _heading(OTHER, 2, "outline"),
        _p(P[1]),
    ]
    shallow = [chapter, _p(TITLE), _p(P[0]), _p(OTHER), _p(P[1])]
    assert _rules(ref, deep, pdf=True) == [("structure.heading_lost", MA)]
    assert _rules(ref, shallow, pdf=True) == [
        ("structure.heading_lost", IN),
        ("structure.heading_lost", IN),
    ]


def test_pdf_details_tell_when_kinds_are_inferred() -> None:
    """A PDF finding that rests on the layout says so in its detail."""
    ref = [
        _heading(TITLE, 2, "outline"),
        _p(P[0]),
        _heading(OTHER, 2, "outline"),
        make_block("pip install example", BlockKind.CODE),
        _p(P[1]),
    ]
    new = [
        _p(TITLE),
        _p(P[0]),
        _heading(OTHER, 2, "outline"),
        _p("pip install example"),
        _p(P[1]),
    ]
    assert [(f.rule, f.detail) for f in _findings(ref, new, pdf=True)] == [
        ("structure.heading_lost", "heading became paragraph"),
        (
            "structure.code_lost",
            "code became paragraph; PDF kinds are inferred from the layout",
        ),
    ]


def _equal(*positions: int) -> list[Pair]:
    """Pair each block with the block at the same position."""
    return [Pair(PairKind.EQUAL, (i,), (i,)) for i in positions]


type Case = tuple[list[Block], list[Block], list[Pair]]


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        (
            ([_heading(TITLE, 2)], [_p(TITLE)], _equal(0)),
            (
                "structure.heading_lost",
                "heading became paragraph",
                f"heading{ARROW}paragraph",
            ),
        ),
        (
            ([_p(TITLE)], [_heading(TITLE, 2)], _equal(0)),
            (
                "structure.kind_changed",
                "paragraph became heading",
                f"paragraph{ARROW}heading",
            ),
        ),
        (
            (
                [_heading(TITLE, 2), _p(P[0])],
                [_p(f"{TITLE} {P[0]}")],
                [Pair(PairKind.MERGE, (0, 1), (0,))],
            ),
            (
                "structure.heading_lost",
                "heading and paragraph became paragraph",
                f"heading+paragraph{ARROW}paragraph",
            ),
        ),
    ],
)
def test_change_of_kind_patterns(
    case: Case,
    expected: tuple[str, str, str],
) -> None:
    """A change of kind is described by its kinds before and after."""
    ref, new, pairs = case
    (draft,) = kind_drafts(pairs, ref, new, pdf=False)[0]
    assert (draft.rule, draft.detail, draft.pattern) == expected


def test_a_partly_lost_list_is_filed_under_its_first_lost_item() -> None:
    """A list that lost some items has one draft, at its first lost item."""
    texts = [
        "Python 3.14 or newer is required.",
        "A working network connection is needed.",
        "The cache directory must be writable.",
    ]
    ref = [make_block(text, BlockKind.LIST_ITEM) for text in texts]
    new = [ref[0], *(_p(text) for text in texts[1:])]
    drafts = kind_drafts(_equal(0, 1, 2), ref, new, pdf=False)
    assert list(drafts) == [1]
    (draft,) = drafts[1]
    assert (draft.rule, draft.detail, draft.pattern) == (
        "structure.list_lost",
        "2 of 3 list items became paragraph",
        f"list_item{ARROW}paragraph",
    )
    assert draft.ref is not None
    assert draft.ref.text == "\n".join(texts[1:])


def _sections(levels: list[int], source: str | None = None) -> list[Block]:
    """Make headings of the given levels, each followed by a paragraph."""
    blocks: list[Block] = []
    for index, level in enumerate(levels):
        blocks += [
            _heading(f"Section number {index} of the guide", level, source),
            _p(f"Paragraph number {index} talks about its own topic."),
        ]
    return blocks


def test_shifted_levels_are_no_change() -> None:
    """Headings all one level deeper keep their structure."""
    assert _rules(_sections([1, 2, 2, 3]), _sections([2, 3, 3, 4])) == []


def test_one_demoted_heading_is_reported() -> None:
    """One heading of a level that the others keep is reported."""
    (finding,) = _findings(_sections([1, 2, 2, 2]), _sections([1, 2, 2, 3]))
    assert (finding.rule, finding.severity) == ("structure.heading_level", MI)
    assert finding.detail == (
        "level 2 became level 3, level 2 was expected from the other headings"
    )


def test_flattened_levels_report_the_deeper_headings() -> None:
    """Sections raised to the level of their chapters are reported."""
    rules = _rules(_sections([1, 2, 2, 1, 2]), _sections([1, 1, 1, 1, 1]))
    assert rules == [("structure.heading_level", MI)] * 3


def test_a_tie_falls_back_on_the_common_shift() -> None:
    """Without a majority, a level keeps the shift of the other headings."""
    assert _rules(_sections([2, 2, 3, 3]), _sections([2, 2, 3, 4])) == [
        ("structure.heading_level", MI),
    ]


def test_the_only_heading_of_a_level_is_never_reported() -> None:
    """A level with one heading has nothing to compare with."""
    assert _rules(_sections([2, 2, 3]), _sections([2, 2, 4])) == []


def test_expected_levels_follow_the_majority() -> None:
    """Each level maps to the level that most of its headings took."""
    ref, new = _sections([1, 2, 2, 2]), _sections([2, 3, 3, 4])
    pairs = align(ref, new, match=0.6, moved_match=0.8)
    assert expected_levels(pairs, ref, new) == {1: 2, 2: 3}


def test_pdf_levels_compare_between_outline_headings_only() -> None:
    """Between PDF files, only outline headings change level at minor."""
    outline = _rules(
        _sections([1, 2, 2, 2], "outline"),
        _sections([1, 2, 2, 3], "outline"),
        pdf=True,
    )
    styles = _rules(
        _sections([1, 2, 2, 2], "style"),
        _sections([1, 2, 2, 3], "style"),
        pdf=True,
    )
    assert outline == [("structure.heading_level", MI)]
    assert styles == [("structure.heading_level", IN)]


def test_split_and_merged_headings_are_left_out_of_the_map() -> None:
    """Only one to one pairs of headings count for the level map."""
    ref, new = _sections([1, 2, 2]), _sections([1, 3, 3])
    heads = [0, 2, 4]
    pairs = [
        Pair(PairKind.SPLIT, (heads[0],), (0, 1)),
        Pair(PairKind.MERGE, (heads[1], heads[2]), (2,)),
    ]
    assert expected_levels(pairs, ref, new) == {}
    assert level_drafts(pairs, ref, new, pdf=False) == {}


def test_expected_levels_without_heading_pairs_is_empty() -> None:
    """Paragraphs only give an empty map."""
    ref = [_p("Only a paragraph here, nothing else.")]
    pairs = [Pair(PairKind.EQUAL, (0,), (0,))]
    assert expected_levels(pairs, ref, ref) == {}


def test_expected_levels_shift_fallback() -> None:
    """Levels without a majority move by the most frequent shift."""
    ref, new = _sections([1, 2, 2, 3]), _sections([2, 3, 4, 4])
    pairs = align(ref, new, match=0.6, moved_match=0.8)
    # Level 2 has no majority (3, 4): shift +1 (twice) applies.
    assert expected_levels(pairs, ref, new) == {1: 2, 2: 3, 3: 4}


def test_expected_levels_are_made_increasing() -> None:
    """A level that does not go deeper than the one above is pushed down."""
    ref, new = _sections([1, 2, 2]), _sections([1, 1, 1])
    pairs = align(ref, new, match=0.6, moved_match=0.8)
    assert expected_levels(pairs, ref, new) == {1: 1, 2: 2}
