# Copyright 2026 Christophe Le Douarec
"""Tests for how the audit shows the document in its findings."""

import time
from dataclasses import replace

import pytest

from doc_checker.audit.accessibility import accessibility_drafts
from doc_checker.audit.links import broken_link_drafts
from doc_checker.audit.sides import LinkMarks, block_side, image_side, quote
from doc_checker.compare.aggregate import PATTERN_MAX_CHARS
from doc_checker.config import AuditThresholds
from doc_checker.images.refs import describe
from doc_checker.model.document import BlockKind, ImageRef, Link
from doc_checker.model.facts import HtmlPageFacts
from doc_checker.text.normalize import ELLIPSIS
from tests.builders import make_audit_document, make_block, make_image_block

# A link text that nearly matches at every place of its block: one long
# word inside a longer one, or many words before a last one that differs.
NEAR_MATCHES = [
    ("a" * 360_000 + "a" * 40_000, "a" * 40_000),
    (" ".join(["a"] * 40_000), " ".join([*["a"] * 3_999, "b"])),
]
HOME = (HtmlPageFacts("index.html", "en", "Home"),)


def _marks(text: str, *links: str) -> list[list[tuple[str, str]]]:
    """Return the segments of the side of each link of a block."""
    marks = LinkMarks(make_block(text), links)
    return [
        [(s.op, s.text) for s in marks.side(index).segments]
        for index in range(len(links))
    ]


def test_links_are_placed_in_their_order() -> None:
    """A text before its link, and an earlier link of it, are not marked."""
    text = "The setup is quick: read the setup notes, then the setup list."
    assert _marks(text, "setup", "setup") == [
        [
            ("equal", "The setup is quick: read the "),
            ("mark", "setup"),
            ("equal", " notes, then the setup list."),
        ],
        [
            ("equal", "The setup is quick: read the setup notes, then the "),
            ("mark", "setup"),
            ("equal", " list."),
        ],
    ]
    assert _marks("See the guide, then here.", "the guide", "here") == [
        [("equal", "See "), ("mark", "the guide"), ("equal", ", then here.")],
        [("equal", "See the guide, then "), ("mark", "here"), ("equal", ".")],
    ]


def test_a_text_after_its_only_link_is_taken_for_it() -> None:
    """The text alone cannot tell a link from a later mention of its text."""
    assert _marks("Read the setup, as the setup says.", "setup") == [
        [
            ("equal", "Read the setup, as the "),
            ("mark", "setup"),
            ("equal", " says."),
        ],
    ]


def test_a_link_not_found_marks_the_last_link_of_its_text() -> None:
    """A link not found after the one before marks that one, if it shares."""
    assert _marks("Only here.", "here", "here") == [
        [("equal", "Only "), ("mark", "here"), ("equal", ".")],
        [("equal", "Only "), ("mark", "here"), ("equal", ".")],
    ]
    assert _marks("Somewhere.", "here") == [[("equal", "Somewhere.")]]
    assert _marks("Go there.", "", "there") == [
        [("equal", "Go there.")],
        [("equal", "Go "), ("mark", "there"), ("equal", ".")],
    ]


def test_link_texts_match_as_whole_words_across_blanks() -> None:
    """Words match whole, any run of blanks matching any run of blanks."""
    assert _marks("Please click\nhere now.", "click here") == [
        [("equal", "Please "), ("mark", "click\nhere"), ("equal", " now.")],
    ]
    assert _marks("A where, a here.", "here") == [
        [("equal", "A where, a "), ("mark", "here"), ("equal", ".")],
    ]
    assert _marks("See (a).", "(a)") == [
        [("equal", "See "), ("mark", "(a)"), ("equal", ".")],
    ]


def test_placing_stays_linear_on_hostile_blocks() -> None:
    """Links not found, or found among many mentions, are placed quickly."""
    cases = [
        (" ".join(["word"] * 50_000), [f"absent{k}" for k in range(20_000)]),
        (" ".join(["here"] * 50_000), ["here"] * 25_000),
        ("here " * 20_000 + "x " * 50_000, ["here", "x"] * 10_000),
    ]
    for text, links in cases:
        block = make_block(text)
        start = time.perf_counter()
        marks = LinkMarks(block, links)
        sides = [marks.side(index) for index in range(len(links))]
        assert time.perf_counter() - start < 1.0
        assert len(sides) == len(links)


def test_quotes_and_sides_of_blocks() -> None:
    """A quote is cut, a block shown whole, a figure as it is described."""
    assert quote("x" * (PATTERN_MAX_CHARS + 5)) == (
        "x" * PATTERN_MAX_CHARS + ELLIPSIS
    )
    block = make_block("A heading.", BlockKind.HEADING)
    side = block_side(block)
    assert (side.text, side.kind, side.source) == (
        "A heading.",
        BlockKind.HEADING,
        block.source,
    )
    image = ImageRef("a.png", "png", "1" * 32, alt="Steps")
    figure = image_side(make_image_block(image))
    assert (figure.text, figure.image) == (describe(image), image)
    assert [s.op for s in figure.segments] == ["equal"]


@pytest.mark.parametrize(
    ("text", "link"),
    NEAR_MATCHES,
    ids=["one-long-word", "many-words"],
)
def test_failed_candidates_count_against_the_budget(
    text: str,
    link: str,
) -> None:
    """A link text that nearly matches everywhere is placed quickly."""
    broken = replace(
        make_block(text),
        links=(Link(link, "gone.html", page="gone.html"),),
    )
    start = time.perf_counter()
    drafts, _ = broken_link_drafts(
        make_audit_document(broken, html_pages=HOME),
    )
    assert time.perf_counter() - start < 1.0
    assert [d.rule for d in drafts] == ["link.page_missing"]
    vague = replace(
        make_block(f"See here {text}"),
        links=(
            Link("here", "a.html", page="a.html", found=True),
            Link(link, "b.html", page="b.html", found=True),
        ),
    )
    start = time.perf_counter()
    (draft,) = accessibility_drafts(
        make_audit_document(vague, html_pages=HOME),
        AuditThresholds(),
    )
    assert time.perf_counter() - start < 1.0
    assert draft.ref is not None
    assert [s.text for s in draft.ref.segments if s.op == "mark"] == ["here"]
