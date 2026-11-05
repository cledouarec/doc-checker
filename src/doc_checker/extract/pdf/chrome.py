# Copyright 2026 Christophe Le Douarec
"""Find the chrome of a PDF: running headers, footers and page numbers.

Two PDF generators never lay out the top and the bottom of a page in the same
way, so the chrome is left out of each document, line by line, before the
comparison. Otherwise a layout difference would become a finding. The chrome
lies at the edges of a page: in its bands, or beyond them on its first or last
line, near the edge and set apart from the text by a gap. A segment there, not
much larger than the body text, is chrome if its text repeats from page to
page, if it is a page number that follows the pages or if it is a running
title. Beyond the bands, a line is chrome whole or not at all, as a row of a
table may hold a cell that looks like chrome.
"""

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Final, Literal

from doc_checker.extract.pdf.layout import LINE_TOLERANCE
from doc_checker.text.normalize import loose_key

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from doc_checker.config import PdfConfig
    from doc_checker.extract.pdf.reader import Bookmark
    from doc_checker.extract.pdf.segments import Segment

type _Zone = Literal["top", "bottom"]

CHROME_MIN_PAGES: Final = 2  # Fewest pages on which a text repeats.
# The sizes are multiples of the body size.
CHROME_MAX_SIZE: Final = 1.15  # Largest size of a text that can be chrome.
HEADING_MIN_SIZE: Final = 1.15  # Smallest size of a text that is a title.
# The first and the last line of a page are looked at beyond the bands, up to a
# share of the page height from its edge: like the bands, and unlike the other
# limits, that reach is not a multiple of a size. There, a line may be chrome
# only if a gap wider than CHROME_GAP sizes separates it from the next row of
# the page, as LaTeX sets its running headers and footers apart from the text,
# while the rows of most tables follow each other at their pitch. The rows of
# an airy table may stand apart too, so a text there only repeats as it is.
OUTER_ZONE: Final = 0.2  # Share of the page height, from each edge.
CHROME_GAP: Final = 2.0  # Gap that an outermost line must exceed, in sizes.
PAGE_NUMBER_DIGITS: Final = 9  # Most digits of a page number.

_ROMAN_VALUES: Final = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100}
_NUMBERS: Final = re.compile(r"\d+")
# The forms of a page number are tested on the neutral key of a text. A number
# may have "page" before it, and "of" and a total after it. The form is not
# enough to tell a page number, as a lone number may be the cell of a table:
# the number must also follow the pages.
_PAGE_NUMBER: Final = re.compile(
    r"^(?:(?:page|p)\s)?#(?:\s(?:of|sur|de)(?:\s#)?)?$",
)
# Two numbers with no word between them have the key of 3/10, but also that of
# 2.5 and of 1,200, so the text must have the slash as well.
_PAGE_FRACTION: Final = re.compile(r"^(?:(?:page|p)\s)?#\s#$")
# Well formed roman numerals from 1 to 99 only, so that the words and the
# acronyms made of the same letters, such as mix, civil, CLI or xml, are not
# page numbers.
_ROMAN_NUMBER: Final = re.compile(
    r"^(?=[ivxl])(?:xc|xl|l?x{0,3})(?:ix|iv|v?i{0,3})$",
)
_TITLE_PREFIX: Final = re.compile(
    r"^(?:(?:chapter|chapitre|part|partie|section|appendix|annexe)\s)?"
    r"(?:#\s?)*",
)


@dataclass(frozen=True, slots=True)
class _RepeatKey:
    """What the same chrome has in common from page to page.

    Attributes:
        zone: The zone of the page that holds the text.
        parity: Parity of the pages among which the text repeats, as the
            facing pages have their own chrome, or None for all the pages.
        text: The text, as `_repeat_text` keeps it.
        size: Font size of the text, in points.
    """

    zone: _Zone
    parity: int | None
    text: str
    size: float


@dataclass(frozen=True, slots=True)
class _Candidate:
    """A segment of a zone that is small enough to be chrome.

    Attributes:
        segment: The segment.
        zone: The zone of the page that holds it.
        banded: Whether it lies in a band, rather than beyond the bands on an
            outermost line of the page.
    """

    segment: Segment
    zone: _Zone
    banded: bool


def neutral_key(text: str) -> str:
    """Return the key of a text without its numbers.

    Args:
        text: Any text.

    Returns:
        The loose key of the text, in which each run of digits is a ``#``, so
        that the footers of two pages have the same key.
    """
    return _NUMBERS.sub("#", loose_key(text))


def title_key(text: str) -> str:
    """Return the key of a title without its chapter word and its numbers.

    Args:
        text: The text of a title, or of a running title.

    Returns:
        The neutral key of the text, without a leading word such as
        ``chapter`` or ``annexe`` and without the numbers that follow, so that
        ``Chapter 2. Getting Started`` and ``1.2 Getting started`` have the
        same key. The key is empty if nothing else is left.
    """
    return _TITLE_PREFIX.sub("", neutral_key(text)).strip()


def band_of(
    segment: Segment,
    height: float,
    cfg: PdfConfig,
) -> Literal["top", "bottom"] | None:
    """Tell in which band of its page a segment lies, if in one.

    A segment lies in a band as soon as its line box reaches into it. The top
    band is the top `header_band` share of the page height and the bottom band
    is the bottom `footer_band` share. The top band wins if both apply.

    Args:
        segment: The segment.
        height: Height of the page, in points.
        cfg: The PDF settings, which give the size of the bands.

    Returns:
        ``"top"`` or ``"bottom"``, or None if the segment is outside both
        bands.
    """
    if segment.top >= height * (1 - cfg.header_band):
        return "top"
    if segment.bottom <= height * cfg.footer_band:
        return "bottom"
    return None


def _stands_apart(
    segment: Segment,
    segments: Sequence[Segment],
    *,
    below: bool,
) -> bool:
    """Return whether a gap separates a segment from the next row inward.

    The next row is the nearest baseline among those of the segments more
    than `LINE_TOLERANCE` times the size of the segment below its baseline, or
    above it. Of the segments on that baseline, the largest one counts.

    Args:
        segment: A segment of the first or the last line of a page.
        segments: The segments of the page.
        below: Whether the next row is below the segment, as for the first
            line, or above it, as for the last line.

    Returns:
        Whether the distance between the two baselines exceeds `CHROME_GAP`
        times the larger size of the two segments. True if there is no next
        row.
    """
    tolerance = LINE_TOLERANCE * segment.size
    if below:
        limit = segment.baseline - tolerance
        lower = [other for other in segments if other.baseline < limit]
        inner = max(lower, key=lambda s: (s.baseline, s.size), default=None)
    else:
        limit = segment.baseline + tolerance
        upper = [other for other in segments if other.baseline > limit]
        inner = min(upper, key=lambda s: (s.baseline, -s.size), default=None)
    if inner is None:
        return True
    gap = abs(segment.baseline - inner.baseline)
    return gap > CHROME_GAP * max(segment.size, inner.size)


def _on_first_line(segment: Segment, highest: float) -> bool:
    """Return whether a segment is on the first line of its page.

    Args:
        segment: The segment.
        highest: The highest baseline of the page.

    Returns:
        Whether its baseline is at most `LINE_TOLERANCE` times its size below
        the highest one.
    """
    return segment.baseline >= highest - LINE_TOLERANCE * segment.size


def _on_last_line(segment: Segment, lowest: float) -> bool:
    """Return whether a segment is on the last line of its page.

    Args:
        segment: The segment.
        lowest: The lowest baseline of the page.

    Returns:
        Whether its baseline is at most `LINE_TOLERANCE` times its size above
        the lowest one.
    """
    return segment.baseline <= lowest + LINE_TOLERANCE * segment.size


def _outer_zone(
    segment: Segment,
    segments: Sequence[Segment],
    *,
    height: float,
    highest: float,
    lowest: float,
) -> _Zone | None:
    """Tell whether a segment is on the first or the last line of its page.

    Args:
        segment: The segment.
        segments: The segments of its page.
        height: Height of the page, in points.
        highest: The highest baseline of the page.
        lowest: The lowest baseline of the page.

    Returns:
        ``"top"`` if the segment is on the first line, as `_on_first_line`
        tells, its line box reaches into the top `OUTER_ZONE` share of the
        page height and a gap separates it from the row below, as
        `_stands_apart` tells. ``"bottom"`` if it is on the last line, as
        `_on_last_line` tells, its line box reaches into the bottom
        `OUTER_ZONE` share and a gap separates it from the row above. None
        otherwise.
    """
    if (
        _on_first_line(segment, highest)
        and segment.top >= height * (1 - OUTER_ZONE)
        and _stands_apart(segment, segments, below=True)
    ):
        return "top"
    if (
        _on_last_line(segment, lowest)
        and segment.bottom <= height * OUTER_ZONE
        and _stands_apart(segment, segments, below=False)
    ):
        return "bottom"
    return None


def _zones(
    segments: Sequence[Segment],
    height: float,
    cfg: PdfConfig,
) -> list[_Zone | None]:
    """Tell in which zone of its page each segment lies, if in one.

    A segment lies in the zone of the band that `band_of` puts it in. Out of
    the bands, it lies in a zone if it is on the first or the last line of the
    page, near the edge and apart from the other rows, as `_outer_zone`
    tells.

    Args:
        segments: The segments of a page.
        height: Height of the page, in points.
        cfg: The PDF settings, which give the size of the bands.

    Returns:
        The zone of each segment, in order: ``"top"``, ``"bottom"``, or None
        for a segment outside both zones.
    """
    if not segments:
        return []
    highest = max(segment.baseline for segment in segments)
    lowest = min(segment.baseline for segment in segments)
    return [
        band_of(segment, height, cfg)
        or _outer_zone(
            segment,
            segments,
            height=height,
            highest=highest,
            lowest=lowest,
        )
        for segment in segments
    ]


def page_number(text: str) -> str | None:
    """Read the number that a page number shows.

    Args:
        text: Any text, such as that of a segment or of a block.

    Returns:
        The first run of digits of the text if it is a number, with or
        without ``page`` before it and ``of`` and a total after it, or two
        numbers around a slash, as in ``3/10``, and if that run has
        `PAGE_NUMBER_DIGITS` digits at most. The numeral, in lower case, if
        the text is a roman numeral from 1 to 99. None for any other text:
        words made of roman letters, acronyms and decimals such as ``2.5`` are
        not page numbers.
    """
    key = neutral_key(text)
    if _PAGE_NUMBER.match(key) or ("/" in text and _PAGE_FRACTION.match(key)):
        # Each # of the key stands for a run of digits of the text.
        number = _NUMBERS.findall(text)[0]
        return number if len(number) <= PAGE_NUMBER_DIGITS else None
    if _ROMAN_NUMBER.match(key):
        return key
    return None


def _number_value(number: str) -> int:
    """Return the value of a page number.

    Args:
        number: A number, as `page_number` reads it: digits, or a roman
            numeral in lower case.

    Returns:
        The integer of the digits, or the value of the numeral: the values of
        its letters added up, except that a letter worth less than the next
        one is taken away.
    """
    if number.isdecimal():
        return int(number)
    values = [_ROMAN_VALUES[letter] for letter in number]
    return sum(
        -value if value < after else value
        for value, after in pairwise([*values, 0])
    )


def _add_title(
    found: dict[str, set[int]],
    text: str,
    page: int | None,
) -> None:
    """Record a title, unless its key is empty.

    Args:
        found: The pages of each title key, which this updates.
        text: The text of the title.
        page: The page where it is a title, if it has one.
    """
    key = title_key(text)
    if not key:
        return
    home = found.setdefault(key, set())
    if page is not None:
        home.add(page)


def title_pages(  # ruff: ignore[too-many-arguments] - one input per source of titles
    pages: Sequence[Sequence[Segment]],
    heights: Sequence[float],
    bookmarks: Sequence[Bookmark],
    doc_title: str | None,
    *,
    body: float,
    cfg: PdfConfig,
) -> dict[str, frozenset[int]]:
    """Find the pages where each title is a real title.

    A running title has the key of a title, so only its page tells it from the
    real one. The titles come from three sources, whose pages are united by
    key. A key that is empty is left out.

    - Each bookmark is a title on the page it points to, if it points to one.
    - The title of the document is a title on no page.
    - Each segment outside the bands that is at least `HEADING_MIN_SIZE` body
      sizes large is a title on its page.

    Args:
        pages: The segments of each page of the document.
        heights: The height of each page, in points.
        bookmarks: The outline of the document.
        doc_title: The title that the document declares, if any.
        body: The body size of the document, in points.
        cfg: The PDF settings, which give the size of the bands.

    Returns:
        The pages of each title key. A title that has no page gets an empty
        set.
    """
    found: dict[str, set[int]] = {}
    for bookmark in bookmarks:
        _add_title(found, bookmark.title, bookmark.page)
    if doc_title is not None:
        _add_title(found, doc_title, None)
    for page, (segments, height) in enumerate(
        zip(pages, heights, strict=True),
    ):
        for segment in segments:
            large = segment.size >= HEADING_MIN_SIZE * body
            if large and band_of(segment, height, cfg) is None:
                _add_title(found, segment.text, page)
    return {key: frozenset(home) for key, home in found.items()}


def _candidates(
    segments: Sequence[Segment],
    height: float,
    *,
    body: float,
    cfg: PdfConfig,
) -> list[_Candidate]:
    """Find the segments of a page that may be chrome.

    Args:
        segments: The segments of a page.
        height: Height of the page, in points.
        body: The body size of the document, in points.
        cfg: The PDF settings, which give the size of the bands.

    Returns:
        The candidates of the page, in order: the segments that lie in a zone,
        as `_zones` tells, and are at most `CHROME_MAX_SIZE` body sizes large.
    """
    found: list[_Candidate] = []
    zones = _zones(segments, height, cfg)
    for segment, zone in zip(segments, zones, strict=True):
        if zone is not None and segment.size <= CHROME_MAX_SIZE * body:
            banded = band_of(segment, height, cfg) is not None
            found.append(_Candidate(segment, zone, banded=banded))
    return found


def _repeat_text(candidate: _Candidate) -> str:
    """Return the text of a candidate, as its repeats share it.

    Args:
        candidate: A candidate.

    Returns:
        In a band, the neutral key of its text, so that the footers of two
        pages repeat whatever their numbers, or the loose key of a lone
        number, which keeps the value: a column of numbers is no repeated
        footer. Beyond the bands, the loose key of its text, numbers
        included: the outermost rows of a table, such as ``Step 2`` and
        ``Step 3``, repeat only once their numbers are left out, while the
        chrome there is a page number, a running title or an exact repeat.
    """
    text = candidate.segment.text
    key = neutral_key(text)
    return key if candidate.banded and key != "#" else loose_key(text)


def _repeat_key(candidate: _Candidate, parity: int | None) -> _RepeatKey:
    """Return what a candidate has in common with its repeats.

    Args:
        candidate: A candidate.
        parity: Parity of the pages among which it repeats, or None for all
            the pages.

    Returns:
        The key of its repeats.
    """
    return _RepeatKey(
        candidate.zone,
        parity,
        _repeat_text(candidate),
        candidate.segment.size,
    )


def _repeat_parity(page: int, page_count: int) -> int | None:
    """Return the parity of the pages among which the texts of a page repeat.

    Args:
        page: Position of the page, from 0.
        page_count: Number of pages of the document.

    Returns:
        The parity of the page, or None, for all the pages, if the document
        has fewer than twice `CHROME_MIN_PAGES` pages, so that a parity may
        have fewer than `CHROME_MIN_PAGES` pages.
    """
    if page_count < 2 * CHROME_MIN_PAGES:
        return None
    return page % 2


def _repeat_threshold(
    parity: int | None,
    page_count: int,
    cfg: PdfConfig,
) -> int:
    """Return the number of pages on which a text must repeat.

    Args:
        parity: Parity of the pages among which the text repeats, or None for
            all the pages.
        page_count: Number of pages of the document.
        cfg: The PDF settings, which give the share of the pages.

    Returns:
        The greater of `CHROME_MIN_PAGES` and the share of those pages,
        rounded up.
    """
    count = page_count if parity is None else len(range(parity, page_count, 2))
    return max(CHROME_MIN_PAGES, math.ceil(cfg.repeat_share * count))


def _repeated_keys(
    candidates: Sequence[Sequence[_Candidate]],
    cfg: PdfConfig,
) -> frozenset[_RepeatKey]:
    """Find the texts of the zones that repeat on enough pages (R1).

    Each candidate is counted among the pages of its parity, or among all the
    pages of a short document, as `_repeat_parity` tells.

    Args:
        candidates: The candidates of each page of the document, as
            `_candidates` finds them.
        cfg: The PDF settings, which give the share of pages.

    Returns:
        The keys of the texts that are on enough distinct pages, among the
        pages of one parity or among all the pages.
    """
    seen: defaultdict[_RepeatKey, set[int]] = defaultdict(set)
    for page, found in enumerate(candidates):
        parity = _repeat_parity(page, len(candidates))
        for candidate in found:
            seen[_repeat_key(candidate, parity)].add(page)
    return frozenset(
        key
        for key, pages in seen.items()
        if len(pages) >= _repeat_threshold(key.parity, len(candidates), cfg)
    )


def _page_numbers(
    candidates: Sequence[Sequence[_Candidate]],
    labels: Sequence[str],
    cfg: PdfConfig,
) -> frozenset[Segment]:
    """Find the page numbers that follow the pages (R2).

    Args:
        candidates: The candidates of each page of the document, as
            `_candidates` finds them.
        labels: The label of each page, in order.
        cfg: The PDF settings, which give the share of pages.

    Returns:
        The candidates in which `page_number` reads a number, if that number
        is the label of their page, without blanks around it and casefolded,
        or if its value minus the position of their page, from 0, is the same
        for such candidates on as many distinct pages as a text that repeats
        among all the pages needs: `CHROME_MIN_PAGES`, and a `repeat_share`
        of the pages. Values of table cells rarely agree by chance on that
        many pages, while page numbers are on almost every page. Roman and
        arabic numbers are counted apart, and roman numerals need only
        `CHROME_MIN_PAGES` pages, as the front matter of a book is short.
    """
    numbers: list[tuple[Segment, bool, bool, int]] = []
    offsets: defaultdict[tuple[bool, int], set[int]] = defaultdict(set)
    for page, (found, label) in enumerate(
        zip(candidates, labels, strict=True),
    ):
        for candidate in found:
            number = page_number(candidate.segment.text)
            if number is None:
                continue
            roman = not number.isdecimal()
            offset = _number_value(number) - page
            offsets[roman, offset].add(page)
            labelled = number == label.strip().casefold()
            numbers.append((candidate.segment, labelled, roman, offset))
    needed = {
        False: _repeat_threshold(None, len(candidates), cfg),
        True: CHROME_MIN_PAGES,
    }
    return frozenset(
        segment
        for segment, labelled, roman, offset in numbers
        if labelled or len(offsets[roman, offset]) >= needed[roman]
    )


def _is_running_title(
    segment: Segment,
    page: int,
    outside: Sequence[Segment],
    titles: Mapping[str, frozenset[int]],
) -> bool:
    """Return whether a segment is a running title (R3).

    Args:
        segment: A candidate of the page.
        page: Position of the page, from 0.
        outside: The segments of the page that are not candidates.
        titles: The pages where each title is a real title.

    Returns:
        Whether the segment has the key of a title, and either the page is not
        one where that title is a real title, or a segment of the page that is
        not a candidate has the same key.
    """
    key = title_key(segment.text)
    if not key or key not in titles:
        return False
    return page not in titles[key] or any(
        title_key(other.text) == key for other in outside
    )


def _whole_lines(
    segments: Sequence[Segment],
    taken: set[Segment],
    *,
    height: float,
    cfg: PdfConfig,
) -> set[Segment]:
    """Keep the chrome of a page whole on its outermost lines beyond the bands.

    Beyond the bands, the first and the last line of a page are chrome whole
    or not at all: a header or a footer holds only chrome, while a row of a
    table holds cells that do not all repeat, and one of them may equal a
    title or the label of its page by chance.

    Args:
        segments: The segments of a page.
        taken: The segments of the page that the rules take for chrome, one
            by one.
        height: Height of the page, in points.
        cfg: The PDF settings, which give the size of the bands.

    Returns:
        The segments of `taken`, less all the segments of the first line of
        the page, as `_on_first_line` tells, if none of them lies in a band
        and one of them is not in `taken`; and less those of the last line,
        as `_on_last_line` tells, on the same terms.
    """
    if not segments:
        return set(taken)
    highest = max(segment.baseline for segment in segments)
    lowest = min(segment.baseline for segment in segments)
    lines = (
        [segment for segment in segments if _on_first_line(segment, highest)],
        [segment for segment in segments if _on_last_line(segment, lowest)],
    )
    kept = set(taken)
    for line in lines:
        banded = any(band_of(segment, height, cfg) for segment in line)
        if not banded and not taken.issuperset(line):
            kept.difference_update(line)
    return kept


def find_chrome(  # ruff: ignore[too-many-arguments] - one input per attribute of the pages, and the settings
    pages: Sequence[Sequence[Segment]],
    heights: Sequence[float],
    titles: Mapping[str, frozenset[int]],
    *,
    labels: Sequence[str],
    body: float,
    cfg: PdfConfig,
) -> frozenset[Segment]:
    """Find the running headers, footers and page numbers of a document.

    The chrome lies in the two zones of a page, top and bottom. A segment lies
    in the zone of the band that `band_of` puts it in. Out of the bands, a
    segment of the first line of the page lies in the top zone if its line box
    reaches into the top `OUTER_ZONE` share of the page height, and a segment
    of the last line lies in the bottom zone if its line box reaches into the
    bottom `OUTER_ZONE` share. A segment is on the first line if its baseline
    is at most `LINE_TOLERANCE` times its size below the highest baseline of
    the page, and on the last line if it is at most as far above the lowest
    one. Such a segment must also stand apart from the next row inward: the
    nearest baseline more than `LINE_TOLERANCE` times its size below it, for
    the first line, or above it, for the last line, must be more than
    `CHROME_GAP` times the larger size of the two segments away, unless there
    is no such row. The rows of a table follow each other closer than that,
    while LaTeX sets its running headers and footers farther from the text.

    A segment of the page ``p`` is a candidate if it lies in a zone and is at
    most `CHROME_MAX_SIZE` body sizes large. A candidate is chrome if it
    follows one of these rules:

    - R1, it repeats: the same text, in the same zone and at the same size,
      is on at least `CHROME_MIN_PAGES` distinct pages, and on at least a
      `repeat_share` of the pages, all of them taken among the pages that
      have the parity of ``p``. In a document of fewer than twice
      `CHROME_MIN_PAGES` pages, where a parity may have fewer than
      `CHROME_MIN_PAGES` pages, they are taken among all the pages. In a
      band, the text is the same once its numbers are left out, unless it is
      a lone number. Beyond the bands, the text is the same with its
      numbers: the outermost rows of a table may stand apart from the others,
      and their cells, such as ``Step 2`` and ``Step 3``, repeat only once
      their numbers are left out.
    - R2, it is a page number that follows the pages: `page_number` reads a
      number in it, and either that number is the label of ``p``, without
      blanks around it and casefolded, or its value minus ``p`` is the same,
      for the numbers read in candidates, on as many distinct pages as R1
      asks among all the pages: at least `CHROME_MIN_PAGES`, and at least a
      `repeat_share` of the pages, ``p`` included. The value of a roman
      numeral is that of its number: ``iv`` is 4. Roman and arabic numbers
      are counted apart, and a roman numeral needs only `CHROME_MIN_PAGES`
      distinct pages, as the front matter of a book is short. A cell of a
      table on the last line of a page is a number, but it does not follow
      the pages.
    - R3, it is a running title: its title key is that of a title, and either
      ``p`` is not a page where that title is a real title, or a segment of
      ``p`` that is not a candidate has the same title key, such as the
      heading of a chapter on its first page.

    Beyond the bands, the first and the last line of a page are then chrome
    whole or not at all: if none of the segments of such a line lies in a
    band, and one of them is not chrome, whether it is not a candidate or no
    rule takes it, none of them is chrome. A header or a footer holds only
    chrome, while a row of a table holds cells that do not all repeat, such
    as a step beside a repeated ``Yes``, and a cell may equal a title or the
    label of its page by chance. The lines are those of the zones: the
    segments whose baseline is at most `LINE_TOLERANCE` times their size away
    from the highest, or the lowest, baseline of the page. In the bands, each
    segment is chrome on its own.

    Args:
        pages: The segments of each page of the document.
        heights: The height of each page, in points.
        titles: The pages where each title is a real title, by title key, as
            `title_pages` finds them.
        labels: The label of each page, in order.
        body: The body size of the document, in points.
        cfg: The PDF settings, which give the bands and the share of pages.

    Returns:
        The chrome segments. The caller never walks this set: it filters the
        segments of the pages in order.
    """
    candidates = [
        _candidates(segments, height, body=body, cfg=cfg)
        for segments, height in zip(pages, heights, strict=True)
    ]
    repeated = _repeated_keys(candidates, cfg)
    numbers = _page_numbers(candidates, labels, cfg)
    chrome: set[Segment] = set()
    for page, (segments, height, found) in enumerate(
        zip(pages, heights, candidates, strict=True),
    ):
        chosen = {candidate.segment for candidate in found}
        outside = [segment for segment in segments if segment not in chosen]
        parity = _repeat_parity(page, len(pages))
        taken: set[Segment] = set()
        for candidate in found:
            segment = candidate.segment
            if (
                _repeat_key(candidate, parity) in repeated
                or segment in numbers
                or _is_running_title(segment, page, outside, titles)
            ):
                taken.add(segment)
        chrome |= _whole_lines(segments, taken, height=height, cfg=cfg)
    return frozenset(chrome)
