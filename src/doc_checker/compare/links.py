# Copyright 2026 Christophe Le Douarec
"""Links lost, broken or redirected between the matched blocks of documents.

Only the regressions of the links of a matched block are reported: a link
that the reference already breaks, a new link that has no counterpart in
the reference, and the links of a missing or added block give no link
finding.

The links of the blocks of a matched pair are paired in order, each link of
the reference with the first new link left that has the same loose key of
its text and a corresponding target, then the same loose key of its text,
then a corresponding target. A link of the reference left without a new
link is ``link.lost``. A new link that leads nowhere, where the link of the
reference led somewhere, is ``link.broken``. A new link whose target does
not correspond to the target of the reference is ``link.target_changed``.

An external link leads to its normalized URL. An internal link leads to the
block that bears its anchor, or to the first block of its page if it has no
anchor, and nowhere if its page does not exist or has no such anchor. A page
of the site that has no block, such as a download, an image or a page left
out, is a file, known by its name and its anchor. The name of a web page is
the last part of its path, without its suffix and without a final
``index.html``, so that ``api/foo.html`` and ``api/foo/index.html`` are both
``foo``; any other file keeps its whole name, whatever folder holds it. Two
URLs correspond if they are equal, and two files if their names and anchors
are. Two blocks correspond if they are matched, or if the headings that open
their sections are: the nearest heading at or before each block, or the
block itself if no heading comes before it. An external target never
corresponds to an internal one. PDF documents carry no links, and give no
link finding.
"""

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Final

from doc_checker.compare.drafts import ARROW, MATCHED, Draft, truncate
from doc_checker.model.document import BlockKind
from doc_checker.model.findings import DiffSegment, Side
from doc_checker.text.normalize import loose_key

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from doc_checker.compare.align import Pair
    from doc_checker.compare.drafts import SegmentOp
    from doc_checker.model.document import Block, Link

_BROKEN: Final = "link.broken"
_LOST: Final = "link.lost"
_TARGET_CHANGED: Final = "link.target_changed"
# The passes that pair links, in order: whether each asks for the same loose
# key of the text, and whether it asks for a target that corresponds.
_PASSES: Final = ((True, True), (True, False), (False, True))
# The suffixes of a web page, and the names of the page of a folder.
_PAGE_SUFFIXES: Final = frozenset({".html", ".htm"})
_INDEX_PAGES: Final = frozenset({"index.html", "index.htm"})

type _AnchorIndex = Mapping[tuple[str, str | None], int]


@dataclass(frozen=True, slots=True)
class _Url:
    """Where an external link leads: its normalized URL."""

    url: str


@dataclass(frozen=True, slots=True)
class _Position:
    """Where an internal link to a block leads: the position of the block."""

    index: int


@dataclass(frozen=True, slots=True)
class _File:
    """Where an internal link to a page without blocks leads."""

    name: str
    fragment: str | None


type _Target = _Url | _Position | _File


@dataclass(frozen=True, slots=True)
class _Located:
    """A link of a block, with where it leads."""

    block: Block
    link: Link
    target: _Target | None


def anchor_index(
    blocks: Sequence[Block],
) -> dict[tuple[str, str | None], int]:
    """Return the position of the block that each page or anchor designates.

    Args:
        blocks: Blocks in reading order.

    Returns:
        Under ``(file, None)``, the position of the first block of each
        file; under ``(file, anchor)``, the position of the first block of
        the file that bears the anchor.
    """
    index: dict[tuple[str, str | None], int] = {}
    for position, block in enumerate(blocks):
        file = block.source.file
        index.setdefault((file, None), position)
        for anchor in block.anchors:
            index.setdefault((file, anchor), position)
    return index


def _file_name(page: str) -> str:
    """Return the name of a page without blocks, whatever its URL layout.

    Args:
        page: Site relative path of the page.

    Returns:
        The last part of the path once a final ``index.html`` or
        ``index.htm`` is left out, without a ``.html`` or ``.htm`` suffix.
        Another file keeps its whole name.
    """
    path = PurePosixPath(page)
    if path.name.casefold() in _INDEX_PAGES:
        path = path.parent
    if path.suffix.casefold() in _PAGE_SUFFIXES:
        return path.stem
    return path.name


def _target(link: Link, index: _AnchorIndex) -> _Target | None:
    """Return where a link leads among blocks, or ``None`` for nowhere."""
    if link.url is not None:
        return _Url(link.url)
    if link.page is None or not link.found:
        return None
    if (link.page, None) not in index:
        return _File(_file_name(link.page), link.fragment)
    position = index.get((link.page, link.fragment))
    return None if position is None else _Position(position)


def _section_heads(blocks: Sequence[Block]) -> list[int]:
    """Return the position of the heading that opens the section of each block.

    A block that no heading comes before opens its own section.
    """
    heads = []
    head: int | None = None
    for position, block in enumerate(blocks):
        if block.kind is BlockKind.HEADING:
            head = position
        heads.append(position if head is None else head)
    return heads


def _side(block: Block, op: SegmentOp, text: str) -> Side:
    """Return a side at a block, that shows one text."""
    return Side(block.source, (DiffSegment(op, truncate(text)),), block.kind)


class _Linker:
    """Compare the links of the matched blocks of two documents."""

    def __init__(
        self,
        pairs: Sequence[Pair],
        ref: Sequence[Block],
        new: Sequence[Block],
    ) -> None:
        self.ref = ref
        self.new = new
        self.ref_index = anchor_index(ref)
        self.new_index = anchor_index(new)
        self.ref_heads = _section_heads(ref)
        self.new_heads = _section_heads(new)
        self.partners = {
            index: frozenset(pair.new)
            for pair in pairs
            if pair.kind in MATCHED
            for index in pair.ref
        }

    def corresponds(
        self,
        before: _Target | None,
        after: _Target | None,
    ) -> bool:
        """Return whether a new target corresponds to a reference target.

        A target that leads nowhere corresponds to none.
        """
        if isinstance(before, _Position) and isinstance(after, _Position):
            partners = self.partners.get(before.index, frozenset())
            heads = self.partners.get(
                self.ref_heads[before.index],
                frozenset(),
            )
            return (
                after.index in partners or self.new_heads[after.index] in heads
            )
        return before is not None and before == after

    def pairing(
        self,
        before: Sequence[_Located],
        after: Sequence[_Located],
    ) -> list[_Located | None]:
        """Return the new link paired with each link of the reference.

        Args:
            before: Links of the reference blocks of a pair, in order.
            after: Links of the new blocks of the pair, in order.

        Returns:
            For each link of ``before``, a link of ``after`` or ``None``.
            Three passes pair them, each one taking the links of ``before``
            left in order, and for each the first link of ``after`` left that
            has: the same loose key of its text and a corresponding target;
            then the same loose key; then a corresponding target.
        """
        texts = [loose_key(located.link.text) for located in after]
        left = list(range(len(after)))
        by_text: dict[str, list[int]] = {}
        for m in left:
            by_text.setdefault(texts[m], []).append(m)
        paired: list[int | None] = [None] * len(before)
        for same_text, same_target in _PASSES:
            for k, located in enumerate(before):
                if paired[k] is not None:
                    continue
                pool = (
                    by_text.get(loose_key(located.link.text), [])
                    if same_text
                    else left
                )
                m = next(
                    (
                        m
                        for m in pool
                        if not same_target
                        or self.corresponds(located.target, after[m].target)
                    ),
                    None,
                )
                if m is not None:
                    paired[k] = m
                    left.remove(m)
                    by_text[texts[m]].remove(m)
        return [None if m is None else after[m] for m in paired]

    def pair_drafts(self, pair: Pair) -> list[Draft]:
        """Return the drafts of the links of one matched pair."""
        before = [
            _Located(block, link, _target(link, self.ref_index))
            for block in (self.ref[i] for i in pair.ref)
            for link in block.links
        ]
        if not before:
            return []
        after = [
            _Located(block, link, _target(link, self.new_index))
            for block in (self.new[j] for j in pair.new)
            for link in block.links
        ]
        first = self.new[pair.new[0]]
        matches = self.pairing(before, after)
        drafts = (
            self.draft(old, match, first)
            for old, match in zip(before, matches, strict=True)
        )
        return [draft for draft in drafts if draft is not None]

    def draft(
        self,
        old: _Located,
        match: _Located | None,
        first: Block,
    ) -> Draft | None:
        """Return the draft of a link of the reference, if it regressed.

        Args:
            old: A link of the reference.
            match: The new link paired with it, if any.
            first: The first new block of the pair.

        Returns:
            ``None`` if the link of the reference leads nowhere or if the
            targets correspond; else the draft of a lost, broken or
            redirected link.
        """
        if old.target is None:
            return None
        link = old.link
        ref_side = _side(old.block, "delete", link.target)
        if match is None:
            return Draft(
                _LOST,
                None,
                ref_side,
                _side(first, "equal", link.text),
                f'"{link.text}" no longer links to {link.target}',
                link.target,
            )
        if self.corresponds(old.target, match.target):
            return None
        new_link = match.link
        new_side = _side(match.block, "insert", new_link.target)
        if match.target is None and link.url is None:
            return Draft(
                _BROKEN,
                None,
                ref_side,
                new_side,
                (
                    f'"{new_link.text}" links to {new_link.href}, which does '
                    f"not exist; it linked to {link.target} before"
                ),
                new_link.href if new_link.page is None else new_link.page,
            )
        return Draft(
            _TARGET_CHANGED,
            None,
            ref_side,
            new_side,
            (
                f'"{new_link.text}" links to {new_link.target} instead of '
                f"{link.target}"
            ),
            f"{link.target}{ARROW}{new_link.target}",
        )


def link_drafts(
    pairs: Sequence[Pair],
    ref: Sequence[Block],
    new: Sequence[Block],
) -> dict[int, list[Draft]]:
    """Return the drafts of the links that matched blocks lost or changed.

    Args:
        pairs: The alignment of the two documents.
        ref: Reference blocks, which the pairs index.
        new: New blocks, which the pairs index.

    Returns:
        The drafts of each matched pair that has some, by position of the
        pair, in the order of the links of the reference.
    """
    linker = _Linker(pairs, ref, new)
    found: dict[int, list[Draft]] = {}
    for position, pair in enumerate(pairs):
        if pair.kind not in MATCHED:
            continue
        drafts = linker.pair_drafts(pair)
        if drafts:
            found[position] = drafts
    return found
