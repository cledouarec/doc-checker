# Copyright 2026 Christophe Le Douarec
"""Turn the alignment of two documents into findings.

Each pair of the alignment gives its drafts: a missing, added, moved, split
or merged block, and the word level differences of matched blocks, which
`classify_hunk` of `doc_checker.compare.classify` classifies, with a
speller that tells a typo from a word replaced or corrected. An
unmatched default admonition title or chapter label gives none, as the
generator adds them, and so does the number of a title that lost its
heading.

The kinds of matched blocks are compared too, by `kind_drafts` of
`doc_checker.compare.structure`: a heading, a table, a code block or the
items of a list that lost their kind are reported, and any other change of
kind is information. Between two PDF documents, whose kinds are inferred
from the layout, those findings are information, except the loss of a
heading of the outline where the new outline has headings of its level.
The levels of matched headings are compared by `level_drafts` of the same
module, with a map of the levels learned from the pairs of headings, so
that a generator that shifts every heading changes nothing. The links of
matched blocks are compared by `link_drafts` of `doc_checker.compare.links`,
which reports the links lost, broken or leading elsewhere.

The images, which the alignment leaves out, are matched and judged by
`image_drafts` of `doc_checker.compare.images`, and their drafts follow
those of the text; an image that either document removed as page chrome is
compared on neither side.

Last, `finalize_drafts` of `doc_checker.compare.aggregate` gives the
drafts the severities of the configuration, leaves out those of the
disabled rules, below the minimum severity or ignored, and replaces the
findings of one rule and pattern, once there are five of them, by one group.
"""

import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final

from doc_checker.compare.aggregate import finalize_drafts, text_pattern
from doc_checker.compare.align import Pair, PairKind, align
from doc_checker.compare.classify import (
    Classification,
    HunkContext,
    classify_hunk,
)
from doc_checker.compare.drafts import MATCHED, Draft, whole
from doc_checker.compare.images import (
    image_drafts,
    match_images,
    placed_images,
)
from doc_checker.compare.links import link_drafts
from doc_checker.compare.structure import kind_drafts, level_drafts
from doc_checker.compare.textdiff import Hunk, diff_tokens, span_text
from doc_checker.errors import DictionaryError
from doc_checker.images.load import PixelLoader
from doc_checker.model.document import (
    CELL_SEPARATOR,
    NUMBER_ATTR,
    SPELLED_KINDS,
    Block,
    BlockKind,
    Document,
    carry_anchors,
)
from doc_checker.model.findings import DiffSegment, Finding, Side
from doc_checker.spell.speller import Speller
from doc_checker.text.lexicon import is_admonition_title, is_chapter_label
from doc_checker.text.normalize import loose_key, printable
from doc_checker.text.numbers import HEADING_NUMBER
from doc_checker.text.tokenize import Token, TokenKind, tokenize

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from doc_checker.compare.drafts import SegmentOp
    from doc_checker.config import Config

CONTEXT_TOKENS: Final = 8

_CONTENTS: Final = frozenset({BlockKind.TOC, BlockKind.INDEX})
# The start of the warnings of the dictionaries that confirm the typos.
_TYPO_CHECK: Final = "typo check: "
_VOCABULARY_ONLY: Final = "told by the reference vocabulary only"


@dataclass(frozen=True, slots=True)
class CompareStats:
    """Figures about a comparison.

    Attributes:
        ref_blocks: Compared reference blocks.
        new_blocks: Compared new blocks.
        matched: Reference blocks found in the new document.
        moved: Pairs out of the common order.
        ref_images: Images of the reference.
        new_images: Images of the new document.
        matched_images: Reference images matched with a new one.
        images: Whether the images were compared.
    """

    ref_blocks: int
    new_blocks: int
    matched: int
    moved: int
    ref_images: int = 0
    new_images: int = 0
    matched_images: int = 0
    images: bool = True


@dataclass(slots=True)
class CompareResult:
    """Outcome of the comparison of two documents."""

    ref: Document
    new: Document
    findings: list[Finding]
    stats: CompareStats


def _keyed(block: Block, patterns: Sequence[re.Pattern[str]]) -> Block:
    """Return a block whose keys lost the ignored patterns, if any."""
    if not patterns:
        return block
    key = block.key_strict
    for pattern in patterns:
        key = pattern.sub("", key)
    if block.kind is not BlockKind.CODE:
        key = " ".join(key.split())
    return replace(block, key_strict=key, key_loose=loose_key(key))


def _prepare(
    blocks: Iterable[Block],
    chrome: frozenset[str],
    patterns: Sequence[re.Pattern[str]],
    *,
    compare_toc: bool,
) -> tuple[list[Block], list[int]]:
    """Return the blocks to compare, with ignored patterns removed.

    Page chrome is left out. So are the table of contents and index blocks,
    unless ``compare_toc`` is set, and the blocks that the ignored patterns
    leave without a loose key, such as the image blocks. The anchors of the
    blocks left out go to the next kept block of their page, as
    `carry_anchors` moves them, so that the links to them still lead
    somewhere.

    Returns:
        The kept blocks, and their positions among ``blocks``.
    """
    skipped = frozenset() if compare_toc else _CONTENTS
    # A block left out loses its loose key, as a block without words.
    keyed = [
        replace(block, key_loose="")
        if block.key_strict in chrome or block.kind in skipped
        else _keyed(block, patterns)
        for block in blocks
    ]
    kept = [index for index, block in enumerate(keyed) if block.key_loose]
    return carry_anchors(keyed, lambda block: bool(block.key_loose)), kept


def _segments(
    text: str,
    tokens: Sequence[Token],
    bounds: tuple[int, int],
    op: SegmentOp,
) -> tuple[DiffSegment, ...]:
    """Return a changed run of tokens with some context around it."""
    start, end = bounds
    if start < end:
        changed = (tokens[start].start, tokens[end - 1].end)
    else:
        point = tokens[start].start if start < len(tokens) else len(text)
        changed = (point, point)
    before = max(0, start - CONTEXT_TOKENS)
    after = min(len(tokens), end + CONTEXT_TOKENS)
    context_start = (
        tokens[before].start if before < len(tokens) else changed[0]
    )
    context_end = tokens[after - 1].end if after else changed[1]
    segments = (
        DiffSegment("equal", text[context_start : changed[0]]),
        DiffSegment(op, text[changed[0] : changed[1]]),
        DiffSegment("equal", text[changed[1] : context_end]),
    )
    return tuple(segment for segment in segments if segment.text)


class _Drafter:
    """Build the draft findings of the pairs of an alignment.

    Attributes:
        speller: The speller that confirms the typos.
        broken: Why a dictionary could not be loaded, each reason once, in
            order.
    """

    def __init__(
        self,
        ref: Sequence[Block],
        new: Sequence[Block],
        langs: tuple[str | None, str | None],
        *,
        pdf: bool,
        speller: Speller,
    ) -> None:
        self.ref = ref
        self.new = new
        self.langs = langs
        self.pdf = pdf
        self.speller = speller
        self.broken: dict[str, None] = {}
        self.vocabulary = frozenset(
            token.text.casefold()
            for block in ref
            for token in tokenize(block.key_strict)
            if token.kind is TokenKind.WORD
        )
        # Whether each document numbers its headings.
        self.numbered = (_numbered(ref), _numbered(new))

    def missing_sections(
        self,
        deleted: Sequence[int],
    ) -> tuple[dict[int, int], set[int]]:
        """Return the fully deleted sections and the blocks they cover."""
        deleted_set = set(deleted)
        heads: dict[int, int] = {}
        covered: set[int] = set()
        for index in sorted(deleted):
            block = self.ref[index]
            if index in covered or block.kind is not BlockKind.HEADING:
                continue
            section = self.section(index)
            if all(position in deleted_set for position in section):
                heads[index] = len(section)
                covered.update(section)
        return heads, covered

    def section(self, start: int) -> range:
        """Return the indices of a heading and of the blocks it heads."""
        level = self.ref[start].level or 0
        end = start + 1
        while end < len(self.ref):
            block = self.ref[end]
            if block.kind is BlockKind.HEADING and (block.level or 0) <= level:
                break
            end += 1
        return range(start, end)

    def hunks(
        self,
        ref_blocks: Sequence[Block],
        new_blocks: Sequence[Block],
    ) -> list[Draft]:
        """Return one draft per word level difference between blocks.

        When the kinds of the blocks differ on the two sides, the cell
        separators of a table are left out: a table whose rows became
        paragraphs lost them with its kind, which `kind_drafts` reports. So
        is the number that starts a block that is no heading, if the other
        side has a heading, and that heading has its number set apart or
        the document of the block numbers its headings: a title that lost
        its heading keeps the number that its generator gave it, and the
        numbers of the headings are never compared. The speller judges the
        differences only when every block of the pair is of
        `SPELLED_KINDS`, and never a word of their inline code.
        """
        flat = {block.kind for block in ref_blocks} != {
            block.kind for block in new_blocks
        }
        ref_numbered = _numbers_left_out(new_blocks, own=self.numbered[0])
        new_numbered = _numbers_left_out(ref_blocks, own=self.numbered[1])
        ref_text = " ".join(
            _diff_key(block, flat=flat, numbered=ref_numbered)
            for block in ref_blocks
        )
        new_text = " ".join(
            _diff_key(block, flat=flat, numbered=new_numbered)
            for block in new_blocks
        )
        ref_tokens, new_tokens = tokenize(ref_text), tokenize(new_text)
        blocks = (*ref_blocks, *new_blocks)
        prose = all(block.kind in SPELLED_KINDS for block in blocks)
        ctx = HunkContext(
            ref_text,
            new_text,
            ref_tokens,
            new_tokens,
            self.vocabulary,
            *self.langs,
            speller=self.speller if prose else None,
            code_words=_code_words(blocks) if prose else frozenset(),
        )
        return [
            self.hunk_draft(hunk, ctx, ref_blocks[0], new_blocks[0])
            for hunk in diff_tokens(ref_tokens, new_tokens)
        ]

    def classify(self, hunk: Hunk, ctx: HunkContext) -> Classification:
        """Return the classification of a difference.

        A dictionary that cannot be loaded is noted in `broken`, and the
        difference is classified without the speller.
        """
        try:
            return classify_hunk(hunk, ctx)
        except DictionaryError as exc:
            self.broken[str(exc)] = None
            return classify_hunk(hunk, replace(ctx, speller=None))

    def hunk_draft(
        self,
        hunk: Hunk,
        ctx: HunkContext,
        ref_block: Block,
        new_block: Block,
    ) -> Draft:
        """Return the draft of one word level difference."""
        classification = self.classify(hunk, ctx)
        ref_segments = _segments(
            ctx.ref_text,
            ctx.ref_tokens,
            (hunk.ref_start, hunk.ref_end),
            "delete",
        )
        new_segments = _segments(
            ctx.new_text,
            ctx.new_tokens,
            (hunk.new_start, hunk.new_end),
            "insert",
        )
        deleted = span_text(
            ctx.ref_text,
            ctx.ref_tokens,
            hunk.ref_start,
            hunk.ref_end,
        )
        inserted = span_text(
            ctx.new_text,
            ctx.new_tokens,
            hunk.new_start,
            hunk.new_end,
        )
        pattern = text_pattern(classification.rule, deleted, inserted)
        return Draft(
            classification.rule,
            classification.severity,
            Side(ref_block.source, ref_segments, ref_block.kind),
            Side(new_block.source, new_segments, new_block.kind),
            pattern=pattern,
        )

    def matched(self, pair: Pair) -> list[Draft]:
        """Return the drafts of a pair of matched blocks."""
        ref_blocks = [self.ref[i] for i in pair.ref]
        new_blocks = [self.new[j] for j in pair.new]
        drafts = []
        if pair.moved:
            drafts.append(
                Draft(
                    "block.moved",
                    None,
                    whole(ref_blocks, "equal"),
                    whole(new_blocks, "equal"),
                ),
            )
        if pair.kind in {PairKind.SPLIT, PairKind.MERGE}:
            drafts.append(
                Draft(
                    "block.split_merged",
                    None,
                    whole(ref_blocks, "equal"),
                    whole(new_blocks, "equal"),
                ),
            )
        if pair.kind is not PairKind.EQUAL:
            drafts += self.hunks(ref_blocks, new_blocks)
        return drafts

    def warnings(self) -> list[str]:
        """Return what the dictionaries could not do, each on one line.

        Returns:
            Each dictionary that could not be loaded, then the spent budget,
            then the words that spylls failed on.
        """
        notes = [
            f"{reason}; the typos that needed it were {_VOCABULARY_ONLY}"
            for reason in self.broken
        ]
        if self.speller.exhausted:
            notes.append(
                f"the dictionaries stopped after {self.speller.lookups} "
                f"distinct words; the other typos were {_VOCABULARY_ONLY}",
            )
        if self.speller.failed:
            notes.append(
                f"{self.speller.failed} word(s) could not be looked up and "
                "were taken as known",
            )
        return [printable(_TYPO_CHECK + note) for note in notes]

    def drafts(self, pairs: Sequence[Pair]) -> list[Draft]:
        """Return the drafts of all pairs, in pair order.

        The changes of kind of a pair follow its other drafts, then its change
        of heading level, then its links that were lost or changed.
        """
        deleted = [p.ref[0] for p in pairs if p.kind is PairKind.DELETED]
        heads, covered = self.missing_sections(deleted)
        kinds = kind_drafts(pairs, self.ref, self.new, pdf=self.pdf)
        levels = level_drafts(pairs, self.ref, self.new, pdf=self.pdf)
        links = link_drafts(pairs, self.ref, self.new)
        drafts: list[Draft] = []
        for position, pair in enumerate(pairs):
            if pair.kind is PairKind.DELETED:
                index = pair.ref[0]
                block = self.ref[index]
                if index in heads:
                    detail = f"{heads[index]} blocks missing in this section"
                    side = whole([block], "delete")
                    drafts.append(
                        Draft("section.missing", None, side, None, detail),
                    )
                elif index not in covered and not _unreported(block):
                    side = whole([block], "delete")
                    drafts.append(
                        Draft(
                            "block.missing",
                            None,
                            side,
                            None,
                            pattern=block.key_loose,
                        ),
                    )
            elif pair.kind is PairKind.INSERTED:
                block = self.new[pair.new[0]]
                if not _unreported(block):
                    side = whole([block], "insert")
                    drafts.append(
                        Draft(
                            "block.added",
                            None,
                            None,
                            side,
                            pattern=block.key_loose,
                        ),
                    )
            else:
                drafts += self.matched(pair)
            drafts += kinds.get(position, [])
            drafts += levels.get(position, [])
            drafts += links.get(position, [])
        return drafts


def _code_words(blocks: Iterable[Block]) -> frozenset[str]:
    """Return the casefolded words of the inline code of blocks."""
    return frozenset(
        token.text.casefold()
        for block in blocks
        for span in block.code_spans
        for token in tokenize(span)
        if token.kind is TokenKind.WORD
    )


def _numbered(blocks: Sequence[Block]) -> bool:
    """Return whether a heading among blocks has its number set apart."""
    return any(
        block.kind is BlockKind.HEADING and NUMBER_ATTR in block.attrs
        for block in blocks
    )


def _numbers_left_out(other: Sequence[Block], *, own: bool) -> bool:
    """Return whether the blocks of a side may lose the number they start with.

    Args:
        other: The blocks of the other side of their pair.
        own: Whether the document of the blocks numbers its headings.

    Returns:
        Whether the other side has a heading, and either that heading has
        its number set apart or the document of the blocks numbers its
        headings.
    """
    headings = [block for block in other if block.kind is BlockKind.HEADING]
    return bool(headings) and (own or _numbered(headings))


def _diff_key(block: Block, *, flat: bool, numbered: bool) -> str:
    """Return the key of a block to diff, without what its kind brought.

    Args:
        block: A block to diff.
        flat: Whether the kinds of the blocks of its pair differ.
        numbered: Whether the block may lose the number it starts with, as
            `_numbers_left_out` tells.

    Returns:
        The strict key. If ``flat`` is set, each cell separator of a table
        is replaced by a blank and, if ``numbered`` is set too, the heading
        number that starts a block that is no heading is left out.
    """
    key = block.key_strict
    if not flat:
        return key
    if block.kind is BlockKind.TABLE:
        key = key.replace(CELL_SEPARATOR, " ")
    if numbered and block.kind is not BlockKind.HEADING:
        match = HEADING_NUMBER.match(key)
        key = key[match.end() :] if match else key
    return key


def _unreported(block: Block) -> bool:
    """Return whether an unmatched block comes from its generator.

    Such a block gives no finding: a default admonition title that is no
    heading, such as ``Note``, or a chapter label, such as ``Chapter 3``.
    The other document folded the label into the number of the heading that
    follows it, unless that title lost its heading.
    """
    return is_chapter_label(block.text) or (
        block.kind is not BlockKind.HEADING and is_admonition_title(block.text)
    )


@dataclass(frozen=True, slots=True)
class _Images:
    """The drafts of the images of two documents, and their counts.

    Attributes:
        drafts: The drafts, in the order of `image_drafts`.
        ref: Images of the reference that are compared.
        new: Images of the new document that are compared.
        matched: Reference images matched with a new one.
    """

    drafts: list[Draft]
    ref: int
    new: int
    matched: int


def _images(
    docs: tuple[Document, Document],
    kept: tuple[Sequence[int], Sequence[int]],
    pairs: Sequence[Pair],
) -> _Images:
    """Match and judge the images of two documents.

    An image that either document removed as page chrome is left out of
    both, as the text of the chrome is.

    Args:
        docs: The reference and the new document.
        kept: Positions of the compared blocks among the blocks of each
            document, as `_prepare` gives them.
        pairs: The alignment of the compared blocks.

    Returns:
        The drafts of the images, after `placed_images`, `match_images` and
        `image_drafts`, with the pixels read only while they are judged.
    """
    ref, new = docs
    logos = ref.chrome_images | new.chrome_images
    ref_images, new_images = (
        [
            placed
            for placed in placed_images(doc.blocks, positions)
            if placed.image.digest not in logos
        ]
        for doc, positions in zip(docs, kept, strict=True)
    )
    matched = [pair for pair in pairs if pair.kind in MATCHED]
    follows = {index: pair.new[-1] for pair in matched for index in pair.ref}
    # The end of the reference blocks faces the end of the new ones.
    precedes = {len(kept[0]): len(kept[1])}
    precedes |= {index: pair.new[0] for pair in matched for index in pair.ref}
    matches = match_images(ref_images, new_images, follows, precedes)
    drafts: list[Draft] = []
    if matches:
        with PixelLoader(ref.path) as first, PixelLoader(new.path) as second:
            drafts = image_drafts(
                ref_images,
                new_images,
                matches,
                (first, second),
                pdf=ref.format == new.format == "pdf",
            )
    return _Images(
        drafts,
        len(ref_images),
        len(new_images),
        sum(m.ref is not None and m.new is not None for m in matches),
    )


def compare_documents(
    ref: Document,
    new: Document,
    cfg: Config,
) -> CompareResult:
    """Compare a reference document with its new version.

    The table of contents and index blocks of PDF documents are neither
    compared nor counted, unless ``cfg.pdf.compare_toc`` is set. Nor are
    the images that either document removed as page chrome, nor any image
    without ``cfg.images``. A word of prose replaced by a close one is
    judged by a speller with the personal words of ``cfg.spelling``, whose
    dictionaries are loaded at the first such word.

    Args:
        ref: Reference document.
        new: New version of the document.
        cfg: Run configuration.

    Returns:
        The findings, in reference order, those of the images after those
        of the text, and statistics. If the dictionaries could not judge
        every such word, the new document is a copy whose warnings end
        with the reasons.
    """
    chrome = ref.chrome_keys | new.chrome_keys
    patterns = [re.compile(pattern) for pattern in cfg.ignore.patterns]
    toc = cfg.pdf.compare_toc
    ref_blocks, ref_kept = _prepare(
        ref.blocks,
        chrome,
        patterns,
        compare_toc=toc,
    )
    new_blocks, new_kept = _prepare(
        new.blocks,
        chrome,
        patterns,
        compare_toc=toc,
    )
    pairs = align(
        ref_blocks,
        new_blocks,
        match=cfg.compare.match,
        moved_match=cfg.compare.moved_match,
    )
    drafter = _Drafter(
        ref_blocks,
        new_blocks,
        (ref.lang, new.lang),
        pdf=ref.format == "pdf",
        speller=Speller(cfg.spelling.words),
    )
    text_drafts = drafter.drafts(pairs)
    notes = drafter.warnings()
    if notes:
        new = replace(new, warnings=[*new.warnings, *notes])
    images = (
        _images((ref, new), (ref_kept, new_kept), pairs)
        if cfg.images
        else _Images([], 0, 0, 0)
    )
    stats = CompareStats(
        ref_blocks=len(ref_blocks),
        new_blocks=len(new_blocks),
        matched=sum(len(pair.ref) for pair in pairs if pair.kind in MATCHED),
        moved=sum(pair.moved for pair in pairs),
        ref_images=images.ref,
        new_images=images.new,
        matched_images=images.matched,
        images=cfg.images,
    )
    drafts = [*text_drafts, *images.drafts]
    return CompareResult(ref, new, finalize_drafts(drafts, cfg), stats)
