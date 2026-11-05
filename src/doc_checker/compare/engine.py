# Copyright 2026 Christophe Le Douarec
"""Turn the alignment of two documents into findings."""

import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final, Literal

from doc_checker.compare.align import Pair, PairKind, align
from doc_checker.compare.classify import HunkContext, classify_hunk
from doc_checker.compare.textdiff import Hunk, diff_tokens
from doc_checker.config import rule_disabled
from doc_checker.model.document import Block, BlockKind, Document
from doc_checker.model.findings import (
    DiffSegment,
    Finding,
    Severity,
    Side,
    finding_id,
)
from doc_checker.rules import get_rule
from doc_checker.text.normalize import loose_key
from doc_checker.text.tokenize import Token, TokenKind, tokenize

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from doc_checker.config import Config

CONTEXT_TOKENS: Final = 8
EXCERPT_MAX_CHARS: Final = 300

_ELLIPSIS: Final = "\U00002026"
_MATCHED: Final = frozenset(
    {PairKind.EQUAL, PairKind.MODIFIED, PairKind.SPLIT, PairKind.MERGE},
)
_CONTENTS: Final = frozenset({BlockKind.TOC, BlockKind.INDEX})

type SegmentOp = Literal["equal", "delete", "insert"]


@dataclass(frozen=True, slots=True)
class CompareStats:
    """Figures about a comparison.

    Attributes:
        ref_blocks: Compared reference blocks.
        new_blocks: Compared new blocks.
        matched: Reference blocks found in the new document.
        moved: Pairs out of the common order.
    """

    ref_blocks: int
    new_blocks: int
    matched: int
    moved: int


@dataclass(slots=True)
class CompareResult:
    """Outcome of the comparison of two documents."""

    ref: Document
    new: Document
    findings: list[Finding]
    stats: CompareStats


@dataclass(frozen=True, slots=True)
class _Draft:
    """A finding before configuration overrides and filters."""

    rule: str
    severity: Severity | None
    ref: Side | None
    new: Side | None
    detail: str | None = None


def _prepare(
    blocks: Iterable[Block],
    chrome: frozenset[str],
    patterns: Sequence[re.Pattern[str]],
    *,
    compare_toc: bool,
) -> list[Block]:
    """Return the blocks to compare, with ignored patterns removed.

    Page chrome is dropped. So are the table of contents and index blocks,
    unless ``compare_toc`` is set.
    """
    skipped = frozenset() if compare_toc else _CONTENTS
    prepared = []
    for block in blocks:
        if block.key_strict in chrome or block.kind in skipped:
            continue
        key = block.key_strict
        for pattern in patterns:
            key = pattern.sub("", key)
        if patterns and block.kind is not BlockKind.CODE:
            key = " ".join(key.split())
        loose = loose_key(key) if patterns else block.key_loose
        if loose:
            prepared.append(replace(block, key_strict=key, key_loose=loose))
    return prepared


def _truncate(text: str) -> str:
    """Return the text cut to ``EXCERPT_MAX_CHARS`` characters."""
    if len(text) <= EXCERPT_MAX_CHARS:
        return text
    return text[:EXCERPT_MAX_CHARS] + _ELLIPSIS


def _whole(blocks: Sequence[Block], op: SegmentOp) -> Side:
    """Return a side showing the whole text of blocks."""
    text = " ".join(block.text for block in blocks)
    return Side(blocks[0].source, (DiffSegment(op, _truncate(text)),))


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
    """Build the draft findings of the pairs of an alignment."""

    def __init__(
        self,
        ref: Sequence[Block],
        new: Sequence[Block],
        langs: tuple[str | None, str | None],
    ) -> None:
        self.ref = ref
        self.new = new
        self.langs = langs
        self.vocabulary = frozenset(
            token.text.casefold()
            for block in ref
            for token in tokenize(block.key_strict)
            if token.kind is TokenKind.WORD
        )

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
    ) -> list[_Draft]:
        """Return one draft per word level difference between blocks."""
        ref_text = " ".join(block.key_strict for block in ref_blocks)
        new_text = " ".join(block.key_strict for block in new_blocks)
        ref_tokens, new_tokens = tokenize(ref_text), tokenize(new_text)
        ctx = HunkContext(
            ref_text,
            new_text,
            ref_tokens,
            new_tokens,
            self.vocabulary,
            *self.langs,
        )
        return [
            self.hunk_draft(hunk, ctx, ref_blocks[0], new_blocks[0])
            for hunk in diff_tokens(ref_tokens, new_tokens)
        ]

    @staticmethod
    def hunk_draft(
        hunk: Hunk,
        ctx: HunkContext,
        ref_block: Block,
        new_block: Block,
    ) -> _Draft:
        """Return the draft of one word level difference."""
        classification = classify_hunk(hunk, ctx)
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
        return _Draft(
            classification.rule,
            classification.severity,
            Side(ref_block.source, ref_segments),
            Side(new_block.source, new_segments),
        )

    def matched(self, pair: Pair) -> list[_Draft]:
        """Return the drafts of a pair of matched blocks."""
        ref_blocks = [self.ref[i] for i in pair.ref]
        new_blocks = [self.new[j] for j in pair.new]
        drafts = []
        if pair.moved:
            drafts.append(
                _Draft(
                    "block.moved",
                    None,
                    _whole(ref_blocks, "equal"),
                    _whole(new_blocks, "equal"),
                ),
            )
        if pair.kind in {PairKind.SPLIT, PairKind.MERGE}:
            drafts.append(
                _Draft(
                    "block.split_merged",
                    None,
                    _whole(ref_blocks, "equal"),
                    _whole(new_blocks, "equal"),
                ),
            )
        if pair.kind is not PairKind.EQUAL:
            drafts += self.hunks(ref_blocks, new_blocks)
        return drafts

    def drafts(self, pairs: Sequence[Pair]) -> list[_Draft]:
        """Return the drafts of all pairs, in pair order."""
        deleted = [p.ref[0] for p in pairs if p.kind is PairKind.DELETED]
        heads, covered = self.missing_sections(deleted)
        drafts: list[_Draft] = []
        for pair in pairs:
            if pair.kind is PairKind.DELETED:
                index = pair.ref[0]
                block = self.ref[index]
                if index in heads:
                    detail = f"{heads[index]} blocks missing in this section"
                    side = _whole([block], "delete")
                    drafts.append(
                        _Draft("section.missing", None, side, None, detail),
                    )
                elif index not in covered:
                    side = _whole([block], "delete")
                    drafts.append(_Draft("block.missing", None, side, None))
            elif pair.kind is PairKind.INSERTED:
                side = _whole([self.new[pair.new[0]]], "insert")
                drafts.append(_Draft("block.added", None, None, side))
            else:
                drafts += self.matched(pair)
        return drafts


def _finalize(drafts: Iterable[_Draft], cfg: Config) -> list[Finding]:
    """Return the findings, with overrides, filters and identifiers."""
    findings = []
    for draft in drafts:
        if rule_disabled(cfg, draft.rule):
            continue
        spec = get_rule(draft.rule)
        default = spec.severity if draft.severity is None else draft.severity
        severity = cfg.severity.get(draft.rule, default)
        if severity < cfg.min_severity:
            continue
        identifier = finding_id(draft.rule, draft.ref, draft.new)
        if identifier in cfg.ignore.findings:
            continue
        findings.append(
            Finding(
                draft.rule,
                severity,
                spec.category,
                spec.title,
                draft.ref,
                draft.new,
                draft.detail,
                identifier,
            ),
        )
    return findings


def compare_documents(
    ref: Document,
    new: Document,
    cfg: Config,
) -> CompareResult:
    """Compare a reference document with its new version.

    The table of contents and index blocks of PDF documents are neither
    compared nor counted, unless ``cfg.pdf.compare_toc`` is set.

    Args:
        ref: Reference document.
        new: New version of the document.
        cfg: Run configuration.

    Returns:
        The findings, in reference order, and statistics.
    """
    chrome = ref.chrome_keys | new.chrome_keys
    patterns = [re.compile(pattern) for pattern in cfg.ignore.patterns]
    toc = cfg.pdf.compare_toc
    ref_blocks = _prepare(ref.blocks, chrome, patterns, compare_toc=toc)
    new_blocks = _prepare(new.blocks, chrome, patterns, compare_toc=toc)
    pairs = align(
        ref_blocks,
        new_blocks,
        match=cfg.compare.match,
        moved_match=cfg.compare.moved_match,
    )
    drafter = _Drafter(ref_blocks, new_blocks, (ref.lang, new.lang))
    stats = CompareStats(
        ref_blocks=len(ref_blocks),
        new_blocks=len(new_blocks),
        matched=sum(len(pair.ref) for pair in pairs if pair.kind in _MATCHED),
        moved=sum(pair.moved for pair in pairs),
    )
    return CompareResult(
        ref,
        new,
        _finalize(drafter.drafts(pairs), cfg),
        stats,
    )
