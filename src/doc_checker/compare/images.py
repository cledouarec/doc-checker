# Copyright 2026 Christophe Le Douarec
"""Match the images of two documents and report how each pair changed.

An image takes its place from the compared blocks: the last one before it,
and the caption next to it. The images are then matched in three passes:
the same content digest; the same perceived content, by the hashes of the
images with pixels or by the texts of two SVGs; last, the same context,
which is a caption, an alternative text, a file name or a place: between
the counterparts of the compared blocks before and after the image. The
second pass runs again over the images left, a few rounds, so that a series
of near-identical images still pairs each image with its own, and then pairs
the images left whose hashes are equal and whose aspect ratios are alike, at
their place first, so that a long series of identical icons pairs whole. An
image left alone is missing or added. The pixels of a pair of raster images
are read again to judge it only when the pair is close in perceived content.
A pair matched by its context whose difference hashes and aspect ratios are
close, but not its perceptual hashes, which a symmetric drawing flips when
its render is a pixel narrower, is measured: its SSIM tells whether it is
close in content after all.
"""

import math
import re
from bisect import bisect_left
from collections import deque
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Final, Literal

import numpy as np

from doc_checker.compare.drafts import ARROW, Draft
from doc_checker.compare.verdicts import SSIM_MIN, judge
from doc_checker.images.hashing import hamming, hamming_matrix
from doc_checker.images.metrics import measure
from doc_checker.images.pixels import EXTERNAL, NOT_READ
from doc_checker.images.refs import describe
from doc_checker.model.document import BlockKind
from doc_checker.model.findings import DiffSegment, Side

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

    from doc_checker.compare.drafts import SegmentOp
    from doc_checker.images.load import PixelLoader
    from doc_checker.images.metrics import Measures
    from doc_checker.model.document import Block, ImageRef

DHASH_MAX: Final = 8
PHASH_MAX: Final = 24
ASPECT_TOLERANCE: Final = 1.15
CANDIDATES_PER_IMAGE: Final = 8
CONTENT_ROUNDS: Final = 4
# The width of the buckets of aspect ratios of identical images, on the
# logarithm of the ratio: about one percent.
ASPECT_BUCKET: Final = 0.01
ROW_CHUNK: Final = 512

# The suffix that a generator adds to a file name to bust caches.
_HASH_SUFFIX: Final = re.compile(r"[-.][0-9a-f]{8,}$")
_SVG: Final = "svg"

type How = Literal["same", "content", "context"]
# The hashes and the aspect ratio of an image with pixels.
type _Signature = tuple[int, int, float]


@dataclass(frozen=True, slots=True)
class Placed:
    """An image block and where it stands among the compared blocks.

    Attributes:
        block: An `IMAGE` block, which has its image.
        slot: Position, among the compared blocks, of the last one before
            the image; -1 without one.
        caption: Loose key of the caption next to the image, or "".
    """

    block: Block
    slot: int
    caption: str

    @property
    def image(self) -> ImageRef:
        """The image of the block.

        Raises:
            ValueError: If the block has no image.
        """
        image = self.block.image
        if image is None:
            msg = f"the block at {self.block.source.href} has no image"
            raise ValueError(msg)
        return image


@dataclass(frozen=True, slots=True)
class Match:
    """A pair of images of the two documents, or an image left alone.

    Attributes:
        ref: Position of the reference image, None for an added image.
        new: Position of the new image, None for a missing image.
        how: The pass that paired the images, None for an image alone.
    """

    ref: int | None
    new: int | None
    how: How | None = None


def _caption(blocks: Sequence[Block], index: int) -> str:
    """Return the loose key of the caption of an image block, if any.

    Args:
        blocks: The blocks of a document.
        index: Position of the image block.

    Returns:
        The key of the next block if it is a caption of the same file and
        page, else that of the previous block on the same condition, else
        "".
    """
    source = blocks[index].source
    for near in (index + 1, index - 1):
        if 0 <= near < len(blocks):
            block = blocks[near]
            if (
                block.kind is BlockKind.CAPTION
                and block.source.file == source.file
                and block.source.page == source.page
            ):
                return block.key_loose
    return ""


def placed_images(
    blocks: Sequence[Block],
    kept: Sequence[int],
) -> list[Placed]:
    """Place the images of a document among its compared blocks.

    Args:
        blocks: The blocks of the document, in reading order.
        kept: Positions in ``blocks`` of the compared blocks, ascending.

    Returns:
        The image blocks, in order. The slot of an image is the position in
        ``kept`` of the last compared block before it, -1 without one.
    """
    placed: list[Placed] = []
    before = 0
    for index, block in enumerate(blocks):
        while before < len(kept) and kept[before] < index:
            before += 1
        if block.kind is BlockKind.IMAGE and block.image is not None:
            placed.append(Placed(block, before - 1, _caption(blocks, index)))
    return placed


def _signature(image: ImageRef) -> _Signature | None:
    """Return the hashes and the aspect ratio of an image with pixels.

    Args:
        image: An image.

    Returns:
        The difference hash, the perceptual hash and the width over the
        height, or None for an image whose pixels were not read.
    """
    if (
        image.dhash is None
        or image.phash is None
        or not image.width
        or not image.height
    ):
        return None
    return image.dhash, image.phash, image.width / image.height


def _twin_key(image: ImageRef) -> tuple[int, int, int] | None:
    """Return what identical images of the same shape share.

    Args:
        image: An image.

    Returns:
        The difference hash, the perceptual hash and the bucket of the
        aspect ratio, `ASPECT_BUCKET` wide on its logarithm, or None for an
        image whose pixels were not read.
    """
    signature = _signature(image)
    if signature is None:
        return None
    dhash, phash, aspect = signature
    return dhash, phash, round(math.log(aspect) / ASPECT_BUCKET)


def _close_signatures(
    ref: ImageRef,
    new: ImageRef,
    *,
    perceptual: bool = True,
) -> bool:
    """Return whether two images are close enough to pair by content.

    Args:
        ref: A reference image.
        new: A new image.
        perceptual: Whether their perceptual hashes must be close too.

    Returns:
        Whether both have pixels, aspect ratios within `ASPECT_TOLERANCE`
        of each other, difference hashes within `DHASH_MAX`, and, if asked,
        perceptual hashes within `PHASH_MAX`.
    """
    first, second = _signature(ref), _signature(new)
    if first is None or second is None:
        return False
    ratio = first[2] / second[2]
    return (
        1 / ASPECT_TOLERANCE <= ratio <= ASPECT_TOLERANCE
        and hamming(first[0], second[0]) <= DHASH_MAX
        and (not perceptual or hamming(first[1], second[1]) <= PHASH_MAX)
    )


@dataclass(frozen=True, slots=True)
class _Hashes:
    """The signatures of some images, ready for numpy.

    Attributes:
        dhash: Difference hashes.
        phash: Perceptual hashes.
        aspect: Aspect ratios.
    """

    dhash: list[int]
    phash: list[int]
    aspect: NDArray[np.float64]

    @classmethod
    def of(cls, signatures: Sequence[_Signature]) -> _Hashes:
        """Gather signatures.

        Args:
            signatures: The signatures of the images, as `_signature` gives
                them.

        Returns:
            Their hashes and aspect ratios.
        """
        return cls(
            [signature[0] for signature in signatures],
            [signature[1] for signature in signatures],
            np.array([signature[2] for signature in signatures]),
        )


def _close(
    rows: _Hashes,
    columns: _Hashes,
    start: int,
) -> tuple[NDArray[np.int64], NDArray[np.intp], NDArray[np.intp]]:
    """Find the pairs of images whose hashes and aspect ratios are close.

    Args:
        rows: The hashes of the reference images.
        columns: The hashes of the new images.
        start: The first of the `ROW_CHUNK` rows to compare.

    Returns:
        The distance, the row and the column of each pair whose hashes are
        within `DHASH_MAX` and `PHASH_MAX` and whose aspect ratios are
        within `ASPECT_TOLERANCE` of each other, row by row and column by
        column; the distance is the sum of the distances of the two hashes.
    """
    stop = start + ROW_CHUNK
    dhash = hamming_matrix(rows.dhash[start:stop], columns.dhash)
    phash = hamming_matrix(rows.phash[start:stop], columns.phash)
    row, column = np.nonzero((dhash <= DHASH_MAX) & (phash <= PHASH_MAX))
    aspect = rows.aspect[start + row] / columns.aspect[column]
    kept = (aspect >= 1 / ASPECT_TOLERANCE) & (aspect <= ASPECT_TOLERANCE)
    row, column = row[kept], column[kept]
    distance = dhash[row, column].astype(np.int64) + phash[row, column]
    return distance, row + start, column


def _candidates(
    rows: Sequence[_Signature],
    columns: Sequence[_Signature],
) -> list[tuple[int, int, int]]:
    """Find the pairs of images whose perceived content is the same.

    The distances are computed by numpy, as `_close` does, `ROW_CHUNK`
    rows at a time, so that the memory stays linear in the number of
    columns.

    Args:
        rows: Signatures of the reference images.
        columns: Signatures of the new images.

    Returns:
        The distance, the row and the column of each pair whose aspect
        ratios are within `ASPECT_TOLERANCE` of each other, whose hashes
        are within `DHASH_MAX` and `PHASH_MAX`, at most
        `CANDIDATES_PER_IMAGE` per row, the nearest ones; the distance is
        the sum of the distances of the two hashes.
    """
    if not rows or not columns:
        return []
    row_hashes, column_hashes = _Hashes.of(rows), _Hashes.of(columns)
    found: list[tuple[int, int, int]] = []
    for start in range(0, len(rows), ROW_CHUNK):
        distance, row, column = _close(row_hashes, column_hashes, start)
        # Row by row, the nearest first; equal distances in column order,
        # as `np.nonzero` gives them.
        order = np.argsort(distance, kind="stable")
        order = order[np.argsort(row[order], kind="stable")]
        row, column, distance = row[order], column[order], distance[order]
        rank = np.arange(row.size) - np.searchsorted(row, row)
        best = rank < CANDIDATES_PER_IMAGE
        found += zip(
            distance[best].tolist(),
            row[best].tolist(),
            column[best].tolist(),
            strict=True,
        )
    return found


def _alt_key(image: ImageRef) -> str | None:
    """Return the alternative text of an image without its outer blanks.

    Args:
        image: An image.

    Returns:
        The text, or None if it is absent or blank.
    """
    return (image.alt or "").strip() or None


def _file_key(name: str) -> str | None:
    """Return the stem of a file name, in lower case, without a hash suffix.

    Args:
        name: The name of an image, such as ``img/setup-3f2a1b9c.png``.

    Returns:
        The stem, such as ``setup``, or None if it is empty.
    """
    return _HASH_SUFFIX.sub("", PurePosixPath(name).stem.lower()) or None


class _Waiting:
    """The new images by a key of their context, in document order."""

    def __init__(self, keys: Sequence[str | int | None]) -> None:
        """Index the new images by their keys.

        Args:
            keys: The key of each new image, None for no key.
        """
        self._queues: dict[str | int, deque[int]] = {}
        for index, key in enumerate(keys):
            if key is not None:
                self._queues.setdefault(key, deque()).append(index)

    def first(
        self,
        key: str | int | None,
        taken: Sequence[bool],
    ) -> int | None:
        """Return the first new image of a key that is not taken yet.

        Args:
            key: The key; None finds nothing.
            taken: Whether each new image is taken.

        Returns:
            The position of the image, or None.
        """
        if key is None:
            return None
        queue = self._queues.get(key)
        while queue and taken[queue[0]]:
            queue.popleft()
        return queue[0] if queue else None


class _Places:
    """The new images by their slots, to find one between two slots."""

    def __init__(self, slots: Sequence[int]) -> None:
        """Sort the new images by slot, then in document order.

        Args:
            slots: The slot of each new image.
        """
        self._order = sorted(range(len(slots)), key=lambda j: (slots[j], j))
        self._slots = [slots[j] for j in self._order]
        # The position, in that order, from which to look for an image that
        # is not taken; a taken image is never free again.
        self._next = list(range(len(slots) + 1))

    def _free(self, position: int, taken: Sequence[bool]) -> int:
        """Return the first position from a given one whose image is free.

        Args:
            position: Where to start, in the order of the slots.
            taken: Whether each new image is taken.

        Returns:
            The position, or the count of the images if none is free.
        """
        last = len(self._order)
        found = position
        while found < last and (
            self._next[found] != found or taken[self._order[found]]
        ):
            if self._next[found] == found:
                self._next[found] = found + 1
            found = self._next[found]
        while position != found:
            self._next[position], position = found, self._next[position]
        return found

    def first(
        self,
        window: tuple[int, int] | None,
        taken: Sequence[bool],
    ) -> int | None:
        """Return the first new image between two slots, not taken yet.

        Args:
            window: The least and the greatest slot, None to find nothing.
            taken: Whether each new image is taken.

        Returns:
            The position of the image of the least slot, the first one in
            document order on a tie, or None.
        """
        if window is None:
            return None
        low, high = window
        found = self._free(bisect_left(self._slots, low), taken)
        if found < len(self._order) and self._slots[found] <= high:
            return self._order[found]
        return None


class _Matcher:
    """The matching of the images of two documents, pass after pass."""

    def __init__(
        self,
        ref: Sequence[Placed],
        new: Sequence[Placed],
        counterparts: tuple[Mapping[int, int], Mapping[int, int]],
    ) -> None:
        """Start with no image matched.

        Args:
            ref: The reference images.
            new: The new images.
            counterparts: For each matched compared block of the reference,
                the position of the last new block of its pair, then that
                of the first one.
        """
        self.ref = ref
        self.new = new
        self.follows, self.precedes = counterparts
        self.partner: list[tuple[int, How] | None] = [None] * len(ref)
        self.taken = [False] * len(new)

    def pair(self, ref: int, new: int, how: How) -> None:
        """Match two images."""
        self.partner[ref] = (new, how)
        self.taken[new] = True

    def alone(self) -> list[int]:
        """Return the reference images not matched yet, in order."""
        return [i for i, partner in enumerate(self.partner) if partner is None]

    def window(self, slot: int) -> tuple[int, int] | None:
        """Return the slots of the new images at the place of a reference one.

        The place lies between the counterparts of the compared blocks
        before and after the reference image: after the last new block of
        the pair of the block before, and before the first new block of the
        pair of the block after.

        Args:
            slot: Slot of a reference image.

        Returns:
            The least and the greatest slot of the new images in that place,
            the counterpart of the block before being -1 for a slot of -1.
            When only one of the two blocks is matched, or when the
            counterpart of the block before comes after that of the block
            after, the place is the one slot next to the counterpart that is
            known, the block before first; None when neither block is
            matched.
        """
        low = -1 if slot < 0 else self.follows.get(slot)
        high = self.precedes.get(slot + 1)
        if low is not None and high is not None and low < high:
            return low, high - 1
        if low is not None:
            return low, low
        if high is not None:
            return high - 1, high - 1
        return None

    def near(self, ref: Placed, new: Placed) -> bool:
        """Return whether two images share a caption or their place."""
        if ref.caption and ref.caption == new.caption:
            return True
        window = self.window(ref.slot)
        return window is not None and window[0] <= new.slot <= window[1]

    def same(self) -> None:
        """Match the images of the same digest, in order."""
        waiting: dict[str, deque[int]] = {}
        for j, placed in enumerate(self.new):
            waiting.setdefault(placed.image.digest, deque()).append(j)
        for i, placed in enumerate(self.ref):
            queue = waiting.get(placed.image.digest)
            if queue:
                self.pair(i, queue.popleft(), "same")

    def content(self) -> None:
        """Match the images of the same perceived content.

        The images with pixels are matched in rounds, at most
        `CONTENT_ROUNDS` of them, until a round matches nothing: the
        candidates of an image may all be taken by others in a round, which
        the next round replaces with the nearest images left. The twins
        come next, as `twins` matches them. The SVGs come last: each one
        takes the first new SVG with the same texts, which are not empty.
        """
        for _ in range(CONTENT_ROUNDS):
            if not self.content_round():
                break
        self.twins()
        svgs: dict[tuple[str, ...], deque[int]] = {}
        for j, placed in enumerate(self.new):
            image = placed.image
            if not self.taken[j] and image.format == _SVG and image.texts:
                svgs.setdefault(image.texts, deque()).append(j)
        for i in self.alone():
            image = self.ref[i].image
            queue = svgs.get(image.texts) if image.format == _SVG else None
            if queue:
                self.pair(i, queue.popleft(), "content")

    def twins(self) -> None:
        """Match the images left with the same hashes and shape.

        A series of more identical images than the rounds can pair, each
        one with `CANDIDATES_PER_IMAGE` candidates at most, is paired whole,
        key by key, as `_twin_key` tells it. Each reference image left first
        takes the first new image left of its key at its place, as `window`
        tells it; the images left then pair in document order. A changed
        image of the series, whose key is another, keeps its place free for
        the context pass, which pairs it with its own counterpart.
        """
        rows: dict[tuple[int, int, int], list[int]] = {}
        for i in self.alone():
            key = _twin_key(self.ref[i].image)
            if key is not None:
                rows.setdefault(key, []).append(i)
        columns: dict[tuple[int, int, int], list[int]] = {}
        for j, placed in enumerate(self.new):
            key = _twin_key(placed.image)
            if not self.taken[j] and key in rows:
                columns.setdefault(key, []).append(j)
        for key, found in columns.items():
            self._twins_of(rows[key], found)

    def _twins_of(self, rows: Sequence[int], columns: Sequence[int]) -> None:
        """Match reference and new images of one key, at their place first.

        Args:
            rows: The reference images left, in order.
            columns: The new images left, in order.
        """
        taken = [False] * len(columns)
        places = _Places([self.new[j].slot for j in columns])
        left = []
        for i in rows:
            found = places.first(self.window(self.ref[i].slot), taken)
            if found is None:
                left.append(i)
            else:
                taken[found] = True
                self.pair(i, columns[found], "content")
        free = (k for k, done in enumerate(taken) if not done)
        for i, k in zip(left, free, strict=False):
            self.pair(i, columns[k], "content")

    def content_round(self) -> bool:
        """Match greedily the images with pixels left, by their hashes.

        The nearest pair comes first, then the one whose images share a
        caption or their place, then in reference and new order.

        Returns:
            Whether the round matched an image.
        """
        rows = [
            (i, signature)
            for i in self.alone()
            if (signature := _signature(self.ref[i].image)) is not None
        ]
        columns = [
            (j, signature)
            for j, placed in enumerate(self.new)
            if not self.taken[j]
            and (signature := _signature(placed.image)) is not None
        ]
        found = _candidates(
            [signature for _, signature in rows],
            [signature for _, signature in columns],
        )
        ranked = sorted(
            (
                distance,
                0 if self.near(self.ref[i], self.new[j]) else 1,
                i,
                j,
            )
            for distance, i, j in (
                (distance, rows[row][0], columns[column][0])
                for distance, row, column in found
            )
        )
        matched = False
        for _, _, i, j in ranked:
            if self.partner[i] is None and not self.taken[j]:
                self.pair(i, j, "content")
                matched = True
        return matched

    def context(self) -> None:
        """Match each image left with the first new one of its context.

        The context is, in this order, the caption, the alternative text,
        the file name without its hash suffix, and the place, as `window`
        tells it.
        """
        waiting = (
            _Waiting([placed.caption or None for placed in self.new]),
            _Waiting([_alt_key(placed.image) for placed in self.new]),
            _Waiting([_file_key(placed.image.name) for placed in self.new]),
        )
        places = _Places([placed.slot for placed in self.new])
        for i in self.alone():
            placed = self.ref[i]
            keys = (
                placed.caption or None,
                _alt_key(placed.image),
                _file_key(placed.image.name),
            )
            found = (
                queue.first(key, self.taken)
                for queue, key in zip(waiting, keys, strict=True)
            )
            j = next((j for j in found if j is not None), None)
            if j is None:
                j = places.first(self.window(placed.slot), self.taken)
            if j is not None:
                self.pair(i, j, "context")

    def matches(self) -> list[Match]:
        """Return the pairs and the images left alone.

        Returns:
            The pairs and the missing images in reference order, then the
            added images in new order.
        """
        found = [
            Match(i, None) if partner is None else Match(i, *partner)
            for i, partner in enumerate(self.partner)
        ]
        found += [
            Match(None, j) for j, taken in enumerate(self.taken) if not taken
        ]
        return found


def match_images(
    ref: Sequence[Placed],
    new: Sequence[Placed],
    follows: Mapping[int, int],
    precedes: Mapping[int, int] | None = None,
) -> list[Match]:
    """Match the images of two documents.

    Args:
        ref: The reference images, as `placed_images` gives them.
        new: The new images.
        follows: For each reference block of a matched pair of the
            alignment, by its position among the compared blocks, the
            position of the last new block of the pair.
        precedes: The same, with the position of the first new block of the
            pair, and the count of the compared new blocks for the count of
            the compared reference blocks; None for none.

    Returns:
        The pairs and the missing images in reference order, then the added
        images in new order. The first pass matches the images of the same
        digest, in order. The second matches the images with pixels whose
        aspect ratios and hashes are close, in rounds, then those left with
        equal hashes and alike aspect ratios, at their place first, and the
        SVGs with the same texts. The third matches each image left with the
        first new one of the same caption, else alternative text, else file
        name, else place: between the counterparts of the blocks before and
        after it, or both at the start.
    """
    matcher = _Matcher(ref, new, (follows, precedes or {}))
    matcher.same()
    matcher.content()
    matcher.context()
    return matcher.matches()


def _side(placed: Placed, op: SegmentOp) -> Side:
    """Return the side of a finding that shows an image."""
    image = placed.image
    segments = (DiffSegment(op, describe(image)),)
    return Side(placed.block.source, segments, BlockKind.IMAGE, image)


def _seen(image: ImageRef) -> bool:
    """Return whether the extraction read an image: pixels or SVG source."""
    return image.problem is None


def _broken(image: ImageRef) -> bool:
    """Return whether an image cannot be read where a reader would see it.

    An image left unread on purpose, as `NOT_READ` tells, is not broken: an
    external image is never fetched, and a vector figure of a PDF that was
    not rendered is drawn on its page all the same.
    """
    return image.problem is not None and image.problem not in NOT_READ


def _pixels_detail(image: ImageRef) -> str | None:
    """Return the size of a raster image, and its resolution if known."""
    if image.width is None or image.height is None:
        return None
    detail = f"{image.width}x{image.height} px"
    if image.dpi is not None:
        detail += f", {image.dpi:.0f} dpi"
    return detail


def _change(rule: str, ref: Placed, new: Placed, detail: str | None) -> Draft:
    """Return the draft of a pair of images whose format may have changed."""
    pattern = f"{ref.image.format}{ARROW}{new.image.format}"
    return Draft(
        rule,
        None,
        _side(ref, "delete"),
        _side(new, "insert"),
        detail,
        pattern,
    )


def _kind_change(ref: Placed, new: Placed) -> Draft | None:
    """Return the draft of a drawing that became a raster image, or back."""
    if ref.image.vector and not new.image.vector:
        return _change("image.rasterized", ref, new, _pixels_detail(new.image))
    if new.image.vector and not ref.image.vector:
        return _change("image.vectorized", ref, new, None)
    return None


def _context_draft(ref: Placed, new: Placed) -> Draft | None:
    """Return the draft of two images matched by their context.

    Args:
        ref: The reference image.
        new: The new image.

    Returns:
        A missing image, with both sides and the reason, if the new image is
        broken and the reference is not, whatever their formats; else None
        if either image is broken or external, as it cannot be judged; else
        a rasterized or a vectorized image; else, when the extraction read
        both images, a replaced image; else None.
    """
    first, second = ref.image, new.image
    if _broken(second):
        if _broken(first):
            return None
        return Draft(
            "image.missing",
            None,
            _side(ref, "delete"),
            _side(new, "insert"),
            f"the new image cannot be read: {second.problem}",
            first.format or "image",
        )
    if _broken(first) or EXTERNAL in {first.problem, second.problem}:
        return None
    changed = _kind_change(ref, new)
    if changed is not None or not (_seen(first) and _seen(second)):
        return changed
    return _change("image.replaced", ref, new, None)


def _alt_draft(ref: Placed, new: Placed) -> Draft | None:
    """Return the draft of an alternative text that a pair of images lost.

    An inline SVG, which has no name, has no alternative text to lose: a
    generator writes inline the SVG that another one writes as an ``img``.
    """
    alt = _alt_key(ref.image)
    image = new.image
    inline = image.format == _SVG and not image.name
    if alt is None or _alt_key(image) is not None or inline:
        return None
    return Draft(
        "image.alt_lost",
        None,
        _side(ref, "equal"),
        _side(new, "equal"),
        f'alt "{alt}" lost',
        "alt",
    )


class _Judge:
    """Judge the pairs of images matched by their perceived content."""

    def __init__(
        self,
        pixels: tuple[PixelLoader, PixelLoader],
        *,
        pdf: bool,
    ) -> None:
        """Keep the loaders of the pixels of the two documents.

        Args:
            pixels: The loaders of the reference and of the new document.
            pdf: Whether both documents are PDF files.
        """
        self.pixels = pixels
        self.pdf = pdf
        self.measured: dict[tuple[str, str], Measures | None] = {}

    def measures(self, ref: ImageRef, new: ImageRef) -> Measures | None:
        """Measure a pair of raster images, once for each pair of digests.

        Args:
            ref: The reference image.
            new: The new image.

        Returns:
            The measures, or None if the pixels of either image cannot be
            read again, or rendered again for a vector figure. The pixels
            are dropped once measured.
        """
        key = (ref.digest, new.digest)
        if key not in self.measured:
            first = self.pixels[0].load(ref)
            second = None if first is None else self.pixels[1].load(new)
            self.measured[key] = (
                None
                if first is None or second is None
                else measure(first, second)
            )
        return self.measured[key]

    def similar(self, ref: ImageRef, new: ImageRef) -> bool:
        """Tell whether the pixels of two images are close, by their SSIM.

        Args:
            ref: The reference image.
            new: The new image.

        Returns:
            Whether both can be read again and their SSIM is `SSIM_MIN` or
            more.
        """
        measures = self.measures(ref, new)
        return measures is not None and measures.ssim >= SSIM_MIN

    def __call__(self, ref: Placed, new: Placed) -> Draft | None:
        """Return the draft of two images close in content.

        Two drawings give none, unless both are SVGs, whose texts are the
        same but not their source; a drawing against a raster image is
        rasterized or vectorized; two raster images, which never share a
        digest after the first pass, are measured and judged.
        """
        first, second = ref.image, new.image
        if first.format == _SVG and second.format == _SVG:
            return _change(
                "image.reencoded",
                ref,
                new,
                "SVG source changed, same texts",
            )
        if first.vector or second.vector:
            return _kind_change(ref, new)
        measures = self.measures(first, second)
        if measures is None:
            return None
        verdict = judge(first, second, measures, pdf=self.pdf)
        if verdict.rule is None:
            return None
        return Draft(
            verdict.rule,
            verdict.severity,
            _side(ref, "delete"),
            _side(new, "insert"),
            verdict.detail,
            verdict.pattern,
        )


def _alone_draft(ref: Placed | None, new: Placed | None) -> Draft | None:
    """Return the draft of a missing or an added image.

    Args:
        ref: The reference image of a match, None for an added image.
        new: Its new image, None for a missing image.

    Returns:
        The draft, or None for a pair.
    """
    if new is None and ref is not None:
        side = _side(ref, "delete")
        pattern = ref.image.format or "image"
        return Draft("image.missing", None, side, None, pattern=pattern)
    if ref is None and new is not None:
        side = _side(new, "insert")
        pattern = new.image.format or "image"
        return Draft("image.added", None, None, side, pattern=pattern)
    return None


def image_drafts(
    ref: Sequence[Placed],
    new: Sequence[Placed],
    matches: Sequence[Match],
    pixels: tuple[PixelLoader, PixelLoader],
    *,
    pdf: bool,
) -> list[Draft]:
    """Return the drafts of the images of two documents.

    Args:
        ref: The reference images, as `placed_images` gives them.
        new: The new images.
        matches: The matches of the images, as `match_images` gives them.
        pixels: The loaders of the pixels of the reference and new
            documents.
        pdf: Whether both documents are PDF files.

    Returns:
        The drafts, in the order of the matches: a missing or an added
        image; for a pair, the change of its images, then the loss of its
        alternative text. A pair matched by its context whose images are
        close in content, which the rounds of the content pass may leave, is
        judged as a pair matched by its content. So is a pair matched by its
        context whose difference hashes and aspect ratios are close, but
        not its perceptual hashes, if its SSIM is `SSIM_MIN` or more; its
        pixels are read for that, and for no other pair of the context.
    """
    judged = _Judge(pixels, pdf=pdf)
    drafts: list[Draft] = []
    for match in matches:
        first = None if match.ref is None else ref[match.ref]
        second = None if match.new is None else new[match.new]
        if first is None or second is None:
            alone = _alone_draft(first, second)
            drafts += [] if alone is None else [alone]
            continue
        a, b = first.image, second.image
        close = match.how == "context" and (
            _close_signatures(a, b)
            or (
                _close_signatures(a, b, perceptual=False)
                and judged.similar(a, b)
            )
        )
        if match.how == "content" or close:
            change = judged(first, second)
        elif match.how == "context":
            change = _context_draft(first, second)
        else:
            change = None
        drafts += [
            draft
            for draft in (change, _alt_draft(first, second))
            if draft is not None
        ]
    return drafts
