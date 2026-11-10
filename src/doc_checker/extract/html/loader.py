# Copyright 2026 Christophe Le Douarec
"""Load an HTML file or a generated HTML site into a document.

The pages are read in two passes: first the raw blocks of every page, then
their normalized blocks, whose links are resolved against all the pages
read, the redirection pages and the base of the site, and whose images are
read under the root of the site.
"""

import posixpath
from collections import Counter
from dataclasses import replace
from itertools import chain, islice, starmap
from typing import TYPE_CHECKING, Final, NamedTuple
from urllib.parse import urlsplit

from doc_checker.errors import InputError
from doc_checker.extract.common import (
    IMAGE_CHROME,
    TINY_IMAGE,
    assign_section_paths,
    image_warnings,
    is_tiny_image,
    normalized_block,
    repeated_images,
    strip_caption_numbers,
    strip_heading_numbers,
)
from doc_checker.extract.html.chrome import CHROME_MIN_SHARE, find_chrome
from doc_checker.extract.html.dom import (
    MAX_DEPTH,
    blocks_from_root,
    mark_code,
    mark_titles,
    prune_deep,
    strip_elements,
)
from doc_checker.extract.html.facts import (
    ContentSpots,
    content_spots,
    page_hrefs,
    page_ids,
    page_title,
)
from doc_checker.extract.html.images import ImageResolver
from doc_checker.extract.html.links import (
    LinkResolver,
    clean_href,
    is_external,
    page_exists,
    site_base,
)
from doc_checker.extract.html.ordering import (
    entry_page,
    nav_links,
    order_pages,
)
from doc_checker.extract.html.profiles import (
    GLOBAL_STRIP,
    GLOBAL_TITLES,
    content_root,
    detect_profile,
    get_profile,
)
from doc_checker.extract.html.site import discover_pages, parse_page
from doc_checker.model.document import (
    Block,
    BlockKind,
    Document,
    SourceRef,
    carry_anchors,
)
from doc_checker.model.facts import HtmlPageFacts
from doc_checker.text.lang import (
    AUTO,
    LANG_SAMPLE_BLOCKS,
    declared_language,
    resolve_language,
)
from doc_checker.text.normalize import normalize_display, printable

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    from lxml.html import HtmlElement

    from doc_checker.config import HtmlConfig
    from doc_checker.extract.html.dom import RawBlock
    from doc_checker.extract.html.images import RawImage
    from doc_checker.extract.html.profiles import Profile
    from doc_checker.extract.html.site import HtmlPage

_BODY_CHROME: Final = ("header", "footer", "aside")
_ROLE: Final = "role"


def _profile_for(page: HtmlPage, cfg: HtmlConfig) -> Profile:
    """Return the forced profile, or the one detected for the page."""
    if cfg.profile != AUTO:
        return get_profile(cfg.profile)
    return detect_profile(page.tree)


def _read_pages(
    path: Path,
    cfg: HtmlConfig,
) -> tuple[list[HtmlPage], dict[str, str]]:
    """Return the content pages, and the URL of each redirection page."""
    pages: list[HtmlPage] = []
    redirects: dict[str, str] = {}
    for page in starmap(parse_page, discover_pages(path, cfg)):
        if page.redirect_href is None:
            pages.append(page)
        else:
            redirects[page.rel_path] = page.redirect_href
    return pages, redirects


def _pruned(pages: Sequence[HtmlPage]) -> list[str]:
    """Remove the content nested too deep in pages, as `prune_deep` does.

    Args:
        pages: Pages, changed in place.

    Returns:
        A warning for each page that lost some content.
    """
    return [
        f"{page.rel_path}: content nested deeper than {MAX_DEPTH} levels "
        "was skipped"
        for page in pages
        if prune_deep(page.tree)
    ]


def _ordered_pages(
    pages: Sequence[HtmlPage],
    cfg: HtmlConfig,
) -> tuple[list[HtmlPage], dict[str, Profile]]:
    """Return the pages in reading order, and the profile of each page."""
    profiles = {page.rel_path: _profile_for(page, cfg) for page in pages}
    entry = entry_page(pages)
    nav = nav_links(entry, profiles[entry.rel_path].nav)
    return order_pages(pages, mode=cfg.page_order, nav=nav), profiles


def _scope(
    page: HtmlPage,
    profile: Profile,
    cfg: HtmlConfig,
) -> tuple[HtmlElement, tuple[str, ...]]:
    """Return the root of the content of a page, and what is removed in it."""
    root = content_root(page.tree, profile, cfg.content_selector)
    mark_titles(root, (*GLOBAL_TITLES, *profile.titles))
    mark_code(root, profile.code)
    selectors = (*GLOBAL_STRIP, *profile.strip, *cfg.strip_selectors)
    if root.tag == "body":
        selectors = (*selectors, *_BODY_CHROME)
    return root, selectors


def _spots(
    page: HtmlPage,
    profile: Profile,
    cfg: HtmlConfig,
    *,
    images: bool,
) -> ContentSpots:
    """Return the places of the content that the audit points at.

    It is read before the chrome of the page is removed.
    """
    root, selectors = _scope(page, profile, cfg)
    return content_spots(root, selectors, images=images)


def _raw_blocks(
    page: HtmlPage,
    profile: Profile,
    cfg: HtmlConfig,
    *,
    images: bool,
    counter: Counter[str],
) -> list[RawBlock]:
    """Return the raw blocks of the content of a page."""
    root, selectors = _scope(page, profile, cfg)
    strip_elements(root, selectors)
    return blocks_from_root(root, images=images, counter=counter)


def _site_root(path: Path) -> Path:
    """Return the directory of a site, or of a single HTML file."""
    return path if path.is_dir() else path.parent


def _resolver(
    path: Path,
    pages: Sequence[HtmlPage],
    redirects: dict[str, str],
    raw: Sequence[Sequence[RawBlock]],
) -> LinkResolver:
    """Return what resolves the links of the blocks of the site.

    Args:
        path: HTML file or site directory.
        pages: Content pages read.
        redirects: URL of each redirection page, as written in it.
        raw: Raw blocks of every page, whose links tell the base of the
            site.

    Returns:
        The resolver of `resolve_links`, its existence check shared with
        the search of the base.
    """
    root = _site_root(path)
    exists = page_exists(frozenset(page.rel_path for page in pages), root)
    hrefs = (
        href for blocks in raw for block in blocks for _, href in block.links
    )
    return LinkResolver(site_base(hrefs, exists), redirects, exists)


def _image_block(
    raw: RawBlock,
    image: RawImage,
    rel_path: str,
    resolve: ImageResolver,
) -> Block:
    """Return the block of an image, without text or keys."""
    return Block(
        kind=BlockKind.IMAGE,
        text="",
        raw_text="",
        source=SourceRef(rel_path, line=raw.line, anchor=raw.anchor),
        image=resolve(image, rel_path),
        anchors=raw.ids,
    )


def _to_block(
    raw: RawBlock,
    rel_path: str,
    counter: Counter[str],
    resolve: LinkResolver,
    images: ImageResolver,
) -> Block:
    """Return the normalized block of a raw block: keys, links, anchors.

    Its inline code is normalized as its text is, but not counted, as the
    text of the block counts it already; a code that has nothing left but
    blanks is left out. The block of an image has its image read instead.
    """
    if raw.image is not None:
        return _image_block(raw, raw.image, rel_path, images)
    block = normalized_block(
        raw.kind,
        raw.text,
        SourceRef(rel_path, line=raw.line, anchor=raw.anchor),
        counter,
        level=raw.level,
        attrs={_ROLE: raw.role} if raw.role else None,
    )
    code = (normalize_display(text, Counter()) for text in raw.code)
    return replace(
        block,
        links=resolve(raw.links, rel_path),
        anchors=raw.ids,
        code_spans=tuple(text for text in code if text.strip()),
    )


def _is_image(block: Block) -> bool:
    """Return whether a block is an image, not a block of text."""
    return block.kind is BlockKind.IMAGE


def _page_blocks(
    path: Path,
    ordered: Sequence[HtmlPage],
    raw: Sequence[Sequence[RawBlock]],
    resolve: LinkResolver,
    counter: Counter[str],
) -> list[list[Block]]:
    """Return the normalized blocks of each page, their images read.

    Args:
        path: HTML file or site directory.
        ordered: Pages in reading order.
        raw: Raw blocks of each page.
        resolve: Resolver of the links, whose base the images share.
        counter: Normalization counter to update.

    Returns:
        The blocks of each page, in page order.
    """
    read = ImageResolver(_site_root(path), resolve.base)
    return [
        [
            _to_block(block, page.rel_path, counter, resolve, read)
            for block in blocks
        ]
        for page, blocks in zip(ordered, raw, strict=True)
    ]


class _Chrome(NamedTuple):
    """What a site repeats on its pages, as page chrome.

    Attributes:
        texts: Texts of the blocks of the chrome, each once, in order.
        keys: Strict keys of the blocks of the chrome.
        images: Digests of the images of the chrome.
    """

    texts: list[str]
    keys: frozenset[str]
    images: frozenset[str]


def _without_chrome(
    per_page: Sequence[Sequence[Block]],
    counter: Counter[str],
) -> tuple[list[Block], _Chrome]:
    """Remove the blocks of text and the images repeated as page chrome.

    The chrome of text is found among the blocks of text only, and the
    chrome of images among the images; the images removed are counted
    (``image_chrome``). The tiny images, as `is_tiny_image` tells, are
    removed too, and counted apart (``tiny_image``).

    Args:
        per_page: Blocks of each page, in page order.
        counter: Normalization counter to update.

    Returns:
        The blocks kept, in order, with the anchors of the others, and the
        chrome.
    """
    texts = [[b for b in blocks if not _is_image(b)] for blocks in per_page]
    chrome_keys = find_chrome(texts)
    chrome = list(
        dict.fromkeys(
            block.text
            for blocks in texts
            for block in blocks
            if block.key_strict in chrome_keys
        ),
    )
    sized = [
        [b for b in blocks if not is_tiny_image(b)] for blocks in per_page
    ]
    chrome_images = repeated_images(sized, share=CHROME_MIN_SHARE)

    def is_chrome(block: Block) -> bool:
        if _is_image(block) and block.image is not None:
            return block.image.digest in chrome_images
        return block.key_strict in chrome_keys

    every = [block for blocks in per_page for block in blocks]
    tiny = sum(is_tiny_image(block) for block in every)
    removed = sum(
        _is_image(block) and not is_tiny_image(block) and is_chrome(block)
        for block in every
    )
    if removed:
        counter[IMAGE_CHROME] += removed
    if tiny:
        counter[TINY_IMAGE] += tiny
    blocks = carry_anchors(
        every,
        lambda block: not (is_tiny_image(block) or is_chrome(block)),
    )
    return blocks, _Chrome(chrome, chrome_keys, chrome_images)


class _PageRead(NamedTuple):
    """What the audit reads of a page before its content is pruned.

    Attributes:
        ids: Identifiers of the whole page.
        hrefs: Targets of the links of the whole page.
        lang: Printable ``lang`` of the page, if it has one.
        title: Printable title of the head, if any.
    """

    ids: frozenset[str]
    hrefs: tuple[str, ...]
    lang: str | None
    title: str | None


def _read_page(page: HtmlPage) -> _PageRead:
    """Return what the audit reads of the whole of a page."""
    lang = page.tree.get("lang")
    return _PageRead(
        page_ids(page.tree),
        page_hrefs(page.tree),
        None if lang is None else printable(lang),
        page_title(page.tree),
    )


def _page_resolver(
    resolve: LinkResolver,
    read: Sequence[_PageRead],
) -> LinkResolver:
    """Return what resolves the links of whole pages, for the audit.

    A site served under a base may write it in its navigation only: the
    base of the links of the whole pages is found from those links, the
    navigation included, as `site_base` finds it. The resolver of the links
    of the blocks keeps the base found from the content.

    Args:
        resolve: The resolver of the links of the blocks.
        read: What the audit read of the whole of each page.

    Returns:
        The resolver, with the redirections and the existence check of
        ``resolve``.
    """
    hrefs = chain.from_iterable(seen.hrefs for seen in read)
    return replace(resolve, base=site_base(hrefs, resolve.exists))


def _links_to(
    resolve: LinkResolver,
    hrefs: Iterable[str],
    page: str,
    known: dict[tuple[str, str], str | None],
) -> frozenset[str]:
    """Return the other pages of the site that the links of a page lead to.

    A target with a path leads to the same page from every page of a
    folder, so it is resolved once per folder and kept in ``known``; a
    target without a path leads to the page that holds it.

    Args:
        resolve: What resolves the links of whole pages.
        hrefs: The targets of the links of the page, without fragment.
        page: Site relative path of the page.
        known: The page each target leads to from each folder, if any, as
            `_target` tells, filled as the targets are resolved.

    Returns:
        The pages of the site that the links lead to, but the page itself.
    """
    folder = posixpath.dirname(page)
    targets: set[str] = set()
    for href in hrefs:
        key = (folder, href)
        if key not in known:
            known[key] = _target(resolve, href, page)
        target = known[key]
        if target is not None and target != page:
            targets.add(target)
    return frozenset(targets)


def _target(resolve: LinkResolver, href: str, page: str) -> str | None:
    """Return the page of the site a target leads to from a folder, if any.

    Args:
        resolve: What resolves the links of whole pages.
        href: The target, without fragment.
        page: Site relative path of a page of the folder.

    Returns:
        The page of the site; ``None`` for a target that leads out of the
        site, or without a path, to the page that holds it.
    """
    cleaned = clean_href(href)
    if not is_external(cleaned) and not urlsplit(cleaned).path:
        return None
    link = resolve.link("", href, page)
    return link.page if link is not None and link.url is None else None


def _page_facts(
    ordered: Sequence[HtmlPage],
    read: Sequence[_PageRead],
    spots: Sequence[ContentSpots],
    resolve: LinkResolver,
) -> tuple[HtmlPageFacts, ...]:
    """Return the facts of each page, the targets of its links resolved.

    The targets are resolved by `_page_resolver`, from ``resolve``, as
    `_links_to` does.
    """
    resolve = _page_resolver(resolve, read)
    known: dict[tuple[str, str], str | None] = {}
    facts = []
    for page, seen, found in zip(ordered, read, spots, strict=True):
        hrefs = dict.fromkeys(href.partition("#")[0] for href in seen.hrefs)
        facts.append(
            HtmlPageFacts(
                file=page.rel_path,
                lang=seen.lang,
                title=seen.title,
                ids=seen.ids,
                links_to=_links_to(resolve, hrefs, page.rel_path, known),
                empty_links=found.empty_links,
                unresolved=found.unresolved,
                bare_tables=found.bare_tables,
                media=found.media,
                blockless=found.blockless,
            ),
        )
    return tuple(facts)


def _sample(blocks: Sequence[Block]) -> str:
    """Return the text of the first blocks, in which the language is read."""
    return " ".join(
        islice(
            (block.text for block in blocks if not _is_image(block)),
            LANG_SAMPLE_BLOCKS,
        ),
    )


def _generator(
    ordered: Sequence[HtmlPage],
    profiles: dict[str, Profile],
) -> str:
    """Return the name of the profile of the most pages, the first on a tie.

    Args:
        ordered: Pages in reading order.
        profiles: Profile of each page.

    Returns:
        The name of the profile.
    """
    names = Counter(profiles[page.rel_path].name for page in ordered)
    return names.most_common(1)[0][0]


def load_html(
    path: Path,
    cfg: HtmlConfig,
    *,
    lang: str,
    images: bool = True,
    audit: bool = False,
) -> Document:
    """Load an HTML file or a generated HTML site.

    Args:
        path: HTML file or site directory.
        cfg: HTML extraction settings.
        lang: Language requested by the user, or ``"auto"``.
        images: Whether the images give blocks and are read.
        audit: Whether to read the facts of each page that only the audit
            uses; without it, `Document.html_pages` is empty.

    Returns:
        The document, its blocks in reading order. A page whose content is
        nested more than `MAX_DEPTH` levels deep loses that content, with a
        warning; the images that cannot be read are a warning.

    Raises:
        InputError: If no HTML page can be found.
    """
    pages, redirects = _read_pages(path, cfg)
    if not pages:
        msg = f"no HTML pages found in {path}"
        raise InputError(msg)
    ordered, profiles = _ordered_pages(pages, cfg)
    read = [_read_page(page) for page in ordered] if audit else []
    deep = _pruned(ordered)
    spots = (
        [
            _spots(page, profiles[page.rel_path], cfg, images=images)
            for page in ordered
        ]
        if audit
        else []
    )
    counter: Counter[str] = Counter()
    raw = [
        _raw_blocks(
            page,
            profiles[page.rel_path],
            cfg,
            images=images,
            counter=counter,
        )
        for page in ordered
    ]
    resolve = _resolver(path, pages, redirects, raw)
    per_page = _page_blocks(path, ordered, raw, resolve, counter)
    blocks, chrome = _without_chrome(per_page, counter)
    blocks = strip_heading_numbers(blocks, counter)
    blocks = strip_caption_numbers(blocks, counter)
    assign_section_paths(blocks)
    declared = declared_language(ordered[0].tree.get("lang"))
    return Document(
        path=path,
        format="html",
        blocks=blocks,
        parts=[page.rel_path for page in ordered],
        lang=resolve_language(lang, declared, _sample(blocks)),
        declared_lang=declared,
        generator=_generator(ordered, profiles),
        normalization=counter,
        chrome=chrome.texts,
        chrome_keys=chrome.keys,
        warnings=[*deep, *image_warnings(blocks)],
        chrome_images=chrome.images,
        html_pages=_page_facts(ordered, read, spots, resolve) if audit else (),
    )
