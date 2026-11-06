# Copyright 2026 Christophe Le Douarec
"""Load an HTML file or a generated HTML site into a document.

The pages are read in two passes: first the raw blocks of every page, then
their normalized blocks, whose links are resolved against all the pages
read, the redirection pages and the base of the site.
"""

from collections import Counter
from dataclasses import replace
from itertools import starmap
from typing import TYPE_CHECKING, Final

from doc_checker.errors import InputError
from doc_checker.extract.common import (
    assign_section_paths,
    normalized_block,
    strip_heading_numbers,
)
from doc_checker.extract.html.chrome import find_chrome
from doc_checker.extract.html.dom import blocks_from_root, strip_elements
from doc_checker.extract.html.links import (
    LinkResolver,
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
    content_root,
    detect_profile,
    get_profile,
)
from doc_checker.extract.html.site import discover_pages, parse_page
from doc_checker.model.document import Document, SourceRef, carry_anchors
from doc_checker.text.lang import AUTO, declared_language, resolve_language

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from doc_checker.config import HtmlConfig
    from doc_checker.extract.html.dom import RawBlock
    from doc_checker.extract.html.profiles import Profile
    from doc_checker.extract.html.site import HtmlPage
    from doc_checker.model.document import Block

_BODY_CHROME: Final = ("header", "footer", "aside")
_LANG_SAMPLE_BLOCKS: Final = 200
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


def _ordered_pages(
    pages: Sequence[HtmlPage],
    cfg: HtmlConfig,
) -> tuple[list[HtmlPage], dict[str, Profile]]:
    """Return the pages in reading order, and the profile of each page."""
    profiles = {page.rel_path: _profile_for(page, cfg) for page in pages}
    entry = entry_page(pages)
    nav = nav_links(entry, profiles[entry.rel_path].nav)
    return order_pages(pages, mode=cfg.page_order, nav=nav), profiles


def _raw_blocks(
    page: HtmlPage,
    profile: Profile,
    cfg: HtmlConfig,
) -> list[RawBlock]:
    """Return the raw blocks of the content of a page."""
    root = content_root(page.tree, profile, cfg.content_selector)
    selectors = (*GLOBAL_STRIP, *profile.strip, *cfg.strip_selectors)
    if root.tag == "body":
        selectors = (*selectors, *_BODY_CHROME)
    strip_elements(root, selectors)
    return blocks_from_root(root)


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
    root = path if path.is_dir() else path.parent
    exists = page_exists(frozenset(page.rel_path for page in pages), root)
    hrefs = (
        href for blocks in raw for block in blocks for _, href in block.links
    )
    return LinkResolver(site_base(hrefs, exists), redirects, exists)


def _to_block(
    raw: RawBlock,
    rel_path: str,
    counter: Counter[str],
    resolve: LinkResolver,
) -> Block:
    """Return the normalized block of a raw block: keys, links, anchors."""
    block = normalized_block(
        raw.kind,
        raw.text,
        SourceRef(rel_path, line=raw.line, anchor=raw.anchor),
        counter,
        level=raw.level,
        attrs={_ROLE: raw.role} if raw.role else None,
    )
    return replace(
        block,
        links=resolve(raw.links, rel_path),
        anchors=raw.ids,
    )


def load_html(path: Path, cfg: HtmlConfig, *, lang: str) -> Document:
    """Load an HTML file or a generated HTML site.

    Args:
        path: HTML file or site directory.
        cfg: HTML extraction settings.
        lang: Language requested by the user, or ``"auto"``.

    Returns:
        The document, its blocks in reading order.

    Raises:
        InputError: If no HTML page can be found.
    """
    pages, redirects = _read_pages(path, cfg)
    if not pages:
        msg = f"no HTML pages found in {path}"
        raise InputError(msg)
    ordered, profiles = _ordered_pages(pages, cfg)
    raw = [_raw_blocks(page, profiles[page.rel_path], cfg) for page in ordered]
    resolve = _resolver(path, pages, redirects, raw)
    counter: Counter[str] = Counter()
    per_page = [
        [_to_block(block, page.rel_path, counter, resolve) for block in blocks]
        for page, blocks in zip(ordered, raw, strict=True)
    ]
    chrome_keys = find_chrome(per_page)
    chrome = list(
        dict.fromkeys(
            block.text
            for blocks in per_page
            for block in blocks
            if block.key_strict in chrome_keys
        ),
    )
    blocks = carry_anchors(
        [block for blocks in per_page for block in blocks],
        lambda block: block.key_strict not in chrome_keys,
    )
    blocks = strip_heading_numbers(blocks, counter)
    assign_section_paths(blocks)
    declared = declared_language(ordered[0].tree.get("lang"))
    sample = " ".join(block.text for block in blocks[:_LANG_SAMPLE_BLOCKS])
    generator = Counter(
        profiles[page.rel_path].name for page in ordered
    ).most_common(1)[0][0]
    return Document(
        path=path,
        format="html",
        blocks=blocks,
        parts=[page.rel_path for page in ordered],
        lang=resolve_language(lang, declared, sample),
        declared_lang=declared,
        generator=generator,
        normalization=counter,
        chrome=chrome,
        chrome_keys=chrome_keys,
        warnings=[],
    )
