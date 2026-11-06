# Copyright 2026 Christophe Le Douarec
"""Load an HTML file or a generated HTML site into a document."""

from collections import Counter
from itertools import starmap
from typing import TYPE_CHECKING, Final

from doc_checker.errors import InputError
from doc_checker.extract.common import (
    assign_section_paths,
    normalized_block,
    strip_heading_numbers,
)
from doc_checker.extract.html.chrome import drop_keys, find_chrome
from doc_checker.extract.html.dom import blocks_from_root, strip_elements
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
from doc_checker.model.document import Document, SourceRef
from doc_checker.text.lang import AUTO, declared_language, resolve_language

if TYPE_CHECKING:
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


def _to_block(raw: RawBlock, rel_path: str, counter: Counter[str]) -> Block:
    """Return the normalized block of a raw block, with its keys."""
    return normalized_block(
        raw.kind,
        raw.text,
        SourceRef(rel_path, line=raw.line, anchor=raw.anchor),
        counter,
        level=raw.level,
        attrs={_ROLE: raw.role} if raw.role else None,
    )


def _page_blocks(
    page: HtmlPage,
    profile: Profile,
    cfg: HtmlConfig,
    counter: Counter[str],
) -> list[Block]:
    """Return the normalized blocks of the content of a page."""
    root = content_root(page.tree, profile, cfg.content_selector)
    selectors = (*GLOBAL_STRIP, *profile.strip, *cfg.strip_selectors)
    if root.tag == "body":
        selectors = (*selectors, *_BODY_CHROME)
    strip_elements(root, selectors)
    return [
        _to_block(raw, page.rel_path, counter)
        for raw in blocks_from_root(root)
    ]


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
    pages = [
        page
        for page in starmap(parse_page, discover_pages(path, cfg))
        if page.redirect_to is None
    ]
    if not pages:
        msg = f"no HTML pages found in {path}"
        raise InputError(msg)
    profiles = {page.rel_path: _profile_for(page, cfg) for page in pages}
    entry = entry_page(pages)
    nav = nav_links(entry, profiles[entry.rel_path].nav)
    ordered = order_pages(pages, mode=cfg.page_order, nav=nav)
    counter: Counter[str] = Counter()
    per_page = [
        _page_blocks(page, profiles[page.rel_path], cfg, counter)
        for page in ordered
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
    blocks = [
        block
        for blocks in per_page
        for block in drop_keys(blocks, chrome_keys)
    ]
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
