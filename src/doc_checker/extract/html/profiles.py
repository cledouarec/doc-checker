# Copyright 2026 Christophe Le Douarec
"""Documentation generator profiles: where the content is, what to remove."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from doc_checker.errors import ConfigError
from doc_checker.extract.html.css import matches, select

if TYPE_CHECKING:
    from lxml.html import HtmlElement


@dataclass(frozen=True, slots=True)
class Profile:
    """How to read the pages of one documentation generator.

    Attributes:
        name: Profile name.
        detect: CSS selector that identifies the generator.
        content: CSS selectors of the content root, tried in order.
        strip: CSS selectors of the elements to remove from the content.
        nav: CSS selectors of the navigation, tried in order, used to
            order the pages of a site.
        titles: CSS selectors of the admonition titles, besides
            `GLOBAL_TITLES`.
        code: CSS selectors of the elements whose text is inline code.
    """

    name: str
    detect: str
    content: tuple[str, ...]
    strip: tuple[str, ...] = ()
    nav: tuple[str, ...] = ()
    titles: tuple[str, ...] = ()
    code: tuple[str, ...] = ()


# Admonition titles of every generator: the icon cell of Asciidoctor, the
# title of an Obsidian callout, the tag of a blockquote admonition. The icon
# cell is found by its parents, not by any ancestor: under nested admonitions,
# libxml2 would find it once per ancestor, in a time that grows as a square.
GLOBAL_TITLES: Final[tuple[str, ...]] = (
    "div.admonitionblock > table > tr > td.icon",
    "div.admonitionblock > table > tbody > tr > td.icon",
    ".callout-title-container",
    "p.blockquote-tag-title",
)


GLOBAL_STRIP: Final[tuple[str, ...]] = (
    "script",
    "style",
    "noscript",
    "template",
    "button",
    "nav",
    "[hidden]",
    # pandoc hides from screen readers the caption that its alternative
    # text repeats, but shows it.
    "[aria-hidden=true]:not(figcaption)",
    "[role=navigation]",
    "[role=search]",
    ".sr-only",
    ".visually-hidden",
    "a.headerlink",
    "a.anchor",
    "a.header-anchor",
    "a.hash-link",
    ".linenos",
    ".linenodiv",
    "td.linenos",
    "a.footnote-backref",
    "a.fn-backref",
    ".toctree-wrapper",
)

_SPHINX_STRIP: Final = (".toctree-wrapper", "div.topic.contents")

PROFILES: Final[tuple[Profile, ...]] = (
    Profile(
        "sphinx-rtd",
        detect="div[itemprop=articleBody]",
        content=("div[itemprop=articleBody]",),
        strip=(".rst-footer-buttons", *_SPHINX_STRIP),
        nav=(".wy-menu-vertical", ".toctree-wrapper"),
    ),
    Profile(
        "sphinx-pydata",
        detect="article.bd-article",
        content=("article.bd-article",),
        strip=(".prev-next-area", *_SPHINX_STRIP),
        nav=(".bd-docs-nav", ".toctree-wrapper"),
    ),
    Profile(
        "mkdocs-material",
        detect="article.md-content__inner",
        content=("article.md-content__inner",),
        strip=(
            "a.md-content__button",
            ".md-source-file",
            ".md-feedback",
            "nav.md-tags",
            ".md-clipboard",
        ),
        nav=("nav.md-nav--primary",),
    ),
    Profile(
        "sphinx",
        detect=(
            "script#documentation_options, "
            "script[src*='documentation_options.js']"
        ),
        content=("[role=main]",),
        strip=(
            ".edit-this-page",
            ".related",
            ".sphinxsidebar",
            *_SPHINX_STRIP,
        ),
        nav=(".sidebar-tree", ".sphinxsidebarwrapper", ".toctree-wrapper"),
    ),
    Profile(
        "docusaurus",
        detect=".theme-doc-markdown",
        content=(".theme-doc-markdown",),
        strip=(
            ".theme-doc-toc-mobile",
            ".theme-doc-toc-desktop",
            ".theme-doc-footer",
            ".theme-doc-version-badge",
        ),
        titles=("[class*=admonitionHeading]",),
        nav=(".theme-doc-sidebar-menu", "nav.menu"),
    ),
    Profile(
        "vitepress",
        detect="meta[name=generator][content^=VitePress], .vp-doc",
        content=(".vp-doc",),
        strip=("span.lang",),
        titles=("p.custom-block-title",),
        nav=("#VPSidebarNav", ".VPSidebar"),
    ),
    Profile(
        "antora",
        detect="meta[name=generator][content^=Antora]",
        content=("article.doc",),
        strip=(".toolbar", "aside.toc"),
        nav=("nav.nav-menu",),
    ),
    Profile(
        "docsy",
        detect=".td-content",
        content=(".td-content",),
        strip=(
            ".td-page-meta__lastmod",
            ".section-index",
            ".feedback--title",
            ".feedback--question",
            ".feedback--answer",
            ".feedback--response",
        ),
        titles=(".alert-heading",),
        nav=("#td-sidebar-menu", ".td-sidebar-nav"),
    ),
    Profile(
        "texinfo",
        detect=(
            "meta[name=Generator][content^=texi2any], "
            "meta[name=Generator][content^=makeinfo], "
            "meta[name=generator][content^=texi2any], "
            "meta[name=generator][content^=makeinfo], "
            "div.top-level-extent"
        ),
        content=(),
        strip=(
            "div.nav-panel",
            "table.nav-panel",
            "div.header",
            "ul.mini-toc",
            "div.menu",
            "table.menu",
            "a.copiable-link",
            "div.printindex",
            "table.index-cp",
            "table.index-fn",
            "div.contents",
            "div.region-contents",
            "div.element-contents",
            "div.shortcontents",
            "div.region-shortcontents",
            "div.element-shortcontents",
            "h2.contents-heading",
            "h2.shortcontents-heading",
        ),
        titles=("blockquote.quotation > p:first-child > b:first-child",),
        nav=(
            "div.contents",
            "div.region-contents",
            "div.element-contents",
            "ul.mini-toc",
            "div.menu",
            "table.menu",
        ),
    ),
    Profile(
        "tex4ht",
        detect=(
            "meta[name=generator][content^=TeX4ht], "
            "meta[name=originator][content^=TeX4ht]"
        ),
        content=(),
        strip=("div.crosslinks", "div.tableofcontents"),
        code=(
            "span[class^=cmtt-]",
            "span[class^=ectt-]",
            "span[class^=ec-lmtt-]",
            "span[class^=ec-lmvtt-]",
            "span[class^=ec-lmtk-]",
            "span[class^=ec-lmtto-]",
        ),
        nav=("div.tableofcontents",),
    ),
    Profile(
        "generic",
        detect="html",
        content=("main", "[role=main]", "article", "#content", "div.content"),
    ),
)
GENERIC: Final = PROFILES[-1]
PROFILE_NAMES: Final = tuple(sorted(profile.name for profile in PROFILES))


def get_profile(name: str) -> Profile:
    """Return a profile by name.

    Args:
        name: Profile name.

    Returns:
        The profile.

    Raises:
        ConfigError: If no profile has this name.
    """
    for profile in PROFILES:
        if profile.name == name:
            return profile
    expected = ", ".join(PROFILE_NAMES)
    msg = f"unknown HTML profile {name!r} (expected: {expected})"
    raise ConfigError(msg)


def detect_profile(tree: HtmlElement) -> Profile:
    """Return the first profile whose detection selector matches.

    Args:
        tree: Parsed page.

    Returns:
        The detected profile, or the generic one.
    """
    for profile in PROFILES:
        if matches(tree, profile.detect):
            return profile
    return GENERIC


def content_root(
    tree: HtmlElement,
    profile: Profile,
    override: str | None,
) -> HtmlElement:
    """Return the element that holds the content of a page.

    Args:
        tree: Parsed page.
        profile: Profile of the page.
        override: User CSS selector, tried first.

    Returns:
        The first element matched by the override or the profile content
        selectors, or the body of the page.
    """
    selectors = ((override,) if override else ()) + profile.content
    for selector in selectors:
        found = select(tree, selector)
        if found:
            return found[0]
    body = tree.find("body")
    return body if body is not None else tree
