# Copyright 2026 Christophe Le Douarec
"""Documentation generator profiles: where the content is, what to remove."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from doc_checker.errors import ConfigError

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
    """

    name: str
    detect: str
    content: tuple[str, ...]
    strip: tuple[str, ...] = ()
    nav: tuple[str, ...] = ()


GLOBAL_STRIP: Final[tuple[str, ...]] = (
    "script",
    "style",
    "noscript",
    "template",
    "button",
    "nav",
    "[hidden]",
    "[aria-hidden=true]",
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
        "generic",
        detect="html",
        content=("main", "[role=main]", "article", "#content", ".content"),
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
        if tree.cssselect(profile.detect):
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
        found = tree.cssselect(selector)
        if found:
            return found[0]
    body = tree.find("body")
    return body if body is not None else tree
