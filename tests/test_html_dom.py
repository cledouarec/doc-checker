# Copyright 2026 Christophe Le Douarec
"""Tests for generator profiles and DOM to block conversion."""

from typing import TYPE_CHECKING

import lxml.html
import pytest

from doc_checker.errors import ConfigError
from doc_checker.extract.html.dom import blocks_from_root, strip_elements
from doc_checker.extract.html.profiles import (
    GLOBAL_STRIP,
    content_root,
    detect_profile,
    get_profile,
)

if TYPE_CHECKING:
    from lxml.html import HtmlElement


def _root(body: str, head: str = "") -> HtmlElement:
    return lxml.html.document_fromstring(
        f"<html><head>{head}</head><body>{body}</body></html>",
    )


def _blocks(body: str) -> list[tuple[str, str]]:
    root = _root(body)
    strip_elements(root, GLOBAL_STRIP)
    return [(b.kind.value, b.text) for b in blocks_from_root(root.body)]


def test_detect_profiles() -> None:
    """Profiles are detected from characteristic elements."""
    rtd = _root('<div itemprop="articleBody"><p>x</p></div>')
    material = _root('<article class="md-content__inner"><p>x</p></article>')
    sphinx = _root(
        '<div role="main"><p>x</p></div>',
        head='<script id="documentation_options"></script>',
    )
    assert detect_profile(rtd).name == "sphinx-rtd"
    assert detect_profile(material).name == "mkdocs-material"
    assert detect_profile(sphinx).name == "sphinx"
    assert detect_profile(_root("<p>x</p>")).name == "generic"


def test_content_root_override_and_fallback() -> None:
    """An override selector wins, then the profile, then the body."""
    tree = _root('<div id="doc"><p>a</p></div><main><p>b</p></main>')
    generic = get_profile("generic")
    assert content_root(tree, generic, "#doc").get("id") == "doc"
    assert content_root(tree, generic, None).tag == "main"
    assert content_root(_root("<p>c</p>"), generic, None).tag == "body"


def test_get_profile_unknown() -> None:
    """An unknown profile name is a configuration error."""
    with pytest.raises(ConfigError, match="unknown HTML profile"):
        get_profile("nope")


def test_blocks_basic_kinds() -> None:
    """Headings, paragraphs, lists, code, tables and captions are found."""
    body = (
        '<h2 id="install">Install<a class="headerlink" href="#install">'
        "\xb6</a></h2>"
        "<p>Run the   installer\nnow.</p>"
        "<ul><li><p>First item</p><ul><li>Nested item</li></ul></li></ul>"
        '<div class="highlight"><pre><span class="linenos">1</span>'
        'pip install x\n<span class="linenos">2</span>x --version</pre></div>'
        "<table><tr><th><p>Option</p></th><th>Default</th></tr>"
        "<tr><td>timeout</td><td>30</td></tr></table>"
        "<figure><figcaption>Figure 1. Overview</figcaption></figure>"
    )
    assert _blocks(body) == [
        ("heading", "Install"),
        ("paragraph", "Run the installer now."),
        ("list_item", "First item"),
        ("list_item", "Nested item"),
        ("code", "pip install x\nx --version"),
        ("table", "Option | Default\ntimeout | 30"),
        ("caption", "Figure 1. Overview"),
    ]


def test_heading_level_and_anchor() -> None:
    """Headings carry their level and blocks the nearest anchor."""
    root = _root(
        '<section id="setup"><h3>Setup</h3><p>Text body.</p></section>',
    )
    blocks = blocks_from_root(root.body)
    assert (blocks[0].level, blocks[0].anchor) == (3, "setup")
    assert blocks[1].anchor == "setup"


def test_anchors_are_printable() -> None:
    """Bidi overrides of an identifier never reach the anchors."""
    root = _root('<h2 id="a&#x202e;b">Title</h2>')
    (block,) = blocks_from_root(root.body)
    assert block.anchor == "a b"


def test_an_identifier_without_printable_characters_is_no_anchor() -> None:
    """An identifier of controls only leaves the current anchor in place."""
    root = _root(
        '<section id="setup"><h3 id="&#x202e;&#x7;">Setup</h3></section>',
    )
    (block,) = blocks_from_root(root.body)
    assert block.anchor == "setup"


def test_admonition_titles() -> None:
    """Default admonition titles are skipped, custom ones are kept."""
    assert _blocks(
        '<div class="admonition note"><p class="admonition-title">Note</p>'
        "<p>Body text here.</p></div>",
    ) == [("paragraph", "Body text here.")]
    root = _root(
        '<div class="admonition">'
        '<p class="admonition-title">Before you start</p></div>',
    )
    (block,) = blocks_from_root(root.body)
    assert (block.text, block.role) == ("Before you start", "admonition-title")


def test_loose_text_in_containers() -> None:
    """Inline text directly in containers becomes paragraphs."""
    assert _blocks("<div>Some <b>bold</b> text<p>Para</p>tail text</div>") == [
        ("paragraph", "Some bold text"),
        ("paragraph", "Para"),
        ("paragraph", "tail text"),
    ]


def test_non_breaking_space_is_kept() -> None:
    """Non-breaking spaces survive whitespace collapsing."""
    assert _blocks("<p>a\xa0: b</p>") == [("paragraph", "a\xa0: b")]


def test_display_math_and_highlighttable() -> None:
    """Display math is a formula and highlight tables are code."""
    assert _blocks('<div class="math">\\[x^2\\]</div>') == [
        ("formula", "\\[x^2\\]"),
    ]
    table = (
        '<table class="highlighttable"><tr><td class="linenos">'
        '<pre>1\n2</pre></td><td class="code"><pre>a\nb</pre></td>'
        "</tr></table>"
    )
    assert _blocks(table) == [("code", "a\nb")]


def test_detect_modern_sphinx() -> None:
    """Sphinx 7.2+ pages are detected despite the cache busting query."""
    head = (
        '<script src="_static/documentation_options.js?v=5929fcd5"></script>'
    )
    tree = _root('<div class="body" role="main"><p>x</p></div>', head=head)
    assert detect_profile(tree).name == "sphinx"


def test_block_content_inside_list_items() -> None:
    """Code, tables and admonitions inside list items are separate blocks."""
    body = (
        "<ol><li><p>Run the command:</p>"
        '<div class="highlight"><pre>example sync --all</pre></div></li>'
        "<li>Check the table:<table><tr><th>Option</th><th>Default</th></tr>"
        "</table></li>"
        '<li>Read the note.<div class="admonition note">'
        '<p class="admonition-title">Note</p><p>Body text.</p></div></li>'
        "<li><p>First paragraph.</p><p>Second paragraph.</p></li></ol>"
    )
    assert _blocks(body) == [
        ("list_item", "Run the command:"),
        ("code", "example sync --all"),
        ("list_item", "Check the table:"),
        ("table", "Option | Default"),
        ("list_item", "Read the note."),
        ("paragraph", "Body text."),
        ("list_item", "First paragraph. Second paragraph."),
    ]


def test_links_of_blocks() -> None:
    """Each block keeps its links, code blocks none."""
    root = _root(
        '<p>Read <a href="a.html">the guide</a> and '
        '<a href="#x"><code>x</code></a>.</p>'
        '<ul><li>See <a href="b.html">b</a><ul><li><a href="c.html">c</a>'
        "</li></ul></li></ul>"
        '<pre>run <a href="d.html">d</a></pre>'
        '<div>Loose <a href="e.html">e</a> text<a href="f.html"></a></div>',
    )
    assert [b.links for b in blocks_from_root(root.body)] == [
        (("the guide", "a.html"), ("x", "#x")),
        (("b", "b.html"),),
        (("c", "c.html"),),
        (),
        (("e", "e.html"),),
    ]


def test_identifiers_point_to_blocks() -> None:
    """Identifiers go to their block, or to the next one for a container."""
    root = _root(
        '<section id="s"><h2>Title</h2><p>Body <span id="mid">text</span>.'
        '</p></section><div id="empty"></div><p><a name="old"></a>Next.</p>',
    )
    assert [(b.text, b.ids) for b in blocks_from_root(root.body)] == [
        ("Title", ("s",)),
        ("Body text.", ("mid",)),
        ("Next.", ("empty", "old")),
    ]


def test_links_and_identifiers_of_a_table_and_its_caption() -> None:
    """A caption keeps its own links, the table those of its cells."""
    root = _root(
        '<table id="t"><caption id="c">See <a href="x.html">x</a></caption>'
        '<tr><td id="cell"><a href="y.html">y</a></td></tr></table>',
    )
    assert [
        (b.kind.value, b.links, b.ids) for b in blocks_from_root(root.body)
    ] == [
        ("caption", (("x", "x.html"),), ("t", "c")),
        ("table", (("y", "y.html"),), ("cell",)),
    ]


def test_identifiers_of_elements_without_text_go_to_the_next_block() -> None:
    """A default admonition title or an empty heading passes its ids on."""
    root = _root(
        '<div class="admonition note"><p class="admonition-title" id="n">'
        'Note</p><p>Body.</p></div><h2 id="empty"></h2><p>After.</p>',
    )
    assert [(b.text, b.ids) for b in blocks_from_root(root.body)] == [
        ("Body.", ("n",)),
        ("After.", ("empty",)),
    ]


def test_link_texts_of_blocks_are_printable() -> None:
    """A link text loses its bidi overrides, a link of none is left out."""
    root = _root(
        '<p><a href="a.html">a&#x202e;b</a> <a href="b.html">&#x202e;</a></p>',
    )
    (block,) = blocks_from_root(root.body)
    assert block.links == (("a b", "a.html"),)
