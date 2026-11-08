# Copyright 2026 Christophe Le Douarec
"""Tests for the facts of HTML pages that only the audit reads."""

import lxml.html

from doc_checker.extract.html.facts import (
    LAYOUT_TABLE_CLASSES,
    content_spots,
    page_hrefs,
    page_ids,
    page_title,
)
from doc_checker.extract.html.images import raw_image
from doc_checker.extract.html.profiles import GLOBAL_STRIP


def _page(body: str) -> lxml.html.HtmlElement:
    return lxml.html.document_fromstring(
        "<html lang='en'><head><title> Guide </title></head>"
        f"<body>{body}</body></html>",
    )


def test_page_ids_cover_the_whole_page() -> None:
    """Identifiers of the navigation count, and so do anchor names."""
    tree = _page(
        '<nav id="menu"></nav><main><h2 id="install">I</h2>'
        '<a name="old"></a></main>',
    )
    assert page_ids(tree) == frozenset({"menu", "install", "old"})


def test_page_title_is_read_from_the_head() -> None:
    """The head title is kept with its blanks collapsed; an SVG title not."""
    assert page_title(_page("<svg><title>Chart</title></svg>")) == "Guide"
    blank = lxml.html.document_fromstring(
        "<html><head><title> </title></head><body><p>x</p></body></html>",
    )
    assert page_title(blank) is None


def test_page_hrefs_are_every_link_of_the_page() -> None:
    """Links of the navigation and of the content, in order."""
    tree = _page(
        '<nav><a href="a.html">A</a></nav>'
        '<p><a href="b.html#x">B</a><a>no target</a></p>',
    )
    assert page_hrefs(tree) == ("a.html", "b.html#x")


def test_links_without_a_name_are_found() -> None:
    """Text, hidden text, a label, a title or a picture names a link."""
    root = _page(
        '<p><a href="#a"></a>'
        '<a href="b.html"><span class="sr-only">Home</span></a>'
        '<a href="c.html" aria-label="Back"></a>'
        '<a href="d.html" title="Next"></a>'
        '<a href="e.html"><img src="x.png" alt="Logo"></a>'
        '<a href="f.html"><i class="icon" aria-hidden="true"></i></a></p>'
        '<nav><a href="g.html"></a></nav>'
        '<p aria-hidden="true"><a href="h.html"></a></p>',
    ).body
    spots = content_spots(root, GLOBAL_STRIP)
    assert [spot.text for spot in spots.empty_links] == ["#a", "f.html"]


def test_unresolved_references_are_found() -> None:
    """Unresolved standard references and docutils problems."""
    root = _page(
        '<p>See <span class="xref std std-ref">missing-label</span>.</p>'
        '<p><a class="reference internal" href="#i">'
        '<span class="std std-ref">Install</span></a></p>'
        '<p><code class="xref py py-class docutils literal">int</code></p>'
        '<p><a href="#id1"><span class="problematic">`bad`_</span></a></p>',
    ).body
    spots = content_spots(root, GLOBAL_STRIP)
    assert [spot.text for spot in spots.unresolved] == [
        "missing-label",
        "`bad`_",
    ]


def test_tables_without_header_cells_are_found() -> None:
    """Data tables need a th; code and layout tables do not."""
    root = _page(
        "<table><tr><th>Option</th><th>Default</th></tr>"
        "<tr><td>a</td><td>1</td></tr></table>"
        "<table><tr><td>timeout</td><td>30</td></tr>"
        "<tr><td>retries</td><td>3</td></tr></table>"
        '<table class="highlighttable"><tr><td>1</td><td>x</td></tr></table>'
        '<table role="presentation"><tr><td>a</td><td>b</td></tr></table>'
        '<table class="docutils footnote"><tr><td>1</td><td>N</td></tr>'
        "</table>",
    ).body
    spots = content_spots(root, GLOBAL_STRIP)
    assert [spot.text for spot in spots.bare_tables] == ["timeout | 30"]


def test_inline_svg_takes_its_label_or_title_as_alt() -> None:
    """The aria-label comes first, then the first title of the drawing."""
    tree = _page(
        '<svg width="300" height="200" aria-label="Flow"><title>T</title>'
        '<rect width="9" height="9"/></svg>'
        '<svg width="300" height="200"><title> Pipeline </title>'
        '<rect width="9" height="9"/></svg>'
        '<svg width="300" height="200"><rect width="9" height="9"/></svg>',
    )
    alts = [getattr(raw_image(svg), "alt", "?") for svg in tree.iter("svg")]
    assert alts == ["Flow", "Pipeline", None]


def test_presentation_image_is_decorative() -> None:
    """An img with the presentation role has an empty alternative text."""
    tree = _page('<img src="a.png" role="presentation"><img src="b.png">')
    alts = [getattr(raw_image(img), "alt", "?") for img in tree.iter("img")]
    assert alts == ["", None]


_SVG = '<svg width="300" height="200">{}</svg>'


def _alts(svgs: str) -> list[str | None]:
    """Return the alt of each inline SVG of a page, never raising."""
    tree = _page(svgs)
    return [getattr(raw_image(svg), "alt", "?") for svg in tree.iter("svg")]


def test_svg_alt_survives_prefixed_children() -> None:
    """Inkscape and Illustrator children, comments and instructions."""
    svgs = (
        _SVG.format("<sodipodi:namedview/><title>T</title>")
        + _SVG.format("<svg:title>P</svg:title>")
        + _SVG.format("<i:pgf/>")
        + _SVG.format("<!-- c --><?pi x?><title>C</title>")
    )
    assert _alts(svgs) == ["T", "P", None, "C"]


def test_role_is_read_as_its_first_token() -> None:
    """The fallback list of roles gives its first one."""
    tree = _page(
        '<img src="a.png" alt="L" role="none presentation">'
        '<img src="b.png" alt="L" role="none">'
        '<img src="c.png" alt="L" role="Img none">',
    )
    alts = [getattr(raw_image(img), "alt", "?") for img in tree.iter("img")]
    assert alts == ["", "", "L"]
    root = _page(
        '<table role="none presentation"><tr><td>a</td></tr></table>'
        '<table role="none"><tr><td>b</td></tr></table>',
    ).body
    assert content_spots(root, GLOBAL_STRIP).bare_tables == ()


def test_root_is_never_removed() -> None:
    """A hidden content root still gives its spots."""
    page = lxml.html.document_fromstring(
        "<html><body><main hidden><a href='x.html'></a>"
        "<table><tr><td>a</td><td>b</td></tr></table></main></body></html>",
    )
    root = page.find("body").find("main")
    spots = content_spots(root, GLOBAL_STRIP)
    assert [s.text for s in spots.empty_links] == ["x.html"]
    assert [s.text for s in spots.bare_tables] == ["a | b"]


def test_exclusions_apply_to_every_spot() -> None:
    """Stripped zones, drawings and formulas hold no spot."""
    root = _page(
        '<footer><span class="xref std">r</span>'
        '<span class="problematic">p</span>'
        "<table><tr><td>a</td></tr></table></footer>"
        '<svg width="300" height="200"><a href="x.html"></a></svg>'
        '<span class="math"><a href="y.html"></a>'
        "<table><tr><td>m</td></tr></table></span>"
        "<math><a href='z.html'></a></math>",
    ).body
    spots = content_spots(root, ("footer",))
    assert not (spots.empty_links)
    assert not (spots.unresolved)
    assert not (spots.bare_tables)


def test_links_name_by_accessible_name_only() -> None:
    """Hidden text and pictures without alt do not name a link."""
    root = _page(
        '<a href="https://x"><svg aria-hidden="true"><path/></svg></a>'
        '<a href="x1.html"><img alt=""></a>'
        '<a href="x2.html"><span aria-hidden="true">go</span></a>'
        '<a href="ok1.html"><svg><title>GitHub</title></svg></a>'
        '<a href="ok2.html" title="Edit"><svg aria-hidden="true"/></a>',
    ).body
    spots = content_spots(root, GLOBAL_STRIP)
    assert [s.text for s in spots.empty_links] == [
        "https://x",
        "x1.html",
        "x2.html",
    ]


def test_code_tables_of_hugo_and_jekyll_are_layout() -> None:
    """Chroma and Rouge line number tables are not data tables."""
    assert {"lntable", "rouge-table"} <= LAYOUT_TABLE_CLASSES
    root = _page(
        '<table class="lntable"><tr><td>1</td><td>x</td></tr></table>'
        '<table class="rouge-table"><tr><td>1</td><td>x</td></tr></table>',
    ).body
    assert content_spots(root, GLOBAL_STRIP).bare_tables == ()


def test_hidden_and_decorative_pictures_do_not_name_a_link() -> None:
    """A picture without alt, hidden or decorative, and hidden text."""
    root = _page(
        '<a href="n1.html"><img src="a.png"></a>'
        '<a href="n2.html"><img src="a.png" alt="A" aria-hidden="true"></a>'
        '<a href="n3.html"><img src="a.png" alt="A" role="presentation"></a>'
        '<a href="n4.html"><span hidden>Go</span></a>',
    ).body
    spots = content_spots(root, ("nav",))
    assert [s.text for s in spots.empty_links] == [
        "n1.html",
        "n2.html",
        "n3.html",
        "n4.html",
    ]
