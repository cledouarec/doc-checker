# Copyright 2026 Christophe Le Douarec
"""Tests for the standalone HTML report."""

import base64
import hashlib
import html
import re
from dataclasses import replace
from typing import TYPE_CHECKING

import lxml.html
import pytest

from doc_checker.compare.engine import CompareResult, compare_documents
from doc_checker.config import Config, HtmlConfig
from doc_checker.extract.html import load_html
from doc_checker.images.svg import parse_svg
from doc_checker.model.document import BlockKind, ImageRef, Link
from doc_checker.report.html import content_security_policy, render_html
from tests.builders import (
    PARAGRAPHS,
    image_result,
    make_block,
    make_document,
    make_image_block,
)
from tests.image_builders import drawing, encode

if TYPE_CHECKING:
    from pathlib import Path

    from lxml.html import HtmlElement

REF_TEXT = (
    "Install the package with the default configuration before you start."
)
CSS_RULE_RE = re.compile(r"([^{}]+)\{([^{}]*)\}")


@pytest.fixture
def result() -> CompareResult:
    """Compare documents with one missing block and one typo."""
    ref = make_document(make_block(REF_TEXT), make_block(PARAGRAPHS[0]))
    new = make_document(
        make_block(REF_TEXT.replace("configuration", "configuraiton")),
    )
    return compare_documents(ref, new, Config())


def _tree(path: Path) -> HtmlElement:
    """Parse a written report."""
    return lxml.html.document_fromstring(path.read_text(encoding="utf-8"))


def _hash(text: str) -> str:
    """Return the CSP source of a text."""
    digest = hashlib.sha256(text.encode()).digest()
    return f"'sha256-{base64.b64encode(digest).decode()}'"


def _isolated(style: str) -> set[str]:
    """Return the selectors of the style rules that isolate their text."""
    return {
        selector.strip()
        for selectors, body in CSS_RULE_RE.findall(style)
        if "unicode-bidi: isolate" in body
        for selector in selectors.split(",")
    }


def test_html_report_structure(tmp_path: Path, result: CompareResult) -> None:
    """The report has its header, its filters and one article per finding."""
    tree = _tree(render_html(result, tmp_path / "report.html"))
    assert tree.findtext(".//title") == "doc-checker report: compare"
    articles = tree.cssselect("article.finding")
    assert [
        (a.get("data-severity"), a.get("data-category"), a.get("data-rule"))
        for a in articles
    ] == [
        ("critical", "content", "block.missing"),
        ("major", "text", "text.typo"),
    ]
    typo = articles[1]
    assert [e.text_content() for e in typo.cssselect("del")] == [
        "configuration",
    ]
    assert [e.text_content() for e in typo.cssselect("ins")] == [
        "configuraiton",
    ]
    typo_id = next(f.id for f in result.findings if f.rule == "text.typo")
    assert typo.get("id") == f"f-{typo_id}"
    boxes = tree.cssselect("form#filters input[name=severity]")
    assert [b.get("value") for b in boxes] == [
        "critical",
        "major",
        "minor",
        "info",
    ]
    rules = tree.cssselect("form#filters select[name=rule] option")
    assert [o.get("value") for o in rules] == [
        "",
        "block.missing",
        "text.typo",
    ]
    assert tree.cssselect("form#filters input[name=q]")


def test_html_report_has_no_external_resources(
    tmp_path: Path,
    result: CompareResult,
) -> None:
    """The report loads nothing and links only to the sources."""
    tree = _tree(render_html(result, tmp_path / "report.html"))
    assert not tree.cssselect("link, img, iframe, object, embed")
    assert len(tree.cssselect("script")) == 1
    for element in tree.iter():
        assert element.get("src") is None
        href = element.get("href") or ""
        assert not href.startswith(("http:", "https:", "//", "javascript:"))
    style = tree.findtext(".//style")
    assert "url(" not in style
    assert "@import" not in style


def test_html_report_policy_matches_its_script_and_style(
    tmp_path: Path,
    result: CompareResult,
) -> None:
    """The policy allows only the inline style and script."""
    tree = _tree(render_html(result, tmp_path / "report.html"))
    (meta,) = tree.cssselect('meta[http-equiv="Content-Security-Policy"]')
    policy = meta.get("content")
    assert policy.startswith("default-src 'none';")
    assert _hash(tree.findtext(".//script")) in policy
    assert _hash(tree.findtext(".//style")) in policy
    assert content_security_policy("a", "b") == (
        f"default-src 'none'; img-src data:; style-src {_hash('a')}; "
        f"script-src {_hash('b')}; base-uri 'none'; form-action 'none'"
    )


def test_html_report_escapes_document_content(tmp_path: Path) -> None:
    """Markup and URLs of a document are shown as text, never run."""
    script = "<script>alert(1)</script> is not run here at all."
    image = '"><img src=x onerror=alert(1)> is not run either.'
    site = Link("site", "https://x.example/", url="//x.example")
    evil_href = 'https://x.example/"><script>alert(2)</script>'
    evil = Link("site", evil_href, url="//x.example/evil")
    ref = make_document(
        make_block(script),
        make_block(image),
        replace(make_block(PARAGRAPHS[0]), links=(site,)),
    )
    new = make_document(replace(make_block(PARAGRAPHS[0]), links=(evil,)))
    result = compare_documents(ref, new, Config())
    tree = _tree(render_html(result, tmp_path / "report.html"))
    assert len(tree.xpath("//script")) == 1
    assert not tree.xpath("//img")
    assert not tree.xpath("//@*[starts-with(name(), 'on')]")
    assert not [v for v in tree.xpath("//@href") if "x.example" in v]
    text = tree.text_content()
    assert "<script>alert(1)</script>" in text
    assert evil_href in text


def test_html_report_keeps_unicode(tmp_path: Path) -> None:
    """Accents, scripts and emoji of a document come through, in UTF-8."""
    quote = "Été 日本語 \U0001f600 is kept as it is written."
    result = compare_documents(
        make_document(make_block(quote)),
        make_document(),
        Config(),
    )
    path = render_html(result, tmp_path / "report.html")
    assert quote in _tree(path).text_content()
    assert "unicode-bidi: isolate" in _tree(path).findtext(".//style")


def test_html_report_groups_and_caps_occurrences(tmp_path: Path) -> None:
    """A group is one article that lists a hundred occurrences at most."""
    block = make_block("Back to the top of the page.")
    result = compare_documents(
        make_document(*[block] * 120),
        make_document(),
        Config(),
    )
    tree = _tree(render_html(result, tmp_path / "report.html"))
    (article,) = tree.cssselect("article.finding")
    assert len(article.cssselect("details.occurrences ol > li")) == 100
    more = article.cssselect("p.more")[0].text_content()
    assert more == "and 20 more occurrences"


def test_html_group_of_blocks_shows_their_text(tmp_path: Path) -> None:
    """A group of missing blocks shows the text of the first one, cut."""
    text = (
        "Back to the top of the page, where the contents list every section "
        "of this guide and of its appendices."
    )
    result = compare_documents(
        make_document(*[make_block(text)] * 6),
        make_document(),
        Config(),
    )
    tree = _tree(render_html(result, tmp_path / "report.html"))
    (article,) = tree.cssselect("article.finding")
    assert article.cssselect("p.detail")[0].text_content() == (
        f"6 occurrences of {text[:80]}\U00002026"
    )
    first = article.cssselect("details.occurrences ol > li")[0]
    assert [h.text_content() for h in first.cssselect("div.side h4")] == [
        "Reference paragraph",
        "New",
    ]


def test_huge_report_stays_bounded(tmp_path: Path) -> None:
    """Fifteen hundred findings make one file of linear size."""
    blocks = [
        make_block(f"Paragraph number {i} talks about its own topic here.")
        for i in range(1500)
    ]
    result = compare_documents(
        make_document(*blocks),
        make_document(),
        Config(),
    )
    path = render_html(result, tmp_path / "report.html")
    assert len(_tree(path).cssselect("article.finding")) == 1500
    assert path.stat().st_size < 1500 * 2500


def test_html_report_dom_ids_are_unique(tmp_path: Path) -> None:
    """Findings that share an identifier get numbered DOM identifiers."""
    twice = make_block("Back to the top of the page.")
    often = make_block("See the previous section for the details.")
    result = compare_documents(
        make_document(*[twice] * 3, *[often] * 6),
        make_document(),
        Config(),
    )
    tree = _tree(render_html(result, tmp_path / "report.html"))
    ids = tree.xpath("//@id")
    assert len(ids) == len(set(ids))
    first, second, third, group = result.findings
    lone = first.id
    assert second.id == third.id == lone
    articles = tree.cssselect("article.finding")
    assert [a.get("id") for a in articles] == [
        f"f-{lone}",
        f"f-{lone}-2",
        f"f-{lone}-3",
        f"f-{group.id}",
    ]
    assert [a.cssselect("p.id code")[0].text for a in articles] == [
        lone,
        lone,
        lone,
        group.id,
    ]
    item = group.occurrences[0].id
    assert [li.get("id") for li in articles[3].cssselect("ol > li")] == [
        f"f-{item}",
        *(f"f-{item}-{n}" for n in range(2, 7)),
    ]


def test_html_report_isolates_document_texts(tmp_path: Path) -> None:
    """Every text of a document is isolated from the bidi text around it."""
    block = replace(
        make_block(PARAGRAPHS[0]),
        links=(Link("site", "https://x.example/", url="//x.example"),),
    )
    ref = replace(
        make_document(block),
        chrome=["Draft \U0000202eexample"],
        warnings=["1 page(s) without text, probably scanned: 2"],
    )
    new = make_document(make_block(PARAGRAPHS[0]))
    result = compare_documents(ref, new, Config())
    tree = _tree(render_html(result, tmp_path / "report.html"))
    assert {".text", ".detail", "a.source", ".quote"} <= _isolated(
        tree.findtext(".//style"),
    )
    assert [e.text for e in tree.cssselect("#chrome .quote")] == [
        "Draft \U0000202eexample",
    ]
    assert [e.text for e in tree.cssselect("#warnings .quote")] == [
        "1 page(s) without text, probably scanned: 2",
    ]
    assert len(tree.cssselect("table.inputs td.quote")) == 14
    (article,) = tree.cssselect("article.finding")
    assert article.cssselect("p.detail")
    assert [a.text for a in article.cssselect("p.where a.source")] == [
        "index.html",
        "index.html",
    ]


def test_html_report_shows_code_as_preformatted(tmp_path: Path) -> None:
    """The sides of a code finding are preformatted, with their kind."""
    code = "def main():\n    return compute(1, 2, 3)"
    result = compare_documents(
        make_document(make_block(code, BlockKind.CODE)),
        make_document(make_block(code.replace("3", "4"), BlockKind.CODE)),
        Config(),
    )
    (finding,) = result.findings
    assert "\n" in finding.ref.text
    tree = _tree(render_html(result, tmp_path / "report.html"))
    (article,) = tree.cssselect("article.finding")
    sides = article.cssselect("div.sides > div.side")
    assert [s.cssselect("h4")[0].text_content() for s in sides] == [
        "Reference code",
        "New code",
    ]
    assert [s.cssselect("pre.text")[0].text_content() for s in sides] == [
        finding.ref.text,
        finding.new.text,
    ]


def test_html_report_header_counts_findings(
    tmp_path: Path,
    result: CompareResult,
) -> None:
    """The header lists the inputs and counts the findings by severity."""
    tree = _tree(render_html(result, tmp_path / "report.html"))
    (header,) = tree.cssselect("header#summary")
    assert header.findtext("h1") == "doc-checker report: compare"
    counts = [
        (li.get("data-severity"), li.cssselect("span.count")[0].text)
        for li in header.cssselect("ul.counts > li")
    ]
    assert counts == [
        ("critical", "1"),
        ("major", "1"),
        ("minor", "0"),
        ("info", "0"),
    ]
    rows = header.cssselect("table.inputs tbody tr")
    assert [row.findtext("th") for row in rows] == [
        "Path",
        "Format",
        "Pages",
        "Generator",
        "Language",
        "Blocks",
        "Images",
    ]
    assert "50.0%" in header.cssselect("p.matched")[0].text_content()
    options = tree.cssselect("form#filters select[name=category] option")
    assert [(o.get("value"), o.text) for o in options] == [
        ("", "All categories"),
        ("content", "Content"),
        ("text", "Text"),
    ]
    sections = tree.cssselect("main#findings > section.severity")
    assert [s.findtext("h2") for s in sections] == [
        "Critical (1)",
        "Major (1)",
    ]


def test_html_report_without_findings(tmp_path: Path) -> None:
    """Identical documents give a report that says there is nothing."""
    block = make_block(PARAGRAPHS[0])
    result = compare_documents(
        make_document(block),
        make_document(block),
        Config(),
    )
    tree = _tree(render_html(result, tmp_path / "report.html"))
    assert tree.cssselect("header#summary p.none")[0].text == "No findings."
    assert not tree.cssselect("main#findings > *")
    assert tree.cssselect("output#shown")[0].text == "0"


def test_html_report_head_and_output_are_stable(
    tmp_path: Path,
    result: CompareResult,
) -> None:
    """The head is in order, nothing is commented and output is repeatable."""
    path = render_html(result, tmp_path / "report.html")
    text = path.read_text(encoding="utf-8")
    assert text.startswith('<!DOCTYPE html>\n<html lang="en">\n')
    tree = _tree(path)
    head = tree.find("head")
    assert [
        (e.tag, e.get("charset") or e.get("http-equiv") or e.get("name"))
        for e in head
    ] == [
        ("meta", "utf-8"),
        ("meta", "Content-Security-Policy"),
        ("meta", "viewport"),
        ("title", None),
        ("style", None),
    ]
    assert tree.find("body")[-1].tag == "script"
    assert not tree.xpath("//comment()")
    again = render_html(result, tmp_path / "again.html")
    assert again.read_text(encoding="utf-8") == text


def test_html_report_images_are_webp_data_urls(tmp_path: Path) -> None:
    """Thumbnails are the only images, WebP in data URLs."""
    tree = _tree(render_html(image_result(), tmp_path / "report.html"))
    images = tree.cssselect("img")
    assert len(images) == 2
    assert tree.xpath("//@src") == [image.get("src") for image in images]
    for image in images:
        assert image.get("class") == "thumb"
        prefix, _, payload = image.get("src").partition(",")
        assert prefix == "data:image/webp;base64"
        data = base64.b64decode(payload, validate=True)
        assert data[:4] == b"RIFF"
        assert data[8:12] == b"WEBP"


def test_html_report_never_inlines_svg(tmp_path: Path) -> None:
    """An SVG with a script is described, never written in the report."""
    source = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 200">'
        "<script>alert(1)</script><text>Evil</text></svg>"
    )
    identity = parse_svg(source.encode())
    svg = ImageRef(
        name="img/<b>evil</b>.svg",
        format="svg",
        digest=identity.digest,
        vector=True,
        texts=identity.texts,
    )
    result = compare_documents(
        make_document(make_block(PARAGRAPHS[0]), make_image_block(svg)),
        make_document(make_block(PARAGRAPHS[0])),
        Config(),
    )
    tree = _tree(render_html(result, tmp_path / "report.html"))
    assert not tree.xpath("//svg | //img | //b")
    assert len(tree.xpath("//script")) == 1
    assert "img/<b>evil</b>.svg, SVG" in tree.text_content()


# An SVG that would run a script if a report ever wrote it out.
EVIL_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 200" '
    'onload="alert(1)"><script>alert(2)</script><text>Evil</text></svg>'
)
EVIL_ALT = '"><svg onload=alert(3)><script>alert(4)</script>'


def test_html_report_of_a_hostile_site_shows_only_thumbnails(
    tmp_path: Path,
) -> None:
    """Images of a hostile site reach the report only as WebP thumbnails."""
    png = base64.b64encode(encode(drawing("hostile"))).decode()
    svg = base64.b64encode(EVIL_SVG.encode()).decode()
    paragraph = f"<p>{PARAGRAPHS[0]}</p>"
    images = (
        f'<img src="data:image/png;base64,{png}" '
        f'alt="{html.escape(EVIL_ALT)}">{EVIL_SVG}'
        f'<img src="data:image/svg+xml;base64,{svg}" alt="drawing">'
    )
    for name, body in (("ref", paragraph + images), ("new", paragraph)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "index.html").write_text(
            f"<html><body><main>{body}</main></body></html>",
            encoding="utf-8",
        )
    ref, new = (
        load_html(tmp_path / name, HtmlConfig(), lang="en")
        for name in ("ref", "new")
    )
    result = compare_documents(ref, new, Config())
    assert [f.rule for f in result.findings] == ["image.missing"] * 3
    tree = _tree(render_html(result, tmp_path / "report.html"))
    (thumb,) = tree.cssselect("img")
    assert thumb.get("class") == "thumb"
    assert thumb.get("alt") == "Reference image"
    assert tree.xpath("//@src") == [thumb.get("src")]
    prefix, _, payload = thumb.get("src").partition(",")
    assert prefix == "data:image/webp;base64"
    data = base64.b64decode(payload, validate=True)
    assert (data[:4], data[8:12]) == (b"RIFF", b"WEBP")
    assert not tree.xpath("//svg | //text | //@onload")
    assert not tree.xpath("//@*[starts-with(name(), 'on')]")
    assert len(tree.xpath("//script")) == 1
    assert EVIL_ALT in tree.text_content()
