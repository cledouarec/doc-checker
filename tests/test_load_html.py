# Copyright 2026 Christophe Le Douarec
"""Tests for loading HTML documents."""

import base64
import dataclasses
import html
from typing import TYPE_CHECKING

import pytest

from doc_checker.config import Config, HtmlConfig
from doc_checker.errors import ConfigError, InputError
from doc_checker.extract import detect_format, load_document
from doc_checker.extract.html import images as html_images
from doc_checker.extract.html import load_html
from doc_checker.model.document import NUMBER_ATTR, BlockKind
from tests.builders import (
    FIGURES_EN,
    SAMPLE_EN,
    SAMPLE_FR,
    build_docusaurus_like,
    build_mkdocs_like,
    build_pandoc_like,
    build_sphinx_like,
    link_text,
    replace_text,
)
from tests.image_builders import drawing, encode

if TYPE_CHECKING:
    from pathlib import Path

    from doc_checker.model.document import Document, ImageRef

BANNER = "Docs for version 2.0."


@pytest.fixture
def en_pair(tmp_path: Path) -> tuple[Document, Document]:
    """Load the English sample built by both generators."""
    ref_site = build_sphinx_like(SAMPLE_EN, tmp_path / "ref")
    new_site = build_mkdocs_like(SAMPLE_EN, tmp_path / "new", banner=BANNER)
    ref = load_html(ref_site, HtmlConfig(), lang="auto")
    new = load_html(new_site, HtmlConfig(), lang="auto")
    return ref, new


def test_generators_yield_same_text(
    en_pair: tuple[Document, Document],
) -> None:
    """Both generators give the same blocks once normalized."""
    ref, new = en_pair
    assert [b.key_strict for b in ref.blocks] == [
        b.key_strict for b in new.blocks
    ]


def test_french_typography_is_normalized(tmp_path: Path) -> None:
    """French typography differences vanish in the strict keys."""
    ref_site = build_sphinx_like(SAMPLE_FR, tmp_path / "r")
    new_site = build_mkdocs_like(SAMPLE_FR, tmp_path / "n")
    ref = load_html(ref_site, HtmlConfig(), lang="auto")
    new = load_html(new_site, HtmlConfig(), lang="auto")
    assert [b.key_strict for b in ref.blocks] == [
        b.key_strict for b in new.blocks
    ]
    assert (ref.lang, ref.normalization["quote"] > 0) == ("fr", True)


def test_mkdocs_pages_follow_nav_order(
    en_pair: tuple[Document, Document],
) -> None:
    """Site pages are read in navigation order."""
    assert en_pair[1].parts == [
        "index.html",
        "getting-started/index.html",
        "user-guide/index.html",
        "reference/index.html",
    ]


def test_generator_language_chrome(en_pair: tuple[Document, Document]) -> None:
    """Generator, language and page chrome are reported."""
    ref, new = en_pair
    assert (ref.generator, new.generator) == ("sphinx-rtd", "mkdocs-material")
    assert (ref.lang, new.lang) == ("en", "en")
    assert new.chrome == [BANNER]
    assert all(b.text != BANNER for b in new.blocks)


def test_section_paths(en_pair: tuple[Document, Document]) -> None:
    """Blocks know their enclosing headings and nearest anchor."""
    block = next(
        b
        for b in en_pair[0].blocks
        if b.text.startswith("Install the package")
    )
    assert block.section_path == ("Getting started", "Installation")
    assert block.source.anchor == "installation"


def test_sphinx_chapters_split(tmp_path: Path) -> None:
    """A Sphinx build split by chapter is read in toctree order."""
    site = build_sphinx_like(SAMPLE_EN, tmp_path / "s", split="chapters")
    doc = load_html(site, HtmlConfig(), lang="auto")
    assert doc.parts == [
        "index.html",
        "getting-started.html",
        "user-guide.html",
        "reference.html",
    ]


def test_load_html_without_pages(tmp_path: Path) -> None:
    """A directory without HTML pages is an input error."""
    with pytest.raises(InputError, match="no HTML pages"):
        load_html(tmp_path, HtmlConfig(), lang="auto")


def test_unknown_profile(tmp_path: Path) -> None:
    """An unknown forced profile is a configuration error."""
    build_mkdocs_like(SAMPLE_EN, tmp_path)
    with pytest.raises(ConfigError, match="unknown HTML profile"):
        load_html(tmp_path, HtmlConfig(profile="nope"), lang="auto")


def test_detect_format_and_damaged_pdf(tmp_path: Path) -> None:
    """Formats are detected and a damaged PDF is an input error."""
    (tmp_path / "x.bin").write_bytes(b"%PDF-1.7")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    assert detect_format(tmp_path) == "html"
    assert detect_format(tmp_path / "x.bin") == "pdf"
    with pytest.raises(InputError, match="unsupported input"):
        detect_format(tmp_path / "notes.txt")
    with pytest.raises(InputError, match="invalid or damaged PDF"):
        load_document(tmp_path / "x.bin", Config())


def test_sphinx_toctree_is_not_content(tmp_path: Path) -> None:
    """The toctree of a modern Sphinx index page is navigation."""
    page = tmp_path / "index.html"
    page.write_text(
        "<html><head>"
        '<script src="_static/documentation_options.js?v=1"></script>'
        '</head><body><div class="document"><div class="body" role="main">'
        '<section id="welcome"><h1>Welcome</h1>'
        "<p>Intro paragraph of the documentation here.</p>"
        '<div class="toctree-wrapper compound"><ul>'
        '<li class="toctree-l1"><a class="reference internal" '
        'href="install.html">Installation</a></li></ul></div>'
        "</section></div></div></body></html>",
        encoding="utf-8",
    )
    doc = load_html(page, HtmlConfig(), lang="auto")
    assert doc.generator == "sphinx"
    assert [b.kind.value for b in doc.blocks] == ["heading", "paragraph"]


@pytest.mark.parametrize(
    ("declared", "expected"),
    [("fr-CA", "fr-CA"), ('en"><script>alert(1)</script>', None), ("", None)],
)
def test_declared_language_must_be_a_language_tag(
    tmp_path: Path,
    declared: str,
    expected: str | None,
) -> None:
    """A declared language that is not a language tag is ignored."""
    page = tmp_path / "a.html"
    page.write_text(
        f'<html lang="{html.escape(declared)}"><body>'
        "<p>Some text here.</p></body></html>",
        encoding="utf-8",
    )
    doc = load_html(page, HtmlConfig(), lang="auto")
    assert doc.declared_lang == expected


def test_builders_render_links(tmp_path: Path) -> None:
    """Each HTML builder links the sections, PDF texts keep the words."""
    sphinx = build_sphinx_like(SAMPLE_EN, tmp_path / "s", split="chapters")
    mkdocs = build_mkdocs_like(SAMPLE_EN, tmp_path / "m")
    pandoc = build_pandoc_like(SAMPLE_EN, tmp_path / "p")
    docusaurus = build_docusaurus_like(
        SAMPLE_EN,
        tmp_path / "d",
        base="/example/",
    )
    pages = {
        'href="reference.html#command-line"': sphinx / "user-guide.html",
        'href="../reference/#command-line"': mkdocs / "user-guide/index.html",
        'href="#command-line"': pandoc / "index.html",
        'href="/example/docs/reference/#command-line"': (
            docusaurus / "docs/user-guide/index.html"
        ),
        'href="https://example.org/cli"': mkdocs / "reference/index.html",
    }
    for href, page in pages.items():
        assert href in page.read_text(encoding="utf-8")
    assert link_text("see [Limits](#limits) now") == "see Limits now"


def test_builders_keep_unknown_slug_on_the_page(tmp_path: Path) -> None:
    """A link to no section stays a same page anchor in every builder."""
    spec = replace_text(
        SAMPLE_EN,
        "[settings](#command-line)",
        "[settings](#nowhere)",
    )
    pages = [
        build_sphinx_like(spec, tmp_path / "s", split="chapters")
        / "user-guide.html",
        build_mkdocs_like(spec, tmp_path / "m") / "user-guide/index.html",
        build_pandoc_like(spec, tmp_path / "p") / "index.html",
        build_docusaurus_like(spec, tmp_path / "d")
        / "docs/user-guide/index.html",
    ]
    for page in pages:
        assert 'href="#nowhere"' in page.read_text(encoding="utf-8")


def test_numbered_headings_lose_their_numbers(tmp_path: Path) -> None:
    """Section numbers are set apart from the titles and counted."""
    site = build_pandoc_like(SAMPLE_EN, tmp_path / "p", numbered=True)
    doc = load_html(site, HtmlConfig(), lang="auto")
    headings = [b for b in doc.blocks if b.kind is BlockKind.HEADING]
    assert [b.text for b in headings] == [s.title for s in SAMPLE_EN.sections]
    assert headings[5].attrs[NUMBER_ATTR] == "2.2"
    assert doc.normalization["heading_number"] == 9


def _images(doc: Document) -> list[ImageRef]:
    """Return the images of a document, in order."""
    return [b.image for b in doc.blocks if b.image is not None]


def test_site_images_are_read(tmp_path: Path) -> None:
    """The files of a site give readable images, the SVG its identity."""
    site = build_sphinx_like(FIGURES_EN, tmp_path / "s", split="chapters")
    doc = load_html(site, HtmlConfig(), lang="auto")
    images = _images(doc)
    assert [(i.name, i.format, i.readable) for i in images] == [
        ("_images/setup.png", "png", True),
        ("_images/dashboard.jpg", "jpeg", True),
        ("_images/pipeline.svg", "svg", False),
        ("_images/quota.png", "png", True),
    ]
    assert (images[0].alt, images[0].width) == ("Installer steps", 400)
    assert images[2].texts == ("pipeline in", "pipeline out")
    assert doc.warnings == []


def test_files_data_urls_and_inline_svg_agree(tmp_path: Path) -> None:
    """A data URL and an inline SVG have the digest of the same file."""
    sites = (
        build_pandoc_like(FIGURES_EN, tmp_path / "f"),
        build_pandoc_like(FIGURES_EN, tmp_path / "e", embed=True),
        build_mkdocs_like(FIGURES_EN, tmp_path / "m"),
    )
    digests = [
        [i.digest for i in _images(load_html(site, HtmlConfig(), lang="en"))]
        for site in sites
    ]
    assert digests[0] == digests[1] == digests[2]
    assert len(digests[0]) == 4


def test_hidden_figure_captions_are_kept(tmp_path: Path) -> None:
    """The caption that pandoc hides from screen readers is read."""
    doc = load_html(
        build_pandoc_like(FIGURES_EN, tmp_path / "p"),
        HtmlConfig(),
        lang="en",
    )
    captions = [b.text for b in doc.blocks if b.kind is BlockKind.CAPTION]
    assert "Steps of the installer" in captions


def _page(site: Path, body: str, name: str = "index.html") -> None:
    """Write one page of a site with a main element."""
    site.mkdir(parents=True, exist_ok=True)
    (site / name).write_text(
        f"<html><body><main>{body}</main></body></html>",
        encoding="utf-8",
    )


def test_image_outside_the_site_is_not_read(tmp_path: Path) -> None:
    """A link out of the site, even through a symbolic link, is not read."""
    outside = tmp_path / "outside.png"
    outside.write_bytes(encode(drawing("secret")))
    site = tmp_path / "site"
    _page(
        site,
        '<p>Text of the page here.</p><img src="link.png">'
        '<img src="../outside.png">',
    )
    try:
        (site / "link.png").symlink_to(outside)
    except OSError:
        pytest.skip("symbolic links are not allowed here")
    doc = load_html(site, HtmlConfig(), lang="en")
    assert [i.problem for i in _images(doc)] == [
        "outside the site",
        "outside the site",
    ]
    assert doc.warnings == [
        "2 image(s) could not be read: link.png, ../outside.png",
    ]


def test_huge_data_url_is_not_decoded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A data URL longer than the limit is not decoded."""
    monkeypatch.setattr(html_images, "DATA_URL_MAX_CHARS", 1_000)
    data = base64.b64encode(encode(drawing("big"))).decode()
    _page(tmp_path / "s", f'<img src="data:image/png;base64,{data}">')
    (image,) = _images(load_html(tmp_path / "s", HtmlConfig(), lang="en"))
    assert (image.problem, image.readable) == ("too large", False)


def test_missing_image_file_is_a_warning(tmp_path: Path) -> None:
    """A missing file is unreadable and reported once, by name."""
    _page(tmp_path / "s", '<p>Text of the page.</p><img src="img/gone.png">')
    doc = load_html(tmp_path / "s", HtmlConfig(), lang="en")
    (image,) = _images(doc)
    assert (image.name, image.problem) == ("img/gone.png", "not found")
    assert doc.warnings == ["1 image(s) could not be read: img/gone.png"]


def test_image_on_most_pages_is_chrome(tmp_path: Path) -> None:
    """A logo repeated on every page of a site is chrome, and counted."""
    logo = encode(drawing("logo", size=(60, 20)))
    for number in range(4):
        _page(
            tmp_path / "s",
            f'<img src="logo.png"><p>Page {number} talks about its topic.</p>'
            f'<img src="p{number}.png">',
            name=f"page{number}.html",
        )
        (tmp_path / "s" / f"p{number}.png").write_bytes(
            encode(drawing(f"p{number}")),
        )
    (tmp_path / "s" / "logo.png").write_bytes(logo)
    doc = load_html(tmp_path / "s", HtmlConfig(), lang="en")
    assert [i.name for i in _images(doc)] == [f"p{n}.png" for n in range(4)]
    assert doc.normalization["image_chrome"] == 4


def test_images_are_left_out_on_request(tmp_path: Path) -> None:
    """Without images, a site has no image block and reads no file."""
    site = build_sphinx_like(FIGURES_EN, tmp_path / "s")
    doc = load_html(site, HtmlConfig(), lang="en", images=False)
    assert _images(doc) == []


def test_long_data_url_keeps_the_rest_of_its_page(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A data URL past ten million characters cuts nothing after it."""
    monkeypatch.setattr(html_images, "DATA_URL_MAX_CHARS", 1_000)
    data = "A" * 10_000_100
    _page(
        tmp_path / "s",
        f'<img src="data:image/png;base64,{data}"><p>After the image.</p>',
    )
    doc = load_html(tmp_path / "s", HtmlConfig(), lang="en")
    (image,) = _images(doc)
    assert image.problem == "too large"
    assert "After the image." in [b.text for b in doc.blocks]


def test_deep_nesting_is_skipped_with_a_warning(tmp_path: Path) -> None:
    """Content nested too deep is left out and reported, with no crash."""
    depth = 1_500
    _page(
        tmp_path / "s",
        "<div>" * depth
        + "<p>Too deep to read.</p>"
        + "</div>" * depth
        + "<p>After the nesting.</p>",
    )
    doc = load_html(tmp_path / "s", HtmlConfig(), lang="en")
    texts = [b.text for b in doc.blocks]
    assert "After the nesting." in texts
    assert "Too deep to read." not in texts
    assert doc.warnings == [
        "index.html: content nested deeper than 250 levels was skipped",
    ]


def test_broken_images_on_every_page_are_not_chrome(tmp_path: Path) -> None:
    """Different broken data URLs stay, and are reported."""
    for number in range(4):
        _page(
            tmp_path / "s",
            f"<p>Page {number} talks about its topic.</p>"
            f'<img src="data:image/png;base64,!!{number}">',
            name=f"page{number}.html",
        )
    doc = load_html(tmp_path / "s", HtmlConfig(), lang="en")
    assert len(_images(doc)) == 4
    assert doc.normalization["image_chrome"] == 0
    assert doc.warnings == ["4 image(s) could not be read: inline image"]


def test_load_document_passes_the_images_option(tmp_path: Path) -> None:
    """The images setting of the configuration reaches the HTML loader."""
    site = build_sphinx_like(FIGURES_EN, tmp_path / "s")
    doc = load_document(site, dataclasses.replace(Config(), images=False))
    assert _images(doc) == []
    assert len(_images(load_document(site, Config()))) == 4
