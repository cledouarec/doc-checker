# Copyright 2026 Christophe Le Douarec
"""Builders of test blocks, documents and documentation sites."""

import base64
import html
import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from hashlib import blake2b
from operator import itemgetter
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from doc_checker.audit.engine import audit_document
from doc_checker.compare.engine import compare_documents
from doc_checker.config import Config
from doc_checker.extract.common import assign_section_paths
from doc_checker.images.refs import raster_ref
from doc_checker.model.document import (
    FROM_OUTLINE,
    FROM_STYLE,
    HEADING_SOURCE,
    Block,
    BlockKind,
    Document,
    ImageRef,
    PixelSource,
    SourceRef,
)
from doc_checker.model.facts import HtmlPageFacts
from doc_checker.model.findings import Severity
from doc_checker.text.normalize import loose_key, normalize_display, strict_key
from tests.image_builders import (
    blurred,
    drawing,
    encode,
    figure_drawing,
    figure_svg,
    grayed,
    pixelated,
    scaled,
)

if TYPE_CHECKING:
    from PIL import Image

    from doc_checker.audit.engine import AuditResult
    from doc_checker.compare.engine import CompareResult
    from doc_checker.model.findings import Finding


def make_block(
    text: str,
    kind: BlockKind = BlockKind.PARAGRAPH,
    *,
    file: str = "index.html",
    level: int | None = None,
) -> Block:
    """Return a normalized block, as an extractor would produce it."""
    counter: Counter[str] = Counter()
    display = normalize_display(text, counter)
    strict = strict_key(display, counter, code=kind is BlockKind.CODE)
    return Block(
        kind=kind,
        text=display,
        raw_text=text,
        source=SourceRef(file),
        level=level,
        key_strict=strict,
        key_loose=loose_key(strict),
    )


type Defect = Literal[
    "jpeg10",
    "jpeg95",
    "x0.4",
    "blur",
    "nearest",
    "gray",
    "raster",
    "other",
    "noalt",
    "dropped",
]


@dataclass(frozen=True)
class Figure:
    """A figure of a section: a drawing, its caption and its alternative text.

    Attributes:
        name: Name of the drawing, which seeds it and names its file.
        caption: The caption.
        alt: The alternative text.
        kind: The format of the file.
        defect: A planted defect, or ``None``.
    """

    name: str
    caption: str
    alt: str
    kind: Literal["png", "jpeg", "svg"] = "png"
    defect: Defect | None = None


@dataclass(frozen=True)
class Section:
    """A heading and the content that follows it.

    The flags that end in ``_as_text`` make a builder write a part of the
    section as plain paragraphs, which loses its kind. Only
    `build_mkdocs_like` and `build_fpdf2_pdf` of `tests.pdf_builders` respect
    them, the latter only ``heading_as_text``.
    """

    title: str
    slug: str
    level: int
    paragraphs: tuple[str, ...] = ()
    bullets: tuple[str, ...] = ()
    code: str | None = None
    table: tuple[tuple[str, ...], ...] = ()
    note: str | None = None
    heading_as_text: bool = False
    bullets_as_text: bool = False
    table_as_text: bool = False
    code_as_text: bool = False
    figures: tuple[Figure, ...] = ()


@dataclass(frozen=True)
class DocSpec:
    """Generator independent description of a documentation."""

    title: str
    lang: str
    sections: tuple[Section, ...]


SAMPLE_EN = DocSpec(
    title="Example",
    lang="en",
    sections=(
        Section("Getting started", "getting-started", 1),
        Section(
            "Installation",
            "installation",
            2,
            paragraphs=(
                (
                    "Install the package with the default "
                    "[configuration](#first-steps) before you start."
                ),
                (
                    "The installer downloads 100 files and requires "
                    "2 GB of free disk space."
                ),
            ),
            bullets=(
                "Python 3.14 or newer is required.",
                "A working internet connection is needed.",
            ),
            code="pip install example\nexample --version",
            note="Do not delete the cache directory while the installer runs.",
        ),
        Section(
            "First steps",
            "first-steps",
            2,
            paragraphs=(
                (
                    "You must restart the service after changing the "
                    "configuration file."
                ),
                (
                    "The efficient scheduler processes 1,000 jobs per "
                    "minute on a single core."
                ),
                "The service's log file is kept for seven days.",
            ),
        ),
        Section("User guide", "user-guide", 1),
        Section(
            "Projects",
            "projects",
            2,
            paragraphs=(
                (
                    "A project groups several documents that share the same "
                    "[settings](#command-line)."
                ),
                "Each project has an owner who can invite other members.",
            ),
            table=(("Option", "Default"), ("timeout", "30"), ("retries", "3")),
        ),
        Section(
            "Exports",
            "exports",
            2,
            paragraphs=(
                (
                    "Exports are written to the output directory in Markdown "
                    "format."
                ),
                "Large exports are split into several files automatically.",
            ),
        ),
        Section("Reference", "reference", 1),
        Section(
            "Command line",
            "command-line",
            2,
            paragraphs=(
                (
                    "The command line accepts options in any order after the "
                    "[subcommand](https://example.org/cli)."
                ),
                'Set the "strict" mode to reject every invalid file.',
            ),
        ),
        Section(
            "Limits",
            "limits",
            2,
            paragraphs=(
                (
                    "The service accepts at most 100 requests per second for "
                    "each client."
                ),
                (
                    "Requests above the limit receive an error and must be "
                    "[retried](#first-steps) later."
                ),
            ),
        ),
    ),
)

SAMPLE_FR = DocSpec(
    title="Exemple",
    lang="fr",
    sections=(
        Section("Prise en main", "prise-en-main", 1),
        Section(
            "Installation",
            "installation",
            2,
            paragraphs=(
                (
                    "Installez le paquet avec la "
                    "[configuration](#premiers-pas) par défaut avant de "
                    "commencer."
                ),
                (
                    "L'installateur télécharge 1 000 fichiers et "
                    "nécessite 2,5 Go d'espace libre."
                ),
                'Attention : le mode "strict" refuse les fichiers invalides.',
            ),
            note=(
                "Ne supprimez pas le répertoire de cache pendant "
                "l'installation."
            ),
        ),
        Section(
            "Premiers pas",
            "premiers-pas",
            2,
            paragraphs=(
                (
                    "Vous devez redémarrer le service après avoir modifié le "
                    "fichier de configuration."
                ),
                "Cette procédure a été validée par l'équipe pendant l'été.",
            ),
        ),
        Section("Guide", "guide", 1),
        Section(
            "Projets",
            "projets",
            2,
            paragraphs=(
                (
                    "Un projet regroupe plusieurs documents qui partagent les "
                    "mêmes [réglages](#installation)."
                ),
                (
                    "Chaque projet possède un propriétaire qui peut inviter "
                    "d'autres membres."
                ),
            ),
        ),
    ),
)


def _esc(text: str) -> str:
    return html.escape(text, quote=False)


def _attr(text: str) -> str:
    return html.escape(text, quote=True)


type FigureFile = tuple[str, bytes]


def _encoded(image: Image.Image, suffix: str, quality: int = 90) -> FigureFile:
    """Encode an image as a PNG or as a JPEG."""
    if suffix == "jpg":
        return "jpg", encode(image, "JPEG", quality=quality)
    return "png", encode(image)


_TRANSFORMS: dict[str, Callable[[Image.Image], Image.Image]] = {
    "x0.4": lambda image: scaled(image, 0.4),
    "blur": blurred,
    "nearest": pixelated,
    "gray": grayed,
}
_RASTER_DEFECTS = frozenset({"jpeg10", "jpeg95", "raster", *_TRANSFORMS})


def figure_file(figure: Figure) -> FigureFile:
    """Return the suffix, without the dot, and the bytes of a figure file.

    A defect that changes the pixels of an SVG figure writes the PNG of its
    drawing, since an SVG has no pixels to change.
    """
    name, defect = figure.name, figure.defect
    other = defect == "other"
    if figure.kind == "svg" and defect not in _RASTER_DEFECTS:
        text = (
            figure_svg(f"{name}-other", flipped=True)
            if other
            else figure_svg(name)
        )
        return "svg", text.encode()
    suffix, quality = {"jpeg": "jpg"}.get(figure.kind, "png"), 90
    if figure.kind == "svg":
        image = figure_drawing(name)
    else:
        image = drawing(f"{name}-other" if other else name)
    if defect in {"jpeg10", "jpeg95"}:
        suffix, quality = "jpg", int(defect.removeprefix("jpeg"))
    elif defect in _TRANSFORMS:
        image = _TRANSFORMS[defect](image)
    return _encoded(image, suffix, quality)


def _figure_numbers(spec: DocSpec) -> dict[str, int]:
    """Number the figures of the documentation from 1, in order."""
    figures = [f.name for section in spec.sections for f in section.figures]
    return {name: k for k, name in enumerate(figures, 1)}


def _write_figures(
    spec: DocSpec,
    folder: Path,
    pattern: str,
    *,
    inline: str | None = None,
) -> None:
    """Write the figure files in a folder, each under its own name.

    Args:
        spec: The documentation.
        folder: The folder, created if it is missing.
        pattern: The file name, with the fields ``name``, ``suffix`` and
            ``h``, the digest of the bytes.
        inline: A suffix whose files are written in the page, not here.
    """
    for section in spec.sections:
        for figure in section.figures:
            if figure.defect == "dropped":
                continue
            suffix, data = figure_file(figure)
            if suffix == inline:
                continue
            digest = blake2b(data, digest_size=4).hexdigest()
            folder.mkdir(parents=True, exist_ok=True)
            file = pattern.format(name=figure.name, suffix=suffix, h=digest)
            (folder / file).write_bytes(data)


_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")


def link_text(text: str) -> str:
    """Replace each ``[text](target)`` link of a spec text by its text."""
    return _LINK.sub(itemgetter(1), text)


type Resolver = Callable[[str], str]


def _render(
    text: str,
    resolve: Resolver,
    *,
    lang: str | None = None,
    reference: bool = False,
) -> str:
    """Return the HTML of a spec text, its links turned into anchors.

    Args:
        text: The spec text, with ``[text](target)`` links.
        resolve: Maps a link target to the href the builder writes.
        lang: The language whose typography applies, or ``None`` for none.
        reference: Whether the anchors carry the Sphinx reference classes.

    Returns:
        The escaped HTML.
    """
    if lang:
        text = smart_typography(text, lang)
    parts: list[str] = []
    position = 0
    for match in _LINK.finditer(text):
        parts.append(_esc(text[position : match.start()]))
        target = match[2]
        css = ""
        if reference:
            kind = "internal" if target.startswith("#") else "external"
            css = f' class="reference {kind}"'
        href = html.escape(resolve(target), quote=True)
        parts.append(f'<a{css} href="{href}">{_esc(match[1])}</a>')
        position = match.end()
    parts.append(_esc(text[position:]))
    return "".join(parts)


def _chapter_of(spec: DocSpec) -> dict[str, str]:
    """Map each section slug to the slug of its chapter."""
    owner: dict[str, str] = {}
    for chapter in _chapters(spec):
        for section in chapter:
            owner[section.slug] = chapter[0].slug
    return owner


def _resolver(spec: DocSpec, here: str, other: str | None) -> Resolver:
    """Return the link resolver of a page.

    Args:
        spec: The documentation.
        here: The slug of the chapter the page shows.
        other: The format of the href to a section of another chapter, with
            the fields ``chapter`` and ``slug``, or ``None`` when every
            section is on the page.

    Returns:
        A function from a link target to an href. External targets and
        targets that are no section of the spec stay as written.
    """
    owner = _chapter_of(spec)

    def resolve(target: str) -> str:
        slug = target.removeprefix("#")
        chapter = owner.get(slug)
        if slug == target or chapter is None:
            return target
        if other is None or chapter == here:
            return f"#{slug}"
        return other.format(chapter=chapter, slug=slug)

    return resolve


def section_numbers(spec: DocSpec) -> dict[str, str]:
    """Number the sections: ``c`` for a chapter, ``c.s`` and ``c.s.t``."""
    numbers: dict[str, str] = {}
    counts = [0, 0, 0]
    for section in spec.sections:
        depth = min(section.level, 3) - 1
        counts[depth] += 1
        counts[depth + 1 :] = [0] * (2 - depth)
        numbers[section.slug] = ".".join(str(c) for c in counts[: depth + 1])
    return numbers


_DOUBLE_QUOTED = re.compile(r'"([^"]*)"')
_FR_COLON = re.compile(r" :")
_FR_HIGH_PUNCT = re.compile(r" ([;!?])")
_DIGIT_GROUP = re.compile(r"(\d) (\d{3})")


def smart_typography(text: str, lang: str) -> str:
    """Apply the typography a Sphinx build would produce."""
    text = text.replace("'", "\U00002019")
    if lang == "fr":
        text = _DOUBLE_QUOTED.sub(lambda m: f"\xab\xa0{m[1]}\xa0\xbb", text)
        text = _FR_COLON.sub("\xa0:", text)
        text = _FR_HIGH_PUNCT.sub(lambda m: f"\U0000202f{m[1]}", text)
        return _DIGIT_GROUP.sub(lambda m: f"{m[1]}\U0000202f{m[2]}", text)
    return _DOUBLE_QUOTED.sub(lambda m: f"\U0000201c{m[1]}\U0000201d", text)


def _chapters(spec: DocSpec) -> list[list[Section]]:
    """Group the sections by chapter, a chapter being a level 1 section."""
    chapters: list[list[Section]] = []
    for section in spec.sections:
        if section.level == 1 or not chapters:
            chapters.append([section])
        else:
            chapters[-1].append(section)
    return chapters


def _sphinx_figure(figure: Figure, k: int | None) -> str:
    """Write a figure as Sphinx does, with a number if ``k`` is given."""
    suffix, _ = figure_file(figure)
    caption = _esc(figure.caption)
    if figure.defect == "dropped":
        return (
            '<figure class="align-default">'
            f"<figcaption>{caption}</figcaption></figure>"
        )
    alt = "" if figure.defect == "noalt" else f'alt="{_attr(figure.alt)}" '
    number = f'<span class="caption-number">Fig. {k} </span>' if k else ""
    return (
        f'<figure class="align-default" id="fig-{figure.name}">'
        f'<img {alt}src="_images/{figure.name}.{suffix}" />'
        f'<figcaption><p>{number}<span class="caption-text">{caption}</span>'
        f'<a class="headerlink" href="#fig-{figure.name}" '
        'title="Link to this image">\xb6</a></p></figcaption></figure>'
    )


def _sphinx_section(
    section: Section,
    lang: str,
    resolve: Resolver,
    number: str | None,
    figure_numbers: dict[str, int],
) -> str:
    tag = f"h{section.level}"
    title = _esc(smart_typography(section.title, lang))
    if number:
        title = f'<span class="section-number">{number}. </span>{title}'
    parts = [
        f'<section id="{section.slug}">',
        (
            f'<{tag}>{title}<a class="headerlink" href="#{section.slug}" '
            f'title="Link to this heading">\xb6</a></{tag}>'
        ),
    ]
    parts += [
        f"<p>{_render(p, resolve, lang=lang, reference=True)}</p>"
        for p in section.paragraphs
    ]
    parts += [
        _sphinx_figure(figure, figure_numbers.get(figure.name))
        for figure in section.figures
    ]
    if section.bullets:
        items = "".join(
            f"<li><p>{_render(b, resolve, lang=lang, reference=True)}</p></li>"
            for b in section.bullets
        )
        parts.append(f'<ul class="simple">{items}</ul>')
    if section.code:
        lines = "".join(
            f'<span class="linenos">{number}</span>{_esc(line)}\n'
            for number, line in enumerate(section.code.split("\n"), 1)
        )
        parts.append(
            '<div class="highlight-default notranslate">'
            '<div class="highlight">'
            f"<pre><span></span>{lines}</pre></div></div>",
        )
    if section.table:
        header, *rows = section.table
        head = "".join(
            f'<th class="head"><p>{_esc(c)}</p></th>' for c in header
        )
        body = "".join(
            "<tr>"
            + "".join(f"<td><p>{_esc(c)}</p></td>" for c in row)
            + "</tr>"
            for row in rows
        )
        parts.append(
            '<table class="docutils align-default"><thead>'
            f'<tr class="row-odd">{head}</tr></thead><tbody>{body}</tbody>'
            "</table>",
        )
    if section.note:
        parts.append(
            '<div class="admonition note"><p class="admonition-title">Note</p>'
            f"<p>{_render(section.note, resolve, lang=lang, reference=True)}"
            "</p></div>",
        )
    parts.append("</section>")
    return "\n".join(parts)


def _sphinx_page(
    spec: DocSpec,
    title: str,
    body: str,
    nav: list[tuple[str, str]],
) -> str:
    links = "".join(
        f'<li class="toctree-l1"><a class="reference internal" '
        f'href="{href}">{_esc(text)}</a></li>'
        for href, text in nav
    )
    return (
        f'<!DOCTYPE html>\n<html lang="{spec.lang}"><head>'
        f'<meta charset="utf-8"><title>{_esc(title)} - {_esc(spec.title)}'
        '</title><script src="_static/documentation_options.js?v=5929fcd5">'
        "</script></head>\n"
        '<body class="wy-body-for-nav"><div class="wy-grid-for-nav">\n'
        '<nav class="wy-nav-side"><div class="wy-menu wy-menu-vertical" '
        f'role="navigation"><ul>{links}</ul></div></nav>\n'
        '<section class="wy-nav-content-wrap"><div class="wy-nav-content">'
        '<div class="rst-content">\n'
        '<div role="navigation" aria-label="Page navigation">'
        '<ul class="wy-breadcrumbs"><li>Docs</li></ul></div>\n'
        '<div role="main" class="document"><div itemprop="articleBody">\n'
        f"{body}\n</div></div>\n"
        '<footer><div class="rst-footer-buttons" role="navigation">'
        '<a href="#">Next</a></div><div role="contentinfo">'
        "<p>\xa9 Copyright 2026.</p></div></footer>\n"
        "</div></div></section></div></body></html>\n"
    )


def build_sphinx_like(
    spec: DocSpec,
    out: Path,
    *,
    split: Literal["single", "chapters"] = "single",
    numbered: bool = False,
) -> Path:
    """Write the documentation as a Sphinx Read the Docs theme build."""
    out.mkdir(parents=True, exist_ok=True)
    chapters = _chapters(spec)
    numbers = section_numbers(spec) if numbered else {}
    figure_numbers = _figure_numbers(spec) if numbered else {}
    _write_figures(spec, out / "_images", "{name}.{suffix}")

    def render(chapter: list[Section], resolve: Resolver) -> str:
        return "\n".join(
            _sphinx_section(
                s,
                spec.lang,
                resolve,
                numbers.get(s.slug),
                figure_numbers,
            )
            for s in chapter
        )

    if split == "single":
        nav = [(f"#{c[0].slug}", c[0].title) for c in chapters]
        body = render(list(spec.sections), _resolver(spec, "", None))
        page = _sphinx_page(spec, spec.title, body, nav)
        (out / "index.html").write_text(page, encoding="utf-8")
        return out
    nav = [(f"{c[0].slug}.html", c[0].title) for c in chapters]
    toctree = "".join(
        f'<li class="toctree-l1"><a class="reference internal" '
        f'href="{href}">{_esc(text)}</a></li>'
        for href, text in nav
    )
    index = f'<div class="toctree-wrapper compound"><ul>{toctree}</ul></div>'
    page = _sphinx_page(spec, spec.title, index, nav)
    (out / "index.html").write_text(page, encoding="utf-8")
    for chapter in chapters:
        resolve = _resolver(spec, chapter[0].slug, "{chapter}.html#{slug}")
        body = render(chapter, resolve)
        page = _sphinx_page(spec, chapter[0].title, body, nav)
        (out / f"{chapter[0].slug}.html").write_text(page, encoding="utf-8")
    return out


def _mkdocs_table(section: Section) -> str:
    header, *rows = section.table
    head = "".join(f"<th>{_esc(c)}</th>" for c in header)
    body = "".join(
        "<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in row) + "</tr>"
        for row in rows
    )
    return (
        f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    )


def _plain_lists(
    section: Section,
    resolve: Resolver,
    *,
    unlisted: bool = False,
    figures: Sequence[str] = (),
) -> list[str]:
    """Return the paragraphs and the bullets, as MkDocs-like builders write.

    With ``unlisted``, each bullet is a paragraph of its own. The figures,
    already written, come after the paragraphs.
    """
    parts = [f"<p>{_render(p, resolve)}</p>" for p in section.paragraphs]
    parts += figures
    if unlisted:
        parts += [f"<p>{_render(b, resolve)}</p>" for b in section.bullets]
    elif section.bullets:
        items = "".join(
            f"<li>{_render(b, resolve)}</li>" for b in section.bullets
        )
        parts.append(f"<ul>{items}</ul>")
    return parts


def _mkdocs_figure(figure: Figure) -> str:
    """Write a figure as Material for MkDocs does, an SVG in line."""
    caption = f"<figcaption>{_esc(figure.caption)}</figcaption>"
    if figure.defect == "dropped":
        return f"<figure>{caption}</figure>"
    suffix, data = figure_file(figure)
    if suffix == "svg":
        return f"<figure>{data.decode()}{caption}</figure>"
    alt = "" if figure.defect == "noalt" else f'alt="{_attr(figure.alt)}" '
    return (
        f'<figure><img {alt}src="../img/{figure.name}.{suffix}" />'
        f"{caption}</figure>"
    )


def _mkdocs_section(section: Section, resolve: Resolver) -> str:
    tag = f"h{section.level}"
    if section.heading_as_text:
        parts = [f"<p>{_esc(section.title)}</p>"]
    else:
        parts = [
            (
                f'<{tag} id="{section.slug}">{_esc(section.title)}'
                f'<a class="headerlink" href="#{section.slug}" '
                f'title="Permanent link">\xb6</a></{tag}>'
            ),
        ]
    parts += _plain_lists(
        section,
        resolve,
        unlisted=section.bullets_as_text,
        figures=[_mkdocs_figure(f) for f in section.figures],
    )
    if section.code and section.code_as_text:
        lines = "<br>".join(_esc(line) for line in section.code.split("\n"))
        parts.append(f"<p>{lines}</p>")
    elif section.code:
        parts.append(
            '<div class="highlight"><pre><span></span>'
            f"<code>{_esc(section.code)}</code></pre></div>",
        )
    if section.table_as_text:
        parts += [f"<p>{_esc(' '.join(row))}</p>" for row in section.table]
    elif section.table:
        parts.append(_mkdocs_table(section))
    if section.note:
        parts.append(
            '<div class="admonition note"><p class="admonition-title">Note</p>'
            f"<p>{_render(section.note, resolve)}</p></div>",
        )
    return "\n".join(parts)


def _mkdocs_page(  # ruff: ignore[too-many-arguments] - test page template
    spec: DocSpec,
    title: str,
    body: str,
    *,
    depth: int,
    slug: str,
    banner: str | None,
) -> str:
    prefix = "../" * depth
    links = "".join(
        f'<li class="md-nav__item"><a href="{prefix}{c[0].slug}/" '
        f'class="md-nav__link">{_esc(c[0].title)}</a></li>'
        for c in _chapters(spec)
    )
    banner_html = f"<p>{_esc(banner)}</p>\n" if banner else ""
    return (
        f'<!doctype html>\n<html lang="{spec.lang}" class="no-js"><head>'
        f'<meta charset="utf-8"><title>{_esc(title)} - {_esc(spec.title)}'
        '</title></head>\n<body dir="ltr">\n'
        '<header class="md-header"><nav class="md-header__inner" '
        'aria-label="Header"><span class="md-header__topic">'
        f"{_esc(spec.title)}</span>"
        '<div class="md-search"><input placeholder="Search"></div></nav>'
        "</header>\n"
        '<div class="md-container"><main class="md-main">'
        '<div class="md-main__inner md-grid">\n'
        '<div class="md-sidebar md-sidebar--primary">'
        '<nav class="md-nav md-nav--primary" aria-label="Navigation">'
        f'<ul class="md-nav__list">{links}</ul></nav></div>\n'
        '<div class="md-content"><article class="md-content__inner '
        'md-typeset">\n'
        f'<a href="https://example.org/edit/{slug}.md" '
        'title="Edit this page" class="md-content__button md-icon">Edit</a>\n'
        f"{banner_html}{body}\n"
        '<aside class="md-source-file"><span class="md-source-file__fact">'
        "Last update: November 4, 2026</span></aside>\n"
        "</article></div></div></main>\n"
        '<footer class="md-footer"><nav class="md-footer__inner" '
        'aria-label="Footer"><a href="#">Next</a></nav>'
        '<div class="md-footer-meta">Made with Material for MkDocs</div>'
        "</footer></div></body></html>\n"
    )


def build_mkdocs_like(
    spec: DocSpec,
    out: Path,
    *,
    banner: str | None = None,
) -> Path:
    """Write the documentation as a Material for MkDocs build."""
    out.mkdir(parents=True, exist_ok=True)
    _write_figures(spec, out / "img", "{name}.{suffix}", inline="svg")
    home = _mkdocs_page(
        spec,
        spec.title,
        "",
        depth=0,
        slug="index",
        banner=banner,
    )
    (out / "index.html").write_text(home, encoding="utf-8")
    for chapter in _chapters(spec):
        slug = chapter[0].slug
        resolve = _resolver(spec, slug, "../{chapter}/#{slug}")
        body = "\n".join(_mkdocs_section(s, resolve) for s in chapter)
        page = _mkdocs_page(
            spec,
            chapter[0].title,
            body,
            depth=1,
            slug=slug,
            banner=banner,
        )
        (out / slug).mkdir(exist_ok=True)
        (out / slug / "index.html").write_text(page, encoding="utf-8")
    return out


_MIME = {"png": "png", "jpg": "jpeg", "svg": "svg+xml"}


def _pandoc_figure(figure: Figure, *, embed: bool) -> str:
    """Write a figure as pandoc does, in line with ``embed``."""
    caption = _esc(figure.caption)
    text = f'<figcaption aria-hidden="true">{caption}</figcaption>'
    if figure.defect == "dropped":
        return f"<figure>{text}</figure>"
    suffix, data = figure_file(figure)
    label = _attr(figure.caption)
    if embed:
        payload = base64.b64encode(data).decode()
        src = f"data:image/{_MIME[suffix]};base64,{payload}"
        extra = (
            ' role="img"'
            if figure.defect == "noalt"
            else f' role="img" aria-label="{label}"'
        )
    else:
        src = f"img/{figure.name}.{suffix}"
        extra = "" if figure.defect == "noalt" else f' alt="{label}"'
    return f'<figure><img src="{src}"{extra} />{text}</figure>'


def _pandoc_section(
    section: Section,
    resolve: Resolver,
    number: str | None,
    code_count: int,
    *,
    embed: bool,
) -> str:
    tag = f"h{section.level}"
    prefix = (
        f'<span class="header-section-number">{number}</span> '
        if number
        else ""
    )
    parts = [
        f'<{tag} id="{section.slug}">{prefix}{_esc(section.title)}</{tag}>',
    ]
    parts += _plain_lists(
        section,
        resolve,
        figures=[_pandoc_figure(f, embed=embed) for f in section.figures],
    )
    if section.code:
        k = code_count + 1
        lines = "\n".join(
            f'<span id="cb{k}-{j}"><a href="#cb{k}-{j}" aria-hidden="true" '
            f'tabindex="-1"></a>{_esc(line)}</span>'
            for j, line in enumerate(section.code.split("\n"), 1)
        )
        parts.append(
            f'<div class="sourceCode" id="cb{k}">'
            '<pre class="sourceCode bash"><code class="sourceCode bash">'
            f"{lines}</code></pre></div>",
        )
    if section.table:
        header, *rows = section.table
        head = "".join(f"<th>{_esc(c)}</th>" for c in header)
        body = "".join(
            f'<tr class="{"odd" if n % 2 else "even"}">'
            + "".join(f"<td>{_esc(c)}</td>" for c in row)
            + "</tr>"
            for n, row in enumerate(rows, 1)
        )
        parts.append(
            f'<table><thead><tr class="header">{head}</tr></thead>'
            f"<tbody>{body}</tbody></table>",
        )
    if section.note:
        parts.append(
            f'<div class="note"><p>{_render(section.note, resolve)}</p></div>',
        )
    return "\n".join(parts)


def build_pandoc_like(
    spec: DocSpec,
    out: Path,
    *,
    numbered: bool = False,
    embed: bool = False,
) -> Path:
    """Write the documentation as one pandoc standalone HTML file.

    With ``embed``, the images are data URLs and no file is written.
    """
    out.mkdir(parents=True, exist_ok=True)
    if not embed:
        _write_figures(spec, out / "img", "{name}.{suffix}")
    numbers = section_numbers(spec) if numbered else {}
    resolve = _resolver(spec, "", None)
    codes = 0
    sections = []
    for section in spec.sections:
        sections.append(
            _pandoc_section(
                section,
                resolve,
                numbers.get(section.slug),
                codes,
                embed=embed,
            ),
        )
        codes += bool(section.code)
    title = _esc(spec.title)
    page = (
        f'<!DOCTYPE html><html lang="{spec.lang}"><head><meta charset="utf-8">'
        f"<title>{title}</title></head><body>"
        f'<header id="title-block-header"><h1 class="title">{title}</h1>'
        f"</header>{chr(10).join(sections)}</body></html>\n"
    )
    (out / "index.html").write_text(page, encoding="utf-8")
    return out


_DOCUSAURUS_NOTE = {"en": "note", "fr": "remarque"}


def _docusaurus_figure(figure: Figure, base: str) -> str:
    """Write a figure as Docusaurus does, its file named with a digest."""
    caption = f"<p><em>{_esc(figure.caption)}</em></p>"
    if figure.defect == "dropped":
        return caption
    suffix, data = figure_file(figure)
    digest = blake2b(data, digest_size=4).hexdigest()
    alt = "" if figure.defect == "noalt" else f'alt="{_attr(figure.alt)}" '
    src = _attr(f"{base}assets/images/{figure.name}-{digest}.{suffix}")
    image = f'<img loading="lazy" {alt}src="{src}" class="img_ev3q">'
    return f"<p>{image}</p>{caption}"


def _docusaurus_section(
    section: Section,
    resolve: Resolver,
    lang: str,
    base: str,
) -> str:
    parts = []
    if section.level > 1:
        tag = f"h{section.level}"
        title = _esc(section.title)
        label = html.escape(section.title, quote=True)
        parts.append(
            f'<{tag} class="anchor anchorWithStickyNavbar" '
            f'id="{section.slug}">{title}<a href="#{section.slug}" '
            f'class="hash-link" aria-label="Direct link to {label}">'
            "\U0000200b</a></" + tag + ">",
        )
    parts += _plain_lists(
        section,
        resolve,
        figures=[_docusaurus_figure(f, base) for f in section.figures],
    )
    if section.code:
        lines = "".join(
            '<span class="token-line"><span class="token plain">'
            f"{_esc(line)}</span><br></span>"
            for line in section.code.split("\n")
        )
        parts.append(
            '<div class="language-bash theme-code-block">'
            '<div class="codeBlockContent"><pre class="prism-code '
            'language-bash"><code class="codeBlockLines">'
            f"{lines}</code></pre>"
            '<div class="buttonGroup"><button type="button">Copy</button>'
            "</div></div></div>",
        )
    if section.table:
        parts.append(_mkdocs_table(section))
    if section.note:
        parts.append(
            '<div class="theme-admonition theme-admonition-note alert">'
            '<div class="admonitionHeading"><span class="admonitionIcon">'
            '<svg viewBox="0 0 14 16"><path d="M6 5h2v2H6z"></path></svg>'
            f"</span>{_DOCUSAURUS_NOTE[lang]}</div>"
            '<div class="admonitionContent">'
            f"<p>{_render(section.note, resolve)}</p></div></div>",
        )
    return "\n".join(parts)


def build_docusaurus_like(
    spec: DocSpec,
    out: Path,
    *,
    base: str = "/",
) -> Path:
    """Write the documentation as a Docusaurus build served under ``base``."""
    out.mkdir(parents=True, exist_ok=True)
    _write_figures(spec, out / "assets" / "images", "{name}-{h}.{suffix}")
    title = _esc(spec.title)
    chapters = _chapters(spec)
    safe = html.escape(base, quote=True)
    navbar = (
        '<nav class="navbar"><a class="navbar__brand" '
        f'href="{safe}">{title}</a></nav>'
    )
    (out / "index.html").write_text(
        f'<!DOCTYPE html><html lang="{spec.lang}" dir="ltr"><head>'
        f'<meta charset="utf-8"><title>{title}</title></head><body>'
        f'{navbar}<div class="main-wrapper"><main></main></div>'
        "</body></html>\n",
        encoding="utf-8",
    )
    sidebar = "".join(
        f'<li><a class="menu__link" href="{safe}docs/{c[0].slug}/">'
        f"{_esc(c[0].title)}</a></li>"
        for c in chapters
    )
    for chapter in chapters:
        head = chapter[0]
        resolve = _resolver(spec, head.slug, base + "docs/{chapter}/#{slug}")
        body = "\n".join(
            _docusaurus_section(s, resolve, spec.lang, base) for s in chapter
        )
        page = (
            f'<!DOCTYPE html><html lang="{spec.lang}" dir="ltr"><head>'
            f'<meta charset="utf-8"><title>{_esc(head.title)} | {title}'
            '</title><meta name="generator" content="Docusaurus v3.9.2">'
            f"</head><body>{navbar}"
            '<div class="main-wrapper"><aside '
            'class="theme-doc-sidebar-container"><ul class="menu__list">'
            f'{sidebar}</ul></aside><main class="docMainContainer">'
            '<article><nav class="theme-doc-breadcrumbs"><ul>'
            f'<li><a href="{safe}">Home</a></li></ul></nav>'
            '<div class="theme-doc-markdown markdown"><header>'
            f"<h1>{_esc(head.title)}</h1></header>\n{body}\n</div>"
            f'<nav class="pagination-nav"><a href="{safe}">Home</a></nav>'
            '</article></main></div><footer class="footer">'
            "<p>\xa9 Copyright 2026.</p></footer></body></html>\n"
        )
        (out / "docs" / head.slug).mkdir(parents=True, exist_ok=True)
        (out / "docs" / head.slug / "index.html").write_text(
            page,
            encoding="utf-8",
        )
    return out


def _not_found(what: str) -> ValueError:
    msg = f"not found in the spec: {what!r}"
    return ValueError(msg)


def map_texts(spec: DocSpec, change: Callable[[str], str]) -> DocSpec:
    """Apply a change to titles, paragraphs, bullets, notes and cells."""
    sections = tuple(
        replace(
            section,
            title=change(section.title),
            paragraphs=tuple(map(change, section.paragraphs)),
            bullets=tuple(map(change, section.bullets)),
            note=change(section.note) if section.note else None,
            table=tuple(tuple(map(change, row)) for row in section.table),
        )
        for section in spec.sections
    )
    return replace(spec, sections=sections)


def replace_text(spec: DocSpec, old: str, new: str) -> DocSpec:
    """Replace a text in titles, paragraphs, bullets, notes and cells."""
    changed = map_texts(spec, lambda text: text.replace(old, new))
    if changed == spec:
        raise _not_found(old)
    return changed


def drop_paragraph(spec: DocSpec, prefix: str) -> DocSpec:
    """Remove the paragraph that starts with ``prefix``."""
    sections = tuple(
        replace(
            section,
            paragraphs=tuple(
                p for p in section.paragraphs if not p.startswith(prefix)
            ),
        )
        for section in spec.sections
    )
    if sections == spec.sections:
        raise _not_found(prefix)
    return replace(spec, sections=sections)


def _index(spec: DocSpec, slug: str) -> int:
    for index, section in enumerate(spec.sections):
        if section.slug == slug:
            return index
    raise _not_found(slug)


def drop_section(spec: DocSpec, slug: str) -> DocSpec:
    """Remove the section with this slug."""
    index = _index(spec, slug)
    sections = spec.sections[:index] + spec.sections[index + 1 :]
    return replace(spec, sections=sections)


def move_section(spec: DocSpec, slug: str, *, after: str) -> DocSpec:
    """Move the section with this slug right after another section."""
    moved = spec.sections[_index(spec, slug)]
    remaining = drop_section(spec, slug)
    target = _index(remaining, after) + 1
    sections = (
        *remaining.sections[:target],
        moved,
        *remaining.sections[target:],
    )
    return replace(spec, sections=sections)


def _replace_section(spec: DocSpec, slug: str, **changes: object) -> DocSpec:
    """Replace fields of the section with this slug."""
    index = _index(spec, slug)
    changed = replace(spec.sections[index], **changes)
    sections = (*spec.sections[:index], changed, *spec.sections[index + 1 :])
    return replace(spec, sections=sections)


def add_paragraph(spec: DocSpec, slug: str, text: str) -> DocSpec:
    """Append a paragraph to the section with this slug."""
    paragraphs = spec.sections[_index(spec, slug)].paragraphs
    return _replace_section(spec, slug, paragraphs=(*paragraphs, text))


def number_titles(spec: DocSpec) -> DocSpec:
    """Start each title with its section number, as a numbered source does.

    Every builder then numbers the titles, those it writes as paragraphs
    with ``heading_as_text`` too, as a Word export or a hand numbered
    Markdown source does.
    """
    numbers = section_numbers(spec)
    sections = tuple(
        replace(section, title=f"{numbers[section.slug]} {section.title}")
        for section in spec.sections
    )
    return replace(spec, sections=sections)


def demote_heading(spec: DocSpec, slug: str) -> DocSpec:
    """Write the title of the section with this slug as a paragraph."""
    return _replace_section(spec, slug, heading_as_text=True)


def set_level(spec: DocSpec, slug: str, level: int) -> DocSpec:
    """Give the section with this slug another heading level."""
    return _replace_section(spec, slug, level=level)


def unlist(spec: DocSpec, slug: str) -> DocSpec:
    """Write each bullet of the section with this slug as a paragraph."""
    return _replace_section(spec, slug, bullets_as_text=True)


def untable(spec: DocSpec, slug: str) -> DocSpec:
    """Write each row of the table of the section as a paragraph."""
    return _replace_section(spec, slug, table_as_text=True)


def uncode(spec: DocSpec, slug: str) -> DocSpec:
    """Write the code of the section with this slug as a paragraph."""
    return _replace_section(spec, slug, code_as_text=True)


SPLIT_WHOLE = (
    "First half of a long paragraph about the installer. "
    "Second half of the same paragraph about settings."
)
SPLIT_PARTS = (
    "First half of a long paragraph about the installer.",
    "Second half of the same paragraph about settings.",
)

PARAGRAPHS = [
    f"Paragraph number {w} talks about topic {w}."
    for w in ("one", "two", "three", "four")
]


def make_document(
    *blocks: Block,
    chrome_keys: frozenset[str] = frozenset(),
) -> Document:
    """Return an English HTML document made of the given blocks."""
    return Document(
        path=Path("doc"),
        format="html",
        blocks=list(blocks),
        parts=["index.html"],
        lang="en",
        declared_lang="en",
        generator="generic",
        normalization=Counter(),
        chrome=[],
        chrome_keys=chrome_keys,
        warnings=[],
    )


def make_heading(
    text: str,
    level: int,
    *,
    file: str = "index.html",
    page: int | None = None,
    outline: bool = False,
) -> Block:
    """Return a normalized heading, of a page of a PDF if a page is given.

    A heading found by a bookmark has `FROM_OUTLINE` as its source, else a
    heading of a PDF page has `FROM_STYLE`; an HTML heading has none.
    """
    if outline:
        attrs = {HEADING_SOURCE: FROM_OUTLINE}
    elif page is not None:
        attrs = {HEADING_SOURCE: FROM_STYLE}
    else:
        attrs = {}
    block = make_block(text, BlockKind.HEADING, file=file, level=level)
    return replace(block, source=SourceRef(file, page=page), attrs=attrs)


def make_audit_document(
    *blocks: Block,
    fmt: Literal["html", "pdf"] = "html",
    parts: Sequence[str] | None = None,
    **facts: object,
) -> Document:
    """Return an English document of the given blocks, as the audit reads it.

    Args:
        blocks: The blocks, in reading order; their section paths are set.
        fmt: Format of the document: an HTML site ``doc`` or ``doc.pdf``.
        parts: Its pages, by default one: ``index.html`` or ``1``.
        facts: Other fields of the document, such as ``html_pages``.

    Returns:
        The document, tagged if it is a PDF.
    """
    pdf = fmt == "pdf"
    default = ["1"] if pdf else ["index.html"]
    doc = replace(
        make_document(*blocks),
        path=Path("doc.pdf" if pdf else "doc"),
        format=fmt,
        parts=list(default if parts is None else parts),
        tagged=True if pdf else None,
    )
    assign_section_paths(doc.blocks)
    return replace(doc, **facts)


def with_figures(spec: DocSpec) -> DocSpec:
    """Give four sections of the English sample a figure each.

    Args:
        spec: A documentation with the sections of `SAMPLE_EN`.

    Returns:
        The documentation with its figures.
    """
    figures = {
        "installation": Figure(
            "setup",
            "Steps of the installer",
            "Installer steps",
        ),
        "projects": Figure(
            "dashboard",
            "Dashboard of a project",
            "Project dashboard",
            "jpeg",
        ),
        "command-line": Figure(
            "pipeline",
            "Pipeline of a command",
            "Command pipeline",
            "svg",
        ),
        "limits": Figure("quota", "Requests per client", "Quota chart"),
    }
    for slug, figure in figures.items():
        spec = _replace_section(spec, slug, figures=(figure,))
    return spec


FIGURES_EN = with_figures(SAMPLE_EN)


def figure_defect(spec: DocSpec, name: str, defect: Defect) -> DocSpec:
    """Plant a defect in the figure with this name."""
    for section in spec.sections:
        for figure in section.figures:
            if figure.name == name:
                figures = tuple(
                    replace(f, defect=defect) if f is figure else f
                    for f in section.figures
                )
                return _replace_section(spec, section.slug, figures=figures)
    raise _not_found(name)


def plant(site: Path, file: str, old: str, new: str) -> None:
    """Replace the one occurrence of a text in a file of a built site.

    Args:
        site: The folder of the site.
        file: The path of the file, relative to the site.
        old: The text to replace, found exactly once in the file.
        new: The text that replaces it.

    Raises:
        ValueError: If the file does not hold ``old`` exactly once.
    """
    path = site / file
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        msg = f"{old!r} found {count} times in {file}, expected once"
        raise ValueError(msg)
    path.write_text(text.replace(old, new), encoding="utf-8")


def rule_counts(
    findings: Iterable[Finding],
    *,
    minimum: Severity = Severity.INFO,
) -> Counter[str]:
    """Count the findings of each rule, from a severity up.

    Args:
        findings: The findings of a result; a group counts as one.
        minimum: The lowest severity counted.

    Returns:
        The number of findings of each rule.
    """
    return Counter(f.rule for f in findings if f.severity >= minimum)


def make_image_block(ref: ImageRef, *, file: str = "index.html") -> Block:
    """Return an image block, as an extractor would produce it."""
    return Block(
        kind=BlockKind.IMAGE,
        text="",
        raw_text="",
        source=SourceRef(file),
        image=ref,
    )


def _data_image_block(image: Image.Image) -> Block:
    """Return the block of a PNG kept in memory, as a data URL keeps it."""
    data = encode(image)
    ref = raster_ref(data, name="a.png", source=PixelSource(data=data))
    return make_image_block(ref)


def image_result(*, scale: float = 0.4) -> CompareResult:
    """Compare a paragraph and an image to the same with the image shrunk.

    Args:
        scale: Factor of both sides of the new image.

    Returns:
        The result, with one `image.degraded` finding whose sides both have
        pixels.
    """
    original = drawing("setup")
    return compare_documents(
        make_document(make_block(PARAGRAPHS[0]), _data_image_block(original)),
        make_document(
            make_block(PARAGRAPHS[0]),
            _data_image_block(scaled(original, scale)),
        ),
        Config(),
    )


def audit_result() -> AuditResult:
    """Audit a one-page site whose one paragraph holds a dead reference.

    Returns:
        The result, with one major `structure.unresolved_reference` finding
        whose side marks the ``??``.
    """
    doc = make_audit_document(
        make_block("See Section ?? for the limits."),
        html_pages=(HtmlPageFacts("index.html", "en", "Home"),),
    )
    return audit_document(doc, Config())


def image_audit_result() -> AuditResult:
    """Audit the images of a page whose one image is blurred.

    Returns:
        The result, with one `image.blurry` finding whose side has pixels.
    """
    block = _data_image_block(blurred(drawing("setup"), 2.0))
    doc = make_audit_document(block)
    return audit_document(doc, Config(checks=("images",)))
