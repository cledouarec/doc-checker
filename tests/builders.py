# Copyright 2026 Christophe Le Douarec
"""Builders of test blocks, documents and documentation sites."""

import html
import re
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from doc_checker.model.document import Block, BlockKind, Document, SourceRef
from doc_checker.text.normalize import loose_key, normalize_display, strict_key


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


@dataclass(frozen=True)
class Section:
    """A heading and the content that follows it."""

    title: str
    slug: str
    level: int
    paragraphs: tuple[str, ...] = ()
    bullets: tuple[str, ...] = ()
    code: str | None = None
    table: tuple[tuple[str, ...], ...] = ()
    note: str | None = None


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
                    "configuration before you start."
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
                    "settings."
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
                    "subcommand."
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
                    "retried later."
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
                    "Installez le paquet avec la configuration "
                    "par défaut avant de commencer."
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
                    "mêmes réglages."
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


_DOUBLE_QUOTED = re.compile(r'"([^"]*)"')
_FR_COLON = re.compile(r" :")
_FR_HIGH_PUNCT = re.compile(r" ([;!?])")
_DIGIT_GROUP = re.compile(r"(\d) (\d{3})")


def _smart(text: str, lang: str) -> str:
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


def _sphinx_section(section: Section, lang: str) -> str:
    tag = f"h{section.level}"
    title = _esc(_smart(section.title, lang))
    parts = [
        f'<section id="{section.slug}">',
        (
            f'<{tag}>{title}<a class="headerlink" href="#{section.slug}" '
            f'title="Link to this heading">\xb6</a></{tag}>'
        ),
    ]
    parts += [f"<p>{_esc(_smart(p, lang))}</p>" for p in section.paragraphs]
    if section.bullets:
        items = "".join(
            f"<li><p>{_esc(_smart(b, lang))}</p></li>" for b in section.bullets
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
            f"<p>{_esc(_smart(section.note, lang))}</p></div>",
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
) -> Path:
    """Write the documentation as a Sphinx Read the Docs theme build."""
    out.mkdir(parents=True, exist_ok=True)
    chapters = _chapters(spec)
    if split == "single":
        nav = [(f"#{c[0].slug}", c[0].title) for c in chapters]
        body = "\n".join(_sphinx_section(s, spec.lang) for s in spec.sections)
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
        body = "\n".join(_sphinx_section(s, spec.lang) for s in chapter)
        page = _sphinx_page(spec, chapter[0].title, body, nav)
        (out / f"{chapter[0].slug}.html").write_text(page, encoding="utf-8")
    return out


def _mkdocs_section(section: Section) -> str:
    tag = f"h{section.level}"
    parts = [
        (
            f'<{tag} id="{section.slug}">{_esc(section.title)}'
            f'<a class="headerlink" href="#{section.slug}" '
            f'title="Permanent link">\xb6</a></{tag}>'
        ),
    ]
    parts += [f"<p>{_esc(p)}</p>" for p in section.paragraphs]
    if section.bullets:
        items = "".join(f"<li>{_esc(b)}</li>" for b in section.bullets)
        parts.append(f"<ul>{items}</ul>")
    if section.code:
        parts.append(
            '<div class="highlight"><pre><span></span>'
            f"<code>{_esc(section.code)}</code></pre></div>",
        )
    if section.table:
        header, *rows = section.table
        head = "".join(f"<th>{_esc(c)}</th>" for c in header)
        body = "".join(
            "<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in row) + "</tr>"
            for row in rows
        )
        parts.append(
            f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody>"
            "</table>",
        )
    if section.note:
        parts.append(
            '<div class="admonition note"><p class="admonition-title">Note</p>'
            f"<p>{_esc(section.note)}</p></div>",
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
        body = "\n".join(_mkdocs_section(s) for s in chapter)
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


def _not_found(what: str) -> ValueError:
    msg = f"not found in the spec: {what!r}"
    return ValueError(msg)


def replace_text(spec: DocSpec, old: str, new: str) -> DocSpec:
    """Replace a text in titles, paragraphs, bullets, notes and cells."""

    def swap(text: str) -> str:
        return text.replace(old, new)

    sections = tuple(
        replace(
            section,
            title=swap(section.title),
            paragraphs=tuple(map(swap, section.paragraphs)),
            bullets=tuple(map(swap, section.bullets)),
            note=swap(section.note) if section.note else None,
            table=tuple(tuple(map(swap, row)) for row in section.table),
        )
        for section in spec.sections
    )
    if sections == spec.sections:
        raise _not_found(old)
    return replace(spec, sections=sections)


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


def add_paragraph(spec: DocSpec, slug: str, text: str) -> DocSpec:
    """Append a paragraph to the section with this slug."""
    index = _index(spec, slug)
    section = spec.sections[index]
    added = replace(section, paragraphs=(*section.paragraphs, text))
    sections = (*spec.sections[:index], added, *spec.sections[index + 1 :])
    return replace(spec, sections=sections)


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
