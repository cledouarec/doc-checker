# Copyright 2026 Christophe Le Douarec
"""Scaling guards and the benchmark tool."""

import subprocess  # ruff: ignore[suspicious-subprocess-import] - runs our own script
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Final

import lxml.html

from doc_checker.compare.engine import compare_documents
from doc_checker.config import Config
from doc_checker.extract import load_document
from doc_checker.extract.html.dom import mark_titles
from doc_checker.extract.html.facts import page_ids
from doc_checker.extract.html.ordering import nav_links, resolve_href
from doc_checker.extract.html.profiles import GLOBAL_TITLES
from doc_checker.extract.html.site import HtmlPage
from doc_checker.model.document import BlockKind
from doc_checker.pipeline import run_audit, run_compare
from tests.builders import (
    DocSpec,
    Section,
    build_sphinx_like,
    make_block,
    make_document,
    map_texts,
    parts_tables,
)
from tests.pdf_builders import build_reportlab_pdf, filler_sections

if TYPE_CHECKING:
    from collections.abc import Callable

    import pytest

SCALE_MAX: Final = 8.0  # A size four times larger costs at most this.
ROOT: Final = Path(__file__).parents[1]


def cost(call: Callable[[], object]) -> float:
    """Return the best processor time of two calls."""
    times = []
    for _ in range(2):
        start = time.process_time()
        call()
        times.append(time.process_time() - start)
    return min(times)


def bench_spec(sections: int) -> DocSpec:
    """Return a documentation of a number of filler sections."""
    texts = filler_sections(sections, paragraphs=8, sentences=4)
    return DocSpec(
        title="Bench",
        lang="en",
        sections=tuple(
            Section(
                f"Part {index}",
                f"part-{index}",
                1 if index % 5 == 0 else 2,
                paragraphs=tuple(texts[index]),
            )
            for index in range(sections)
        ),
    )


def test_pdf_extraction_scales_linearly(tmp_path: Path) -> None:
    """Four times the pages cost at most eight times the time."""
    small = build_reportlab_pdf(bench_spec(10), tmp_path / "small.pdf")
    large = build_reportlab_pdf(bench_spec(40), tmp_path / "large.pdf")
    ratio = cost(lambda: load_document(large, Config())) / cost(
        lambda: load_document(small, Config()),
    )
    assert ratio <= SCALE_MAX


def test_html_extraction_scales_linearly(tmp_path: Path) -> None:
    """Four times the pages of a site cost at most eight times the time."""
    small = build_sphinx_like(bench_spec(160), tmp_path / "small")
    large = build_sphinx_like(bench_spec(640), tmp_path / "large")
    ratio = cost(lambda: load_document(large, Config())) / cost(
        lambda: load_document(small, Config()),
    )
    assert ratio <= SCALE_MAX


def _edited(spec: DocSpec) -> DocSpec:
    """Return the documentation with one word changed in every text."""
    return map_texts(spec, lambda text: text.replace(" the ", " a ", 1))


def test_compare_scales_linearly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Comparing four times larger builds costs at most eight times more."""
    # The time of this process must hold the reading of the inputs.
    monkeypatch.setattr("doc_checker.pipeline.PARALLEL_MIN_BYTES", 10**18)
    sites = {
        size: (
            build_sphinx_like(bench_spec(size), tmp_path / f"r{size}"),
            build_sphinx_like(
                _edited(bench_spec(size)),
                tmp_path / f"n{size}",
            ),
        )
        for size in (80, 320)
    }
    ratio = cost(lambda: run_compare(*sites[320], Config())) / cost(
        lambda: run_compare(*sites[80], Config()),
    )
    assert ratio <= SCALE_MAX


def test_audit_scales_linearly(tmp_path: Path) -> None:
    """Auditing a four times larger site costs at most eight times more."""
    small = build_sphinx_like(bench_spec(40), tmp_path / "small")
    large = build_sphinx_like(bench_spec(160), tmp_path / "large")
    ratio = cost(lambda: run_audit(large, Config())) / cost(
        lambda: run_audit(small, Config()),
    )
    assert ratio <= SCALE_MAX


def test_benchmark_tool_runs(tmp_path: Path) -> None:
    """The benchmark tool measures the stages of a run."""
    pdf = build_reportlab_pdf(bench_spec(2), tmp_path / "doc.pdf")
    done = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] - our own script, our own file
        [sys.executable, "tools/benchmark.py", "audit", str(pdf)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr
    for stage in ("extract (audit facts)", "structure", "text", "reports"):
        assert stage in done.stdout


def _benchmark(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the benchmark tool with arguments."""
    return subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] - our own script, our own file
        [sys.executable, "tools/benchmark.py", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )


def test_benchmark_tool_compares_into_a_folder(tmp_path: Path) -> None:
    """The benchmark tool compares two documents and writes in -o."""
    ref = build_reportlab_pdf(bench_spec(2), tmp_path / "ref.pdf")
    new = build_reportlab_pdf(_edited(bench_spec(2)), tmp_path / "new.pdf")
    out = tmp_path / "reports"
    done = _benchmark(
        "compare",
        str(ref),
        str(new),
        "--no-images",
        "-o",
        str(out),
    )
    assert done.returncode == 0, done.stderr
    for stage in ("extract ref", "extract new", "compare", "reports"):
        assert stage in done.stdout
    assert (out / "report.html").is_file()


def test_benchmark_tool_rejects_a_wrong_number_of_paths(
    tmp_path: Path,
) -> None:
    """The benchmark tool exits with 2 when compare gets one path."""
    done = _benchmark("compare", str(tmp_path))
    assert done.returncode == 2


def _image_page(folder: Path, count: int) -> Path:
    """Write a page of many images of one missing file."""
    folder.mkdir()
    images = "".join(
        f'<img src="a.png" alt="i{i}"> w{i} ' for i in range(count)
    )
    (folder / "index.html").write_text(
        '<!DOCTYPE html><html lang="en"><head><title>I</title></head>'
        f"<body><main><p>{images}</p></main></body></html>",
        encoding="utf-8",
    )
    return folder


def test_many_images_on_one_page_scale_linearly(tmp_path: Path) -> None:
    """Four times the images of a page cost at most eight times the time."""
    small = _image_page(tmp_path / "small", 4_000)
    large = _image_page(tmp_path / "large", 16_000)
    ratio = cost(lambda: load_document(large, Config())) / cost(
        lambda: load_document(small, Config()),
    )
    assert ratio <= SCALE_MAX


def _page(folder: Path, head: str, body: str) -> Path:
    """Write a site of one page, of extra head elements and a body."""
    folder.mkdir()
    (folder / "index.html").write_text(
        f'<!DOCTYPE html><html lang="en"><head><title>P</title>{head}</head>'
        f"<body>{body}</body></html>",
        encoding="utf-8",
    )
    return folder


def _nested_admonitions(folder: Path, cells: int) -> Path:
    """Write a page of admonition tables under 200 nested admonitions."""
    table = (
        '<table><tr><td class="icon">Note</td>'
        '<td class="content">x</td></tr></table>'
    )
    blocks = '<div class="admonitionblock note">' * 200
    return _page(
        folder,
        "",
        f"<main>{blocks}{table * cells}{'</div>' * 200}</main>",
    )


def test_nested_admonitions_scale_linearly(tmp_path: Path) -> None:
    """Four times the icon cells of nested admonitions cost at most 8 times."""
    small = _nested_admonitions(tmp_path / "small", 4_000)
    large = _nested_admonitions(tmp_path / "large", 16_000)
    ratio = cost(lambda: load_document(large, Config())) / cost(
        lambda: load_document(small, Config()),
    )
    assert ratio <= SCALE_MAX


def _quotation(folder: Path, paragraphs: int) -> Path:
    """Write a texinfo page of a quotation of paragraphs led by a bold word."""
    return _page(
        folder,
        '<meta name="Generator" content="makeinfo 7.2">',
        '<blockquote class="quotation">'
        f"{'<p><b>Note</b> word</p>' * paragraphs}</blockquote>",
    )


def test_long_quotations_scale_linearly(tmp_path: Path) -> None:
    """Four times the paragraphs of a quotation cost at most eight times."""
    small = _quotation(tmp_path / "small", 10_000)
    large = _quotation(tmp_path / "large", 40_000)
    ratio = cost(lambda: load_document(large, Config())) / cost(
        lambda: load_document(small, Config()),
    )
    assert ratio <= SCALE_MAX


def _marked_twice(folder: Path, count: int) -> Path:
    """Write a page that two selectors of the texinfo group find often."""
    marks = (
        '<meta name="generator" content="makeinfo 7.2">'
        '<div class="top-level-extent"></div>'
    )
    return _page(folder, "", f"<main>{marks * count}<p>Text.</p></main>")


def test_groups_of_selectors_scale_linearly(tmp_path: Path) -> None:
    """Four times what each selector of a group finds costs at most 8 times."""
    small = _marked_twice(tmp_path / "small", 10_000)
    large = _marked_twice(tmp_path / "large", 40_000)
    ratio = cost(lambda: load_document(large, Config())) / cost(
        lambda: load_document(small, Config()),
    )
    assert ratio <= SCALE_MAX


def _identified(count: int) -> lxml.html.HtmlElement:
    """Parse a page of elements with an id and anchors with a name."""
    pairs = "".join(
        f'<p id="p{i}">w</p><a name="a{i}">x</a>' for i in range(count)
    )
    return lxml.html.fromstring(f"<html><body>{pairs}</body></html>")


def test_identifiers_scale_linearly() -> None:
    """Four times the ids and names of a page cost at most eight times."""
    small, large = _identified(10_000), _identified(40_000)
    ratio = cost(lambda: page_ids(large)) / cost(lambda: page_ids(small))
    assert ratio <= SCALE_MAX


def test_nested_navigation_is_walked_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each link of nested navigation menus is read once, in order."""
    resolved: list[str] = []

    def recorded(href: str, page_rel: str) -> str | None:
        resolved.append(href)
        return resolve_href(href, page_rel)

    monkeypatch.setattr(
        "doc_checker.extract.html.ordering.resolve_href",
        recorded,
    )
    links = "".join(f'<a href="p{i}.html">{i}</a>' for i in range(100))
    nested = f"{'<div class=menu>' * 200}{links}{'</div>' * 200}"
    page = HtmlPage(
        "index.html",
        lxml.html.fromstring(f"<html><body>{nested}</body></html>"),
    )
    found = nav_links(page, ["div.menu"])
    assert found == [f"p{i}.html" for i in range(100)]
    assert len(resolved) == 100


def _table_pair(rows: int) -> Callable[[], object]:
    """Return the comparison of a long table, a third of its rows changed."""
    ref_text, new_text, _ = parts_tables(rows, 0.3)
    ref = make_document(make_block(ref_text, BlockKind.TABLE))
    new = make_document(make_block(new_text, BlockKind.TABLE))
    return lambda: compare_documents(ref, new, Config())


def test_a_modified_long_table_scales_linearly() -> None:
    """Four times the rows of a changed table cost at most eight times."""
    ratio = cost(_table_pair(4_000)) / cost(_table_pair(1_000))
    assert ratio <= SCALE_MAX


def _nested_titles(depth: int) -> lxml.html.HtmlElement:
    """Parse nested block titles over 20,000 inline elements and a block."""
    titles = '<div class="callout-title-container">' * depth
    body = f"{titles}{'<span>w</span>' * 20_000}<p>x</p>{'</div>' * depth}"
    return lxml.html.fromstring(
        f"<html><body><main>{body}</main></body></html>",
    )


def test_nested_titles_are_marked_once() -> None:
    """Two hundred levels of nested titles cost at most eight times one."""
    deep, flat = _nested_titles(200), _nested_titles(1)
    ratio = cost(lambda: mark_titles(deep, GLOBAL_TITLES)) / cost(
        lambda: mark_titles(flat, GLOBAL_TITLES),
    )
    assert ratio <= SCALE_MAX
