# Copyright 2026 Christophe Le Douarec
"""Scaling guards and the benchmark tool."""

import subprocess  # ruff: ignore[suspicious-subprocess-import] - runs our own script
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Final

from doc_checker.config import Config
from doc_checker.extract import load_document
from doc_checker.pipeline import run_audit, run_compare
from tests.builders import DocSpec, Section, build_sphinx_like, map_texts
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
