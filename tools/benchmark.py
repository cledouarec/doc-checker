# Copyright 2026 Christophe Le Douarec
"""Measure the time and the peak memory of the stages of a run.

Usage: ``uv run python tools/benchmark.py {extract,audit,compare} PATH
[PATH] [--no-images] [--pdf-tables] [-o DIR]``. One line per stage: its
seconds and the peak memory of the process so far.
"""

import argparse
import sys
import tempfile
import time
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from doc_checker.audit import engine
from doc_checker.compare.engine import compare_documents
from doc_checker.config import Config, PdfConfig
from doc_checker.extract import load_document
from doc_checker.pipeline import write_reports

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

try:
    import resource
except ImportError:  # Windows has no resource module.
    resource = None

# Audit stage name -> the names of the engine that run in it.
FAMILIES = {
    "structure": ("structure_drafts", "broken_link_drafts"),
    "a11y": ("accessibility_drafts",),
    "images": ("image_quality_drafts",),
    "text": ("text_drafts",),
}


def peak_memory() -> str:
    """Return the peak memory of the process, in MB, or ``n/a``."""
    if resource is None:
        return "     n/a"
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux counts kibibytes, macOS bytes.
    mebibytes = peak / 1048576 if sys.platform == "darwin" else peak / 1024
    return f"{mebibytes:7.0f}"


def show(stage: str, seconds: float) -> None:
    """Print the line of a stage."""
    print(f"{stage:<28} {seconds:8.2f} s   peak {peak_memory()} MB")


def timed[T](stage: str, call: Callable[[], T]) -> T:
    """Run a call and print the line of its stage.

    Returns:
        The result of the call.
    """
    start = time.perf_counter()
    result = call()
    show(stage, time.perf_counter() - start)
    return result


@contextmanager
def family_timers() -> Generator[dict[str, float]]:
    """Time the families of the audit by proxies around their functions.

    Yields:
        The seconds spent in each family, filled while the context runs.
    """
    spent = dict.fromkeys(FAMILIES, 0.0)
    originals: dict[str, Callable[..., object]] = {}

    def proxy(
        stage: str,
        function: Callable[..., object],
    ) -> Callable[..., object]:
        def measured(*args: object, **kwargs: object) -> object:
            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                spent[stage] += time.perf_counter() - start

        return measured

    for stage, names in FAMILIES.items():
        for name in names:
            originals[name] = getattr(engine, name)
            setattr(engine, name, proxy(stage, originals[name]))
    try:
        yield spent
    finally:
        for name, original in originals.items():
            setattr(engine, name, original)


def bench_extract(paths: Sequence[Path], cfg: Config) -> None:
    """Measure the extraction of a document."""
    timed("extract", lambda: load_document(paths[0], cfg))


def bench_audit(paths: Sequence[Path], cfg: Config) -> None:
    """Measure the stages of the audit of a document, then its reports."""
    doc = timed(
        "extract (audit facts)",
        lambda: load_document(paths[0], cfg, audit=True),
    )
    with family_timers() as spent:
        result = engine.audit_document(doc, cfg)
    for stage, seconds in spent.items():
        show(stage, seconds)
    timed("reports", lambda: write_reports(result, cfg))


def bench_compare(paths: Sequence[Path], cfg: Config) -> None:
    """Measure the stages of the comparison of two documents."""
    ref = timed("extract ref", lambda: load_document(paths[0], cfg))
    new = timed("extract new", lambda: load_document(paths[1], cfg))
    result = timed("compare", lambda: compare_documents(ref, new, cfg))
    timed("reports", lambda: write_reports(result, cfg))


def parse(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("extract", "audit", "compare"))
    parser.add_argument("paths", nargs="+", type=Path, metavar="PATH")
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="do not read the images",
    )
    parser.add_argument(
        "--pdf-tables",
        action="store_true",
        help="read the ruled tables of PDF documents (pdf-tables extra)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="folder of the reports (default: a temporary folder)",
    )
    args = parser.parse_args(argv)
    wanted = 2 if args.command == "compare" else 1
    if len(args.paths) != wanted:
        parser.error(f"{args.command} takes {wanted} PATH")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark.

    Args:
        argv: Command line arguments, ``sys.argv[1:]`` by default.

    Returns:
        The exit code.
    """
    args = parse(argv)
    runs = {
        "extract": bench_extract,
        "audit": bench_audit,
        "compare": bench_compare,
    }
    with ExitStack() as stack:
        output = args.output or Path(
            stack.enter_context(tempfile.TemporaryDirectory()),
        )
        cfg = replace(
            Config(),
            images=not args.no_images,
            output=output,
            pdf=PdfConfig(tables=args.pdf_tables),
        )
        runs[args.command](args.paths, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
