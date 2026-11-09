# Copyright 2026 Christophe Le Douarec
"""Tests for the end to end runs of the comparison and the audit."""

import errno
import os
import signal
import subprocess  # ruff: ignore[suspicious-subprocess-import] - runs our own script
import sys
import tempfile
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, NoReturn, Self

import pytest

from doc_checker.config import Config
from doc_checker.errors import ConfigError, InputError
from doc_checker.model.findings import Severity
from doc_checker.pipeline import (
    check_wordlist_path,
    input_size,
    load_pair,
    make_output_folder,
    run_audit,
    run_compare,
    write_wordlist,
)
from doc_checker.report.markdown import render_markdown
from tests.builders import (
    FIGURES_EN,
    SAMPLE_EN,
    build_mkdocs_like,
    build_sphinx_like,
)
from tests.pdf_builders import build_reportlab_pdf

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping
    from pathlib import Path

    from doc_checker.compare.engine import CompareResult


IMAGES_ONLY = Config(checks=("images",))
# A comparison that is killed once its two reading processes are started;
# it prints their process numbers first.
KILLED_COMPARE = """
import multiprocessing
import os
import signal
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from doc_checker import pipeline
from doc_checker.config import Config


class KilledPool(ProcessPoolExecutor):
    pools = 0

    def submit(self, *args, **kwargs):
        future = super().submit(*args, **kwargs)
        KilledPool.pools += 1
        if KilledPool.pools == 2:
            for child in multiprocessing.active_children():
                print(child.pid, flush=True)
            os.kill(os.getpid(), signal.SIGKILL)
        return future


if __name__ == "__main__":
    pipeline.PARALLEL_MIN_BYTES = 0
    pipeline.ProcessPoolExecutor = KilledPool
    pipeline.run_compare(Path(sys.argv[1]), Path(sys.argv[2]), Config())
"""


def test_print_profile_raises_the_resolution_needed(tmp_path: Path) -> None:
    """Images at 102 dpi are minor for the screen and major for print."""
    pdf = build_reportlab_pdf(FIGURES_EN, tmp_path / "doc.pdf")
    for cfg, severity in (
        (IMAGES_ONLY, Severity.MINOR),
        (replace(IMAGES_ONLY, profile="print"), Severity.MAJOR),
    ):
        findings = run_audit(pdf, cfg).findings
        low = {f.severity for f in findings if f.rule == "image.low_dpi"}
        assert low == {severity}


def test_write_wordlist(tmp_path: Path) -> None:
    """One word per line, UTF-8, with newline ends; empty for no word."""
    path = tmp_path / "w.txt"
    assert write_wordlist(("caf\xe9", "Zorblax"), path) == path
    assert path.read_bytes() == "caf\xe9\nZorblax\n".encode()
    write_wordlist((), path)
    assert path.read_bytes() == b""
    missing = tmp_path / "missing" / "w.txt"
    with pytest.raises(ConfigError, match="missing"):
        write_wordlist(("a",), missing)


def test_check_wordlist_path(tmp_path: Path) -> None:
    """A list goes in a folder that exists or that the run creates."""
    out = tmp_path / "a" / "b" / "out"
    for path in (
        tmp_path / "w.txt",
        out / "w.txt",
        tmp_path / "a" / "w.txt",
        out / ".." / "w.txt",
    ):
        check_wordlist_path(path, out)
    (tmp_path / "file").write_text("", encoding="utf-8")
    for path, code in (
        (tmp_path, errno.EISDIR),
        (tmp_path / "missing" / "w.txt", errno.ENOENT),
        (out / "sub" / "w.txt", errno.ENOENT),
        (tmp_path / "file" / "w.txt", errno.ENOTDIR),
    ):
        message = f"cannot write the word list to {path}: {os.strerror(code)}"
        with pytest.raises(ConfigError) as caught:
            check_wordlist_path(path, out)
        assert str(caught.value) == message


def test_make_output_folder(tmp_path: Path) -> None:
    """The output folder and those above it are made; a file refuses."""
    out = tmp_path / "a" / "b"
    make_output_folder(replace(Config(), output=out))
    make_output_folder(replace(Config(), output=out))
    assert out.is_dir()
    (tmp_path / "file").write_text("", encoding="utf-8")
    blocked = tmp_path / "file" / "out"
    with pytest.raises(ConfigError, match="cannot write reports to"):
        make_output_folder(replace(Config(), output=blocked))


def _sites(tmp_path: Path) -> tuple[Path, Path]:
    """Build two small sites of the same documentation."""
    return (
        build_sphinx_like(SAMPLE_EN, tmp_path / "ref"),
        build_mkdocs_like(SAMPLE_EN, tmp_path / "new"),
    )


class FailingPool:
    """A pool that reads in this process, and fails on some inputs.

    Attributes:
        failures: The error of the reading of each input that fails.
        refused: The inputs whose process cannot start.
        spoiled: What happens to the document file of some inputs once it
            is written, before the command reads it.
        results: The files where the documents were asked to be written.
        terminated: Whether the pool was told to stop its process.
    """

    def __init__(
        self,
        failures: Mapping[Path, BaseException],
        refused: Collection[Path],
        spoiled: Mapping[Path, Callable[[Path], object]],
    ) -> None:
        """Make a pool that fails on some inputs."""
        self.failures = failures
        self.refused = refused
        self.spoiled = spoiled
        self.results: list[Path] = []
        self.terminated = False

    def __enter__(self) -> Self:
        """Return the pool."""
        return self

    def __exit__(self, *_: object) -> None:
        """Do nothing: the pool has no process."""

    def submit(
        self,
        function: Callable[[Path, Config, Path], Path],
        path: Path,
        cfg: Config,
        result: Path,
    ) -> Future[Path]:
        """Read an input here, or fail as its reading process.

        Raises:
            OSError: If the process of the input cannot start.
        """
        if path in self.refused:
            raise OSError(errno.EAGAIN, os.strerror(errno.EAGAIN))
        self.results.append(result)
        future: Future[Path] = Future()
        if path in self.failures:
            future.set_exception(self.failures[path])
        else:
            future.set_result(function(path, cfg, result))
            if path in self.spoiled:
                self.spoiled[path](result)
        return future

    def terminate_workers(self) -> None:
        """Remember that the pool was told to stop its process."""
        self.terminated = True


def _failing(
    monkeypatch: pytest.MonkeyPatch,
    failures: Mapping[Path, BaseException],
    refused: Collection[Path] = (),
    spoiled: Mapping[Path, Callable[[Path], object]] | None = None,
) -> list[FailingPool]:
    """Read inputs of any size in pools that fail on some inputs.

    Returns:
        The pools made by the run, in order, once it has made them.
    """
    pools: list[FailingPool] = []

    def make(*_: object, **__: object) -> FailingPool:
        pools.append(FailingPool(failures, refused, spoiled or {}))
        return pools[-1]

    monkeypatch.setattr("doc_checker.pipeline.ProcessPoolExecutor", make)
    monkeypatch.setattr("doc_checker.pipeline.PARALLEL_MIN_BYTES", 0)
    return pools


def test_parallel_loading_gives_the_same_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two processes give the findings and the report of one."""
    ref, new = _sites(tmp_path)
    serial = run_compare(ref, new, Config(output=tmp_path / "a"))
    monkeypatch.setattr("doc_checker.pipeline.PARALLEL_MIN_BYTES", 0)
    parallel = run_compare(ref, new, Config(output=tmp_path / "b"))
    assert parallel.findings == serial.findings
    first = render_markdown(serial, tmp_path / "a.md", None).read_bytes()
    second = render_markdown(parallel, tmp_path / "b.md", None).read_bytes()
    assert first == second


def test_small_inputs_are_loaded_in_this_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Small inputs never start a process."""

    def refuse(*_: object, **__: object) -> None:
        raise AssertionError

    monkeypatch.setattr("doc_checker.pipeline.ProcessPoolExecutor", refuse)
    ref, new = _sites(tmp_path)
    assert run_compare(ref, new, Config()).stats.matched > 0


def test_input_error_of_a_worker_keeps_its_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreadable input fails the same way in a worker."""
    good = build_reportlab_pdf(SAMPLE_EN, tmp_path / "good.pdf")
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.7\nnot a pdf")
    with pytest.raises(InputError) as serial:
        run_compare(good, bad, Config())
    monkeypatch.setattr("doc_checker.pipeline.PARALLEL_MIN_BYTES", 0)
    with pytest.raises(InputError) as parallel:
        run_compare(good, bad, Config())
    assert str(parallel.value) == str(serial.value)


def test_broken_worker_is_an_input_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reading process that stops is an unreadable input."""
    ref, new = _sites(tmp_path)
    stopped = BrokenProcessPool("stopped")
    _failing(monkeypatch, {ref: stopped, new: stopped})
    with pytest.raises(InputError) as caught:
        run_compare(ref, new, Config())
    message = f"cannot read {ref}: the reading process stopped"
    assert str(caught.value) == message


def test_broken_new_worker_names_the_new_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A process of the new input that stops names the new input."""
    ref, new = _sites(tmp_path)
    _failing(monkeypatch, {new: BrokenProcessPool("stopped")})
    with pytest.raises(InputError) as caught:
        run_compare(ref, new, Config())
    message = f"cannot read {new}: the reading process stopped"
    assert str(caught.value) == message


@pytest.mark.parametrize(
    ("side", "error"),
    [
        (0, BrokenProcessPool("stopped")),
        (1, BrokenProcessPool("stopped")),
        (0, InputError("unreadable")),
        (1, KeyboardInterrupt()),
    ],
)
def test_a_failed_reading_stops_both_processes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    side: int,
    error: BaseException,
) -> None:
    """When an input cannot be read, no reading process is left running."""
    sites = _sites(tmp_path)
    pools = _failing(monkeypatch, {sites[side]: error})
    with pytest.raises((InputError, KeyboardInterrupt)):
        run_compare(*sites, Config())
    assert [pool.terminated for pool in pools] == [True, True]


def test_read_inputs_let_their_processes_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inputs read without error let their processes end by themselves."""
    ref, new = _sites(tmp_path)
    temp = _temporary_folder(tmp_path, monkeypatch)
    pools = _failing(monkeypatch, {})
    assert run_compare(ref, new, Config()).stats.matched > 0
    assert [pool.terminated for pool in pools] == [False, False]
    # The documents went through a folder of the temporary folder.
    results = [result for pool in pools for result in pool.results]
    assert [result.parent.parent for result in results] == [temp, temp]
    assert list(temp.iterdir()) == []


def _temporary_folder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Make the temporary folder of this process an empty folder.

    Returns:
        The folder.
    """
    temp = tmp_path / "temp"
    temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp))
    return temp


def write_part_and_die(result: Path) -> NoReturn:
    """Write the start of a document, then end the process at once."""
    result.write_bytes(b"\x80\x05\x95")
    os._exit(1)


class RealPool(ProcessPoolExecutor):
    """A pool of processes that records the files of its documents.

    Its process dies while it writes the document of the input it is told,
    or writes it to a folder that does not exist.

    Attributes:
        results: The files where the documents were asked to be written.
        dying: The input whose process dies while it writes its document.
        unwritable: The input whose document cannot be written.
    """

    def __init__(
        self,
        *args: object,
        results: list[Path],
        dying: Path | None = None,
        unwritable: Path | None = None,
        **kwargs: object,
    ) -> None:
        """Make the pool."""
        super().__init__(*args, **kwargs)
        self.results = results
        self.dying = dying
        self.unwritable = unwritable

    def submit(
        self,
        fn: Callable[..., object],
        /,
        *args: object,
        **kwargs: object,
    ) -> Future[object]:
        """Read an input in the process, or die writing its document."""
        path, cfg, result = args
        self.results.append(result)
        if path == self.dying:
            return super().submit(write_part_and_die, result)
        if path == self.unwritable:
            missing = result.parent / "missing" / result.name
            return super().submit(fn, path, cfg, missing)
        return super().submit(fn, *args, **kwargs)


def _real_pools(
    monkeypatch: pytest.MonkeyPatch,
    dying: Path | None = None,
    unwritable: Path | None = None,
) -> list[Path]:
    """Read inputs of any size in real pools that record their files.

    Returns:
        The files where the documents are asked to be written, once asked.
    """
    results: list[Path] = []
    monkeypatch.setattr(
        "doc_checker.pipeline.ProcessPoolExecutor",
        partial(
            RealPool,
            results=results,
            dying=dying,
            unwritable=unwritable,
        ),
    )
    monkeypatch.setattr("doc_checker.pipeline.PARALLEL_MIN_BYTES", 0)
    return results


@pytest.mark.parametrize("dying", [0, 1])
def test_a_worker_dying_in_its_result_leaves_no_folder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dying: int,
) -> None:
    """A process that dies while it writes its document stops its input."""
    sites = _sites(tmp_path)
    temp = _temporary_folder(tmp_path, monkeypatch)
    results = _real_pools(monkeypatch, sites[dying])
    with pytest.raises(InputError) as caught:
        load_pair(*sites, Config())
    message = f"cannot read {sites[dying]}: the reading process stopped"
    assert str(caught.value) == message
    assert [result.parent.parent for result in results] == [temp, temp]
    assert list(temp.iterdir()) == []


def _report(result: CompareResult, path: Path) -> bytes:
    """Return the Markdown report of a comparison."""
    return render_markdown(result, path, None).read_bytes()


@pytest.mark.parametrize("unwritable", [0, 1])
def test_a_document_that_cannot_be_written_is_read_here(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unwritable: int,
) -> None:
    """When a process cannot write its document, both inputs are read here."""
    sites = _sites(tmp_path)
    serial = run_compare(*sites, Config())
    temp = _temporary_folder(tmp_path, monkeypatch)
    results = _real_pools(monkeypatch, unwritable=sites[unwritable])
    parallel = run_compare(*sites, Config())
    assert len(results) == 2
    first = _report(serial, tmp_path / "a.md")
    assert _report(parallel, tmp_path / "b.md") == first
    assert list(temp.iterdir()) == []


def _remove(result: Path) -> None:
    """Remove a document file."""
    result.unlink()


def _cut(result: Path) -> None:
    """Keep the first half of a document file."""
    data = result.read_bytes()
    result.write_bytes(data[: len(data) // 2])


def _empty(result: Path) -> None:
    """Empty a document file."""
    result.write_bytes(b"")


@pytest.mark.parametrize("spoil", [_remove, _cut, _empty])
@pytest.mark.parametrize("side", [0, 1])
def test_a_spoiled_document_file_is_read_here(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    spoil: Callable[[Path], None],
    side: int,
) -> None:
    """A document file gone or cut before it is read: inputs read here."""
    sites = _sites(tmp_path)
    serial = run_compare(*sites, Config())
    temp = _temporary_folder(tmp_path, monkeypatch)
    pools = _failing(monkeypatch, {}, spoiled={sites[side]: spoil})
    parallel = run_compare(*sites, Config())
    first = _report(serial, tmp_path / "a.md")
    assert _report(parallel, tmp_path / "b.md") == first
    assert [pool.terminated for pool in pools] == [True, True]
    assert list(temp.iterdir()) == []


def _figure_sites(tmp_path: Path) -> tuple[Path, Path]:
    """Build two sites with figures, of two generators."""
    return (
        build_sphinx_like(FIGURES_EN, tmp_path / "ref"),
        build_mkdocs_like(FIGURES_EN, tmp_path / "new"),
    )


def _figure_pdfs(tmp_path: Path) -> tuple[Path, Path]:
    """Build two PDFs with figures, the first with a warning."""
    return (
        build_reportlab_pdf(
            FIGURES_EN,
            tmp_path / "ref.pdf",
            scanned_page=True,
        ),
        build_reportlab_pdf(FIGURES_EN, tmp_path / "new.pdf"),
    )


@pytest.mark.parametrize("build", [_figure_sites, _figure_pdfs])
def test_documents_are_the_same_in_two_processes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    build: Callable[[Path], tuple[Path, Path]],
) -> None:
    """The documents read in two processes equal those read here."""
    pair = build(tmp_path)
    monkeypatch.setattr("doc_checker.pipeline.PARALLEL_MIN_BYTES", 10**18)
    here = load_pair(*pair, Config())
    results = _real_pools(monkeypatch)
    apart = load_pair(*pair, Config())
    assert len(results) == 2
    assert apart == here
    assert all(document.blocks for document in apart)
    if build is _figure_pdfs:
        assert apart[0].warnings == [
            "1 page(s) without text, probably scanned: 10",
        ]


@pytest.mark.parametrize(
    "error",
    [OSError(errno.EAGAIN, "no process"), NotImplementedError()],
)
def test_pools_that_cannot_start_read_here(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    """When no pool of processes can be made, the inputs are read here."""

    def refuse(*_: object, **__: object) -> NoReturn:
        raise error

    sites = _sites(tmp_path)
    here = load_pair(*sites, Config())
    monkeypatch.setattr("doc_checker.pipeline.ProcessPoolExecutor", refuse)
    monkeypatch.setattr("doc_checker.pipeline.PARALLEL_MIN_BYTES", 0)
    assert load_pair(*sites, Config()) == here


def test_a_process_that_cannot_start_reads_here(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a process cannot start, the other stops; both are read here."""
    sites = _sites(tmp_path)
    here = load_pair(*sites, Config())
    temp = _temporary_folder(tmp_path, monkeypatch)
    pools = _failing(monkeypatch, {}, refused=(sites[1],))
    assert load_pair(*sites, Config()) == here
    assert [pool.terminated for pool in pools] == [True, True]
    assert list(temp.iterdir()) == []


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="SIGKILL and its exit status are POSIX",
)
def test_reading_processes_end_with_a_killed_command(tmp_path: Path) -> None:
    """A command killed while it reads leaves no reading process behind."""
    ref, new = _sites(tmp_path)
    script = tmp_path / "killed.py"
    script.write_text(KILLED_COMPARE, encoding="utf-8")
    temp = tmp_path / "temp"
    temp.mkdir()
    try:
        # The output ends once every process that holds it has ended.
        done = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] - our own script, our own files
            [sys.executable, str(script), str(ref), str(new)],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
            env={**os.environ, "TMPDIR": str(temp)},
        )
    except subprocess.TimeoutExpired as expired:
        # The output of an expired run is bytes, whatever the mode.
        for pid in bytes(expired.stdout or b"").split():
            os.kill(int(pid), signal.SIGKILL)
        pytest.fail("a reading process outlived its command")
    assert done.returncode == -signal.SIGKILL
    assert len(done.stdout.split()) == 2
    assert list(temp.iterdir()) == []


def test_input_size(tmp_path: Path) -> None:
    """A file counts its bytes, a site those of its pages, at any depth."""
    (tmp_path / "a.html").write_text("x" * 10, encoding="utf-8")
    (tmp_path / "b.HTM").write_text("y" * 5, encoding="utf-8")
    (tmp_path / "c.png").write_bytes(b"z" * 100)
    deeper = tmp_path / "sub" / "deeper"
    deeper.mkdir(parents=True)
    (deeper / "d.htm").write_text("w" * 3, encoding="utf-8")
    # A folder with the suffix of a page counts the pages it holds.
    (tmp_path / "x.html").mkdir()
    (tmp_path / "x.html" / "e.html").write_text("v" * 2, encoding="utf-8")
    assert input_size(tmp_path / "a.html") == 10
    assert input_size(tmp_path / "x.html") == 2
    assert input_size(tmp_path) == 20
    assert input_size(tmp_path / "missing") == 0


def test_input_size_leaves_out_a_broken_link(tmp_path: Path) -> None:
    """A link to no file is no page."""
    (tmp_path / "a.html").write_text("x" * 10, encoding="utf-8")
    try:
        (tmp_path / "b.html").symlink_to(tmp_path / "missing.html")
    except OSError, NotImplementedError:
        pytest.skip("symbolic links are not supported here")
    assert input_size(tmp_path) == 10
