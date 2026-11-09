# Copyright 2026 Christophe Le Douarec
"""End to end runs shared by the command line and the tests."""

import errno
import os
import pickle  # ruff: ignore[suspicious-pickle-import] - only the documents of our own reading processes
import signal
import threading
from concurrent.futures import ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from contextlib import ExitStack, contextmanager, suppress
from multiprocessing import get_context, parent_process
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Final

from doc_checker.audit.engine import audit_document
from doc_checker.compare.engine import compare_documents
from doc_checker.errors import ConfigError, InputError
from doc_checker.extract import detect_format, load_document
from doc_checker.report.context import Thumbnails
from doc_checker.report.html import render_html
from doc_checker.report.markdown import render_markdown

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence
    from concurrent.futures import Future
    from multiprocessing.process import BaseProcess

    from doc_checker.audit.engine import AuditResult
    from doc_checker.compare.engine import CompareResult
    from doc_checker.config import Config
    from doc_checker.model.document import Document

# Size, in bytes, from which each of the two inputs of a comparison is read
# in a process of its own, as `input_size` tells it: below, the start of a
# process costs more than the reading of the input saves.
PARALLEL_MIN_BYTES: Final = 1_000_000
_PAGE_SUFFIXES: Final = frozenset({".html", ".htm"})
# Seconds that the command waits for its stopped reading processes to end,
# before it removes the folder where they write their documents.
_STOP_SECONDS: Final = 30.0
_SIDES: Final = ("ref", "new")


class _NoProcessError(Exception):
    """The reading processes cannot start here, or cannot hand over."""


def input_size(path: Path) -> int:
    """Return the size of an input, which tells how long it takes to read.

    Args:
        path: A file, or the folder of a site.

    Returns:
        The bytes of a file; for a folder, the bytes of its files with an
        HTML suffix, ``.html`` or ``.htm`` in any case, at any depth; 0 if
        the input cannot be read.
    """
    try:
        if not path.is_dir():
            return path.stat().st_size
        return sum(
            page.stat().st_size
            for page in path.rglob("*")
            if page.suffix.lower() in _PAGE_SUFFIXES and page.is_file()
        )
    except OSError:
        return 0


def _end_with(parent: BaseProcess, result: Path) -> None:
    """End this process as soon as its parent has ended.

    The file of its document goes first, then the folder of the files if it
    is empty: the parent can no longer remove them.

    Args:
        parent: The parent process.
        result: The file where this process writes its document.
    """
    parent.join()
    with suppress(OSError):
        result.unlink(missing_ok=True)
    with suppress(OSError):
        result.parent.rmdir()
    os._exit(1)


def _start_reading(result: Path) -> None:
    """Prepare a reading process, in the process itself.

    It ignores the interrupts of the terminal, which reach every process of
    the command: the command stops it. It ends as soon as the command ends,
    however the command ends, even killed, as `_end_with` tells: the process
    does not wait, forever, for the command to take its document.

    Args:
        result: The file where the process writes its document.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    parent = parent_process()
    if parent is not None:
        threading.Thread(
            target=_end_with,
            args=(parent, result),
            daemon=True,
        ).start()


def _read_to_file(path: Path, cfg: Config, result: Path) -> Path:
    """Load a document, in a reading process, and write it to a file.

    Only the name of the file goes back to the command, in a message too
    short to be cut: a process stopped while it writes its document leaves
    a part of the file, never a part of a message that the command would
    wait for the end of.

    Args:
        path: The input.
        cfg: Run configuration.
        result: The file to write.

    Returns:
        The file.

    Raises:
        _NoProcessError: If the file cannot be written, as on a full disk:
            the command then reads its inputs itself.
    """
    document = load_document(path, cfg)
    try:
        with result.open("wb") as stream:
            pickle.dump(document, stream, protocol=pickle.HIGHEST_PROTOCOL)
    except OSError as exc:
        raise _NoProcessError from exc
    return result


@contextmanager
def _reading_process(result: Path) -> Generator[ProcessPoolExecutor]:
    """Give a pool of one process, started afresh, to read an input.

    The process ends with the command, as `_start_reading` tells. If the
    body fails, or is interrupted, the process is stopped here rather than
    left to finish its reading. A Ctrl-C while the process starts, before it
    ignores the interrupts, may make it print a traceback; the command still
    ends as it does for any Ctrl-C.

    Args:
        result: The file where the process writes its document.

    Yields:
        The pool.
    """
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=get_context("spawn"),
        initializer=_start_reading,
        initargs=(result,),
    ) as pool:
        try:
            yield pool
        except BaseException:
            pool.terminate_workers()
            raise


def _document(future: Future[Path], path: Path) -> Document:
    """Return the document that a reading process wrote.

    Args:
        future: The reading of the input, which gives the file of its
            document.
        path: The input.

    Returns:
        The document.

    Raises:
        InputError: If the input cannot be read, or if its process stops
            before it gives the document.
        _NoProcessError: If the file of the document cannot be written, or
            read whole, as when it is removed: the command then reads its
            inputs itself.
    """
    try:
        result = future.result()
    except BrokenProcessPool:
        msg = f"cannot read {path}: the reading process stopped"
        raise InputError(msg) from None
    try:
        with result.open("rb") as stream:
            document: Document = pickle.load(stream)  # ruff: ignore[suspicious-pickle-usage] - written by our own reading process, in a folder of our own
    except (OSError, EOFError, pickle.UnpicklingError) as exc:
        raise _NoProcessError from exc
    return document


def _load_apart(
    ref_path: Path,
    new_path: Path,
    cfg: Config,
) -> tuple[Document, Document]:
    """Load the two builds of a comparison, each in a process of its own.

    Each process writes its document to a file of a temporary folder. The
    folder goes once both processes are done or stopped, however the
    reading ends.

    Args:
        ref_path: Reference build.
        new_path: New build.
        cfg: Run configuration.

    Returns:
        The reference document and the new document.

    Raises:
        _NoProcessError: If the folder cannot be made, a process cannot
            start, or a document cannot go through its file.
    """
    futures: list[Future[Path]] = []
    with ExitStack() as stack:
        try:
            folder = Path(
                stack.enter_context(
                    TemporaryDirectory(
                        prefix="doc-checker-",
                        ignore_cleanup_errors=True,
                    ),
                ),
            )
            # Run when the pools are closed, before the folder goes.
            stack.callback(wait, futures, timeout=_STOP_SECONDS)
            for side, path in zip(_SIDES, (ref_path, new_path), strict=True):
                result = folder / f"{side}.pickle"
                pool = stack.enter_context(_reading_process(result))
                futures.append(pool.submit(_read_to_file, path, cfg, result))
        except (OSError, NotImplementedError) as exc:
            raise _NoProcessError from exc
        return _document(futures[0], ref_path), _document(futures[1], new_path)


def load_pair(
    ref_path: Path,
    new_path: Path,
    cfg: Config,
) -> tuple[Document, Document]:
    """Load the two builds of a comparison, at the same time if both are large.

    When each input has at least `PARALLEL_MIN_BYTES`, as `input_size`
    tells, each is read in a process of its own, started afresh on every
    system; smaller inputs are read here, one after the other, and so are
    large ones where the processes cannot start, or cannot hand over their
    documents through the temporary folder. The documents are the same
    either way, and so is the error of an input that cannot be read: that of
    the reference comes first. A process that stops before it gives its
    document makes its input unreadable. When an input fails, or the run is
    interrupted, no reading process is left running.

    Args:
        ref_path: Reference build.
        new_path: New build.
        cfg: Run configuration.

    Returns:
        The reference document and the new document.
    """
    if (
        input_size(ref_path) >= PARALLEL_MIN_BYTES
        and input_size(new_path) >= PARALLEL_MIN_BYTES
    ):
        with suppress(_NoProcessError):
            return _load_apart(ref_path, new_path, cfg)
    return load_document(ref_path, cfg), load_document(new_path, cfg)


def run_compare(ref_path: Path, new_path: Path, cfg: Config) -> CompareResult:
    """Load two documentation builds and compare them.

    Two large builds are read in processes started afresh, as `load_pair`
    tells. Each process imports the main script again, so a script that
    calls this function must run its own code under
    ``if __name__ == "__main__":``.

    Args:
        ref_path: Reference build.
        new_path: New build.
        cfg: Run configuration.

    Returns:
        The comparison result.

    Raises:
        InputError: If the builds have different formats or cannot be read.
    """
    ref_format = detect_format(ref_path)
    new_format = detect_format(new_path)
    if ref_format != new_format:
        msg = (
            f"REF is {ref_format} but NEW is {new_format}: "
            "both must have the same format"
        )
        raise InputError(msg)
    ref, new = load_pair(ref_path, new_path, cfg)
    return compare_documents(ref, new, cfg)


def run_audit(path: Path, cfg: Config) -> AuditResult:
    """Load a documentation build with its audit facts and audit it.

    Args:
        path: The build, a site directory, an HTML file or a PDF.
        cfg: Run configuration.

    Returns:
        The audit result.
    """
    return audit_document(load_document(path, cfg, audit=True), cfg)


def _render_reports(
    result: CompareResult | AuditResult,
    cfg: Config,
) -> list[Path]:
    """Write the reports of the formats of a run, with shared thumbnails.

    Args:
        result: Comparison or audit result.
        cfg: Run configuration, with the output directory and formats.

    Returns:
        The written report files.
    """
    cfg.output.mkdir(parents=True, exist_ok=True)
    written = []
    with Thumbnails(result) as thumbnails:
        if "html" in cfg.formats:
            html = cfg.output / "report.html"
            written.append(render_html(result, html, thumbnails))
        if "md" in cfg.formats:
            markdown = cfg.output / "report.md"
            written.append(render_markdown(result, markdown, thumbnails))
    return written


def write_reports(
    result: CompareResult | AuditResult,
    cfg: Config,
) -> list[Path]:
    """Write the reports of a comparison or an audit in the output directory.

    Args:
        result: Comparison or audit result.
        cfg: Run configuration, with the output directory and formats.

    Returns:
        The written report files: ``report.html``, then ``report.md``,
        for the formats of the configuration. Both show the same
        thumbnails, made once.

    Raises:
        ConfigError: If the reports cannot be written.
    """
    try:
        written = _render_reports(result, cfg)
    except OSError as exc:
        msg = f"cannot write reports to {cfg.output}: {exc.strerror}"
        raise ConfigError(msg) from None
    return written


def check_wordlist_path(path: Path, output: Path) -> None:
    """Tell, before a run, whether a word list can be written to a path.

    Its folder must exist, or be the output folder or a folder above it,
    which the run creates before it writes the list.

    Args:
        path: The file of the word list.
        output: The output folder of the run.

    Raises:
        ConfigError: If the path is a folder, or if its folder does not
            exist and the run does not create it.
    """
    folder = path.parent
    if path.is_dir():
        reason = os.strerror(errno.EISDIR)
    elif folder.is_dir():
        return
    elif folder.exists():
        reason = os.strerror(errno.ENOTDIR)
    elif output.resolve().is_relative_to(folder.resolve()):
        return
    else:
        reason = os.strerror(errno.ENOENT)
    msg = f"cannot write the word list to {path}: {reason}"
    raise ConfigError(msg)


def make_output_folder(cfg: Config) -> None:
    """Create the output folder of a run, and the folders above it.

    Args:
        cfg: Run configuration, with the output folder.

    Raises:
        ConfigError: If the folder cannot be created.
    """
    try:
        cfg.output.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        msg = f"cannot write reports to {cfg.output}: {exc.strerror}"
        raise ConfigError(msg) from None


def write_wordlist(words: Sequence[str], path: Path) -> Path:
    """Write a list of words, one per line, in UTF-8.

    Args:
        words: The words, as `AuditResult.words` lists them.
        path: The file to write; it is written even when there is no word.

    Returns:
        The file written.

    Raises:
        ConfigError: If the file cannot be written.
    """
    try:
        path.write_text(
            "".join(f"{word}\n" for word in words),
            encoding="utf-8",
            newline="\n",
        )
    except OSError as exc:
        msg = f"cannot write the word list to {path}: {exc.strerror}"
        raise ConfigError(msg) from None
    return path
