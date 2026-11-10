# Copyright 2026 Christophe Le Douarec
"""Run the search of the tables of a PDF in a bounded process.

The search of `tables` runs in a process of its own, started afresh on every
system, whose CPU time, memory and duration are limited for the whole
document, where the system allows it, and which is killed past them: no file
can make the search run longer. The limits are read when `read_tables` is
called, `MAX_TABLE_PAGE_CHARS` from `tables`.
"""

import math
import os
import signal
import threading
from contextlib import ExitStack, suppress
from multiprocessing import get_context, parent_process
from pathlib import Path
from typing import TYPE_CHECKING, Final

from doc_checker.extract.pdf import tables

if TYPE_CHECKING:
    from collections.abc import Sequence
    from multiprocessing.connection import Connection

    from doc_checker.extract.pdf.tables import TablePage, TableReading

try:
    import resource
except ImportError:  # Windows has no resource module.
    resource = None

# CPU seconds that the reading process of `read_tables` may spend on the
# tables of a document, from the start of the search: a base, then for each
# page searched a share, and a share for each of its characters, as the work of
# pdfplumber grows with the text of a page. The documents measured keep 5 times
# what they need at least, in all and on their pages of text: the PGF manual,
# the ruled samples, and tables of 6 pt, of 4.5 pt and of 20,000 characters a
# page, which take 0.14 to 0.32 s a page. Pages that would keep pdfminer for
# minutes stop after a few seconds.
TABLES_BASE_SECONDS: Final = 2.0
TABLES_SECONDS_PER_PAGE: Final = 0.3
TABLES_SECONDS_PER_CHAR: Final = 0.00012
# Most bytes of address space that the reading process may add to what it holds
# when the search starts: the PGF manual adds 154 MB.
TABLES_MEMORY_BYTES: Final = 2 * 1024**3
# How many times its CPU seconds the command waits for the reading process, by
# the clock, before it stops it: the bound where the system cannot limit the
# CPU time of a process, and for a process that waits rather than computes.
TABLES_WALL_FACTOR: Final = 3
# Seconds that the command waits for a reading process to end once it has
# given its tables, before it kills it.
_END_SECONDS: Final = 10.0
_STOPPED: Final = (
    f"{tables.NOT_READ}the table reader went over its limits or stopped"
)
_NOT_STARTED: Final = f"{tables.NOT_READ}the table reader could not start"


def _end_with_parent() -> None:
    """End this process as soon as its parent has ended."""
    parent = parent_process()
    if parent is not None:
        parent.join()
    os._exit(1)


def _address_space() -> int | None:
    """Return the bytes of address space of this process, where they show."""
    with suppress(OSError, ValueError, IndexError):
        pages = Path("/proc/self/statm").read_text(encoding="ascii").split()
        return int(pages[0]) * os.sysconf("SC_PAGE_SIZE")
    return None


def _lower_limit(kind: int, soft: int, hard: int) -> None:
    """Lower a resource limit of this process, if it can.

    Args:
        kind: The resource, such as ``resource.RLIMIT_CPU``.
        soft: The soft limit, which never rises above the hard one.
        hard: The hard limit, which never rises above the hard limit in
            force.
    """
    if resource is None:
        return
    current = resource.getrlimit(kind)[1]
    if current != resource.RLIM_INFINITY:
        hard = min(hard, current)
    with suppress(ValueError, OSError):
        resource.setrlimit(kind, (min(soft, hard), hard))


def _end_at_limit(signum: int, _: object) -> None:
    """End this process at once, as a shell tells a death by a signal.

    Args:
        signum: The signal received.
    """
    os._exit(128 + signum)


def _limit_table_reader(seconds: float, memory: int) -> None:
    """Bound what this process may still spend, where the system allows it.

    Its CPU time may grow by `seconds`, rounded up: there the system signals
    it, and the process ends at the signal, without a core file. macOS sends
    the signal once and never kills; Linux sends it, then kills the process
    one second of CPU time later, at the hard limit, if it still runs. Its
    address space may grow by `memory` bytes, where it shows, past which the
    process fails to allocate. The clock of the command bounds the process
    where the system sets no limit.

    Args:
        seconds: The CPU seconds that the process may still spend.
        memory: The bytes of address space that it may still add.
    """
    if resource is None:
        return
    usage = resource.getrusage(resource.RUSAGE_SELF)
    signal.signal(signal.SIGXCPU, _end_at_limit)
    limit = math.ceil(usage.ru_utime + usage.ru_stime + seconds)
    _lower_limit(resource.RLIMIT_CPU, limit, limit + 1)
    size = _address_space()
    if size is not None:
        _lower_limit(resource.RLIMIT_AS, size + memory, size + memory)


def _bounded_tables(
    path: Path,
    pages: Sequence[TablePage],
    page_count: int,
    limits: tuple[float, int],
) -> TableReading:
    """Find the tables of a document within limits, in the reading process.

    Args:
        path: The PDF file.
        pages: The pages to search, as the reader found them.
        page_count: The number of pages that the reader found in the file.
        limits: The CPU seconds and the bytes of memory that the search may
            spend, as `_limit_table_reader` takes them.

    Returns:
        What `find_tables` returns; or, if memory ran out, wherever it did,
        only the warning that the reader went over its limits: the search
        ends there.
    """
    _limit_table_reader(*limits)
    try:
        return tables.find_tables(path, pages, page_count=page_count)
    except MemoryError:
        return {}, [_STOPPED], {}


def _table_reader(
    sender: Connection,
    path: Path,
    pages: Sequence[TablePage],
    page_count: int,
    limits: tuple[float, int],
) -> None:
    """Run the reading process of the tables, and send what it finds.

    The process ignores the interrupts of the terminal, which reach every
    process of the command: the command stops it. It ends as soon as its
    parent ends, however the parent ends, as `_end_with_parent` tells.

    Args:
        sender: The end of the pipe to the command, which this closes.
        path: The PDF file.
        pages: The pages to search, as the reader found them.
        page_count: The number of pages that the reader found in the file.
        limits: The CPU seconds and the bytes of memory that the search may
            spend.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    threading.Thread(target=_end_with_parent, daemon=True).start()
    with sender:
        try:
            found = _bounded_tables(path, pages, page_count, limits)
        except Exception as exc:  # ruff: ignore[blind-except] - the command raises it again
            sender.send((None, exc))
        else:
            sender.send((found, None))


def _received(receiver: Connection, seconds: float) -> TableReading:
    """Wait for the tables that a reading process sends.

    Args:
        receiver: The end of the pipe from the process.
        seconds: How long to wait, by the clock.

    Returns:
        The tables, or the warning that the reader went over its limits or
        stopped, if they do not come in time or the pipe closes first. The
        error that the search raised in the process, if any, is raised
        again.
    """
    if not receiver.poll(seconds):
        return {}, [_STOPPED], {}
    try:
        found, error = receiver.recv()
    except EOFError, OSError:
        return {}, [_STOPPED], {}
    if error is not None:
        raise error
    return found


def read_tables(
    path: Path,
    pages: Sequence[TablePage],
    *,
    page_count: int,
) -> TableReading:
    """Find the ruled tables of some pages of a PDF, in a bounded process.

    The search of `find_tables` runs in a process of its own, started afresh
    on every system, which may spend `TABLES_BASE_SECONDS` of CPU time, plus
    `TABLES_SECONDS_PER_PAGE` for each page searched and
    `TABLES_SECONDS_PER_CHAR` for each character that the page holds, as
    `TablePage.chars` counts them, up to `MAX_TABLE_PAGE_CHARS`, and add
    `TABLES_MEMORY_BYTES` of memory, where the system limits them; the
    command waits for it `TABLES_WALL_FACTOR` times its CPU seconds at most,
    by the clock. However this call ends, the process is killed if it still
    runs, at once if it passed a limit or the call fails or is interrupted,
    after a short wait if it has sent its tables; then it is waited for, so
    that it is gone when this returns. It ends, too, if the command ends. It
    may start from a reading process of a comparison, as a process of its
    own. The process imports the main script again, as every process that
    multiprocessing spawns: a script that calls this function must run its
    own code under ``if __name__ == "__main__":``.

    Args:
        path: The PDF file.
        pages: The pages to search, as the reader found them.
        page_count: The number of pages that the reader found in the file.

    Returns:
        What `find_tables` returns; or, if the process passes a limit or
        stops, no table and only the warning ``tables not read: the table
        reader went over its limits or stopped``; or, if it cannot start,
        only the warning ``tables not read: the table reader could not
        start``: the search never runs without its limits. The constants are
        read when this is called. A page that is not one of the pages of the
        file raises a `ValueError`, as `check_pages` tells.
    """
    tables.check_pages(pages, page_count)
    tables.require_pdfplumber()
    if not pages:
        return {}, [], {}
    seconds = TABLES_BASE_SECONDS + sum(
        TABLES_SECONDS_PER_PAGE
        + TABLES_SECONDS_PER_CHAR
        * min(page.chars, tables.MAX_TABLE_PAGE_CHARS)
        for page in pages
    )
    context = get_context("spawn")
    with ExitStack() as stack:
        try:
            receiver, sender = context.Pipe(duplex=False)
            stack.enter_context(receiver)
            with sender:
                process = context.Process(
                    target=_table_reader,
                    args=(
                        sender,
                        path,
                        tuple(pages),
                        page_count,
                        (seconds, TABLES_MEMORY_BYTES),
                    ),
                    name="doc-checker tables",
                )
                process.start()
        # A daemonic process may not start a process: multiprocessing
        # asserts it.
        except OSError, NotImplementedError, AssertionError:
            return {}, [_NOT_STARTED], {}
        # Run last first: kill, then wait, then close, however this ends.
        stack.callback(process.close)
        stack.callback(process.join)
        stack.callback(process.kill)
        found = _received(receiver, TABLES_WALL_FACTOR * seconds)
        if found[1] != [_STOPPED]:
            process.join(_END_SECONDS)
        return found
