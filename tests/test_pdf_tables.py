# Copyright 2026 Christophe Le Douarec
"""Tests for the ruled tables of PDF documents (pdf-tables extra)."""

import contextlib
import multiprocessing
import os
import signal
import subprocess  # ruff: ignore[suspicious-subprocess-import] - runs this interpreter on a fixed command
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pdfplumber
import pytest
from pdfplumber.table import TableFinder, TableSettings
from reportlab.lib.units import mm

from doc_checker.config import Config, PdfConfig
from doc_checker.errors import ConfigError
from doc_checker.extract import load_document
from doc_checker.extract.pdf.figures import MAX_PAGE_PATHS
from doc_checker.extract.pdf.reader import open_pdf, read_page
from doc_checker.extract.pdf.table_process import read_tables
from doc_checker.extract.pdf.tables import (
    PdfTable,
    TablePage,
    continues,
    find_tables,
    require_pdfplumber,
    rule_crossings,
)
from doc_checker.model.document import BlockKind
from doc_checker.pipeline import run_compare
from tests.builders import FIGURES_EN
from tests.pdf_builders import (
    SETTING_ROWS,
    ProseStyle,
    build_accent_table_pdf,
    build_actual_text_table_pdf,
    build_bad_checksum_table_pdf,
    build_busy_page_pdf,
    build_close_words_table_pdf,
    build_cropped_table_pdf,
    build_crowded_table_pdf,
    build_dense_paths_pdf,
    build_dense_table_pdf,
    build_digit_grid_pdf,
    build_drawn_grid_pdf,
    build_form_fanout_pdf,
    build_framed_code_pdf,
    build_labelled_grid_pdf,
    build_listing_beside_table_pdf,
    build_nested_forms_pdf,
    build_overflowing_table_pdf,
    build_prose_pdf,
    build_quoted_draws_pdf,
    build_reportlab_pdf,
    build_resource_reuse_pdf,
    build_ruled_grid_pdf,
    build_run_length_page_pdf,
    build_shared_cmap_fonts_pdf,
    build_split_table_pdf,
    build_table_pdf,
    build_turned_pdf,
    build_turned_table_pdf,
    build_two_column_table_pdf,
    build_unmapped_table_pdf,
    filler_sections,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from doc_checker.extract.pdf.tables import TableReading
    from doc_checker.model.document import Document

TABLES = Config(pdf=PdfConfig(tables=True))
A4 = (595.2756, 841.8898)
CROPPED = TablePage(0, A4[0] - 50, A4[1] - 50, (), 0)
RAW = TablePage(0, 595, 842, (), 0)  # The page of a PDF of raw objects.
# Most processor seconds that the count of a page beyond its budget takes:
# without the count, pdfminer reads each of these pages in 5 s or more.
QUICK_SECONDS = 1.0
STOPPED = "tables not read: the table reader went over its limits or stopped"
MODULE = "doc_checker.extract.pdf.tables"
PROCESS = "doc_checker.extract.pdf.table_process"
# An introduction that a table goes on, without a full stop.
SUCH_AS = (
    "The settings of the service are listed below with their default values,"
    " such as"
)
KEPT_AS_TEXT = (
    "1 table(s) kept as text, their cells not matching the text of their"
    " page: 1"
)


def _children() -> set[multiprocessing.process.BaseProcess]:
    """Return the processes of this one that still run."""
    return set(multiprocessing.active_children())


def _pages(pdf: Path) -> list[TablePage]:
    """Return every page of an A4 PDF without figures."""
    count = len(load_document(pdf, Config()).parts)
    return [TablePage(i, *A4, (), 0) for i in range(count)]


def _find(
    pdf: Path,
) -> tuple[dict[int, list[PdfTable]], list[str], dict[str, int]]:
    """Find the tables of every page of an A4 PDF without figures."""
    pages = _pages(pdf)
    return find_tables(pdf, pages, page_count=len(pages))


def test_tables_option_needs_the_extra(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without pdfplumber, asking for tables is a configuration error."""
    monkeypatch.setitem(sys.modules, "pdfplumber", None)
    pdf = build_table_pdf(tmp_path / "t.pdf", margin=25 * mm, grid=True)
    with pytest.raises(ConfigError, match="pdf-tables extra"):
        load_document(pdf, TABLES)
    with pytest.raises(ConfigError):
        require_pdfplumber()


def test_reading_processes_without_the_extra_fail_the_same(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the extra, a reading process fails as the command does."""
    shadow = tmp_path / "shadow" / "pdfplumber"
    shadow.mkdir(parents=True)
    (shadow / "__init__.py").write_text(
        'raise ImportError("not installed")\n',
        encoding="utf-8",
    )
    # The reading processes start with the paths of this one.
    monkeypatch.syspath_prepend(shadow.parent)
    monkeypatch.delitem(sys.modules, "pdfplumber", raising=False)
    pdf = build_cropped_table_pdf(tmp_path / "c.pdf")
    with pytest.raises(ConfigError) as serial:
        run_compare(pdf, pdf, TABLES)

    def here(*_: object, **__: object) -> None:
        raise AssertionError

    monkeypatch.setattr("doc_checker.pipeline.load_document", here)
    monkeypatch.setattr("doc_checker.pipeline.PARALLEL_MIN_BYTES", 0)
    with pytest.raises(ConfigError) as parallel:
        run_compare(pdf, pdf, TABLES)
    assert str(parallel.value) == str(serial.value)
    assert "pip install 'doc-checker[pdf-tables]'" in str(serial.value)


def test_base_install_never_imports_pdfplumber(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reading a PDF without the option leaves pdfplumber unimported."""
    monkeypatch.delitem(sys.modules, "pdfplumber", raising=False)
    pdf = build_table_pdf(tmp_path / "t.pdf", margin=25 * mm, grid=True)
    load_document(pdf, Config())
    assert "pdfplumber" not in sys.modules


def test_the_base_commands_never_import_pdfplumber(tmp_path: Path) -> None:
    """Extract and audit without the option leave pdfplumber unimported."""
    pdf = build_table_pdf(tmp_path / "t.pdf", margin=25 * mm, grid=True)
    code = (
        "import sys; from click.testing import CliRunner;"
        " from doc_checker.cli import main;"
        " pdf, out = sys.argv[1], sys.argv[2];"
        " codes = [CliRunner().invoke(main, args).exit_code for args in"
        " (['extract', pdf], ['audit', pdf, '-o', out])];"
        " print(codes, sorted(name for name in sys.modules"
        " if name.split('.')[0] in ('pdfplumber', 'pdfminer')))"
    )
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] - a fixed command of this interpreter
        [sys.executable, "-c", code, str(pdf), str(tmp_path / "out")],
        capture_output=True,
        check=True,
        encoding="utf-8",
    )
    assert result.stdout == "[0, 0] []\n"


def test_ruled_tables_are_found(tmp_path: Path) -> None:
    """A ruled table is found on each page, its header row first."""
    pdf = build_table_pdf(
        tmp_path / "t.pdf",
        margin=25 * mm,
        grid=True,
        repeat_header=True,
    )
    tables, notes, skipped = _find(pdf)
    assert (notes, skipped) == ([], {})
    assert len(tables) >= 3
    for found in tables.values():
        assert found[0].rows[0] == ("Option", "Default")
        assert len(found[0].columns) == 3


def test_unruled_tables_are_not_found(tmp_path: Path) -> None:
    """A table without rules stays text."""
    pdf = build_table_pdf(tmp_path / "t.pdf", margin=25 * mm)
    assert _find(pdf) == ({}, [], {})


def test_a_ruled_strip_is_no_table(tmp_path: Path) -> None:
    """A single row of ruled cells, each with text, is no table."""
    pdf = build_ruled_grid_pdf(tmp_path / "s.pdf", [("a", "b", "c", "d")])
    with pdfplumber.open(pdf) as plumbed:
        assert len(plumbed.pages[0].find_tables()) == 1
    assert _find(pdf) == ({}, [], {})


@pytest.mark.parametrize(
    ("rows", "table"),
    [
        ([("a", "b", ""), ("", "", "f")], True),
        ([("a", "", ""), ("", "", "f")], False),
        ([("a", ""), ("", "d")], True),
        ([("a", ""), ("", "")], False),
    ],
)
def test_half_of_the_cells_hold_text(
    tmp_path: Path,
    rows: list[tuple[str, ...]],
    table: bool,  # ruff: ignore[boolean-type-hint-positional-argument] - pytest parameter
) -> None:
    """A grid is a table when half of its cells at least hold text."""
    pdf = build_ruled_grid_pdf(tmp_path / "g.pdf", rows)
    tables = _find(pdf)[0]
    assert (tables[0][0].rows if table else tables) == (
        tuple(rows) if table else {}
    )


def test_frames_and_drawings_are_not_tables(tmp_path: Path) -> None:
    """A frame around code and a grid of empty cells are no tables."""
    pdf = build_framed_code_pdf(tmp_path / "f.pdf")
    assert _find(pdf) == ({}, [], {})


@pytest.mark.parametrize("labels", ["over", "against"])
def test_grids_under_labels_are_not_tables(
    tmp_path: Path,
    labels: Literal["over", "against"],
) -> None:
    """A grid whose labels lie over its lines is a drawing, not a table."""
    plot = build_labelled_grid_pdf(tmp_path / "p.pdf", labels=labels)
    assert _find(plot) == ({}, [], {})


def test_grids_with_padded_text_are_tables(tmp_path: Path) -> None:
    """The same grid is a table when its text keeps off its lines."""
    plain = build_labelled_grid_pdf(tmp_path / "g.pdf", labels=None)
    tables = _find(plain)[0]
    assert tables[0][0].rows == (
        ("north", "west", "east"),
        ("south", "up", "down"),
    )


def test_table_boxes_are_in_crop_box_coordinates(tmp_path: Path) -> None:
    """The box of a table is measured from the corner of the crop box."""
    pdf = build_cropped_table_pdf(tmp_path / "c.pdf")
    table = find_tables(pdf, [CROPPED], page_count=1)[0][0][0]
    expected = (50.0, 450.0, 250.0, 650.0)
    assert all(
        abs(found - wanted) <= 2.0
        for found, wanted in zip(table.box, expected, strict=True)
    )
    assert table.rows == (
        ("Option", "Default", "Unit"),
        ("timeout", "30", "s"),
    )
    assert len(table.columns) == 4
    assert abs(table.columns[0] - 50.0) <= 2.0
    assert abs(table.columns[-1] - 250.0) <= 2.0


def test_table_boxes_ignore_the_origin_of_the_media_box(
    tmp_path: Path,
) -> None:
    """A media box that starts below 0 moves no box of a table."""
    pdf = build_cropped_table_pdf(tmp_path / "m.pdf", media=(0, -100, *A4))
    table = find_tables(pdf, [CROPPED], page_count=1)[0][0][0]
    expected = (50.0, 450.0, 250.0, 650.0)
    assert all(
        abs(found - wanted) <= 2.0
        for found, wanted in zip(table.box, expected, strict=True)
    )


def test_tables_out_of_the_crop_box_are_not_found(tmp_path: Path) -> None:
    """A table that runs out of the crop box is no table."""
    pdf = build_cropped_table_pdf(tmp_path / "c.pdf", crop=(150, 50, *A4))
    page = TablePage(0, A4[0] - 150, A4[1] - 50, (), 0)
    assert find_tables(pdf, [page], page_count=1) == ({}, [], {})


def test_tables_over_a_figure_are_not_found(tmp_path: Path) -> None:
    """A table that overlaps a vector figure belongs to the figure."""
    pdf = build_cropped_table_pdf(tmp_path / "c.pdf")
    over = TablePage(
        0,
        CROPPED.width,
        CROPPED.height,
        ((40, 600, 80, 700),),
        0,
    )
    aside = TablePage(
        0,
        CROPPED.width,
        CROPPED.height,
        ((300, 50, 400, 90),),
        0,
    )
    assert find_tables(pdf, [over], page_count=1) == ({}, [], {})
    assert 0 in find_tables(pdf, [aside], page_count=1)[0]


def test_tables_of_unmapped_glyphs_are_not_found(tmp_path: Path) -> None:
    """A table whose text has no Unicode mapping stays text."""
    pdf = build_unmapped_table_pdf(tmp_path / "u.pdf")
    page = TablePage(0, 595, 842, (), 0)
    assert find_tables(pdf, [page], page_count=1) == ({}, [], {})


def test_reading_the_tables_says_nothing(tmp_path: Path) -> None:
    """The reading of the tables prints nothing, even on a damaged font."""
    pdf = build_unmapped_table_pdf(tmp_path / "u.pdf")
    code = (
        "import sys; from pathlib import Path;"
        " from doc_checker.extract.pdf.tables import TablePage, find_tables;"
        " page = TablePage(0, 595, 842, (), 0);"
        " print(find_tables(Path(sys.argv[1]), [page], page_count=1))"
    )
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] - a fixed command of this interpreter
        [sys.executable, "-c", code, str(pdf)],
        capture_output=True,
        check=True,
        encoding="utf-8",
    )
    assert (result.stdout, result.stderr) == ("({}, [], {})\n", "")


def test_turned_and_resized_pages_are_counted(tmp_path: Path) -> None:
    """A turned page, or one of another size, is counted, not searched."""
    turned = build_cropped_table_pdf(tmp_path / "t.pdf", rotate=90)
    flipped = TablePage(0, CROPPED.height, CROPPED.width, (), 0)
    assert find_tables(turned, [flipped], page_count=1) == (
        {},
        [],
        {"rotated": 1},
    )
    plain = build_cropped_table_pdf(tmp_path / "c.pdf")
    whole = TablePage(0, *A4, (), 0)
    assert find_tables(plain, [whole], page_count=1) == ({}, [], {"size": 1})


@pytest.mark.parametrize(
    ("limit", "reason"),
    [
        ("MAX_TABLE_PAGE_CHARS", "dense"),
        ("MAX_TABLE_PAGE_EDGES", "rules"),
        ("MAX_TABLE_PAGE_CROSSINGS", "rules"),
    ],
)
def test_pages_beyond_a_budget_are_counted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit: str,
    reason: str,
) -> None:
    """A page beyond a budget of the search is counted, not searched."""
    monkeypatch.setattr(f"doc_checker.extract.pdf.tables.{limit}", 10)
    pdf = build_table_pdf(tmp_path / "t.pdf", margin=25 * mm, grid=True)
    pages = _pages(pdf)
    assert _find(pdf) == ({}, [], {reason: len(pages)})


@pytest.mark.parametrize(
    ("lines", "skipped"),
    [(40, {}), (41, {"rules": 5}), (100, {"rules": 5})],
    ids=["1,600 crossings", "1,681 crossings", "10,000 crossings"],
)
def test_grids_past_the_crossings_are_counted_not_stopped(
    tmp_path: Path,
    lines: int,
    skipped: dict[str, int],
) -> None:
    """A grid at the most crossings is read in time; one past them, counted."""
    pdf = build_digit_grid_pdf(tmp_path / "g.pdf", lines=lines, pages=5)
    pages = _reader_pages(pdf)
    before = _children()
    tables, notes, counted = read_tables(pdf, pages, page_count=len(pages))
    assert (notes, counted) == ([], skipped)
    assert len(tables) == (0 if skipped else len(pages))
    assert _children() <= before


def test_crossings_are_those_of_pdfplumber(tmp_path: Path) -> None:
    """The rules counted are those that pdfplumber crosses, once merged."""
    rows = [("a", "b", "c", "d")] * 3
    boxed = build_ruled_grid_pdf(tmp_path / "b.pdf", rows, boxed=True)
    framed = build_framed_code_pdf(tmp_path / "f.pdf")
    settings = TableSettings()
    found = []
    for pdf in (boxed, framed):
        with pdfplumber.open(pdf) as plumbed:
            page = plumbed.pages[0]
            edges = TableFinder(page, settings).edges
            vertical = sum(edge["orientation"] == "v" for edge in edges)
            crossings = vertical * (len(edges) - vertical)
            assert rule_crossings(page, settings) == crossings
            found.append(crossings)
    # The 48 sides of the 12 boxes merge into 5 columns and 4 rows of rules.
    assert found[0] == 5 * 4


@pytest.mark.parametrize("find", [find_tables, read_tables])
@pytest.mark.parametrize("index", [-1, 1])
def test_pages_must_be_pages_of_the_file(
    tmp_path: Path,
    index: int,
    find: Callable[..., object],
) -> None:
    """A page that is not one of the pages of the file is refused."""
    pdf = build_cropped_table_pdf(tmp_path / "c.pdf")
    page = TablePage(index, CROPPED.width, CROPPED.height, (), 0)
    with pytest.raises(ValueError, match="not one of 1 pages"):
        find(pdf, [page], page_count=1)


@pytest.mark.parametrize(
    "build",
    [
        lambda out: build_busy_page_pdf(out, pairs=1_000_000),
        lambda out: build_form_fanout_pdf(out, calls=5_000),
        lambda out: build_form_fanout_pdf(out, calls=5_000, escaped=True),
        lambda out: build_form_fanout_pdf(
            out,
            calls=2_000,
            form_bytes=0,
            cmap_bytes=36_000,
        ),
        lambda out: build_form_fanout_pdf(out, calls=5_000, comment=True),
        lambda out: build_resource_reuse_pdf(out, calls=1_000),
        lambda out: build_busy_page_pdf(
            out,
            pairs=16 * 1_048_576,
            predictor=True,
        ),
        lambda out: build_form_fanout_pdf(
            out,
            calls=5_000,
            comment=True,
            inline=True,
        ),
        lambda out: build_form_fanout_pdf(
            out,
            calls=5_000,
            comment=True,
            ascii85="/F /A85",
        ),
        lambda out: build_form_fanout_pdf(
            out,
            calls=5_000,
            comment=True,
            ascii85="/F /A#385",
        ),
        lambda out: build_form_fanout_pdf(
            out,
            calls=5_000,
            comment=True,
            ascii85="/#46 /A85",
        ),
    ],
    ids=[
        "busy",
        "fan-out",
        "escaped",
        "font",
        "comment",
        "reused address",
        "predictor",
        "inline image",
        "ascii85 image",
        "escaped filter",
        "escaped key",
    ],
)
def test_pages_of_too_much_work_are_counted_quickly(
    tmp_path: Path,
    build: Callable[[Path], Path],
) -> None:
    """A page that pdfminer would take seconds to read is counted as dense."""
    pdf = build(tmp_path / "w.pdf")
    start = time.process_time()
    result = find_tables(pdf, [RAW], page_count=1)
    assert time.process_time() - start < QUICK_SECONDS
    assert result == ({}, [], {"dense": 1})


@pytest.mark.parametrize(
    ("depth", "cycle", "skipped"),
    [(8, False, {}), (9, False, {"dense": 1}), (2, True, {"dense": 1})],
)
def test_forms_within_forms_have_a_depth(
    tmp_path: Path,
    depth: int,
    cycle: bool,  # ruff: ignore[boolean-type-hint-positional-argument] - pytest parameter
    skipped: dict[str, int],
) -> None:
    """Forms drawn deeper than the limit, or in a cycle, are too much work."""
    pdf = build_nested_forms_pdf(tmp_path / "n.pdf", depth=depth, cycle=cycle)
    assert find_tables(pdf, [RAW], page_count=1) == ({}, [], skipped)


def test_words_that_read_do_draw_nothing(tmp_path: Path) -> None:
    """Do in a string or a comment is no drawing of a form."""
    pdf = build_quoted_draws_pdf(
        tmp_path / "q.pdf",
        form_bytes=100_000,
        lines=40,
    )
    assert find_tables(pdf, [RAW], page_count=1) == ({}, [], {})


def test_a_damaged_checksum_keeps_the_table(tmp_path: Path) -> None:
    """A content whose checksum is spoiled still gives its table."""
    pdf = build_bad_checksum_table_pdf(tmp_path / "b.pdf")
    tables, notes, skipped = find_tables(pdf, [RAW], page_count=1)
    assert (notes, skipped) == ([], {})
    assert tables[0][0].rows == (("Option", "Default"), ("timeout", "30"))


def test_tables_are_read_in_a_process_of_their_own(tmp_path: Path) -> None:
    """The bounded reading finds the tables, and leaves no process."""
    pdf = build_cropped_table_pdf(tmp_path / "c.pdf")
    before = _children()
    assert read_tables(pdf, [CROPPED], page_count=1) == find_tables(
        pdf,
        [CROPPED],
        page_count=1,
    )
    assert _children() <= before


def test_tables_are_read_from_a_reading_process(tmp_path: Path) -> None:
    """A reading process of a comparison starts the bounded reading too."""
    pdf = build_cropped_table_pdf(tmp_path / "c.pdf")
    before = _children()
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
        future = pool.submit(read_tables, pdf, [CROPPED], page_count=1)
        tables, notes, _ = future.result(timeout=120)
    assert notes == []
    assert tables[0][0].rows[0] == ("Option", "Default", "Unit")
    assert _children() <= before


@pytest.mark.parametrize(
    ("base", "factor"),
    [(0.5, 1_000.0), (1_000.0, 0.002)],
    ids=["processor", "clock"],
)
def test_hostile_documents_stop_at_the_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    base: float,
    factor: float,
) -> None:
    """A document that pdfminer would read for long is stopped: a warning."""
    if base < 1 and sys.platform == "win32":
        pytest.skip("Windows limits no processor time")
    monkeypatch.setattr(f"{PROCESS}.TABLES_BASE_SECONDS", base)
    monkeypatch.setattr(f"{PROCESS}.TABLES_SECONDS_PER_PAGE", 0.0)
    monkeypatch.setattr(f"{PROCESS}.TABLES_WALL_FACTOR", factor)
    pdf = build_shared_cmap_fonts_pdf(
        tmp_path / "s.pdf",
        fonts=1_000,
        cmap_bytes=36_000,
    )
    before = _children()
    assert read_tables(pdf, [RAW], page_count=1) == ({}, [STOPPED], {})
    assert _children() <= before


def _processor_limits(pid: int) -> tuple[str, str]:
    """Return the soft and hard processor limits of a process, on Linux."""
    limits = Path(f"/proc/{pid}/limits").read_text(encoding="ascii")
    for line in limits.splitlines():
        if line.startswith("Max cpu time"):
            soft, hard = line.split()[3:5]
            return soft, hard
    raise AssertionError(limits)


def _signalled_reading(pdf: Path) -> tuple[TableReading, tuple[str, str]]:
    """Read the tables of a page, the reader sent its limit signal once.

    The reader is signalled, as macOS signals the processor limit, once its
    limits are set and long before it would reach them; the system never
    signals it itself.

    Returns:
        What the reading gives, and the processor limits of the reader.
    """
    limits: list[tuple[str, str]] = []

    def signal_reader() -> None:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and not limits:
            for child in multiprocessing.active_children():
                if child.name != "doc-checker tables" or child.pid is None:
                    continue
                with contextlib.suppress(OSError):
                    found = _processor_limits(child.pid)
                    if found[0] != "unlimited":
                        limits.append(found)
                        os.kill(child.pid, signal.SIGXCPU)
            time.sleep(0.01)

    watcher = threading.Thread(target=signal_reader, daemon=True)
    watcher.start()
    reading = read_tables(pdf, [RAW], page_count=1)
    watcher.join()
    assert limits
    return reading, limits[0]


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="the limits of a process show on Linux",
)
def test_the_reader_ends_at_its_first_processor_signal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The signal of the CPU limit stops the reader: macOS never kills."""
    monkeypatch.setattr(f"{PROCESS}.TABLES_BASE_SECONDS", 1_000.0)
    pdf = build_shared_cmap_fonts_pdf(
        tmp_path / "s.pdf",
        fonts=1_000,
        cmap_bytes=36_000,
    )
    before = _children()
    reading, _ = _signalled_reading(pdf)
    assert reading == ({}, [STOPPED], {})
    assert _children() <= before


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="the limits of a process show on Linux",
)
def test_the_system_kills_the_reader_a_second_past_its_signal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hard processor limit is one second past the soft one, on Linux."""
    monkeypatch.setattr(f"{PROCESS}.TABLES_BASE_SECONDS", 1_000.0)
    pdf = build_shared_cmap_fonts_pdf(
        tmp_path / "s.pdf",
        fonts=1_000,
        cmap_bytes=36_000,
    )
    _, (soft, hard) = _signalled_reading(pdf)
    assert int(hard) == int(soft) + 1


def test_a_page_buys_no_more_time_than_the_most_characters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A page said to hold more characters than are searched gets no more."""
    monkeypatch.setattr(f"{PROCESS}.TABLES_BASE_SECONDS", 0.5)
    monkeypatch.setattr(f"{PROCESS}.TABLES_SECONDS_PER_PAGE", 0.0)
    monkeypatch.setattr(f"{MODULE}.MAX_TABLE_PAGE_CHARS", 1_000)
    pdf = build_shared_cmap_fonts_pdf(
        tmp_path / "s.pdf",
        fonts=1_000,
        cmap_bytes=36_000,
    )
    claimed = TablePage(0, 595, 842, (), 10**9)
    before = _children()
    assert read_tables(pdf, [claimed], page_count=1) == ({}, [STOPPED], {})
    assert _children() <= before


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="the address space of a process shows on Linux",
)
def test_memory_has_a_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A document that needs too much memory gives a warning, no table."""
    monkeypatch.setattr(f"{PROCESS}.TABLES_MEMORY_BYTES", 64 * 2**20)
    pdf = build_run_length_page_pdf(tmp_path / "l.pdf", size=32 * 2**20)
    before = _children()
    tables, notes, _ = read_tables(pdf, [RAW], page_count=1)
    assert tables == {}
    assert notes == [STOPPED]
    assert _children() <= before


def _reader_pages(pdf: Path) -> list[TablePage]:
    """Return every page of a PDF, as the reader finds it, without figures."""
    with open_pdf(pdf) as document:
        texts = [
            read_page(document, index, images=False)
            for index in range(len(document))
        ]
    return [
        TablePage(text.index, text.width, text.height, (), len(text.chars))
        for text in texts
    ]


def test_the_limits_grow_with_the_text_of_the_pages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pages of the most text that the search takes have time for it."""
    monkeypatch.setattr(f"{PROCESS}.TABLES_BASE_SECONDS", 0.5)
    monkeypatch.setattr(f"{PROCESS}.TABLES_SECONDS_PER_PAGE", 0.0)
    pdf = build_dense_table_pdf(
        tmp_path / "d.pdf",
        shape="full registers",
        rows=1_000,
    )
    pages = _reader_pages(pdf)
    assert len(pages) >= 5
    assert max(page.chars for page in pages) > 19_000
    tables, notes, _ = read_tables(pdf, pages, page_count=len(pages))
    assert notes == []
    assert sorted(tables) == [page.index for page in pages]


@pytest.mark.parametrize(
    "shape",
    ["parts", "boxed parts", "registers", "full registers"],
)
def test_dense_shapes_hold_their_text(tmp_path: Path, shape: str) -> None:
    """Each dense shape holds the text that its builder tells."""
    pdf = build_dense_table_pdf(tmp_path / "d.pdf", shape=shape, rows=200)
    chars = _reader_pages(pdf)[0].chars
    least = {"registers": 8_000, "full registers": 15_000}.get(shape, 4_000)
    assert chars > least


def test_a_reader_that_cannot_start_reads_no_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no process can start, no table is read, never without limits."""

    class _NoPipes:
        """A context of processes that cannot open a pipe."""

        @staticmethod
        def Pipe(*_: object, **__: object) -> None:  # ruff: ignore[invalid-function-name] - the name of multiprocessing
            """Fail as a system without free descriptors does."""
            raise OSError

    monkeypatch.setattr(f"{PROCESS}.get_context", lambda _: _NoPipes())
    pdf = build_cropped_table_pdf(tmp_path / "c.pdf")
    assert read_tables(pdf, [CROPPED], page_count=1) == (
        {},
        ["tables not read: the table reader could not start"],
        {},
    )


def _interrupt(*_: object) -> None:
    """Interrupt the command, as Ctrl-C does."""
    raise KeyboardInterrupt


def test_an_interrupted_wait_for_the_end_kills_the_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An interrupt while the process ends after its tables still kills it."""
    join = BaseProcess.join
    kill = BaseProcess.kill
    killed: list[BaseProcess] = []

    def interrupted(
        process: BaseProcess,
        timeout: float | None = None,
    ) -> None:
        if timeout is not None:
            raise KeyboardInterrupt
        join(process, timeout)

    def recorded(process: BaseProcess) -> None:
        killed.append(process)
        kill(process)

    monkeypatch.setattr(BaseProcess, "join", interrupted)
    monkeypatch.setattr(BaseProcess, "kill", recorded)
    pdf = build_cropped_table_pdf(tmp_path / "c.pdf")
    before = _children()
    with pytest.raises(KeyboardInterrupt):
        read_tables(pdf, [CROPPED], page_count=1)
    assert len(killed) == 1
    assert _children() <= before


@pytest.mark.skipif(sys.platform == "win32", reason="no alarm on Windows")
def test_an_interrupted_reading_leaves_no_process(tmp_path: Path) -> None:
    """A reading interrupted while it waits kills its process, and ends."""
    pdf = build_shared_cmap_fonts_pdf(
        tmp_path / "s.pdf",
        fonts=1_000,
        cmap_bytes=36_000,
    )
    before = _children()
    previous = signal.signal(signal.SIGALRM, _interrupt)
    try:
        signal.setitimer(signal.ITIMER_REAL, 1.0)
        with pytest.raises(KeyboardInterrupt):
            read_tables(pdf, [RAW], page_count=1)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert _children() <= before


def test_a_failing_page_is_a_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An error of pdfplumber on a page is a warning, not a trace."""
    pdf = build_table_pdf(tmp_path / "t.pdf", margin=25 * mm, grid=True)

    def fail(*_: object, **__: object) -> None:
        msg = "broken stream"
        raise ValueError(msg)

    monkeypatch.setattr("pdfplumber.page.Page.find_tables", fail)
    pages = _pages(pdf)
    tables, notes, skipped = find_tables(
        pdf,
        pages[:1],
        page_count=len(pages),
    )
    assert (tables, skipped) == ({}, {})
    assert notes == ["page 1: tables not read: ValueError: broken stream"]


def test_warnings_hide_memory_addresses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An address in an error never reaches a warning: reports are stable."""
    pdf = build_cropped_table_pdf(tmp_path / "c.pdf")

    def fail(*_: object, **__: object) -> None:
        msg = "<PDFObjRef at 0x7f3aBC12>"
        raise KeyError(msg)

    monkeypatch.setattr("pdfplumber.page.Page.find_tables", fail)
    notes = find_tables(pdf, [CROPPED], page_count=1)[1]
    assert notes == [
        "page 1: tables not read: KeyError: '<PDFObjRef at 0x\U00002026>'",
    ]


def test_an_unreadable_file_is_one_warning(tmp_path: Path) -> None:
    """A file that pdfplumber cannot open gives one warning, no table."""
    pdf = tmp_path / "bad.pdf"
    pdf.write_bytes(b"%PDF-1.7\nnot a PDF\n")
    tables, notes, skipped = find_tables(pdf, [CROPPED], page_count=1)
    assert (tables, skipped) == ({}, {})
    assert len(notes) == 1
    assert notes[0].startswith("tables not read: PdfminerException: ")


def test_pages_that_do_not_match_give_no_table(tmp_path: Path) -> None:
    """When pdfplumber counts other pages than the reader, nothing is read."""
    pdf = build_cropped_table_pdf(tmp_path / "c.pdf")
    assert find_tables(pdf, [CROPPED], page_count=2) == (
        {},
        ["tables not read: the pages do not match"],
        {},
    )


def test_continuation_needs_the_same_columns() -> None:
    """A table continues the one of the page before with the same columns."""
    first = PdfTable(
        0,
        (50, 100, 300, 700),
        (50.0, 170.0, 300.0),
        (("a", "b"),),
    )
    second = PdfTable(
        1,
        (51, 400, 301, 780),
        (51.0, 171.0, 301.0),
        (("a", "b"),),
    )
    other = PdfTable(
        1,
        (50, 400, 300, 780),
        (50.0, 120.0, 300.0),
        (("a", "b"),),
    )
    far = PdfTable(
        2,
        (51, 400, 301, 780),
        (51.0, 171.0, 301.0),
        (("a", "b"),),
    )
    wider = PdfTable(
        1,
        (50, 400, 400, 780),
        (50.0, 170.0, 300.0, 400.0),
        (("a", "b", "c"),),
    )
    assert continues(first, second)
    assert not continues(first, other)
    assert not continues(first, far)
    assert not continues(first, wider)


def _tables(doc: Document) -> list[str]:
    """Return the texts of the table blocks of a document."""
    return [b.text for b in doc.blocks if b.kind is BlockKind.TABLE]


def _blocks(doc: Document) -> list[tuple[BlockKind, str]]:
    """Return the kind and the text of each block of a document."""
    return [(b.kind, b.text) for b in doc.blocks]


def _searched(monkeypatch: pytest.MonkeyPatch) -> list[list[TablePage]]:
    """Record the pages that each search of the tables is given."""
    calls: list[list[TablePage]] = []

    def search(
        path: Path,
        pages: Sequence[TablePage],
        *,
        page_count: int,
    ) -> TableReading:
        calls.append(list(pages))
        return read_tables(path, pages, page_count=page_count)

    monkeypatch.setattr(f"{PROCESS}.read_tables", search)
    return calls


def test_table_becomes_one_block(tmp_path: Path) -> None:
    """A ruled table over several pages is one block, its header once."""
    pdf = build_table_pdf(
        tmp_path / "t.pdf",
        margin=25 * mm,
        grid=True,
        repeat_header=True,
    )
    doc = load_document(pdf, TABLES)
    tables = [b for b in doc.blocks if b.kind is BlockKind.TABLE]
    assert len(tables) == 1
    rows = tables[0].text.split("\n")
    assert rows[0] == "Option | Default"
    assert rows.count("Option | Default") == 1
    assert len(rows) == 161
    assert doc.normalization["table_header"] == len(doc.parts) - 1


def test_table_is_placed_between_its_paragraphs(tmp_path: Path) -> None:
    """The table block comes between the sentences around the table."""
    pdf = build_table_pdf(tmp_path / "t.pdf", margin=25 * mm, grid=True)
    doc = load_document(pdf, TABLES)
    kinds = [b.kind for b in doc.blocks]
    assert kinds == [BlockKind.PARAGRAPH, BlockKind.TABLE, BlockKind.PARAGRAPH]
    # Without a repeated header, each piece keeps its first row.
    assert len(_tables(doc)[0].split("\n")) == 161
    assert doc.normalization["table_header"] == 0


def test_table_rows_in_the_bands_are_kept(tmp_path: Path) -> None:
    """Narrow margins never turn table rows into running headers."""
    pdf = build_table_pdf(
        tmp_path / "t.pdf",
        margin=12 * mm,
        grid=True,
        repeat_header=True,
    )
    table = next(
        b
        for b in load_document(pdf, TABLES).blocks
        if b.kind is BlockKind.TABLE
    )
    assert len(table.text.split("\n")) == 161


def test_without_the_option_tables_stay_text(tmp_path: Path) -> None:
    """The base reading gives no table block."""
    pdf = build_table_pdf(tmp_path / "t.pdf", margin=25 * mm, grid=True)
    doc = load_document(pdf, Config())
    assert BlockKind.TABLE not in {b.kind for b in doc.blocks}


def test_dense_pages_are_not_searched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A page of more characters than the limit is read as text."""
    monkeypatch.setattr(
        "doc_checker.extract.pdf.tables.MAX_TABLE_PAGE_CHARS",
        10,
    )
    pdf = build_table_pdf(tmp_path / "t.pdf", margin=25 * mm, grid=True)
    doc = load_document(pdf, TABLES)
    assert BlockKind.TABLE not in {b.kind for b in doc.blocks}
    pages = len(doc.parts)
    assert doc.warnings == [
        f"{pages} page(s) not searched for tables: {pages} dense",
    ]


def test_pages_of_many_rules_are_not_searched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A page whose drawings the figure search skips is not searched."""
    pdf = build_crowded_table_pdf(tmp_path / "c.pdf", paths=MAX_PAGE_PATHS + 1)
    searched = _searched(monkeypatch)
    doc = load_document(pdf, TABLES)
    assert [[page.index for page in pages] for pages in searched] == [[1]]
    assert [
        b.source.page for b in doc.blocks if b.kind is BlockKind.TABLE
    ] == [2]
    assert doc.warnings == [
        "1 page(s) not searched for tables: 1 of many rules",
    ]


def test_turned_pages_are_not_searched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A page that the reader turns upright is not searched, square or not."""
    pdf = build_turned_table_pdf(tmp_path / "t.pdf")
    searched = _searched(monkeypatch)
    doc = load_document(pdf, TABLES)
    assert [[page.index for page in pages] for pages in searched] == [[1]]
    assert [
        b.source.page for b in doc.blocks if b.kind is BlockKind.TABLE
    ] == [2]
    turned = " ".join(b.text for b in doc.blocks if b.source.page == 1)
    assert "timeout 30" in turned
    assert doc.warnings == ["1 page(s) not searched for tables: 1 turned"]


def test_figures_guard_the_grids_without_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without images, the grid of a drawing is still no table."""
    pdf = build_drawn_grid_pdf(tmp_path / "d.pdf")
    searched = _searched(monkeypatch)
    no_images = Config(images=False, pdf=PdfConfig(tables=True))
    doc = load_document(pdf, no_images)
    assert _blocks(doc) == _blocks(load_document(pdf, Config(images=False)))
    assert [len(page.figures) for page in searched[0]] == [1]
    # The grid is a part of the figure, not a table kept as text.
    assert doc.warnings == []


def test_ruled_running_headers_stay_chrome(tmp_path: Path) -> None:
    """A ruled table in the header band of each page is page chrome."""
    pdf = build_prose_pdf(
        filler_sections(3, paragraphs=6, sentences=4),
        tmp_path / "h.pdf",
        ProseStyle(ruled_header=True),
    )
    plain = load_document(pdf, Config())
    doc = load_document(pdf, TABLES)
    assert "Page #" in plain.chrome
    assert (_blocks(doc), doc.chrome) == (_blocks(plain), plain.chrome)


def test_tables_read_otherwise_stay_text(tmp_path: Path) -> None:
    """A table whose cells pdfminer reads otherwise than PDFium stays text."""
    pdf = build_actual_text_table_pdf(tmp_path / "a.pdf")
    doc = load_document(pdf, TABLES)
    assert _blocks(doc) == _blocks(load_document(pdf, Config()))
    assert "timeout 30" in " ".join(b.text for b in doc.blocks)
    assert doc.warnings == [KEPT_AS_TEXT]


@pytest.mark.parametrize(
    "beyond",
    ["seconds", "seconds, or 60 when the service has been idle for a day"],
    ids=["middle-in", "middle-out"],
)
def test_tables_that_cut_a_line_stay_text(tmp_path: Path, beyond: str) -> None:
    """A table whose edge cuts a line of text in two leaves the line whole."""
    pdf = build_overflowing_table_pdf(tmp_path / "o.pdf", beyond=beyond)
    doc = load_document(pdf, TABLES)
    assert _blocks(doc) == _blocks(load_document(pdf, Config()))
    assert f"17 or fewer {beyond}" in " ".join(b.text for b in doc.blocks)
    assert doc.warnings == [KEPT_AS_TEXT]


@pytest.mark.parametrize("row", [0, 5])
def test_a_misread_row_of_a_large_table_stays_text(
    tmp_path: Path,
    row: int,
) -> None:
    """One row that pdfminer misreads keeps a table of 36 rows as text."""
    pdf = build_actual_text_table_pdf(
        tmp_path / "m.pdf",
        rows=SETTING_ROWS[:36],
        misread={row},
    )
    doc = load_document(pdf, TABLES)
    assert _blocks(doc) == _blocks(load_document(pdf, Config()))
    text = " ".join(b.text for b in doc.blocks)
    assert " ".join(SETTING_ROWS[row]) in text
    assert doc.warnings == [KEPT_AS_TEXT]


def test_repeated_headers_left_out_as_artifacts_still_join(
    tmp_path: Path,
) -> None:
    """A piece whose repeated header the reader leaves out still joins."""
    pdf = build_split_table_pdf(tmp_path / "s.pdf", artifact_header=True)
    doc = load_document(pdf, TABLES)
    assert _tables(doc) == [
        (
            "Option | Default\ntimeout | 30\nretries | 3\nworkers | 4"
            "\ncache | 256\nlevel | info"
        ),
    ]
    # The reader leaves the repeated header out: the join has none to remove.
    assert doc.normalization["artifact"] == len("OptionDefault")
    assert doc.normalization["table_header"] == 0
    assert doc.warnings == []


def test_a_header_that_only_pdfminer_reads_is_left_out(
    tmp_path: Path,
) -> None:
    """The header row of an artifact, unless repeated, never becomes text."""
    pdf = build_split_table_pdf(
        tmp_path / "s.pdf",
        artifact_header=True,
        note="The other settings follow.",
    )
    doc = load_document(pdf, TABLES)
    assert _tables(doc) == [
        "Option | Default\ntimeout | 30\nretries | 3\nworkers | 4",
        "cache | 256\nlevel | info",
    ]
    assert doc.normalization["table_header"] == 0
    assert doc.warnings == []


def test_an_image_between_two_tables_keeps_them_apart(tmp_path: Path) -> None:
    """An image under the end of a page's table keeps the next one apart."""
    pdf = build_split_table_pdf(tmp_path / "s.pdf", image=True)
    kinds = [b.kind for b in load_document(pdf, TABLES).blocks]
    assert kinds == [
        BlockKind.PARAGRAPH,
        BlockKind.TABLE,
        BlockKind.IMAGE,
        BlockKind.TABLE,
        BlockKind.PARAGRAPH,
    ]


@pytest.mark.parametrize(
    ("intro", "outro", "kind", "text"),
    [
        (
            "Table 1: Default settings of the service",
            "All of them can be changed in the configuration file.",
            BlockKind.CAPTION,
            "Default settings of the service",
        ),
        (
            SUCH_AS,
            "and all of them can be changed in the configuration file.",
            BlockKind.PARAGRAPH,
            SUCH_AS,
        ),
    ],
    ids=["caption", "such-as"],
)
def test_text_never_joins_across_a_table_over_pages(
    tmp_path: Path,
    intro: str,
    outro: str,
    kind: BlockKind,
    text: str,
) -> None:
    """The text above a table over two pages never joins the text after it."""
    pdf = build_table_pdf(
        tmp_path / "t.pdf",
        margin=25 * mm,
        rows=SETTING_ROWS[:61],
        grid=True,
        repeat_header=True,
        intro=intro,
        outro=outro,
    )
    doc = load_document(pdf, TABLES)
    assert len(doc.parts) == 2
    assert [
        (b.kind, b.text) for b in doc.blocks if b.text != _tables(doc)[0]
    ] == [
        (kind, text),
        (BlockKind.PARAGRAPH, outro),
    ]
    assert [b.kind for b in doc.blocks][1] is BlockKind.TABLE


def test_tables_apart_are_not_joined(tmp_path: Path) -> None:
    """A sentence between two tables of the same columns keeps them apart."""
    pdf = build_split_table_pdf(
        tmp_path / "s.pdf",
        note="The other settings follow.",
    )
    doc = load_document(pdf, TABLES)
    assert _tables(doc) == [
        "Option | Default\ntimeout | 30\nretries | 3\nworkers | 4",
        "Option | Default\ncache | 256\nlevel | info",
    ]
    assert doc.normalization["table_header"] == 0


def test_figure_boxes_are_read_without_images(tmp_path: Path) -> None:
    """The reader gives the boxes of the figures, images read or not."""
    pdf = build_reportlab_pdf(FIGURES_EN, tmp_path / "f.pdf")
    with open_pdf(pdf) as document:
        pages = [
            (
                read_page(document, index),
                read_page(document, index, images=False),
            )
            for index in range(len(document))
        ]
    boxes = [
        tuple(
            (image.x0, image.bottom, image.x1, image.top)
            for image in read.images
            if image.ref.vector
        )
        for read, _ in pages
    ]
    assert any(boxes)
    assert [read.figure_boxes for read, _ in pages] == boxes
    assert [bare.figure_boxes for _, bare in pages] == boxes


@pytest.mark.parametrize(
    ("paths", "crowded"),
    [(MAX_PAGE_PATHS, False), (MAX_PAGE_PATHS + 1, True)],
)
def test_crowded_pages_are_told(
    tmp_path: Path,
    paths: int,
    crowded: bool,  # ruff: ignore[boolean-type-hint-positional-argument] - pytest parameter
) -> None:
    """The reader tells a page whose paths are too many to look for figures."""
    pdf = build_dense_paths_pdf(tmp_path / "d.pdf", paths=paths)
    with open_pdf(pdf) as document:
        assert read_page(document, 0, images=False).crowded is crowded


def test_turns_are_told(tmp_path: Path) -> None:
    """The reader tells the pages that it turned upright."""
    lines = [["The service starts.", "It stops at night."]] * 2
    pdf = build_turned_pdf(lines, tmp_path / "t.pdf", turned={1})
    with open_pdf(pdf) as document:
        turns = [
            read_page(document, index, images=False).turn
            for index in range(len(document))
        ]
    assert turns[0] == 0
    assert turns[1] != 0


def test_text_beside_a_table_keeps_its_order(tmp_path: Path) -> None:
    """Reading a table never reorders the lines of the text beside it."""
    pdf = build_listing_beside_table_pdf(tmp_path / "l.pdf")
    # The lines of the listing, without their numbers, in their order.
    lines = [
        "for value in",
        "total +=",
        "value * weight",
        "save(total",
        "report(total",
    ]

    def order(doc: Document) -> list[str]:
        # The spaces read inside monospaced code depend on the font that
        # PDFium draws, which a system chooses for a font that the PDF does
        # not embed, as the README tells: the lines are found without them.
        text = "\n".join(
            b.text for b in doc.blocks if b.kind is not BlockKind.TABLE
        ).replace(" ", "")
        return sorted(
            lines,
            key=lambda line: text.index(line.replace(" ", "")),
        )

    doc = load_document(pdf, TABLES)
    assert _tables(doc) == ["Time | Mon Tue Wed\nAfternoon | used used"]
    assert order(doc) == order(load_document(pdf, Config())) == lines


def test_cells_keep_the_words_of_the_reader(tmp_path: Path) -> None:
    """A cell keeps the blanks that PDFium reads, where pdfminer has none."""
    pdf = build_close_words_table_pdf(tmp_path / "w.pdf")
    doc = load_document(pdf, TABLES)
    assert _tables(doc) == ["Cell long text | value\na | b"]


def test_text_never_joins_across_a_table_and_a_column_break(
    tmp_path: Path,
) -> None:
    """The text above a table at the foot of a column ends with the column."""
    pdf = build_two_column_table_pdf(tmp_path / "c.pdf", intro=SUCH_AS)
    texts = [
        b.text
        for b in load_document(pdf, TABLES).blocks
        if b.kind is not BlockKind.TABLE
    ]
    assert texts[1] == SUCH_AS
    assert texts[2].startswith("and all of them can be changed")


def test_cells_keep_the_words_of_the_reader_past_an_accent(
    tmp_path: Path,
) -> None:
    """A decomposed accent never costs a cell the blanks of the reader."""
    pdf = build_accent_table_pdf(tmp_path / "a.pdf")
    doc = load_document(pdf, TABLES)
    assert _tables(doc) == ["Caf\xe9 long text | value\na | b"]
