# Copyright 2026 Christophe Le Douarec
"""Command line interface for Doc Checker."""

import os
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import click

from doc_checker import __version__
from doc_checker.config import (
    AVAILABLE_FORMATS,
    CliOverrides,
    Config,
    apply_overrides,
    load_config,
)
from doc_checker.errors import ConfigError, InputError
from doc_checker.extract import load_document
from doc_checker.model.findings import Severity
from doc_checker.pipeline import run_compare, write_reports
from doc_checker.report.terminal import render_blocks, render_summary

CONFIG_EXIT_CODE: Final = 2
INPUT_EXIT_CODE: Final = 3
_LANGUAGES: Final = click.Choice(["auto", "fr", "en"])
_SEVERITIES: Final = click.Choice([severity.label for severity in Severity])
_NO_COLOR: Final = "NO_COLOR"


class _CliError(click.ClickException):
    """A user facing error with its own exit code."""

    def __init__(self, message: str, exit_code: int) -> None:
        super().__init__(message)
        self.exit_code = exit_code


@contextmanager
def _exit_on_error() -> Generator[None]:
    """Turn configuration and input errors into exit codes 2 and 3.

    Yields:
        Nothing; the body runs under the error translation.

    Raises:
        _CliError: For configuration and input errors.
    """
    try:
        yield
    except ConfigError as exc:
        raise _CliError(str(exc), CONFIG_EXIT_CODE) from exc
    except InputError as exc:
        raise _CliError(str(exc), INPUT_EXIT_CODE) from exc


def _config_options[F: Callable[..., object]](function: F) -> F:
    """Return the command with the configuration options added."""
    function = click.option(
        "--exclude",
        multiple=True,
        metavar="GLOB",
        help="Skip the site pages matching this pattern.",
    )(function)
    function = click.option(
        "--content-selector",
        metavar="CSS",
        help="CSS selector of the content of each HTML page.",
    )(function)
    function = click.option(
        "--lang",
        type=_LANGUAGES,
        help="Language of the documentation.",
    )(function)
    return click.option(
        "-c",
        "--config",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        help="Configuration file.",
    )(function)


def _split_formats(
    _ctx: click.Context,
    _param: click.Parameter,
    value: str | None,
) -> tuple[str, ...] | None:
    """Return the report formats of a comma separated list.

    Raises:
        click.BadParameter: If a format is unknown or the list is empty.
    """
    if value is None:
        return None
    formats = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = [name for name in formats if name not in AVAILABLE_FORMATS]
    if unknown or not formats:
        available = ", ".join(AVAILABLE_FORMATS)
        msg = f"unknown format {value!r} (available: {available})"
        raise click.BadParameter(msg)
    return formats


def _load(config: Path | None, overrides: CliOverrides) -> Config:
    """Return the configuration with the command line options applied."""
    return apply_overrides(load_config(config, cwd=Path.cwd()), overrides)


@click.group(
    invoke_without_command=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.version_option(__version__, prog_name="doc-checker")
@click.pass_context
def main(ctx: click.Context) -> None:
    """Compare two documentation builds or audit one."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@main.command()
@click.argument("doc", type=click.Path(exists=True, path_type=Path))
@_config_options
def extract(
    *,
    doc: Path,
    config: Path | None,
    lang: str | None,
    content_selector: str | None,
    exclude: tuple[str, ...],
) -> None:
    """Print the normalized blocks of DOC, to tune selectors."""
    with _exit_on_error():
        cfg = _load(
            config,
            CliOverrides(
                lang=lang,
                content_selector=content_selector,
                exclude=exclude,
            ),
        )
        document = load_document(doc, cfg)
    click.echo(render_blocks(document))


@main.command()
@click.argument("ref", type=click.Path(exists=True, path_type=Path))
@click.argument("new", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o",
    "--output",
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory where the reports are written.",
)
@click.option(
    "--format",
    "formats",
    metavar="LIST",
    callback=_split_formats,
    help="Comma separated report formats: md.",
)
@click.option(
    "--min-severity",
    type=_SEVERITIES,
    help="Leave out findings below this severity.",
)
@click.option(
    "--disable",
    multiple=True,
    metavar="RULE",
    help="Disable a rule, glob patterns allowed.",
)
@click.option(
    "--ignore-moves",
    is_flag=True,
    help="Do not report blocks that only changed place.",
)
@_config_options
def compare(  # ruff: ignore[too-many-arguments] - Click passes options as keywords
    *,
    ref: Path,
    new: Path,
    output: Path | None,
    formats: tuple[str, ...] | None,
    min_severity: str | None,
    disable: tuple[str, ...],
    ignore_moves: bool,
    config: Path | None,
    lang: str | None,
    content_selector: str | None,
    exclude: tuple[str, ...],
) -> None:
    """Compare the NEW build of a documentation with its REF build."""
    with _exit_on_error():
        cfg = _load(
            config,
            CliOverrides(
                lang=lang,
                min_severity=(
                    Severity.parse(min_severity) if min_severity else None
                ),
                formats=formats,
                output=output,
                content_selector=content_selector,
                exclude=exclude,
                disable=disable,
                ignore_moves=ignore_moves,
            ),
        )
        result = run_compare(ref, new, cfg)
        reports = write_reports(result, cfg)
    color = False if _NO_COLOR in os.environ else None
    click.echo(render_summary(result, reports), color=color)
