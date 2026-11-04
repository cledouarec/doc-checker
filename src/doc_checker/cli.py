# Copyright 2026 Christophe Le Douarec
"""Command line interface for Doc Checker."""

import click

from doc_checker import __version__


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
