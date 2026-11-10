# Copyright 2026 Christophe Le Douarec
"""Tests that the README tells the options and the profiles."""

import re
from pathlib import Path
from typing import Final

import click

from doc_checker.cli import main
from doc_checker.extract.html.profiles import PROFILE_NAMES

README: Final = (Path(__file__).parents[1] / "README.md").read_text(
    encoding="utf-8",
)
# The heading of the section that describes each command, but `extract`,
# whose options one paragraph of the usage names.
COMMAND_SECTIONS: Final = {
    "compare": "## Usage",
    "audit": "### Audit",
}
EXTRACT_OPTIONS: Final = "`extract` takes those that tell how to read"
_FENCE: Final = "```"
_OPTION: Final = re.compile(r"`(--?[a-z][-a-z]*)")


def _depth(line: str) -> int:
    """Return the level of a Markdown heading, 0 for another line."""
    depth = len(line) - len(line.lstrip("#"))
    return depth if line[depth : depth + 1] == " " else 0


def _section_intro(heading: str) -> str:
    """Return a section of the README, up to its first subsection.

    The lines of code blocks are never taken for headings.
    """
    lines = README.splitlines()
    start = lines.index(heading)
    fenced = False
    for end in range(start + 1, len(lines)):
        if lines[end].startswith(_FENCE):
            fenced = not fenced
        elif not fenced and _depth(lines[end]):
            break
    else:
        end = len(lines)
    return "\n".join(lines[start:end])


def test_readme_names_every_option() -> None:
    """Each long option of each command appears in the README."""
    assert isinstance(main, click.Group)
    for command in main.commands.values():
        for parameter in command.params:
            for option in parameter.opts:
                if option.startswith("--"):
                    assert f"`{option}" in README, (command.name, option)


def test_section_intro_stops_at_its_first_subsection() -> None:
    """The intro of a section ends before its first subsection."""
    intro = _section_intro("### Audit")
    assert intro.startswith("### Audit")
    assert "--emit-wordlist" in intro
    assert "#### Text checks" not in intro


def _paragraph(start: str) -> str:
    """Return the paragraph of the README that starts with a text."""
    for paragraph in README.split("\n\n"):
        if paragraph.startswith(start):
            return paragraph
    raise AssertionError(start)


def _options(command: click.Command) -> set[str]:
    """Return the options of a command, short and long."""
    return {
        option
        for parameter in command.params
        for option in (*parameter.opts, *parameter.secondary_opts)
        if option.startswith("-")
    }


def test_each_command_section_names_its_options() -> None:
    """The section of each command names all its options, short or long."""
    assert isinstance(main, click.Group)
    assert {*COMMAND_SECTIONS, "extract"} == set(main.commands)
    for name, heading in COMMAND_SECTIONS.items():
        intro = _section_intro(heading)
        for option in _options(main.commands[name]):
            assert f"`{option}" in intro, (name, option)


def test_extract_paragraph_names_exactly_its_options() -> None:
    """The paragraph of extract names all its options, and no other."""
    assert isinstance(main, click.Group)
    paragraph = _paragraph(EXTRACT_OPTIONS)
    named = set(_OPTION.findall(paragraph))
    assert named == _options(main.commands["extract"])


def test_readme_names_every_profile() -> None:
    """Each HTML profile is named in the README."""
    for name in PROFILE_NAMES:
        assert f"`{name}`" in README, name


def test_readme_tells_the_extra() -> None:
    """The README tells how to install the pdf-tables extra."""
    assert "doc-checker[pdf-tables]" in README
