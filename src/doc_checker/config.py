# Copyright 2026 Christophe Le Douarec
"""Configuration: TOML files and command line overrides."""

import fnmatch
import re
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, cast

from doc_checker.errors import ConfigError
from doc_checker.model.findings import Severity
from doc_checker.rules import RULES

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

type PageOrder = Literal["auto", "nav", "path"]

AVAILABLE_FORMATS: Final = ("md",)
LANGUAGES: Final = ("auto", "fr", "en")
PAGE_ORDERS: Final = ("auto", "nav", "path")
DEFAULT_HTML_INCLUDE: Final = ("**/*.html", "**/*.htm")
DEFAULT_HTML_EXCLUDE: Final = (
    "**/genindex.html",
    "**/search.html",
    "**/404.html",
    "**/py-modindex.html",
    "**/_static/**",
    "**/_sources/**",
    "**/_modules/**",
    "**/_images/**",
    "**/assets/**",
    "**/search/**",
)
DEFAULT_OUTPUT: Final = Path("doc-checker-report")
CONFIG_FILE: Final = "doc-checker.toml"
PYPROJECT_FILE: Final = "pyproject.toml"
MOVED_RULE: Final = "block.moved"

_SECTIONS: Final = ("general", "html", "thresholds", "ignore", "severity")


@dataclass(frozen=True, slots=True)
class HtmlConfig:
    """HTML extraction settings.

    Attributes:
        profile: Generator profile name, or ``"auto"`` to detect it.
        content_selector: CSS selector of the content, overriding profiles.
        strip_selectors: Extra CSS selectors of elements to remove.
        include: Glob patterns of the pages to read in a site directory.
        exclude: Glob patterns of the pages to skip.
        page_order: How pages are ordered: navigation, path, or automatic.
    """

    profile: str = "auto"
    content_selector: str | None = None
    strip_selectors: tuple[str, ...] = ()
    include: tuple[str, ...] = DEFAULT_HTML_INCLUDE
    exclude: tuple[str, ...] = DEFAULT_HTML_EXCLUDE
    page_order: PageOrder = "auto"


@dataclass(frozen=True, slots=True)
class CompareThresholds:
    """Similarity thresholds used by the comparison.

    Attributes:
        match: Minimum similarity to pair two blocks in place.
        moved_match: Minimum similarity to pair a moved block.
    """

    match: float = 0.6
    moved_match: float = 0.8


@dataclass(frozen=True, slots=True)
class IgnoreConfig:
    """What to leave out of the reports.

    Attributes:
        patterns: Regular expressions removed from texts before comparison.
        rules: Rule identifiers or glob patterns to disable.
        findings: Identifiers of findings to drop.
    """

    patterns: tuple[str, ...] = ()
    rules: tuple[str, ...] = ()
    findings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Config:
    """Complete configuration of a run.

    Attributes:
        lang: Document language, or ``"auto"`` to detect it.
        min_severity: Findings below this severity are not reported.
        formats: Report formats to write.
        output: Directory where the reports are written.
        html: HTML extraction settings.
        compare: Comparison thresholds.
        ignore: What to leave out of the reports.
        severity: Severity overrides by rule identifier.
    """

    lang: str = "auto"
    min_severity: Severity = Severity.INFO
    formats: tuple[str, ...] = ("md",)
    output: Path = DEFAULT_OUTPUT
    html: HtmlConfig = field(default_factory=HtmlConfig)
    compare: CompareThresholds = field(default_factory=CompareThresholds)
    ignore: IgnoreConfig = field(default_factory=IgnoreConfig)
    severity: Mapping[str, Severity] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CliOverrides:
    """Values given on the command line, ``None`` when not given."""

    lang: str | None = None
    min_severity: Severity | None = None
    formats: tuple[str, ...] | None = None
    output: Path | None = None
    content_selector: str | None = None
    exclude: tuple[str, ...] = ()
    disable: tuple[str, ...] = ()
    ignore_moves: bool = False


class _Section:
    """Typed and validated access to one table of a configuration file."""

    def __init__(
        self,
        data: object,
        prefix: str,
        source: Path,
    ) -> None:
        self.prefix = prefix
        self.source = source
        if not isinstance(data, dict):
            msg = ""
            raise self.error(msg, "expected a table")
        self.data = cast("dict[str, object]", data)

    def error(self, key: str, problem: str) -> ConfigError:
        """Return an error naming the file and the full key."""
        name = f"{self.prefix}.{key}" if key else self.prefix
        return ConfigError(f"{self.source}: {name}: {problem}")

    def check_keys(self, known: Iterable[str]) -> None:
        """Reject the keys that are not in ``known``."""
        for key in sorted(set(self.data) - set(known)):
            raise self.error(key, "unknown key")

    def text(self, key: str, default: str) -> str:
        """Return a string value."""
        value = self.data.get(key, default)
        if not isinstance(value, str):
            raise self.error(key, "expected a string")
        return value

    def choice(self, key: str, default: str, choices: Iterable[str]) -> str:
        """Return a string value restricted to ``choices``."""
        value = self.text(key, default)
        allowed = tuple(choices)
        if value not in allowed:
            expected = ", ".join(allowed)
            raise self.error(key, f"{value!r} is not one of {expected}")
        return value

    def texts(self, key: str, default: tuple[str, ...]) -> tuple[str, ...]:
        """Return a list of strings as a tuple."""
        value = self.data.get(key, default)
        items = cast("list[object]", value)
        if not isinstance(value, (list, tuple)) or not all(
            isinstance(item, str) for item in items
        ):
            raise self.error(key, "expected a list of strings")
        return tuple(cast("list[str]", items))

    def ratio(self, key: str, default: float) -> float:
        """Return a number in ``]0, 1]``."""
        value = self.data.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise self.error(key, "expected a number")
        if not 0 < value <= 1:
            raise self.error(key, "must be in ]0, 1]")
        return float(value)

    def severity(self, key: str, default: Severity) -> Severity:
        """Return a severity name parsed as a severity."""
        try:
            return Severity.parse(self.text(key, default.label))
        except ValueError as exc:
            raise self.error(key, str(exc)) from None


def _parse_general(section: _Section, base_dir: Path, cwd: Path) -> Config:
    section.check_keys(("lang", "min_severity", "formats", "output"))
    formats = section.texts("formats", ("md",))
    for name in formats:
        if name not in AVAILABLE_FORMATS:
            expected = ", ".join(AVAILABLE_FORMATS)
            msg = f"unknown format {name!r} (expected: {expected})"
            msg_0 = "formats"
            raise section.error(msg_0, msg)
    output = (
        base_dir / section.text("output", "")
        if "output" in section.data
        else cwd / DEFAULT_OUTPUT
    )
    return Config(
        lang=section.choice("lang", "auto", LANGUAGES),
        min_severity=section.severity("min_severity", Severity.INFO),
        formats=formats,
        output=output,
    )


def _parse_html(section: _Section) -> HtmlConfig:
    section.check_keys(
        (
            "profile",
            "content_selector",
            "strip_selectors",
            "include",
            "exclude",
            "page_order",
        ),
    )
    selector = section.text("content_selector", "")
    return HtmlConfig(
        profile=section.text("profile", "auto"),
        content_selector=selector or None,
        strip_selectors=section.texts("strip_selectors", ()),
        include=section.texts("include", DEFAULT_HTML_INCLUDE),
        exclude=section.texts("exclude", DEFAULT_HTML_EXCLUDE),
        page_order=cast(
            "PageOrder",
            section.choice("page_order", "auto", PAGE_ORDERS),
        ),
    )


def _parse_thresholds(section: _Section) -> CompareThresholds:
    section.check_keys(("compare",))
    compare = _Section(
        section.data.get("compare", {}),
        f"{section.prefix}.compare",
        section.source,
    )
    compare.check_keys(("match", "moved_match"))
    return CompareThresholds(
        match=compare.ratio("match", 0.6),
        moved_match=compare.ratio("moved_match", 0.8),
    )


def _check_rule_patterns(patterns: Iterable[str], where: str) -> None:
    """Reject the rule patterns that match no known rule.

    Args:
        patterns: Rule identifiers or glob patterns.
        where: Origin of the patterns, used in the error message.

    Raises:
        ConfigError: If a pattern matches no rule.
    """
    for pattern in patterns:
        if not fnmatch.filter(RULES, pattern):
            msg = f"{where}: no rule matches {pattern!r}"
            raise ConfigError(msg)


def _parse_ignore(section: _Section) -> IgnoreConfig:
    section.check_keys(("patterns", "rules", "findings"))
    patterns = section.texts("patterns", ())
    for pattern in patterns:
        try:
            re.compile(pattern)
        except re.error as exc:
            msg = f"invalid regular expression {pattern!r}: {exc}"
            msg_0 = "patterns"
            raise section.error(msg_0, msg) from None
    rules = section.texts("rules", ())
    _check_rule_patterns(rules, f"{section.source}: ignore.rules")
    return IgnoreConfig(
        patterns=patterns,
        rules=rules,
        findings=section.texts("findings", ()),
    )


def _parse_severity(section: _Section) -> dict[str, Severity]:
    overrides: dict[str, Severity] = {}
    for rule_id in sorted(section.data):
        if rule_id not in RULES:
            raise section.error(rule_id, "unknown rule")
        overrides[rule_id] = section.severity(rule_id, Severity.INFO)
    return overrides


def config_from_mapping(
    data: Mapping[str, object],
    *,
    source: Path,
    cwd: Path,
) -> Config:
    """Build a configuration from the content of a configuration file.

    Args:
        data: Parsed TOML tables.
        source: Path of the file, used in error messages and to resolve
            relative paths.
        cwd: Working directory, used for the default output directory.

    Returns:
        The validated configuration.

    Raises:
        ConfigError: If a section, a key or a value is invalid.
    """
    for name in sorted(set(data) - set(_SECTIONS)):
        msg = f"{source}: {name}: unknown section"
        raise ConfigError(msg)

    def section(name: str) -> _Section:
        return _Section(data.get(name, {}), name, source)

    general = _parse_general(section("general"), source.parent, cwd)
    return replace(
        general,
        html=_parse_html(section("html")),
        compare=_parse_thresholds(section("thresholds")),
        ignore=_parse_ignore(section("ignore")),
        severity=_parse_severity(section("severity")),
    )


def _read_toml(path: Path) -> dict[str, object]:
    """Read a TOML file.

    Args:
        path: File to read.

    Returns:
        The parsed tables.

    Raises:
        ConfigError: If the file cannot be read or is not valid TOML.
    """
    try:
        with path.open("rb") as stream:
            return tomllib.load(stream)
    except OSError as exc:
        msg = f"{path}: cannot read file: {exc.strerror}"
        raise ConfigError(msg) from None
    except tomllib.TOMLDecodeError as exc:
        msg = f"{path}: invalid TOML: {exc}"
        raise ConfigError(msg) from None


def load_config(path: Path | None, *, cwd: Path) -> Config:
    """Load the configuration of a run.

    The first source found wins: the explicit ``path``, then
    ``doc-checker.toml`` in ``cwd``, then the ``[tool.doc-checker]`` table of
    ``pyproject.toml`` in ``cwd``, then the defaults.

    Args:
        path: Explicit configuration file, if any.
        cwd: Working directory.

    Returns:
        The validated configuration.
    """
    if path is not None:
        return config_from_mapping(_read_toml(path), source=path, cwd=cwd)
    project = cwd / CONFIG_FILE
    if project.is_file():
        return config_from_mapping(
            _read_toml(project),
            source=project,
            cwd=cwd,
        )
    pyproject = cwd / PYPROJECT_FILE
    if pyproject.is_file():
        tool = _read_toml(pyproject).get("tool", {})
        if isinstance(tool, dict) and "doc-checker" in tool:
            table = cast("dict[str, object]", tool["doc-checker"])
            return config_from_mapping(table, source=pyproject, cwd=cwd)
    return Config(output=cwd / DEFAULT_OUTPUT)


def apply_overrides(cfg: Config, overrides: CliOverrides) -> Config:
    """Apply command line options on top of a configuration.

    Values replace the configured ones, except ``exclude`` and ``disable``
    which extend them.

    Args:
        cfg: Configuration loaded from files.
        overrides: Command line values.

    Returns:
        The resulting configuration.
    """
    _check_rule_patterns(overrides.disable, "--disable")
    selector = overrides.content_selector or cfg.html.content_selector
    html = replace(
        cfg.html,
        content_selector=selector,
        exclude=cfg.html.exclude + overrides.exclude,
    )
    moved = (MOVED_RULE,) if overrides.ignore_moves else ()
    ignore = replace(
        cfg.ignore,
        rules=cfg.ignore.rules + overrides.disable + moved,
    )
    changes: dict[str, object] = {"html": html, "ignore": ignore}
    for name in ("lang", "min_severity", "formats", "output"):
        value = getattr(overrides, name)
        if value is not None:
            changes[name] = value
    return replace(cfg, **changes)


def rule_disabled(cfg: Config, rule_id: str) -> bool:
    """Tell whether a rule is disabled by the configuration.

    Args:
        cfg: Configuration.
        rule_id: Rule identifier.

    Returns:
        Whether a pattern of ``cfg.ignore.rules`` matches the rule.
    """
    return any(
        fnmatch.fnmatchcase(rule_id, pattern) for pattern in cfg.ignore.rules
    )
