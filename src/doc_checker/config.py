# Copyright 2026 Christophe Le Douarec
"""Configuration: TOML files and command line overrides."""

import fnmatch
import math
import re
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, cast

from doc_checker.errors import ConfigError
from doc_checker.model.findings import Severity
from doc_checker.rules import RULES
from doc_checker.spell.words import read_word_file

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

type PageOrder = Literal["auto", "nav", "path"]

AVAILABLE_FORMATS: Final = ("html", "md")
DEFAULT_FORMATS: Final = ("html", "md")
AVAILABLE_CHECKS: Final = ("images", "structure", "a11y", "text")
PROFILES: Final = ("screen", "print")
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
MAX_JPEG_QUALITY: Final = 100

_SECTIONS: Final = (
    "general",
    "html",
    "pdf",
    "thresholds",
    "ignore",
    "severity",
    "spelling",
)


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
class PdfConfig:
    """PDF extraction and comparison settings.

    Attributes:
        header_band: Share of the page height, from the top, where each line
            may be a running header. The first line of a page is looked at
            beyond the band, down to 20 % of the height, if a gap of more
            than two sizes separates it from the next line.
        footer_band: Share of the page height, from the bottom, where each
            line may be a running footer. The last line of a page is looked
            at beyond the band, up to 20 % of the height, if a gap of more
            than two sizes separates it from the line before. Page numbers
            are looked for at the top as well as at the bottom of the page.
        repeat_share: Minimum share of the pages on which a text must repeat
            to be a running header or footer. Page numbers that differ from
            the labels of their pages need the same share of the pages,
            unlike running titles.
        compare_toc: Whether tables of contents and indexes are compared.
    """

    header_band: float = 0.08
    footer_band: float = 0.08
    repeat_share: float = 0.3
    compare_toc: bool = False


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
class AuditThresholds:
    """Thresholds used by the audit.

    Attributes:
        dpi_major: Effective resolution below which an image is a major
            finding on screen.
        dpi_minor: Effective resolution below which an image is a minor
            finding on screen.
        print_dpi_major: Same as ``dpi_major`` for the print profile.
        print_dpi_minor: Same as ``dpi_minor`` for the print profile.
        heavy_minor: Size in bytes above which an image is heavy.
        heavy_major: Size in bytes above which an image is very heavy.
        jpeg_quality_minor: Estimated JPEG quality below which it is minor.
        jpeg_quality_major: Estimated JPEG quality below which it is major.
        upscale: Enlargement ratio above which an image is upscaled.
        blur: Sharpness ratio below which an image is blurred.
        blockiness: Block artefact ratio above which an image is blocky.
        unmapped_major: Share of the characters of a PDF without Unicode
            mapping that is a major finding.
        long_sentence_minor: Number of words above which a sentence is long.
        long_sentence_major: Number of words above which a sentence is very
            long.
    """

    dpi_major: float = 96.0
    dpi_minor: float = 150.0
    print_dpi_major: float = 150.0
    print_dpi_minor: float = 300.0
    heavy_minor: int = 1_000_000
    heavy_major: int = 3_000_000
    jpeg_quality_minor: int = 50
    jpeg_quality_major: int = 25
    upscale: float = 1.1
    blur: float = 0.10
    blockiness: float = 2.0
    unmapped_major: float = 0.01
    long_sentence_minor: int = 40
    long_sentence_major: int = 60

    def dpi(self, profile: str) -> tuple[float, float]:
        """Return the major and minor resolution limits of a profile.

        Args:
            profile: ``"screen"`` or ``"print"``.

        Returns:
            The ``(major, minor)`` limits in dots per inch.
        """
        if profile == "print":
            return (self.print_dpi_major, self.print_dpi_minor)
        return (self.dpi_major, self.dpi_minor)


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
class SpellingConfig:
    """Spell checking settings.

    Attributes:
        words: Personal words: those of the configuration, then those of its
            word files, then those of the command line.
    """

    words: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Config:
    """Complete configuration of a run.

    Attributes:
        lang: Document language, or ``"auto"`` to detect it.
        min_severity: Findings below this severity are not reported.
        formats: Report formats to write.
        output: Directory where the reports are written.
        images: Whether the images are extracted and compared.
        html: HTML extraction settings.
        pdf: PDF extraction and comparison settings.
        compare: Comparison thresholds.
        checks: Audit check families to run.
        profile: Audit profile, ``"screen"`` or ``"print"``.
        audit: Audit thresholds.
        spelling: Spell checking settings.
        ignore: What to leave out of the reports.
        severity: Severity overrides by rule identifier.
    """

    lang: str = "auto"
    min_severity: Severity = Severity.INFO
    formats: tuple[str, ...] = DEFAULT_FORMATS
    output: Path = DEFAULT_OUTPUT
    images: bool = True
    html: HtmlConfig = field(default_factory=HtmlConfig)
    pdf: PdfConfig = field(default_factory=PdfConfig)
    compare: CompareThresholds = field(default_factory=CompareThresholds)
    checks: tuple[str, ...] = AVAILABLE_CHECKS
    profile: str = "screen"
    audit: AuditThresholds = field(default_factory=AuditThresholds)
    spelling: SpellingConfig = field(default_factory=SpellingConfig)
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
    no_images: bool = False
    checks: tuple[str, ...] | None = None
    profile: str | None = None
    words: tuple[Path, ...] = ()


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

    def ratio(self, key: str, default: float, *, upper: float = 1.0) -> float:
        """Return a number in ``]0, upper]``."""
        value = self.data.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise self.error(key, "expected a number")
        if not 0 < value <= upper:
            raise self.error(key, f"must be in ]0, {upper:g}]")
        return float(value)

    def positive(self, key: str, default: float) -> float:
        """Return a number above 0."""
        value = self.data.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise self.error(key, "expected a number")
        if not math.isfinite(value):
            raise self.error(key, "expected a finite number")
        if value <= 0:
            raise self.error(key, "must be above 0")
        return float(value)

    def count(self, key: str, default: int) -> int:
        """Return an integer above 0."""
        value = self.data.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise self.error(key, "expected an integer")
        if value <= 0:
            raise self.error(key, "must be above 0")
        return value

    def flag(self, key: str, *, default: bool) -> bool:
        """Return a boolean value."""
        value = self.data.get(key, default)
        if not isinstance(value, bool):
            raise self.error(key, "expected a boolean")
        return value

    def severity(self, key: str, default: Severity) -> Severity:
        """Return a severity name parsed as a severity."""
        try:
            return Severity.parse(self.text(key, default.label))
        except ValueError as exc:
            raise self.error(key, str(exc)) from None


def _parse_checks(section: _Section) -> tuple[str, ...]:
    key = "checks"
    checks = section.texts("checks", AVAILABLE_CHECKS)
    for name in checks:
        if name not in AVAILABLE_CHECKS:
            expected = ", ".join(AVAILABLE_CHECKS)
            msg = f"unknown check {name!r} (expected: {expected})"
            raise section.error(key, msg)
    if not checks:
        msg = "expected at least one check"
        raise section.error(key, msg)
    return tuple(dict.fromkeys(checks))


def _parse_general(section: _Section, base_dir: Path, cwd: Path) -> Config:
    section.check_keys(
        (
            "lang",
            "min_severity",
            "formats",
            "output",
            "images",
            "checks",
            "profile",
        ),
    )
    formats = section.texts("formats", DEFAULT_FORMATS)
    for name in formats:
        if name not in AVAILABLE_FORMATS:
            expected = ", ".join(AVAILABLE_FORMATS)
            msg = f"unknown format {name!r} (expected: {expected})"
            msg_0 = "formats"
            raise section.error(msg_0, msg)
    checks = _parse_checks(section)
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
        images=section.flag("images", default=True),
        checks=checks,
        profile=section.choice("profile", "screen", PROFILES),
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


def _parse_pdf(section: _Section) -> PdfConfig:
    section.check_keys(
        ("header_band", "footer_band", "repeat_share", "compare_toc"),
    )
    return PdfConfig(
        header_band=section.ratio("header_band", 0.08, upper=0.5),
        footer_band=section.ratio("footer_band", 0.08, upper=0.5),
        repeat_share=section.ratio("repeat_share", 0.3),
        compare_toc=section.flag("compare_toc", default=False),
    )


def _parse_audit(section: _Section) -> AuditThresholds:
    defaults = AuditThresholds()
    default_unmapped = defaults.unmapped_major
    section.check_keys(AuditThresholds.__slots__)
    values = AuditThresholds(
        dpi_major=section.positive("dpi_major", defaults.dpi_major),
        dpi_minor=section.positive("dpi_minor", defaults.dpi_minor),
        print_dpi_major=section.positive(
            "print_dpi_major",
            defaults.print_dpi_major,
        ),
        print_dpi_minor=section.positive(
            "print_dpi_minor",
            defaults.print_dpi_minor,
        ),
        heavy_minor=section.count("heavy_minor", defaults.heavy_minor),
        heavy_major=section.count("heavy_major", defaults.heavy_major),
        jpeg_quality_minor=section.count(
            "jpeg_quality_minor",
            defaults.jpeg_quality_minor,
        ),
        jpeg_quality_major=section.count(
            "jpeg_quality_major",
            defaults.jpeg_quality_major,
        ),
        upscale=section.positive("upscale", defaults.upscale),
        blur=section.ratio("blur", defaults.blur),
        blockiness=section.positive("blockiness", defaults.blockiness),
        unmapped_major=section.ratio("unmapped_major", default_unmapped),
        long_sentence_minor=section.count(
            "long_sentence_minor",
            defaults.long_sentence_minor,
        ),
        long_sentence_major=section.count(
            "long_sentence_major",
            defaults.long_sentence_major,
        ),
    )
    for key in ("jpeg_quality_minor", "jpeg_quality_major"):
        if getattr(values, key) > MAX_JPEG_QUALITY:
            raise section.error(key, "must be in [1, 100]")
    for key in ("upscale", "blockiness"):
        if getattr(values, key) <= 1:
            raise section.error(key, "must be above 1")
    for low, high in (
        ("dpi_major", "dpi_minor"),
        ("print_dpi_major", "print_dpi_minor"),
        ("heavy_minor", "heavy_major"),
        ("jpeg_quality_major", "jpeg_quality_minor"),
        ("long_sentence_minor", "long_sentence_major"),
    ):
        if getattr(values, low) >= getattr(values, high):
            raise section.error(low, f"must be below {high}")
    return values


def _parse_thresholds(
    section: _Section,
) -> tuple[CompareThresholds, AuditThresholds]:
    section.check_keys(("compare", "audit"))

    def table(name: str) -> _Section:
        return _Section(
            section.data.get(name, {}),
            f"{section.prefix}.{name}",
            section.source,
        )

    compare = table("compare")
    compare.check_keys(("match", "moved_match"))
    return (
        CompareThresholds(
            match=compare.ratio("match", 0.6),
            moved_match=compare.ratio("moved_match", 0.8),
        ),
        _parse_audit(table("audit")),
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


def _parse_spelling(section: _Section, base_dir: Path) -> SpellingConfig:
    section.check_keys(("words", "files"))
    words = list(section.texts("words", ()))
    for name in section.texts("files", ()):
        words.extend(read_word_file(base_dir / name))
    return SpellingConfig(words=tuple(words))


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
    compare, audit = _parse_thresholds(section("thresholds"))
    return replace(
        general,
        html=_parse_html(section("html")),
        pdf=_parse_pdf(section("pdf")),
        compare=compare,
        audit=audit,
        spelling=_parse_spelling(section("spelling"), source.parent),
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
    if overrides.words:
        extra = [w for p in overrides.words for w in read_word_file(p)]
        changes["spelling"] = SpellingConfig(cfg.spelling.words + tuple(extra))
    if overrides.no_images:
        changes["images"] = False
    for name in (
        "lang",
        "min_severity",
        "formats",
        "output",
        "checks",
        "profile",
    ):
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
