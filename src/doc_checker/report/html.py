# Copyright 2026 Christophe Le Douarec
"""Standalone HTML report of a comparison.

The report is one file that opens from the disk: its style sheet and its
script are inline, and a content security policy lets them run, but nothing
else. The texts of the documents are escaped, and no URL of a document ever
becomes the target of a link or the source of a resource.
"""

import base64
import hashlib
import importlib.resources
from collections import Counter
from typing import TYPE_CHECKING

from doc_checker.model.findings import Severity
from doc_checker.report.context import (
    CATEGORY_TITLES,
    SEVERITY_TITLES,
    TEMPLATES,
    ReportContext,
    is_code,
)

if TYPE_CHECKING:
    from pathlib import Path

    from doc_checker.compare.engine import CompareResult


def _hash_source(text: str) -> str:
    """Return the source expression that allows one inline text."""
    digest = hashlib.sha256(text.encode()).digest()
    return f"'sha256-{base64.b64encode(digest).decode()}'"


def content_security_policy(css: str, script: str) -> str:
    """Return the content security policy of a report.

    Args:
        css: Inline style sheet, exactly as its ``style`` element holds it.
        script: Inline script, exactly as its ``script`` element holds it.

    Returns:
        A policy that allows the style sheet and the script by their SHA-256
        hashes, and images in data URLs, but no other resource, no base URL
        and no form submission.
    """
    return (
        "default-src 'none'; img-src data:; "
        f"style-src {_hash_source(css)}; "
        f"script-src {_hash_source(script)}; "
        "base-uri 'none'; form-action 'none'"
    )


def _asset(name: str) -> str:
    """Return the text of a style sheet or a script of the report."""
    resource = importlib.resources.files("doc_checker.report").joinpath(
        f"templates/{name}",
    )
    return resource.read_text(encoding="utf-8")


class _DomIds:
    """Identifiers of the DOM, unique even when finding identifiers repeat."""

    def __init__(self) -> None:
        """Start with no identifier given out."""
        self._given: Counter[str] = Counter()

    def __call__(self, finding_id: str) -> str:
        """Return the next DOM identifier of a finding.

        Args:
            finding_id: Identifier of the finding, which identical findings
                share.

        Returns:
            ``f-{id}`` the first time, then ``f-{id}-2``, ``f-{id}-3``...
        """
        self._given[finding_id] += 1
        count = self._given[finding_id]
        return f"f-{finding_id}" if count == 1 else f"f-{finding_id}-{count}"


def render_html(result: CompareResult, path: Path) -> Path:
    """Write the standalone HTML report of a comparison.

    Args:
        result: Comparison result.
        path: Report file to write.

    Returns:
        The written file.
    """
    context = ReportContext(result, path.parent)
    css = _asset("report.css")
    script = _asset("report.js")
    severities = sorted(Severity, reverse=True)
    text = TEMPLATES.get_template("report.html.j2").render(
        policy=content_security_policy(css, script),
        css=css,
        script=script,
        inputs=context.inputs(),
        matched=context.matched(),
        moved=result.stats.moved,
        severities=severities,
        counts=[
            (severity, len(context.findings_of(severity)))
            for severity in severities
        ],
        categories=context.categories(),
        rules=sorted({finding.rule for finding in result.findings}),
        summary=context.summary(),
        normalization=context.normalization(),
        chrome=context.chrome(),
        warnings=context.warnings(),
        sections=[
            (severity, [f for _, findings in groups for f in findings])
            for severity, groups in context.sections()
        ],
        total=len(result.findings),
        category_titles=CATEGORY_TITLES,
        severity_titles=SEVERITY_TITLES,
        source=context.source,
        shown=context.shown,
        hidden=context.hidden,
        is_code=is_code,
        dom_id=_DomIds(),
    )
    path.write_text(text, encoding="utf-8", newline="\n")
    return path
