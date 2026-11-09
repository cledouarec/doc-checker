# Copyright 2026 Christophe Le Douarec
"""Audit a document: run the families of checks it is asked for.

The family `STRUCTURE` checks the headings, the references, the captions
and the pages of the document, by `structure_drafts` of
`doc_checker.audit.structure`, then its links, by `broken_link_drafts` of
`doc_checker.audit.links`. The drafts of the families then become findings
as those of `compare` do, by `finalize_drafts` of
`doc_checker.compare.aggregate`: severities, filters, identifiers, groups.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from doc_checker.audit.links import broken_link_drafts
from doc_checker.audit.structure import structure_drafts
from doc_checker.compare.aggregate import finalize_drafts
from doc_checker.model.document import BlockKind

if TYPE_CHECKING:
    from doc_checker.compare.drafts import Draft
    from doc_checker.config import Config
    from doc_checker.model.document import Document
    from doc_checker.model.findings import Finding

STRUCTURE: Final = "structure"
A11Y: Final = "a11y"
IMAGES: Final = "images"


@dataclass(frozen=True, slots=True)
class AuditStats:
    """Figures about an audit.

    Attributes:
        pages: Pages of the document: HTML pages or PDF pages.
        blocks: Blocks that are not images.
        images: Image blocks.
        links: Links checked.
        checks: Families of checks run.
        profile: Profile of the audit, ``"screen"`` or ``"print"``.
        images_read: Whether the images were read.
    """

    pages: int
    blocks: int
    images: int
    links: int
    checks: tuple[str, ...]
    profile: str
    images_read: bool


@dataclass(slots=True)
class AuditResult:
    """Outcome of the audit of a document."""

    doc: Document
    findings: list[Finding]
    stats: AuditStats


def audit_document(doc: Document, cfg: Config) -> AuditResult:
    """Audit a document.

    Args:
        doc: The document, loaded with its audit facts.
        cfg: Run configuration: its families of checks, its profile, its
            severities and its filters.

    Returns:
        The findings, those of the structure first, in the order of the
        document, and statistics.
    """
    drafts: list[Draft] = []
    links = 0
    if STRUCTURE in cfg.checks:
        drafts += structure_drafts(doc)
        broken, links = broken_link_drafts(doc)
        drafts += broken
    images = sum(block.kind is BlockKind.IMAGE for block in doc.blocks)
    stats = AuditStats(
        pages=len(doc.parts),
        blocks=len(doc.blocks) - images,
        images=images,
        links=links,
        checks=cfg.checks,
        profile=cfg.profile,
        images_read=cfg.images,
    )
    return AuditResult(doc, finalize_drafts(drafts, cfg), stats)
