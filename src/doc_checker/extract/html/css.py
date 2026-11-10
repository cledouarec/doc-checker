# Copyright 2026 Christophe Le Douarec
"""Find the elements that CSS selectors match, in a time linear in the page.

lxml turns a group of selectors, such as ``a, b``, into one XPath union,
which libxml2 merges in a time that grows as the product of what each side
finds: a page of 40,000 elements of each side takes seconds. Each selector of
a group is therefore run alone, and what they find is merged here. And
``:first-child``, which cssselect turns into a count of the siblings before
an element, quadratic in their number, is written as the first child of the
parent, which libxml2 finds at once.
"""

from functools import cache
from typing import TYPE_CHECKING, override

from cssselect import parse
from lxml import etree
from lxml.cssselect import LxmlHTMLTranslator

if TYPE_CHECKING:
    from cssselect.xpath import XPathExpr
    from lxml.html import HtmlElement


class _Translator(LxmlHTMLTranslator):
    """The translator of lxml for HTML, with a linear ``:first-child``."""

    @override
    def xpath_first_child_pseudo(self, xpath: XPathExpr) -> XPathExpr:
        """Keep an element that is the first element child of its parent.

        Args:
            xpath: The path of the element.

        Returns:
            The path, with that condition.
        """
        return xpath.add_condition("count(. | ../*[1]) = 1")


@cache
def _paths(selector: str) -> tuple[etree.XPath, ...]:
    """Return the XPath of each selector of a group, as lxml would run it.

    A selector that is not valid CSS, or not one that lxml can run, raises
    the error of cssselect that lxml raises.
    """
    translator = _Translator()
    return tuple(
        etree.XPath(
            translator.selector_to_xpath(
                part,
                translate_pseudo_elements=True,
            ),
        )
        for part in parse(selector)
    )


def select(root: HtmlElement, selector: str) -> list[HtmlElement]:
    """Return the elements that a CSS selector matches, as lxml does.

    Args:
        root: The element searched, itself included, with its descendants.
        selector: A CSS selector, or a group of them separated by commas.

    Returns:
        The elements, each once, in document order.
    """
    found = [nodes for path in _paths(selector) if (nodes := path(root))]
    if len(found) <= 1:
        return found[0] if found else []
    matched = {node for nodes in found for node in nodes}
    return [node for node in root.getroottree().iter() if node in matched]


def matches(root: HtmlElement, selector: str) -> bool:
    """Return whether a CSS selector matches an element or a descendant.

    Args:
        root: The element searched, itself included, with its descendants.
        selector: A CSS selector, or a group of them separated by commas.

    Returns:
        Whether one selector of the group matches.
    """
    return any(path(root) for path in _paths(selector))
