# Copyright 2026 Christophe Le Douarec
"""Errors raised by Doc Checker."""


class DocCheckerError(Exception):
    """Base class of the errors reported to the user."""


class ConfigError(DocCheckerError):
    """Invalid configuration file or option value."""


class InputError(DocCheckerError):
    """Unreadable or unsupported input document."""
