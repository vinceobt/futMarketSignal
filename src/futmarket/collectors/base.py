"""Shared error type for the data collectors."""

from __future__ import annotations


class SourceError(RuntimeError):
    """Raised when a collector cannot fetch or parse what it needs."""
