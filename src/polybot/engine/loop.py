"""Task E. Main loop: scan -> fair price -> edge -> risk -> paper execution -> journal."""

from __future__ import annotations

from polybot.config import Settings


def run_once(settings: Settings) -> None:
    raise NotImplementedError


def run(settings: Settings) -> None:
    """Loop forever; survive API errors with backoff; stop cleanly on Ctrl+C."""
    raise NotImplementedError
