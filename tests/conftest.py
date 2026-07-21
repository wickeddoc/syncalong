"""Shared pytest fixtures for the syncalong test suite."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clear_syncalong_env(monkeypatch):
    """Clear SYNCALONG_* env vars so the suite is independent of the shell.

    The server and CLI read these as fallbacks; a developer who exported them
    (as the docs suggest) would otherwise see spurious auth/model/server
    failures. Tests that need a value set it explicitly via monkeypatch.
    """
    for var in (
        "SYNCALONG_TOKEN",
        "SYNCALONG_SERVER",
        "SYNCALONG_MODEL",
        "SYNCALONG_DEVICE",
    ):
        monkeypatch.delenv(var, raising=False)
