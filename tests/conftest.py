"""Shared fixtures. Tests never touch the network (P1-18)."""

import socket

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any socket connection fail, so a test cannot reach the network by accident."""

    def guard(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("Network access in tests is not allowed; use recorded fixtures.")

    monkeypatch.setattr(socket.socket, "connect", guard)


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded random generator, so randomised tests are reproducible."""
    return np.random.default_rng(20261012)
