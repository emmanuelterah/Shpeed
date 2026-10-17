"""Shared fixtures. Tests never touch the network (P1-18)."""

import socket
from typing import Any

import numpy as np
import pytest

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", ""})


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any socket connection fail, so a test cannot reach the network by accident.

    Loopback connections stay allowed: on Windows the asyncio event loop builds a
    loopback self-pipe when the API test client starts, and that is not network access.
    """
    real_connect = socket.socket.connect

    def guard(sock: socket.socket, address: Any, *args: Any, **kwargs: Any) -> None:
        host = address[0] if isinstance(address, tuple) and address else address
        if isinstance(host, str) and host in _LOOPBACK_HOSTS:
            real_connect(sock, address, *args, **kwargs)
            return
        raise RuntimeError("Network access in tests is not allowed; use recorded fixtures.")

    monkeypatch.setattr(socket.socket, "connect", guard)


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded random generator, so randomised tests are reproducible."""
    return np.random.default_rng(20261012)
