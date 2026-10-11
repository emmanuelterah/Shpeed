"""The container health check command."""

import urllib.error
from types import TracebackType

import pytest

from f1sim.api import healthcheck


class FakeResponse:
    """Stands in for the object ``urllib.request.urlopen`` returns."""

    def __init__(self, status: int) -> None:
        """Answer with ``status``."""
        self.status = status

    def __enter__(self) -> "FakeResponse":
        """Enter the ``with`` block."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Leave the ``with`` block."""


def test_healthy_api_exits_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(healthcheck.urllib.request, "urlopen", lambda *_a, **_k: FakeResponse(200))
    assert healthcheck.main() == 0


def test_error_status_is_unhealthy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(healthcheck.urllib.request, "urlopen", lambda *_a, **_k: FakeResponse(503))
    assert not healthcheck.check()


def test_unreachable_api_exits_one(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: object, **_kwargs: object) -> FakeResponse:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(healthcheck.urllib.request, "urlopen", refuse)
    assert healthcheck.main() == 1
