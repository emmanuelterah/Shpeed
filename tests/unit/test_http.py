from pathlib import Path

import pytest

httpx = pytest.importorskip("httpx")

from f1sim.adapters.sources.http import HttpClient, RateLimiter
from f1sim.core.errors import DataError


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def _client(handler, tmp_path: Path | None = None, **kw) -> tuple[HttpClient, FakeClock]:  # type: ignore[no-untyped-def]
    clock = FakeClock()
    limiter = RateLimiter(
        per_second=100, burst=100, per_hour=10_000, clock=clock, sleep=clock.sleep
    )
    client = HttpClient(
        cache_dir=tmp_path,
        limiter=limiter,
        transport=httpx.MockTransport(handler),
        sleep=clock.sleep,
        jitter=lambda: 0.0,
        **kw,
    )
    return client, clock


def test_retries_server_errors_with_exponential_backoff() -> None:
    calls = {"n": 0}

    def handler(request):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        return httpx.Response(503) if calls["n"] < 3 else httpx.Response(200, json={"ok": True})

    client, clock = _client(handler)
    assert client.get_json("https://example.test/a", cache=False) == {"ok": True}
    assert clock.slept == [1.0, 2.0]


def test_honours_retry_after_on_429() -> None:
    responses = iter(
        [httpx.Response(429, headers={"Retry-After": "7"}), httpx.Response(200, json=[])]
    )
    client, clock = _client(lambda r: next(responses))  # type: ignore[no-untyped-def]
    client.get_json("https://example.test/a", cache=False)
    assert clock.slept == [7.0]


def test_client_errors_fail_immediately() -> None:
    client, clock = _client(lambda r: httpx.Response(404))  # type: ignore[no-untyped-def]
    with pytest.raises(DataError, match="404"):
        client.get_json("https://example.test/missing", cache=False)
    assert clock.slept == []


def test_gives_up_after_max_retries() -> None:
    client, _ = _client(lambda r: httpx.Response(500), max_retries=2)  # type: ignore[no-untyped-def]
    with pytest.raises(DataError, match="3 attempts"):
        client.get_json("https://example.test/a", cache=False)


def test_cache_serves_repeat_requests_without_network(tmp_path: Path) -> None:
    calls = {"n": 0}

    def handler(request):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        return httpx.Response(200, json={"v": 1})

    client, _ = _client(handler, tmp_path)
    for _ in range(3):
        assert client.get_json("https://example.test/a", {"limit": 100}) == {"v": 1}
    assert calls["n"] == 1
    assert len(list(tmp_path.rglob("*.json"))) == 1


def test_rate_limiter_spaces_out_bursts() -> None:
    clock = FakeClock()
    limiter = RateLimiter(per_second=4, burst=4, per_hour=500, clock=clock, sleep=clock.sleep)
    for _ in range(8):
        limiter.acquire()
    assert sum(clock.slept) == pytest.approx(1.0)


def test_rate_limiter_enforces_hourly_cap() -> None:
    clock = FakeClock()
    limiter = RateLimiter(per_second=1000, burst=1000, per_hour=3, clock=clock, sleep=clock.sleep)
    for _ in range(4):
        limiter.acquire()
    assert clock.now >= 3600
