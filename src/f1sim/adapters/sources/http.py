"""Shared HTTP layer (P1-04): disk cache, rate limiting, retry with backoff and jitter.

Jolpica-F1 allows short bursts of about 4 requests a second and about 500 an hour for
unauthenticated clients, so both limits are enforced client-side. Responses for finished
sessions never change, so callers cache them; schedules for a live season are not cached.
"""

import hashlib
import json
import random
import time
from collections import deque
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import httpx
import structlog

from f1sim import __version__
from f1sim.core.errors import DataError

log = structlog.get_logger(__name__)

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
HOUR_S = 3600
FIRST_ERROR_STATUS = 400


class RateLimiter:
    """Token bucket for bursts plus a sliding one-hour window."""

    def __init__(
        self,
        *,
        per_second: float,
        burst: int,
        per_hour: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._rate = per_second
        self._burst = burst
        self._per_hour = per_hour
        self._clock = clock
        self._sleep = sleep
        self._tokens = float(burst)
        self._last = clock()
        self._hour: deque[float] = deque()

    def acquire(self) -> None:
        now = self._clock()
        while self._hour and now - self._hour[0] >= HOUR_S:
            self._hour.popleft()
        if len(self._hour) >= self._per_hour:
            wait = HOUR_S - (now - self._hour[0])
            log.info("http.hourly_limit_wait", seconds=round(wait, 1))
            self._sleep(wait)
            now = self._clock()
            self._hour.popleft()
        self._tokens = min(self._burst, self._tokens + (now - self._last) * self._rate)
        self._last = now
        if self._tokens < 1:
            self._sleep((1 - self._tokens) / self._rate)
            now = self._clock()
            self._tokens = 1.0
            self._last = now
        self._tokens -= 1
        self._hour.append(now)


class HttpClient:
    def __init__(
        self,
        *,
        cache_dir: Path | None,
        limiter: RateLimiter | None = None,
        max_retries: int = 5,
        backoff_s: float = 1.0,
        max_backoff_s: float = 60.0,
        timeout_s: float = 30.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        self._cache_dir = cache_dir
        self._limiter = limiter or RateLimiter(per_second=4.0, burst=4, per_hour=500, sleep=sleep)
        self._max_retries = max_retries
        self._backoff = backoff_s
        self._max_backoff = max_backoff_s
        self._sleep = sleep
        self._jitter = jitter
        self._client = httpx.Client(
            timeout=timeout_s,
            transport=transport,
            headers={"User-Agent": f"shpeed/{__version__} (open-source F1 simulator)"},
            follow_redirects=True,
        )

    def get_json(
        self, url: str, params: Mapping[str, str | int] | None = None, *, cache: bool = True
    ) -> Any:
        """GET ``url`` and decode JSON. Raises ``DataError`` once retries are exhausted."""
        path = self._cache_path(url, params) if cache else None
        if path is not None and path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        data = self._fetch(url, params)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data), encoding="utf-8")
            tmp.replace(path)  # atomic: a crash never leaves a half-written cache entry
        return data

    def _fetch(self, url: str, params: Mapping[str, str | int] | None) -> Any:
        last_error = ""
        for attempt in range(self._max_retries + 1):
            self._limiter.acquire()
            try:
                response = self._client.get(url, params=dict(params or {}))
            except httpx.TransportError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                retry_after = None
            else:
                if response.status_code < FIRST_ERROR_STATUS:
                    return response.json()
                if response.status_code not in RETRYABLE_STATUS:
                    raise DataError(f"GET {response.url} returned {response.status_code}")
                last_error = f"HTTP {response.status_code}"
                retry_after = _retry_after(response)
            if attempt == self._max_retries:
                break
            wait = retry_after if retry_after is not None else self._backoff_for(attempt)
            log.warning(
                "http.retry", url=url, attempt=attempt + 1, wait_s=round(wait, 2), error=last_error
            )
            self._sleep(wait)
        raise DataError(f"GET {url} failed after {self._max_retries + 1} attempts: {last_error}")

    def _backoff_for(self, attempt: int) -> float:
        return min(self._max_backoff, self._backoff * 2**attempt + self._jitter() * self._backoff)

    def _cache_path(self, url: str, params: Mapping[str, str | int] | None) -> Path | None:
        if self._cache_dir is None:
            return None
        key = url + "?" + "&".join(f"{k}={v}" for k, v in sorted((params or {}).items()))
        digest = hashlib.sha256(key.encode()).hexdigest()[:32]
        return self._cache_dir / digest[:2] / f"{digest}.json"

    def close(self) -> None:
        self._client.close()


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None
