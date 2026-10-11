"""Type stubs for the parts of FastF1 that Shpeed uses.

FastF1 ships without type information, so pyright (and Pylance) would treat every value from
it as unknown. These stubs describe only the calls made in
``f1sim.adapters.sources.fastf1_adapter``; extend them when the adapter uses more of FastF1.
Signatures follow fastf1 3.8.
"""

from typing import Literal

import pandas as pd

_Backend = Literal["fastf1", "f1timing", "ergast"] | None

class Cache:
    """FastF1's on-disk request cache."""

    @classmethod
    def enable_cache(
        cls,
        cache_dir: str,
        ignore_version: bool = ...,
        force_renew: bool = ...,
        use_requests_cache: bool = ...,
    ) -> None: ...

class Session:
    """One session of a race weekend, populated by ``load``."""

    laps: pd.DataFrame
    results: pd.DataFrame
    weather_data: pd.DataFrame
    track_status: pd.DataFrame
    date: pd.Timestamp
    t0_date: pd.Timestamp

    def load(
        self,
        *,
        laps: bool = ...,
        telemetry: bool = ...,
        weather: bool = ...,
        messages: bool = ...,
    ) -> None: ...

def get_session(
    year: int,
    gp: str | int,
    identifier: int | str | None = ...,
    *,
    backend: _Backend = ...,
    force_ergast: bool = ...,
) -> Session: ...
def get_event_schedule(
    year: int,
    *,
    include_testing: bool = ...,
    backend: _Backend = ...,
    force_ergast: bool = ...,
) -> pd.DataFrame: ...
