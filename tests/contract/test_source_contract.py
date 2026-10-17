"""Contract suite every ``SourceAdapter`` must pass (Liskov substitution, DIP)."""

import contextlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import pytest

from f1sim.adapters.sources.fastf1_adapter import FastF1Adapter
from f1sim.adapters.sources.http import HttpClient, RateLimiter
from f1sim.adapters.sources.jolpica_adapter import JolpicaAdapter
from f1sim.adapters.store.tables import TABLES
from f1sim.core.domain.entities import EventKey, SessionKey, SessionType
from f1sim.core.errors import DataError
from f1sim.core.ports.sources import SourceAdapter
from tests.fixtures.fastf1_fakes import fake_session, loaders
from tests.fixtures.jolpica_transport import jolpica_transport

pytestmark = pytest.mark.contract

NOW = datetime(2025, 1, 1)
RACE = SessionKey(EventKey(2024, 1), SessionType.RACE)


@dataclass(frozen=True)
class SourceCase:
    """One source adapter together with a session it can fetch.

    Attributes:
        source: The adapter under test.
        season: A season with finished sessions it can list.
        session: A session the adapter can fetch.
    """

    source: SourceAdapter
    season: int
    session: SessionKey


@pytest.fixture(params=["jolpica", "fastf1"])
def case(request: pytest.FixtureRequest) -> SourceCase:
    """Each source adapter in turn; every test below runs against both."""
    if request.param == "jolpica":
        limiter = RateLimiter(per_second=1e6, burst=1000, per_hour=10**6, sleep=lambda _s: None)
        http = HttpClient(cache_dir=None, limiter=limiter, transport=jolpica_transport())
        return SourceCase(JolpicaAdapter(http, clock=lambda: NOW), 2024, RACE)
    load_session, load_schedule = loaders({(2024, 1, "R"): fake_session()})
    source = FastF1Adapter(
        load_session=load_session, load_schedule=load_schedule, clock=lambda: NOW
    )
    return SourceCase(source, 2024, RACE)


def test_name_is_a_non_empty_string(case: SourceCase) -> None:
    assert isinstance(case.source.name, str)
    assert case.source.name


def test_sessions_are_unique_keys_of_the_requested_season(case: SourceCase) -> None:
    found = case.source.sessions(case.season)
    assert isinstance(found, Sequence)
    keys = list(found)
    assert keys
    assert len(keys) == len(set(keys))
    assert all(isinstance(key, SessionKey) and key.event.season == case.season for key in keys)


def test_every_listed_session_is_fetchable_or_raises_data_error(case: SourceCase) -> None:
    for key in case.source.sessions(case.season):
        with contextlib.suppress(DataError):
            case.source.fetch(key)


def test_fetch_returns_canonical_tables(case: SourceCase) -> None:
    tables = case.source.fetch(case.session)
    assert tables
    for name, frame in tables.items():
        assert name in TABLES
        assert isinstance(frame, pd.DataFrame)
        assert set(TABLES[name].columns) <= set(frame.columns)


def test_fetch_of_an_unknown_session_raises_data_error(case: SourceCase) -> None:
    unknown = SessionKey(EventKey(case.season, 99), SessionType.RACE)
    with pytest.raises(DataError):
        case.source.fetch(unknown)
