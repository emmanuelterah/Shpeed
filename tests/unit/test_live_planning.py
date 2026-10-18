"""Planning a live forecast: qualifying first, latest weekend, calendar lookups (P8-06)."""

from datetime import UTC, datetime

import pandas as pd
import pytest

from f1sim.adapters.store.memory_store import InMemoryStore
from f1sim.core.domain.entities import EventKey, SnapshotId
from f1sim.core.errors import DataError
from f1sim.core.usecases.planning import (
    event_details,
    latest_raced,
    latest_unraced,
    official_classification,
    plan_event,
)
from tests.support import synthetic_results


def _store(results: pd.DataFrame, **tables: pd.DataFrame) -> tuple[InMemoryStore, SnapshotId]:
    """A store holding ``results`` and any extra tables."""
    store = InMemoryStore()
    return store, store.commit({"results": results, **tables}, run_id="t")


def _season(rounds: int = 4, sprint_rounds: tuple[int, ...] = ()) -> pd.DataFrame:
    """A synthetic 2024 season."""
    return synthetic_results(seasons=(2024,), rounds=rounds, sprint_rounds=sprint_rounds)


def _without(df: pd.DataFrame, round_: int, *sessions: str) -> pd.DataFrame:
    """Drop some sessions of one round, as if they had not happened yet."""
    gone = (df["round"] == round_) & df["session"].isin(list(sessions))
    return df.loc[~gone].reset_index(drop=True)


def _events(start: datetime) -> pd.DataFrame:
    """A one-row calendar for 2024 round 4."""
    return pd.DataFrame(
        [
            {
                "season": 2024,
                "round": 4,
                "event_name": "Test Grand Prix",
                "circuit_id": "test",
                "locality": "Testville",
                "country": "Testland",
                "event_date": start,
                "event_format": "conventional",
                "available_at": datetime(2024, 1, 1),
            }
        ]
    )


def test_a_weekend_past_qualifying_can_be_planned_before_the_race() -> None:
    store, snap = _store(_without(_season(), 4, "R"))
    plan = plan_event(store, snap, EventKey(2024, 4))
    assert len(plan.entries) == 10
    assert not plan.classified


def test_a_sprint_result_is_never_used_as_the_live_grid() -> None:
    store, snap = _store(_without(_season(sprint_rounds=(4,)), 4, "Q", "R"))
    with pytest.raises(DataError, match="qualifying"):
        plan_event(store, snap, EventKey(2024, 4))


def test_latest_unraced_is_the_newest_qualified_weekend_without_a_result() -> None:
    store, snap = _store(_without(_season(), 4, "R"))
    assert latest_unraced(store, snap, 2024) == EventKey(2024, 4)


def test_latest_unraced_waits_for_qualifying_on_a_sprint_weekend() -> None:
    store, snap = _store(_without(_season(sprint_rounds=(4,)), 4, "Q", "R"))
    assert latest_unraced(store, snap, 2024) is None


def test_nothing_is_waiting_when_every_weekend_is_raced() -> None:
    store, snap = _store(_season())
    assert latest_unraced(store, snap, 2024) is None
    assert latest_raced(store, snap, 2024) == EventKey(2024, 4)
    assert latest_raced(store, snap, 2025) is None


def test_official_classification_needs_a_race_result() -> None:
    store, snap = _store(_without(_season(), 4, "R"))
    assert len(official_classification(store, snap, EventKey(2024, 3))) >= 1
    with pytest.raises(DataError):
        official_classification(store, snap, EventKey(2024, 4))


def test_event_details_reads_name_and_lights_out_as_naive_utc() -> None:
    start = datetime(2024, 3, 24, 15, 0, tzinfo=UTC)
    store, snap = _store(_season(), events=_events(start))
    details = event_details(store, snap, EventKey(2024, 4))
    assert details.name == "Test Grand Prix"
    assert details.lights_out == datetime(2024, 3, 24, 15, 0)


def test_event_details_are_empty_without_a_calendar_row() -> None:
    store, snap = _store(_season())
    details = event_details(store, snap, EventKey(2024, 4))
    assert details.name is None
    assert details.lights_out is None
