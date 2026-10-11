"""The leakage guard, race planning and the walk-forward backtester."""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from f1sim.adapters.store.memory_store import InMemoryStore
from f1sim.core.domain import PositionMatrix
from f1sim.core.domain.entities import EventKey, SnapshotId
from f1sim.core.domain.race import RaceEntry
from f1sim.core.errors import DataError, InvariantError, LeakageError
from f1sim.core.ports.predictor import ModelSpec, Predictor
from f1sim.core.ports.store import DataReader
from f1sim.core.usecases.backtest import BacktestResult, plan_races, race_seed, walk_forward
from f1sim.core.usecases.readers import FAR_FUTURE, CutoffReader, FrameReader, as_naive_utc
from tests.support import UniformModel, synthetic_results


def _store(
    seasons: tuple[int, ...], rounds: int, sprint_rounds: tuple[int, ...] = ()
) -> tuple[InMemoryStore, SnapshotId]:
    store = InMemoryStore()
    df = synthetic_results(seasons=seasons, rounds=rounds, sprint_rounds=sprint_rounds)
    return store, store.commit({"results": df}, run_id="t")


Uniform = UniformModel


class Cheater(Uniform):
    """Tries to read the race result before it happens."""

    name = "cheater"

    def fit(self, reader: DataReader, *, snapshot: SnapshotId, cutoff: datetime) -> None:
        """Read a day past the cutoff."""
        reader.read("results", snapshot=snapshot, as_of=cutoff + timedelta(days=1))


class Shuffler(Uniform):
    """Returns rows in the wrong order."""

    name = "shuffler"

    def predict(self, event: EventKey, entries: Sequence[RaceEntry]) -> PositionMatrix:
        """Forecast the entries in reverse order."""
        return super().predict(event, list(reversed(entries)))


def _spec(cls: type[Uniform]) -> ModelSpec:
    def _build(seed: int) -> Predictor:
        return cls(seed)

    return ModelSpec(cls.name, _build)


def test_cutoff_reader_refuses_reads_past_the_cutoff() -> None:
    store, snap = _store(seasons=(2024,), rounds=2)
    cutoff = datetime(2024, 3, 1, 15, 0)
    guarded = CutoffReader(store, cutoff)
    assert guarded.cutoff == cutoff
    assert guarded.latest_snapshot() == snap
    rows = guarded.read("results", snapshot=snap, as_of=cutoff)
    assert set(rows["session"]) == {"Q"}
    with pytest.raises(LeakageError):
        guarded.read("results", snapshot=snap, as_of=cutoff + timedelta(seconds=1))


def test_cutoff_reader_catches_a_store_that_ignores_as_of() -> None:
    class Leaky:
        """A broken store that returns every row whatever the cutoff."""

        def latest_snapshot(self) -> SnapshotId:
            """Report a fixed snapshot."""
            return SnapshotId(1)

        def read(self, table: str, *, snapshot: SnapshotId, as_of: datetime) -> pd.DataFrame:
            """Ignore the arguments and return a whole weekend."""
            del table, snapshot, as_of
            return synthetic_results(seasons=(2024,), rounds=1)

    with pytest.raises(LeakageError):
        CutoffReader(Leaky(), datetime(2024, 1, 1)).read(
            "results", snapshot=SnapshotId(1), as_of=datetime(2024, 1, 1)
        )


def test_frame_reader_serves_one_snapshot_from_memory() -> None:
    store, snap = _store(seasons=(2024,), rounds=1)
    frames = FrameReader.load(store, snap, ("results",))
    assert frames.latest_snapshot() == snap
    aware = datetime(2024, 3, 1, 15, 0, tzinfo=UTC)
    assert len(frames.read("results", snapshot=snap, as_of=aware)) == 10
    with pytest.raises(DataError):
        frames.read("results", snapshot=SnapshotId(99), as_of=FAR_FUTURE)
    with pytest.raises(DataError):
        frames.read("laps", snapshot=snap, as_of=FAR_FUTURE)


def test_as_naive_utc_converts_aware_times() -> None:
    aware = datetime(2024, 1, 1, 12, tzinfo=UTC)
    assert as_naive_utc(aware) == datetime(2024, 1, 1, 12)


def test_plan_uses_qualifying_cutoff_and_grid() -> None:
    store, snap = _store(seasons=(2024,), rounds=3)
    plans, skipped = plan_races(store, snap, [2024])
    assert [p.event.round for p in plans] == [1, 2, 3]
    assert not skipped
    first = plans[0]
    assert first.cutoff == datetime(2024, 3, 1, 15, 0)
    assert [e.grid for e in first.entries] == list(range(1, 11))
    assert len(first.classified) <= 10


def test_sprint_sets_the_grid_in_2021_and_2022_only() -> None:
    for season, expected_cutoff_hours in ((2022, 24), (2024, 24)):
        store, snap = _store(seasons=(season,), rounds=1, sprint_rounds=(1,))
        plan = plan_races(store, snap, [season])[0][0]
        results = store.read("results", snapshot=snap, as_of=FAR_FUTURE)
        grid_session = "S" if season == 2022 else "Q"
        source = results.loc[(results["session"] == grid_session) & results["position"].notna()]
        leader = source.sort_values("position")["driver_id"].iloc[0]
        assert plan.entries[0].driver_id == leader
        assert plan.cutoff == datetime(season, 3, 1, 15) + timedelta(hours=expected_cutoff_hours)


def test_race_without_pre_race_results_is_skipped() -> None:
    df = synthetic_results(seasons=(2024,), rounds=2)
    df = df.loc[~((df["round"] == 2) & (df["session"] == "Q"))]
    store = InMemoryStore()
    snap = store.commit({"results": df}, run_id="t")
    plans, skipped = plan_races(store, snap, [2024])
    assert [p.event.round for p in plans] == [1]
    assert skipped[0].event == EventKey(2024, 2)


def test_walk_forward_scores_every_race_and_summarises() -> None:
    store, snap = _store(seasons=(2024,), rounds=4)
    result = walk_forward(store, snapshot=snap, seasons=(2024,), model=_spec(Uniform))
    assert result.model == "uniform"
    assert len(result.races) == 4
    summary = result.summary()
    assert summary["races"] == 4.0
    assert 0 < summary["rps"] < 1
    assert set(result.calibration) == {"win", "podium", "top10"}
    assert BacktestResult("x", (2024,), snap, []).summary() == {}


def test_a_model_that_peeks_fails_the_backtest() -> None:
    store, snap = _store(seasons=(2024,), rounds=2)
    with pytest.raises(LeakageError):
        walk_forward(store, snapshot=snap, seasons=(2024,), model=_spec(Cheater))


def test_rows_out_of_entry_order_are_rejected() -> None:
    store, snap = _store(seasons=(2024,), rounds=1)
    with pytest.raises(InvariantError):
        walk_forward(store, snapshot=snap, seasons=(2024,), model=_spec(Shuffler))


def test_race_seeds_differ_per_race_and_base() -> None:
    seeds = {race_seed(b, EventKey(s, r)) for b in (0, 1) for s in (2023, 2024) for r in (1, 2)}
    assert len(seeds) == 8
