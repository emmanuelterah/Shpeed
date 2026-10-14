"""Pace observations, the Kalman filter, the pace predictor and its fitted-state store."""

from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from f1sim.adapters.config_files import (
    load_circuits,
    load_engine_suppliers,
    load_pace_hyper,
    load_team_lineage,
)
from f1sim.adapters.models.pace import PacePredictor
from f1sim.adapters.models.pace_checks import predictive_check
from f1sim.adapters.models.pace_data import (
    QUALI,
    RACE,
    ObservationCache,
    SessionObs,
    build_observations,
    event_cluster,
)
from f1sim.adapters.models.pace_filter import PaceFilter, PaceHyper, PaceState
from f1sim.adapters.models.registry import ModelOptions, get_model, pace_context
from f1sim.adapters.store.memory_store import InMemoryStore
from f1sim.adapters.store.state_files import FileStateStore
from f1sim.core.domain.entities import EventKey, SnapshotId
from f1sim.core.domain.race import RaceEntry
from f1sim.core.domain.reference import CircuitTable, EngineSuppliers
from f1sim.core.errors import DataError
from f1sim.core.ports.state import StateKey
from f1sim.core.usecases.backtest import walk_forward
from tests.pace_world import DRIVERS, World, build_world
from tests.support import synthetic_results

FITTED = PaceHyper.from_dict(
    {
        "team_sd": 0.6,
        "team_step": 0.1,
        "team_season_step": 0.7,
        "quali_offset_sd": 0.03,
        "cluster_sd": 0.02,
        "driver_quali_sd": 0.15,
        "driver_race_sd": 0.19,
        "driver_corr": 0.6,
        "driver_season_step": 0.01,
        "noise_q": 0.04,
        "noise_r": 0.07,
        "noise_fp": 0.32,
        "order_noise": 0.1,
    }
)
"""Roughly what NUTS finds on the synthetic world, so the tests do not need NUTS."""


@pytest.fixture(scope="module")
def world() -> World:
    """Three synthetic seasons of six rounds."""
    return build_world(seasons=(2022, 2023, 2024), rounds=6)


@pytest.fixture(scope="module")
def observations(world: World) -> list[SessionObs]:
    """Pace observations for the whole world."""
    return build_observations(world.tables["laps"], world.tables["events"], load_circuits())


@pytest.fixture(scope="module")
def store(world: World) -> InMemoryStore:
    """The world committed as one snapshot."""
    s = InMemoryStore()
    s.commit(world.tables, run_id="world")
    return s


def _truth(world: World, obs: SessionObs) -> np.ndarray:
    kind = 0 if obs.kind == QUALI else 1
    values = np.array(
        [
            world.team_pace[(obs.season, obs.round, d.rsplit("_", 1)[0])]
            + world.driver_pace[d][kind]
            for d in obs.drivers
        ]
    )
    return values - np.median(values)


def _filter(hyper: PaceHyper = FITTED) -> PaceFilter:
    return PaceFilter(hyper, load_team_lineage(), load_engine_suppliers())


def test_observations_track_the_true_pace(world: World, observations: list[SessionObs]) -> None:
    by_session = {(o.season, o.round, o.session): o for o in observations}
    assert len(by_session) == 3 * 6 * 3
    for (_, _, session), obs in by_session.items():
        truth = _truth(world, obs)
        floor = 0.7 if session == "FP2" else 0.98
        assert np.corrcoef(truth, obs.values)[0, 1] > floor, session
    kinds = {o.session: o.kind for o in observations}
    assert kinds == {"FP2": RACE, "Q": QUALI, "R": RACE}
    times = [o.available_at for o in observations]
    assert times == sorted(times)


def test_observation_groups_and_clusters(observations: list[SessionObs]) -> None:
    assert {o.group for o in observations} == {"FP", "Q", "R"}
    clusters = {o.circuit: o.cluster for o in observations}
    assert clusters["monaco"] == "low"
    assert clusters["monza"] == "high"
    assert clusters["bahrain"] == "medium"


def test_cache_and_window_give_the_same_observations(world: World) -> None:
    laps, events = world.tables["laps"], world.tables["events"]
    cache = ObservationCache()
    circuits = load_circuits()
    first = build_observations(laps, events, circuits, cache=cache, store_key=1)
    again = build_observations(laps, events, circuits, cache=cache, store_key=1)
    assert [o.values.tolist() for o in first] == [o.values.tolist() for o in again]
    late = build_observations(laps, events, circuits, window=((2024, 3), None))
    assert {(o.season, o.round) for o in late} == {(2024, r) for r in range(4, 7)}
    assert not build_observations(laps.iloc[0:0], events, circuits)


def test_event_cluster_falls_back_to_medium(world: World) -> None:
    table = CircuitTable({})
    assert event_cluster(world.tables["laps"], (2024, 1), "nowhere", table) == "medium"
    assert event_cluster(world.tables["laps"].iloc[0:0], (2024, 1), "", load_circuits()) == "medium"


def test_filter_recovers_teammate_gaps(world: World, observations: list[SessionObs]) -> None:
    pace = _filter()
    for obs in observations:
        pace.update(obs)
    for team in ("mercedes", "ferrari", "red_bull", "mclaren", "williams"):
        first, second = pace.effect(f"dr:{team}_1"), pace.effect(f"dr:{team}_2")
        true_gap = world.driver_pace[f"{team}_1"][1] - world.driver_pace[f"{team}_2"][1]
        assert first[0] - second[0] == pytest.approx(true_gap, abs=0.1), team
    assert pace.effect("dr:nobody") == (0.0, pytest.approx(float("nan"), nan_ok=True))


def test_fitted_hyperparameters_pass_the_predictive_check(
    observations: list[SessionObs],
) -> None:
    context = pace_context(ModelOptions(pace_hyper=FITTED))
    check = predictive_check(observations, context, [2024])
    assert check.passed, check
    loose = predictive_check(observations, context.with_hyper(PaceHyper()), [2024])
    assert not loose.passed
    with pytest.raises(DataError):
        predictive_check(observations, context, [2030])


def test_season_start_widens_team_uncertainty(observations: list[SessionObs]) -> None:
    pace = _filter()
    for obs in observations:
        if obs.season == 2022:
            pace.update(obs)
    before = pace.effect("team:woking")[1]
    pace.advance(2023, 1)
    assert pace.effect("team:woking")[1] > before
    pace.advance(2023, 1)
    assert pace.effect("team:woking")[1] == pytest.approx(pace.effect("team:woking")[1])


def test_state_round_trips_and_rejects_bad_shapes(observations: list[SessionObs]) -> None:
    pace = _filter()
    pace.update(observations[0])
    saved = pace.state.to_dict()
    restored = PaceState.from_dict(saved)
    assert restored.labels == pace.state.labels
    assert np.allclose(restored.cov, pace.state.cov)
    assert restored.last_event == pace.state.last_event
    with pytest.raises(DataError):
        PaceState.from_dict(saved | {"labels": ["x"]})


def test_race_pace_forecast_shapes(observations: list[SessionObs]) -> None:
    pace = _filter()
    for obs in observations[:6]:
        pace.update(obs)
    mean, cov = pace.race_pace(2022, "medium", ["mercedes_1", "rookie"], ["mercedes", "haas"])
    assert mean.shape == (2,)
    assert cov.shape == (2, 2)
    assert cov[1, 1] > cov[0, 0]


def test_hyperparameters_round_trip_and_validate(tmp_path: Path) -> None:
    assert PaceHyper.from_dict(FITTED.to_dict()) == FITTED
    assert FITTED.noise_sd("FP") == FITTED.noise.fp
    with pytest.raises(DataError):
        PaceHyper.from_dict({"team_sd": "fast"})
    with pytest.raises(DataError):
        PaceHyper.from_dict({"team_sd": -1})
    with pytest.raises(DataError):
        PaceHyper.from_dict({"driver_corr": 1.5})
    assert load_pace_hyper(tmp_path / "missing.json") == PaceHyper()
    path = tmp_path / "hyper.json"
    path.write_text('{"hyper": {"team_sd": 2.0}}')
    assert load_pace_hyper(path).team.sd == 2.0
    path.write_text("{oops")
    with pytest.raises(DataError):
        load_pace_hyper(path)


def test_reference_tables_load() -> None:
    circuits = load_circuits()
    speed = circuits.average_speed("monza", 79.0)
    assert speed is not None
    assert circuits.cluster(speed) == "high"
    assert circuits.average_speed("nowhere", 80.0) is None
    suppliers = load_engine_suppliers()
    assert suppliers.supplier_of(2024, "williams") == "mercedes"
    assert suppliers.supplier_of(2018, "red_bull") == "renault"
    assert suppliers.supplier_of(2024, "new_team") == "new_team"
    assert EngineSuppliers({}).supplier_of(2024, "x") == "x"


def test_pace_model_beats_the_baselines(store: InMemoryStore) -> None:
    snap = store.latest_snapshot()
    options = ModelOptions(pace_hyper=FITTED)
    scores = {
        name: walk_forward(
            store, snapshot=snap, seasons=(2024,), model=get_model(name, options)
        ).summary()["rps"]
        for name in ("pace", "plackett_luce", "grid_noise")
    }
    assert scores["pace"] < scores["plackett_luce"]
    assert scores["pace"] < scores["grid_noise"]


def _entries(store: InMemoryStore, round_: int) -> list[RaceEntry]:
    results = store.read("results", snapshot=store.latest_snapshot(), as_of=datetime(2100, 1, 1))
    quali = results.loc[(results["season"] == 2024) & (results["round"] == round_)]
    quali = quali.loc[quali["session"] == "Q"].sort_values("position")
    return [
        RaceEntry(str(d), str(t), i + 1)
        for i, (d, t) in enumerate(zip(quali["driver_id"], quali["team_id"], strict=True))
    ]


def test_weekend_update_resumes_from_the_saved_state(store: InMemoryStore, tmp_path: Path) -> None:
    snap = store.latest_snapshot()
    context = replace(
        pace_context(ModelOptions(pace_hyper=FITTED)), state_store=FileStateStore(tmp_path)
    )
    after_quali = datetime(2024, 4, 27, 16, 0)  # round 5 qualifying has ended
    first = PacePredictor(0, context)
    first.fit(store, snapshot=snap, cutoff=after_quali)
    assert first.report.saved == EventKey(2024, 4)
    assert first.report.resumed_from is None
    second = PacePredictor(0, context)
    second.fit(store, snapshot=snap, cutoff=after_quali)
    assert second.report.resumed_from == EventKey(2024, 4)
    assert second.report.sessions == 2
    assert np.allclose(first.filter.state.mean, second.filter.state.mean)
    entries = _entries(store, 5)
    a = first.predict(EventKey(2024, 5), entries)
    b = second.predict(EventKey(2024, 5), entries)
    assert np.allclose(a.p_position, b.p_position)
    assert a.drivers == tuple(DRIVERS[i] for i in [DRIVERS.index(e.driver_id) for e in entries])


def test_changed_history_is_refitted_not_resumed(world: World, tmp_path: Path) -> None:
    context = replace(
        pace_context(ModelOptions(pace_hyper=FITTED)), state_store=FileStateStore(tmp_path)
    )
    cutoff = datetime(2024, 4, 27, 16, 0)
    original = InMemoryStore()
    original.commit(world.tables, run_id="a")
    PacePredictor(0, context).fit(original, snapshot=original.latest_snapshot(), cutoff=cutoff)
    laps = world.tables["laps"].copy()
    times = laps["lap_time_s"].to_numpy(copy=True)
    times[0] += 1.0
    laps["lap_time_s"] = times
    edited = InMemoryStore()
    edited.commit({**world.tables, "laps": laps}, run_id="b")
    model = PacePredictor(0, context)
    model.fit(edited, snapshot=edited.latest_snapshot(), cutoff=cutoff)
    assert model.report.resumed_from is None


def test_state_files_are_atomic_and_pick_the_newest_snapshot(tmp_path: Path) -> None:
    files = FileStateStore(tmp_path)
    event = EventKey(2024, 4)
    for snap in (3, 5, 9):
        files.save(StateKey("pace", "1-x", event, SnapshotId(snap)), {"snap": snap})
    found = files.load_latest("pace", "1-x", event, SnapshotId(6))
    assert found == (SnapshotId(5), {"snap": 5})
    assert files.load_latest("pace", "1-x", EventKey(2024, 5), SnapshotId(9)) is None
    assert files.load_latest("pace", "2-y", event, SnapshotId(9)) is None
    assert not list(tmp_path.rglob(".tmp-*"))
    files.path(StateKey("pace", "1-x", event, SnapshotId(9))).write_text("not json")
    assert files.load_latest("pace", "1-x", event, SnapshotId(9)) is None
    with pytest.raises(DataError):
        files.save(StateKey("../pace", "1", event, SnapshotId(1)), {})


def test_pace_model_without_laps_is_uninformed() -> None:
    empty = InMemoryStore()
    empty.commit({"results": synthetic_results(seasons=(2023,), rounds=1)}, run_id="none")
    model = PacePredictor(0, pace_context(ModelOptions()))
    model.fit(empty, snapshot=empty.latest_snapshot(), cutoff=datetime(2024, 1, 1))
    entries = [RaceEntry("a", "t1", 1), RaceEntry("b", "t2", 2)]
    matrix = model.predict(EventKey(2024, 1), entries)
    assert matrix.p_win()[0] == pytest.approx(matrix.p_win()[1], abs=0.05)
