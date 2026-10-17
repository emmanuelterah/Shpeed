"""The tyre model recovers the degradation, warm-up, pit loss and stop time in the laps."""

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from f1sim.adapters.config_files import load_tyre_model
from f1sim.adapters.models.pit_stops import race_stops
from f1sim.adapters.models.tyres import (
    AGE_GRID,
    WARMUP_LAPS,
    CompoundCurve,
    DegradationModel,
    TemperatureCheck,
    TyreModel,
    attach_track_temperature,
    fit_degradation,
    fit_degradation_from_laps,
    fit_race_degradation,
    fit_tyre_model,
    held_out_stint_rmse,
    race_laps,
    temperature_gain,
)
from f1sim.core.errors import DataError, InvariantError
from tests.tyres_world import (
    FUEL_S_PER_LAP,
    PIT_LANE_S,
    TEAM_STOP_OFFSET,
    TyresWorld,
    build_tyres_world,
)


@pytest.fixture(scope="module")
def world() -> TyresWorld:
    """Two synthetic seasons of two-stop races."""
    return build_tyres_world()


@pytest.fixture(scope="module")
def fitted(world: TyresWorld) -> TyreModel:
    """The tyre model fitted on every synthetic race."""
    tables = world.tables
    return fit_tyre_model(tables["laps"], tables["events"], tables["pit_stops"])


def _slope(model: TyreModel, compound: str, circuit: str = "bahrain") -> float:
    high = model.degradation.loss(compound, circuit, 30.0, None)
    low = model.degradation.loss(compound, circuit, 5.0, None)
    return (high - low) / 25.0


def test_race_laps_keeps_only_clean_dry_race_laps(world: TyresWorld) -> None:
    kept = race_laps(world.tables["laps"])
    assert set(kept["session"]) == {"R"}
    assert kept["clean_lap"].all()
    assert (kept["tyre_life"] >= 1.0).all()
    assert len(kept) < len(world.tables["laps"])


def test_wear_is_ordered_soft_then_medium_then_hard(fitted: TyreModel) -> None:
    assert _slope(fitted, "SOFT") > _slope(fitted, "MEDIUM") > _slope(fitted, "HARD")


def test_curves_start_at_zero_and_rise(fitted: TyreModel) -> None:
    for compound in fitted.degradation.compounds:
        values = fitted.degradation.curve(compound, "bahrain").values
        assert values[0] == 0.0
        assert np.all(np.diff(values) >= -1e-9)
        assert values[WARMUP_LAPS] > values[0]
        assert values[30] > values[5] >= 0.0


def test_fuel_slope_is_recovered(fitted: TyreModel) -> None:
    assert fitted.fuel_s_per_lap == pytest.approx(FUEL_S_PER_LAP, abs=0.01)


def test_pit_loss_is_recovered_per_circuit(fitted: TyreModel) -> None:
    assert fitted.pit_loss_s("bahrain") == pytest.approx(PIT_LANE_S, abs=0.5)
    assert fitted.pit_loss_s("monza") == pytest.approx(PIT_LANE_S, abs=0.5)
    assert np.isfinite(fitted.pit_loss_s("unknown"))


def test_stop_time_is_recovered_per_team(fitted: TyreModel) -> None:
    expected = PIT_LANE_S + TEAM_STOP_OFFSET["mercedes"]
    assert fitted.stop_time_s("mercedes") == pytest.approx(expected, abs=0.3)
    assert fitted.stop_time_s("nobody") == pytest.approx(fitted.stops.default.mean(), abs=1e-9)
    assert fitted.stop_time_s("mclaren") > fitted.stop_time_s("mercedes")


def test_warmup_penalties_are_positive_and_decreasing(fitted: TyreModel) -> None:
    for compound in fitted.degradation.compounds:
        first = fitted.warmup.penalty(compound, 1.0)
        second = fitted.warmup.penalty(compound, 2.0)
        assert first > second > 0.0
        assert first == pytest.approx(0.7, abs=0.4)


def test_loss_uses_warmup_below_three_laps(fitted: TyreModel) -> None:
    assert fitted.loss("SOFT", "bahrain", 1.0) == fitted.warmup.penalty("SOFT", 1.0)
    assert fitted.loss(None, "bahrain", 10.0) == 0.0
    assert fitted.loss("SOFT", "bahrain", 10.0) == fitted.degradation.loss(
        "SOFT", "bahrain", 10.0, None
    )


def test_model_round_trips_through_a_dict(fitted: TyreModel) -> None:
    assert TyreModel.from_dict(fitted.to_dict()).to_dict() == fitted.to_dict()


def test_held_out_rmse_is_finite(world: TyresWorld, fitted: TyreModel) -> None:
    rmse = held_out_stint_rmse(
        world.tables["laps"], world.tables["events"], fitted.degradation, (2024,)
    )
    assert np.isfinite(rmse)
    assert rmse >= 0.0


def test_temperature_term_is_not_warranted_without_temperature_data(
    world: TyresWorld,
) -> None:
    check = temperature_gain(
        world.tables["laps"],
        world.tables["events"],
        train_seasons=(2023,),
        holdout_seasons=(2024,),
    )
    assert isinstance(check, TemperatureCheck)
    assert check.rmse_with == check.rmse_without
    assert check.warranted is False


def test_fit_degradation_from_laps_builds_global_curves(world: TyresWorld) -> None:
    model = fit_degradation_from_laps(world.tables["laps"], world.tables["events"])
    assert isinstance(model, DegradationModel)
    assert set(model.compounds) == {"SOFT", "MEDIUM", "HARD"}
    for compound in model.compounds:
        assert np.isfinite(model.loss(compound, "bahrain", 20.0, None))


def test_pooling_nothing_raises() -> None:
    with pytest.raises(InvariantError):
        fit_degradation([])


def test_fitting_a_world_without_races_raises(world: TyresWorld) -> None:
    empty = world.tables["laps"].iloc[0:0]
    with pytest.raises(InvariantError):
        fit_degradation_from_laps(empty, world.tables["events"])


def test_a_too_short_race_gives_no_fit(world: TyresWorld) -> None:
    race = world.tables["laps"].loc[world.tables["laps"]["session"] == "R"]
    assert fit_race_degradation(race.head(20), "bahrain") is None


def test_track_temperature_is_matched_backwards(world: TyresWorld) -> None:
    laps = world.tables["laps"].loc[world.tables["laps"]["season"] == 2023]
    first = laps["lap_start_utc"].min()
    weather = pd.DataFrame(
        [
            {
                "season": 2023,
                "round": 1,
                "session": "R",
                "time_utc": first,
                "track_temp_c": 42.0,
            }
        ]
    )
    matched = attach_track_temperature(laps, weather)
    assert matched["track_temp_c"].notna().all()
    assert float(matched["track_temp_c"].iloc[0]) == 42.0
    late = weather.assign(time_utc=first + timedelta(days=1))
    assert attach_track_temperature(laps, late)["track_temp_c"].isna().all()


def _temperature_laps(track_temp: float | None) -> pd.DataFrame:
    ages = list(range(1, 11))
    frame = pd.DataFrame(
        {
            "season": 2020,
            "round": 1,
            "session": "R",
            "driver_id": "VER",
            "stint": 1,
            "compound": "SOFT",
            "tyre_life": [float(age) for age in ages],
            "lap_time_s": [0.1 * age for age in ages],
            "clean_lap": True,
            "track_status": "1",
            "is_accurate": True,
        }
    )
    if track_temp is not None:
        frame["track_temp_c"] = track_temp
    return frame


def _linear_model() -> DegradationModel:
    values = 0.1 * AGE_GRID
    curve = CompoundCurve(
        compound="SOFT",
        values=values,
        se=np.zeros_like(values),
        offset=0.0,
        offset_se=0.0,
        temperature=-0.05,
        temperature_se=0.01,
        n_laps=100,
    )
    return DegradationModel(("SOFT",), {"SOFT": curve}, {("bahrain", "SOFT"): curve})


def test_held_out_rmse_uses_track_temperature() -> None:
    events = pd.DataFrame([{"season": 2020, "round": 1, "circuit_id": "bahrain"}])
    model = _linear_model()
    cool = held_out_stint_rmse(_temperature_laps(40.0), events, model, (2020,))
    warm = held_out_stint_rmse(_temperature_laps(60.0), events, model, (2020,))
    assert cool == pytest.approx(0.0, abs=1e-9)
    assert warm > 0.1


def test_held_out_rmse_is_nan_without_holdout_data(world: TyresWorld, fitted: TyreModel) -> None:
    rmse = held_out_stint_rmse(
        world.tables["laps"], world.tables["events"], fitted.degradation, (1999,)
    )
    assert np.isnan(rmse)


def test_race_stops_keeps_only_race_sessions() -> None:
    frame = pd.DataFrame({"session": ["R", "Q", "FP1"], "lap": [1, 2, 3]})
    assert list(race_stops(frame)["session"]) == ["R"]


def test_track_temperature_tolerates_missing_lap_times(world: TyresWorld) -> None:
    laps = world.tables["laps"].loc[world.tables["laps"]["season"] == 2023].copy()
    first = laps["lap_start_utc"].min()
    laps.loc[laps.index[0], "lap_start_utc"] = pd.NaT
    weather = pd.DataFrame(
        [{"season": 2023, "round": 1, "session": "R", "time_utc": first, "track_temp_c": 42.0}]
    )
    matched = attach_track_temperature(laps, weather)
    assert len(matched) == len(laps)
    assert int(matched["track_temp_c"].notna().sum()) == len(laps) - 1


def test_fitting_with_temperature_tolerates_missing_temperatures(world: TyresWorld) -> None:
    laps = world.tables["laps"].copy()
    laps["track_temp_c"] = np.where(laps["tyre_life"] > 5.0, 45.0, np.nan)
    model = fit_degradation_from_laps(laps, world.tables["events"], temperature=True)
    assert np.isfinite(model.loss("SOFT", "bahrain", 20.0, 45.0))


def test_fitting_falls_back_to_plain_when_no_temperatures(world: TyresWorld) -> None:
    laps = world.tables["laps"].copy()
    laps["track_temp_c"] = np.nan
    model = fit_degradation_from_laps(laps, world.tables["events"], temperature=True)
    assert np.isfinite(model.loss("SOFT", "bahrain", 20.0, 45.0))


def test_tyre_model_round_trips_through_a_file(fitted: TyreModel, tmp_path: Path) -> None:
    path = tmp_path / "tyre_model.json"
    path.write_text(json.dumps(fitted.to_dict()), encoding="utf-8")
    assert load_tyre_model(path).to_dict() == fitted.to_dict()


def test_missing_tyre_model_file_raises(tmp_path: Path) -> None:
    with pytest.raises(DataError):
        load_tyre_model(tmp_path / "nope.json")


def test_malformed_tyre_model_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "tyre_model.json"
    path.write_text("{not a model}", encoding="utf-8")
    with pytest.raises(DataError):
        load_tyre_model(path)
