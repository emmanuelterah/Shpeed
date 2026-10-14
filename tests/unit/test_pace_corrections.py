"""Fuel, tyre and track-evolution corrections recover the effects built into synthetic laps."""

import numpy as np
import pandas as pd
import pytest

from f1sim.adapters.models.pace_corrections import (
    EVOLUTION_PRIOR,
    FUEL_PRIOR,
    CircuitHistory,
    Estimate,
    dry_laps,
    fit_push_laps,
    fit_race_laps,
    pool,
    push_laps,
)
from tests.pace_world import COMPOUNDS, EVOLUTION_S_PER_MIN, FUEL_S_PER_LAP, build_world


@pytest.fixture(scope="module")
def laps() -> pd.DataFrame:
    """One synthetic season of laps."""
    return build_world(seasons=(2024,), rounds=3).tables["laps"]


def _session(laps: pd.DataFrame, round_: int, session: str) -> pd.DataFrame:
    selected: pd.DataFrame = laps.loc[(laps["round"] == round_) & (laps["session"] == session)]
    return selected


def test_race_fit_recovers_fuel_and_tyre_wear(laps: pd.DataFrame) -> None:
    fit = fit_race_laps(_session(laps, 1, "R"))
    assert fit is not None
    assert fit.slope is not None
    assert -fit.slope.value == pytest.approx(FUEL_S_PER_LAP, abs=3 * fit.slope.se)
    assert fit.wear["MEDIUM"] == pytest.approx(COMPOUNDS["MEDIUM"][1], abs=0.01)
    assert fit.wear["HARD"] == pytest.approx(COMPOUNDS["HARD"][1], abs=0.01)
    hard_vs_medium = COMPOUNDS["HARD"][0] - COMPOUNDS["MEDIUM"][0]
    assert fit.offsets["MEDIUM"] == pytest.approx(-hard_vs_medium, abs=0.2)
    assert fit.residual_sd == pytest.approx(0.3, abs=0.05)


def test_known_fuel_is_removed_instead_of_fitted(laps: pd.DataFrame) -> None:
    fit = fit_race_laps(_session(laps, 1, "R"), fuel=FUEL_S_PER_LAP)
    assert fit is not None
    assert fit.slope is None
    assert len(fit.drivers) == 10


def test_qualifying_fit_recovers_track_evolution(laps: pd.DataFrame) -> None:
    fit = fit_push_laps(_session(laps, 2, "Q"))
    assert fit is not None
    assert fit.slope is not None
    assert fit.slope.value == pytest.approx(EVOLUTION_S_PER_MIN, abs=0.005)
    fixed = fit_push_laps(_session(laps, 2, "Q"), evolution=EVOLUTION_S_PER_MIN)
    assert fixed is not None
    assert fixed.slope is None


def test_wet_and_unclean_laps_are_dropped(laps: pd.DataFrame) -> None:
    race = _session(laps, 1, "R").copy()
    race.loc[race.index[:50], "compound"] = "INTERMEDIATE"
    kept = dry_laps(race)
    assert len(kept) == int(race["clean_lap"].sum()) - int(race["clean_lap"].iloc[:50].sum())
    assert "INTERMEDIATE" not in set(kept["compound"])


def test_push_laps_are_close_to_each_drivers_best(laps: pd.DataFrame) -> None:
    quali = _session(laps, 1, "Q").copy()
    times = quali["lap_time_s"].to_numpy(copy=True)
    times[1] += 20.0
    quali["lap_time_s"] = times
    pushed = push_laps(quali)
    assert quali.index[1] not in pushed.index
    assert pushed["minutes"].min() >= 0
    assert push_laps(quali.iloc[0:0]).empty


def test_too_few_laps_give_no_fit(laps: pd.DataFrame) -> None:
    assert fit_race_laps(_session(laps, 1, "R").head(10)) is None
    assert fit_push_laps(_session(laps, 1, "Q").iloc[0:0]) is None


def test_pool_weights_by_precision() -> None:
    prior = pool([], FUEL_PRIOR)
    assert (prior.value, prior.se) == pytest.approx(FUEL_PRIOR)
    pooled = pool([Estimate(0.10, 0.001), Estimate(float("nan"), 0.1)], FUEL_PRIOR)
    assert pooled.value == pytest.approx(0.10, abs=1e-3)
    assert pooled.se < 0.001 + 1e-9


def test_circuit_history_pools_and_round_trips() -> None:
    history = CircuitHistory()
    history.add("monza", fuel=Estimate(0.05, 0.005), evolution=None)
    history.add("monza", fuel=None, evolution=Estimate(-0.02, 0.002))
    assert history.fuel_at("monza").value == pytest.approx(0.05, abs=0.002)
    assert history.evolution_at("monaco").value == EVOLUTION_PRIOR[0]
    again = CircuitHistory.from_dict(history.to_dict())
    assert again == history
    assert np.isfinite(again.evolution_at("monza").se)
