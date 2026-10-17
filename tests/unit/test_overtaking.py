"""The pass-probability model and its fit from historical position changes (P5-04)."""

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest

from f1sim.adapters.models.overtaking import (
    DEFAULT_AID_BONUS,
    default_overtaking,
    fit_overtaking,
)

SEASON = 2023
CIRCUIT = "bahrain"
START = datetime(2023, 3, 5, 15, 0)
PACE = {"a": 90.0, "b": 90.15, "c": 90.3, "d": 90.45}
NOISE = 1.0


def race_laps(
    rounds: int = 6, laps: int = 30, pace: dict[str, float] | None = None
) -> pd.DataFrame:
    """Race laps where the fastest cars start at the back, so they have to pass.

    Positions and lap start times follow the cars' cumulative times, as in real timing data.
    """
    rng = np.random.default_rng(4)
    pace = pace or PACE
    order = list(reversed(pace))  # a is fastest; the grid is reversed
    rows = []
    for round_ in range(1, rounds + 1):
        elapsed = {driver: 0.3 * grid for grid, driver in enumerate(order)}
        for lap in range(1, laps + 1):
            times = {d: pace[d] + float(rng.normal(0.0, NOISE)) for d in order}
            starts = dict(elapsed)
            for driver in order:
                elapsed[driver] += times[driver]
            ranked = sorted(order, key=elapsed.__getitem__)
            for driver in order:
                rows.append(
                    _lap_row(round_, driver, lap, times[driver], starts[driver])
                    | {"position": ranked.index(driver) + 1}
                )
    return pd.DataFrame(rows)


def _lap_row(round_: int, driver: str, lap: int, time_s: float, start_s: float) -> dict[str, Any]:
    """One canonical race lap, started ``start_s`` seconds after lights out."""
    return {
        "season": SEASON,
        "round": round_,
        "session": "R",
        "driver_id": driver,
        "team_id": f"team_{driver}",
        "lap_number": lap,
        "lap_time_s": time_s,
        "stint": 1,
        "compound": "MEDIUM",
        "tyre_life": float(lap),
        "fresh_tyre": True,
        "pit_in": False,
        "pit_out": False,
        "track_status": "1",
        "is_accurate": True,
        "deleted": False,
        "clean_lap": True,
        "lap_start_utc": START + timedelta(seconds=start_s),
        "available_at": START,
    }


def events(rounds: int = 6) -> pd.DataFrame:
    """One circuit's events for the weekends in the laps."""
    return pd.DataFrame(
        [
            {
                "season": SEASON,
                "round": round_,
                "event_name": "Bahrain Grand Prix",
                "circuit_id": CIRCUIT,
                "available_at": START,
            }
            for round_ in range(1, rounds + 1)
        ]
    )


def test_pass_probability_rises_with_the_advantage() -> None:
    model = default_overtaking()
    low = model.pass_probability(np.array([0.0, 0.5, 1.0]))
    assert np.all(np.diff(low) > 0)
    assert np.all((low > 0.0) & (low < 1.0))


def test_aid_and_circuit_shift_the_probability() -> None:
    model = default_overtaking()
    base = model.pass_probability(np.array([0.7]))
    with_drs = model.pass_probability(np.array([0.7]), aid="drs")
    assert with_drs[0] > base[0]
    assert "drs" in DEFAULT_AID_BONUS


def test_fit_is_positive_in_the_pace_advantage() -> None:
    model = fit_overtaking(race_laps(), events())
    assert model.pace_coef > 0.0
    assert model.pass_probability(np.array([1.0]))[0] > model.pass_probability(np.array([0.0]))[0]


def test_too_little_history_falls_back_to_the_default() -> None:
    empty = race_laps().iloc[0:0]
    model = fit_overtaking(empty, events())
    assert model == default_overtaking()


def test_pass_probability_handles_a_scalar() -> None:
    model = default_overtaking()
    value = model.pass_probability(0.5, circuit="bahrain", aid="overtake_mode")
    assert 0.0 < float(value) < 1.0


def test_circuit_difficulty_is_learned() -> None:
    model = fit_overtaking(race_laps(), events())
    assert CIRCUIT in model.difficulty
    assert -2.0 <= model.difficulty[CIRCUIT] <= 2.0


def test_missing_events_are_skipped() -> None:
    model = fit_overtaking(race_laps(), events().assign(round=99))
    assert model == default_overtaking()


def test_laps_behind_the_safety_car_or_in_the_pits_are_not_chances() -> None:
    laps = race_laps()
    yellow = laps.assign(track_status="4")
    assert fit_overtaking(yellow, events()) == default_overtaking()
    pitting = laps.assign(pit_in=True)
    assert fit_overtaking(pitting, events()) == default_overtaking()


def test_only_close_pairs_count_as_pass_chances() -> None:
    far = race_laps(pace={"a": 90.0, "b": 95.0, "c": 100.0, "d": 105.0})
    assert fit_overtaking(far, events()) == default_overtaking()


@pytest.mark.parametrize("rounds", [1])
def test_single_round_fit_still_runs(rounds: int) -> None:
    model = fit_overtaking(race_laps(rounds=rounds), events(rounds=rounds))
    assert model.pace_coef >= 0.0
