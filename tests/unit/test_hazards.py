"""Hazard fitting: safety cars, retirements, launches and priors (P6-01 to P6-06)."""

import numpy as np
import pandas as pd

from f1sim.adapters.models.hazards import (
    CRASH_PRIOR_PER_RACE,
    RED_FLAG_PRIOR_PER_RACE,
    default_hazards,
    fit_hazards,
    fit_lap1,
    fit_retirements,
    fit_safety_car,
)


def _laps(statuses: list[str], drivers: tuple[str, ...] = ("a", "b")) -> pd.DataFrame:
    columns = ["season", "round", "session", "driver_id", "lap_number", "track_status", "position"]
    rows = [
        {
            "season": 2024,
            "round": 1,
            "session": "R",
            "driver_id": driver,
            "lap_number": lap,
            "track_status": status,
            "position": lap,
        }
        for driver in drivers
        for lap, status in enumerate(statuses, start=1)
    ]
    return pd.DataFrame(rows, columns=columns)


def _events() -> pd.DataFrame:
    return pd.DataFrame([{"season": 2024, "round": 1, "circuit_id": "bahrain"}])


def _results(rows: list[tuple[str, str, int | None, str, int]]) -> pd.DataFrame:
    columns = [
        "season",
        "round",
        "session",
        "driver_id",
        "team_id",
        "grid",
        "position",
        "status",
        "laps",
    ]
    return pd.DataFrame(
        [
            {
                "season": 2024,
                "round": 1,
                "session": "R",
                "driver_id": driver,
                "team_id": team,
                "grid": position or 1,
                "position": position,
                "status": status,
                "laps": laps,
            }
            for driver, team, position, status, laps in rows
        ],
        columns=columns,
    )


def test_default_hazards_are_the_priors() -> None:
    hazards = default_hazards()
    assert hazards.safety_car.circuits == {}
    assert hazards.safety_car.red_flag_per_race == RED_FLAG_PRIOR_PER_RACE
    assert hazards.retirement.field_crash == CRASH_PRIOR_PER_RACE
    assert hazards.retirement.p_mechanical("unknown") == hazards.retirement.field_mechanical
    assert hazards.lap1.launch("unknown") == 0.0


def test_safety_car_counts_one_deployment_and_its_length() -> None:
    laps = _laps(["1", "1", "1", "4", "4", "4", "1", "1", "1", "1"])
    hazard = fit_safety_car(laps, {(2024, 1): "bahrain"})
    assert hazard.deploy_per_lap > 0.0
    assert hazard.circuits["bahrain"] > 0.0
    assert hazard.duration_laps == 3.0
    assert hazard.red_flag_per_race < RED_FLAG_PRIOR_PER_RACE


def test_a_red_flag_is_seen() -> None:
    laps = _laps(["1", "1", "5", "1", "1"])
    hazard = fit_safety_car(laps, {(2024, 1): "bahrain"})
    assert hazard.red_flag_per_race > 0.0


def test_safety_car_without_data_is_the_prior() -> None:
    hazard = fit_safety_car(_laps([]), {})
    assert hazard.deploy_per_lap == default_hazards().safety_car.deploy_per_lap


def test_mechanical_hazard_is_per_team_and_crash_per_driver() -> None:
    results = _results(
        [
            ("x", "team_a", 1, "Finished", 57),
            ("x", "team_a", None, "Gearbox", 20),
            ("y", "team_b", 1, "Finished", 57),
            ("y", "team_b", None, "Collision", 10),
        ]
    )
    hazard = fit_retirements(results)
    assert hazard.p_mechanical("team_a") > hazard.p_mechanical("team_b")
    assert hazard.p_crash("y") > hazard.field_crash
    assert hazard.p_crash("x") < hazard.field_crash
    assert hazard.p_retire("team_a", "x") > 0.0


def test_retirements_without_races_are_the_field() -> None:
    hazard = fit_retirements(_results([]))
    assert hazard.p_crash("x") == CRASH_PRIOR_PER_RACE
    assert hazard.p_mechanical("team_a") == hazard.field_mechanical


def test_launch_is_places_gained_on_lap_one() -> None:
    results = _results(
        [
            ("a", "team", 1, "Finished", 57),
            ("b", "team", 3, "Finished", 57),
        ]
    )
    lap1 = pd.DataFrame(
        [
            {
                "season": 2024,
                "round": 1,
                "session": "R",
                "driver_id": "a",
                "lap_number": 1,
                "position": 3,
            },
            {
                "season": 2024,
                "round": 1,
                "session": "R",
                "driver_id": "b",
                "lap_number": 1,
                "position": 1,
            },
        ]
    )
    hazard = fit_lap1(lap1, results)
    assert hazard.launch("b") > 0.0
    assert hazard.launch("a") < 0.0


def test_first_corner_incident_is_fitted() -> None:
    results = _results(
        [
            ("a", "team", 1, "Finished", 57),
            ("b", "team", 2, "Finished", 57),
            ("c", "team", None, "Accident", 0),
        ]
    )
    hazard = fit_lap1(_laps(["1", "1"]), results)
    assert hazard.incident > 0.0


def test_fit_hazards_bundles_every_part() -> None:
    laps = _laps(["1", "1", "4", "4", "1"])
    results = _results(
        [
            ("a", "team", 1, "Finished", 57),
            ("b", "team", None, "Collision", 2),
        ]
    )
    hazards = fit_hazards(laps, _events(), results)
    assert hazards.safety_car.deploy_rate("bahrain") > 0.0
    assert np.isfinite(hazards.retirement.p_retire("team", "a"))
    assert hazards.penalty.per_race > 0.0
