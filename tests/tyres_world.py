"""A synthetic race weekend with known tyre behaviour, for testing the tyre and strategy models.

Every driver runs one stop: a soft or medium first stint onto a hard second stint, long enough
that each compound is seen past the last spline knot. Lap times carry a known compound offset,
a known wear rate per lap of age, a known warm-up penalty on the first two laps of a stint and a
known fuel burn. Pit laps add a known extra time on top of the driver's clean-lap baseline, so the
measured pit loss is known, and ``pit_stops`` records a per-team pit-lane duration with a known
offset, so tests can check every P4 estimate recovers it.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

TEAMS = ("mercedes", "ferrari", "red_bull", "mclaren", "williams")
DRIVERS = tuple(f"{team}_{i}" for team in TEAMS for i in (1, 2))
CIRCUITS = ("bahrain", "monza")
BASE_LAP_S = {"bahrain": 95.0, "monza": 85.0}
COMPOUNDS = {"SOFT": (-0.6, 0.09), "MEDIUM": (0.0, 0.05), "HARD": (0.4, 0.03)}
"""Compound to (pace offset in seconds, wear in seconds per lap of tyre age)."""

FUEL_S_PER_LAP = 0.06
RACE_LAPS = 100
STOP1_LAP = 32
STOP2_LAP = 66
STOP_SPREAD = 4
WARMUP_PENALTY = {1: 0.7, 2: 0.3}
PIT_IN_EXTRA_S = 15.0
PIT_OUT_EXTRA_S = 8.5
PIT_LANE_S = PIT_IN_EXTRA_S + PIT_OUT_EXTRA_S
"""The extra time of the in- and out-laps over the baseline, and so the measured pit loss."""
NOISE_S = 0.15
TEAM_STOP_OFFSET = {
    "mercedes": -0.6,
    "ferrari": 0.4,
    "red_bull": -0.2,
    "mclaren": 0.8,
    "williams": 0.2,
}
"""Team to its pit-lane duration offset from the field, seconds; stored in ``pit_stops``."""
PLANS = (
    ("SOFT", "MEDIUM", "HARD"),
    ("MEDIUM", "HARD", "SOFT"),
    ("HARD", "SOFT", "MEDIUM"),
)
"""Stint compounds per plan, rotated across drivers so each compound is seen in every stint."""


@dataclass
class TyresWorld:
    """The generated tables and the truth behind them.

    Attributes:
        tables: ``events``, ``laps`` and ``pit_stops`` rows, ready to commit.
        stop_laps: ``(season, round, driver)`` to the driver's two stop laps.
        driver_pace: Driver to its true race pace, seconds against the circuit base.
    """

    tables: dict[str, pd.DataFrame]
    stop_laps: dict[tuple[int, int, str], tuple[int, int]] = field(
        default_factory=dict[tuple[int, int, str], tuple[int, int]]
    )
    driver_pace: dict[str, float] = field(default_factory=dict[str, float])


def build_tyres_world(seasons: tuple[int, ...] = (2023, 2024), seed: int = 5) -> TyresWorld:
    """Generate whole synthetic seasons for the tyre and strategy models.

    Args:
        seasons: Seasons to generate; each is one race at its own circuit.
        seed: Random seed.

    Returns:
        The tables and the truth behind them.
    """
    rng = np.random.default_rng(seed)
    world = TyresWorld(tables={})
    for driver in DRIVERS:
        world.driver_pace[driver] = float(rng.normal(0.0, 0.3))
    rows: dict[str, list[dict[str, Any]]] = {"events": [], "laps": [], "pit_stops": []}
    for index, season in enumerate(seasons):
        circuit = CIRCUITS[index % len(CIRCUITS)]
        start = datetime(season, 3, 1, 15, 0)
        rows["events"].append(_event(season, circuit, start))
        for slot, driver in enumerate(DRIVERS):
            _driver_race(
                world,
                rows,
                rng,
                season=season,
                circuit=circuit,
                start=start,
                driver=driver,
                slot=slot,
            )
    world.tables = {name: pd.DataFrame(r) for name, r in rows.items()}
    return world


def _event(season: int, circuit: str, start: datetime) -> dict[str, Any]:
    return {
        "season": season,
        "round": 1,
        "event_name": f"{circuit.title()} Grand Prix",
        "circuit_id": circuit,
        "locality": circuit,
        "country": "Nowhere",
        "event_date": start,
        "event_format": "conventional",
        "available_at": start - timedelta(days=30),
    }


def _driver_race(  # pylint: disable=too-many-arguments,too-many-locals
    world: TyresWorld,
    rows: dict[str, list[dict[str, Any]]],
    rng: np.random.Generator,
    *,
    season: int,
    circuit: str,
    start: datetime,
    driver: str,
    slot: int,
) -> None:
    team = driver.rsplit("_", 1)[0]
    pace = BASE_LAP_S[circuit] + world.driver_pace[driver]
    stops = (STOP1_LAP + slot % STOP_SPREAD, STOP2_LAP + (slot + 2) % STOP_SPREAD)
    world.stop_laps[(season, 1, driver)] = stops
    plan = PLANS[slot % len(PLANS)]
    times: dict[int, float] = {}
    for lap in range(1, RACE_LAPS + 1):
        stint, life = _stint_and_life(lap, stops)
        compound = plan[stint - 1]
        offset, wear = COMPOUNDS[compound]
        warmup = WARMUP_PENALTY.get(life, 0.0) if life < 3 else 0.0
        noise = float(rng.normal(0.0, NOISE_S))
        times[lap] = (
            pace + offset + wear * life + warmup + FUEL_S_PER_LAP * (RACE_LAPS - lap) + noise
        )
    skip = set(stops) | {stop + 1 for stop in stops}
    baseline = float(np.median([t for lap, t in times.items() if lap not in skip]))
    for stop in stops:
        times[stop] = baseline + PIT_IN_EXTRA_S
        times[stop + 1] = baseline + PIT_OUT_EXTRA_S
    _append_laps(
        rows,
        start,
        season=season,
        driver=driver,
        team=team,
        stops=stops,
        plan=plan,
        times=times,
    )
    for number, stop in enumerate(stops, start=1):
        rows["pit_stops"].append(
            {
                "season": season,
                "round": 1,
                "session": "R",
                "driver_id": driver,
                "stop": number,
                "lap": stop,
                "duration_s": PIT_LANE_S + TEAM_STOP_OFFSET[team] + float(rng.normal(0.0, 0.1)),
                "available_at": start + timedelta(hours=2),
            }
        )


def _stint_and_life(lap: int, stops: tuple[int, int]) -> tuple[int, int]:
    if lap <= stops[0]:
        return 1, lap
    if lap <= stops[1]:
        return 2, lap - stops[0]
    return 3, lap - stops[1]


def _append_laps(  # pylint: disable=too-many-arguments
    rows: dict[str, list[dict[str, Any]]],
    start: datetime,
    *,
    season: int,
    driver: str,
    team: str,
    stops: tuple[int, int],
    plan: tuple[str, str, str],
    times: dict[int, float],
) -> None:
    elapsed = 0.0
    for lap in range(1, RACE_LAPS + 1):
        elapsed += times[lap]
        stint, life = _stint_and_life(lap, stops)
        pit_in = lap in stops
        pit_out = lap - 1 in stops
        rows["laps"].append(
            {
                "season": season,
                "round": 1,
                "session": "R",
                "driver_id": driver,
                "driver_code": driver[:3].upper(),
                "team_id": team,
                "lap_number": lap,
                "lap_time_s": times[lap],
                "sector1_s": times[lap] / 3,
                "sector2_s": times[lap] / 3,
                "sector3_s": times[lap] / 3,
                "stint": stint,
                "compound": plan[stint - 1],
                "tyre_life": float(life),
                "fresh_tyre": True,
                "pit_in": pit_in,
                "pit_out": pit_out,
                "track_status": "1",
                "position": None,
                "is_accurate": True,
                "deleted": False,
                "lap_start_utc": start + timedelta(seconds=elapsed),
                "clean_lap": not (pit_in or pit_out),
                "available_at": start + timedelta(hours=2),
            }
        )


def event_rows(world: TyresWorld) -> pd.DataFrame:
    """The ``events`` table of a world."""
    return world.tables["events"]


def iter_stops(world: TyresWorld) -> Iterator[tuple[int, str, int]]:
    """Yield ``(season, driver, stop_lap)`` for every stop in a world."""
    for (season, _round, driver), stops in world.stop_laps.items():
        for stop in stops:
            yield season, driver, stop
