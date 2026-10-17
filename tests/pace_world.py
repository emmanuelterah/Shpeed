"""A synthetic championship with known car and driver pace, for testing the pace model.

Five teams of two drivers race on a rotation of real circuits. Each weekend has a long-run
practice session (FP2), qualifying and a race, with laps shaped by the things the pace
model has to remove: fuel burn, tyre wear by compound, track evolution and noise. The
true pace of every car and driver is kept, so tests can check the model recovers it.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from f1sim.adapters.store.memory_store import InMemoryStore
from f1sim.core.domain.entities import EventKey
from f1sim.core.domain.race import RaceEntry
from f1sim.core.usecases.readers import FAR_FUTURE
from tests.support import classification

TEAMS = ("mercedes", "ferrari", "red_bull", "mclaren", "williams")
DRIVERS = tuple(f"{team}_{i}" for team in TEAMS for i in (1, 2))
CIRCUITS = ("bahrain", "monaco", "monza", "silverstone", "hungaroring", "spa")
BASE_LAP_S = {
    "bahrain": 92.0,
    "monaco": 74.0,
    "monza": 82.0,
    "silverstone": 88.0,
    "hungaroring": 78.0,
    "spa": 106.0,
}
COMPOUNDS = {"SOFT": (-0.6, 0.09), "MEDIUM": (0.0, 0.05), "HARD": (0.4, 0.03)}
"""Compound to (pace offset in seconds, wear in seconds per lap of tyre age)."""

FUEL_S_PER_LAP = 0.06
EVOLUTION_S_PER_MIN = -0.015
RACE_LAPS = 40


@dataclass
class World:
    """The generated tables and the truth behind them.

    Attributes:
        tables: ``events``, ``results`` and ``laps`` rows, ready to commit.
        team_pace: ``(season, round, team)`` to true race pace, in percent of lap time.
        driver_pace: ``driver`` to true ``(qualifying, race)`` effects, in percent.
    """

    tables: dict[str, pd.DataFrame]
    team_pace: dict[tuple[int, int, str], float] = field(
        default_factory=dict[tuple[int, int, str], float]
    )
    driver_pace: dict[str, tuple[float, float]] = field(
        default_factory=dict[str, tuple[float, float]]
    )


def build_world(seasons: tuple[int, ...] = (2023, 2024), rounds: int = 6, seed: int = 11) -> World:
    """Generate whole synthetic seasons.

    Args:
        seasons: Seasons to generate.
        rounds: Races per season.
        seed: Random seed.

    Returns:
        The tables and the true pace behind them.
    """
    rng = np.random.default_rng(seed)
    team = dict(zip(TEAMS, rng.normal(0.0, 0.8, len(TEAMS)).tolist(), strict=True))
    world = World(tables={})
    for driver in DRIVERS:
        race = float(rng.normal(0.0, 0.25))
        world.driver_pace[driver] = (0.7 * race + float(rng.normal(0.0, 0.1)), race)
    rows: dict[str, list[dict[str, Any]]] = {"events": [], "results": [], "laps": []}
    for season in seasons:
        for name in TEAMS:
            team[name] += float(rng.normal(0.0, 0.4))
        for round_ in range(1, rounds + 1):
            for name in TEAMS:
                team[name] += float(rng.normal(0.0, 0.1))
                world.team_pace[(season, round_, name)] = team[name]
            weekend = _Weekend(season, round_, CIRCUITS[(round_ - 1) % len(CIRCUITS)], rng)
            weekend.build(world, rows)
    world.tables = {name: pd.DataFrame(r) for name, r in rows.items()}
    return world


class _Weekend:
    def __init__(self, season: int, round_: int, circuit: str, rng: np.random.Generator):
        self.season = season
        self.round = round_
        self.circuit = circuit
        self.rng = rng
        self.start = datetime(season, 3, 1, 12, 0) + timedelta(days=14 * (round_ - 1))

    def pace(self, world: World, driver: str, kind: int) -> float:
        """True pace of a driver this weekend: ``kind`` 0 is qualifying, 1 is race."""
        team = driver.rsplit("_", 1)[0]
        return world.team_pace[(self.season, self.round, team)] + world.driver_pace[driver][kind]

    def build(self, world: World, rows: dict[str, list[dict[str, Any]]]) -> None:
        """Add the weekend's event, laps and results rows."""
        rows["events"].append(
            {
                "season": self.season,
                "round": self.round,
                "event_name": f"{self.circuit.title()} Grand Prix",
                "circuit_id": self.circuit,
                "locality": self.circuit,
                "country": "Nowhere",
                "event_date": self.start + timedelta(days=2),
                "event_format": "conventional",
                "available_at": self.start - timedelta(days=30),
            }
        )
        self._practice(world, rows)
        grid = self._qualifying(world, rows)
        self._race(world, rows, grid)

    def _lap(self, session: str, driver: str, number: int, time_s: float) -> dict[str, Any]:
        return {
            "season": self.season,
            "round": self.round,
            "session": session,
            "driver_id": driver,
            "driver_code": driver[:3].upper(),
            "team_id": driver.rsplit("_", 1)[0],
            "lap_number": number,
            "lap_time_s": time_s,
            "sector1_s": time_s / 3,
            "sector2_s": time_s / 3,
            "sector3_s": time_s / 3,
            "track_status": "1",
            "position": None,
            "is_accurate": True,
            "deleted": False,
        }

    def _practice(self, world: World, rows: dict[str, list[dict[str, Any]]]) -> None:
        base = BASE_LAP_S[self.circuit]
        begin = self.start + timedelta(hours=2)
        for driver in DRIVERS:
            pace = base * (1 + self.pace(world, driver, 1) / 100)
            fuel_laps = float(self.rng.uniform(15, 30))
            for lap in range(1, 13):
                life = float(lap)
                offset, wear = COMPOUNDS["MEDIUM"]
                noise = float(self.rng.normal(0, 0.25))
                fuel = FUEL_S_PER_LAP * (fuel_laps - lap)
                time_s = pace + offset + wear * life + fuel + noise
                row = self._lap("FP2", driver, lap, time_s)
                row |= _tyre(1, "MEDIUM", life, clean=lap > 1)
                row |= _timing(begin + timedelta(minutes=2 * lap), begin + timedelta(hours=1))
                rows["laps"].append(row)

    def _qualifying(self, world: World, rows: dict[str, list[dict[str, Any]]]) -> list[str]:
        base = BASE_LAP_S[self.circuit]
        begin = self.start + timedelta(days=1, hours=3)
        best: dict[str, float] = {}
        for driver in DRIVERS:
            pace = base * (1 + self.pace(world, driver, 0) / 100) + COMPOUNDS["SOFT"][0]
            for lap in range(1, 7):
                minute = 8.0 * lap + float(self.rng.uniform(0, 4))
                time_s = pace + EVOLUTION_S_PER_MIN * minute + float(self.rng.normal(0, 0.08))
                best[driver] = min(best.get(driver, np.inf), time_s)
                row = self._lap("Q", driver, lap, time_s) | _tyre(lap, "SOFT", 1.0, clean=lap > 1)
                row |= _timing(begin + timedelta(minutes=minute), begin + timedelta(hours=1))
                rows["laps"].append(row)
        order = sorted(DRIVERS, key=lambda d: best[d])
        for position, driver in enumerate(order, start=1):
            rows["results"].append(self._result("Q", driver, 0, position, begin))
        return order

    def _race(self, world: World, rows: dict[str, list[dict[str, Any]]], grid: list[str]) -> None:
        begin = self.start + timedelta(days=2, hours=3)
        totals = {
            driver: self._race_laps(world, rows["laps"], driver, (slot, begin))
            for slot, driver in enumerate(grid)
        }
        retired = {d for d in DRIVERS if self.rng.random() < 0.05}
        finishers = sorted((d for d in DRIVERS if d not in retired), key=lambda d: totals[d])
        for driver in DRIVERS:
            position = finishers.index(driver) + 1 if driver in finishers else None
            rows["results"].append(
                self._result("R", driver, grid.index(driver) + 1, position, begin)
            )

    def _race_laps(
        self, world: World, laps: list[dict[str, Any]], driver: str, start: tuple[int, datetime]
    ) -> float:
        """One driver's race: medium tyres, one stop onto hards. Returns the race time."""
        pace = BASE_LAP_S[self.circuit] * (1 + self.pace(world, driver, 1) / 100)
        stop = int(self.rng.integers(15, 26))
        total = 0.4 * start[0]
        for lap in range(1, RACE_LAPS + 1):
            compound, stint, life = ("MEDIUM", 1, lap) if lap <= stop else ("HARD", 2, lap - stop)
            offset, wear = COMPOUNDS[compound]
            time_s = pace + offset + wear * life + FUEL_S_PER_LAP * (RACE_LAPS - lap)
            time_s += float(self.rng.normal(0, 0.3)) + (20.0 if lap == stop else 0.0)
            total += time_s
            row = self._lap("R", driver, lap, time_s)
            row |= _tyre(stint, compound, life, clean=lap not in (1, stop, stop + 1))
            row |= _timing(start[1] + timedelta(seconds=total), start[1] + timedelta(hours=2))
            row["pit_in"], row["pit_out"] = lap == stop, lap == stop + 1
            laps.append(row)
        return total

    def _result(
        self, session: str, driver: str, grid: int, position: int | None, begin: datetime
    ) -> dict[str, Any]:
        team = driver.rsplit("_", 1)[0]
        return {
            "season": self.season,
            "round": self.round,
            "session": session,
            "driver_id": driver,
            "driver_code": driver[:3].upper(),
            "driver_number": DRIVERS.index(driver) + 1,
            "team_id": team,
            "team_name": team.title(),
            **classification(grid, position),
            "laps": RACE_LAPS,
            "available_at": begin + timedelta(hours=2 if session == "R" else 1),
        }


def _tyre(stint: int, compound: str, life: float, clean: bool) -> dict[str, Any]:
    return {
        "stint": stint,
        "compound": compound,
        "tyre_life": life,
        "fresh_tyre": True,
        "pit_in": False,
        "pit_out": False,
        "clean_lap": clean,
    }


def _timing(start: datetime, available_at: datetime) -> dict[str, Any]:
    return {"lap_start_utc": start, "available_at": available_at}


def world_store(seasons: tuple[int, ...] = (2024,), rounds: int = 4) -> InMemoryStore:
    """A synthetic championship committed to an in-memory store as one snapshot."""
    world = build_world(seasons=seasons, rounds=rounds)
    committed = InMemoryStore()
    committed.commit(world.tables, run_id="world")
    return committed


def quali_entries(store: InMemoryStore, event: EventKey) -> list[RaceEntry]:
    """Race entries for one weekend, on the grid in qualifying order."""
    results = store.read("results", snapshot=store.latest_snapshot(), as_of=FAR_FUTURE)
    quali = results.loc[
        (results["season"] == event.season)
        & (results["round"] == event.round)
        & (results["session"] == "Q")
    ].sort_values("position")
    return [
        RaceEntry(str(driver), str(team), slot)
        for slot, (driver, team) in enumerate(
            zip(quali["driver_id"], quali["team_id"], strict=True), start=1
        )
    ]
