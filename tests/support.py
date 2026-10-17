"""Small builders for valid canonical rows, shared by tests."""

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from f1sim.core.domain import PositionMatrix
from f1sim.core.domain.entities import EventKey, SnapshotId
from f1sim.core.domain.race import RaceEntry
from f1sim.core.ports.store import DataReader

T0 = datetime(2025, 3, 16, 4, 0)


def lap_rows(
    *,
    season: int = 2025,
    round_: int = 1,
    session: str = "R",
    drivers: tuple[str, ...] = ("max_verstappen", "lando_norris"),
    n_laps: int = 5,
    available_at: datetime = T0,
    **overrides: Any,
) -> pd.DataFrame:
    """Build valid ``laps`` rows: a pit stop on lap 3, so laps 1, 3 and 4 are not clean."""
    rows: list[dict[str, Any]] = []
    for d in drivers:
        for lap in range(1, n_laps + 1):
            rows.append(
                {
                    "season": season,
                    "round": round_,
                    "session": session,
                    "driver_id": d,
                    "driver_code": d[:3].upper(),
                    "team_id": "red_bull" if d == "max_verstappen" else "mclaren",
                    "lap_number": lap,
                    "lap_time_s": 90.0 + lap * 0.1,
                    "sector1_s": 30.0,
                    "sector2_s": 30.0,
                    "sector3_s": 30.0,
                    "stint": 1 if lap <= 3 else 2,
                    "compound": "MEDIUM" if lap <= 3 else "HARD",
                    "tyre_life": float(lap),
                    "fresh_tyre": True,
                    "pit_in": lap == 3,
                    "pit_out": lap == 4,
                    "track_status": "1",
                    "position": 1,
                    "is_accurate": True,
                    "deleted": False,
                    "lap_start_utc": T0,
                    "clean_lap": lap not in (1, 3, 4),
                    "available_at": available_at,
                }
            )
    df = pd.DataFrame(rows)
    for key, value in overrides.items():
        df[key] = value
    return df


def result_rows(
    *, season: int = 2025, round_: int = 1, session: str = "R", available_at: datetime = T0
) -> pd.DataFrame:
    """Build valid ``results`` rows for a two-driver race."""
    return pd.DataFrame(
        [
            {
                "season": season,
                "round": round_,
                "session": session,
                "driver_id": driver,
                "driver_code": driver[:3].upper(),
                "driver_number": num,
                "team_id": team,
                "team_name": team.title(),
                "grid": pos,
                "position": pos,
                "position_text": str(pos),
                "status": "Finished",
                "points": pts,
                "laps": 57,
                "time_s": 5400.0 + pos,
                "q1_s": None,
                "q2_s": None,
                "q3_s": None,
                "available_at": available_at,
            }
            for driver, num, team, pos, pts in [
                ("max_verstappen", 1, "red_bull", 1, 25.0),
                ("lando_norris", 4, "mclaren", 2, 18.0),
            ]
        ]
    )


def ledger_rows(
    source: str = "fastf1",
    *,
    season: int = 2025,
    round_: int = 1,
    session: str = "R",
    run_id: str = "r",
    ingested_at: datetime = T0,
) -> pd.DataFrame:
    """Build valid ``_ingested`` ledger rows for one session."""
    return pd.DataFrame(
        [
            {
                "source": source,
                "season": season,
                "round": round_,
                "session": session,
                "run_id": run_id,
                "ingested_at": ingested_at,
            }
        ]
    )


SYNTH_DRIVERS = tuple(f"driver_{i:02d}" for i in range(10))
"""Synthetic drivers, fastest first: ``driver_00`` is the strongest."""


def synthetic_results(
    *,
    seasons: tuple[int, ...] = (2023, 2024),
    rounds: int = 8,
    seed: int = 7,
    skill_spread: float = 2.5,
    dnf_rate: float = 0.08,
    sprint_rounds: tuple[int, ...] = (),
) -> pd.DataFrame:
    """Build ``results`` for whole synthetic seasons with a known strength order.

    Each weekend has qualifying (and optionally a sprint) a day before the race; orders
    are Plackett-Luce draws whose log strengths fall linearly from ``skill_spread`` to 0.
    """
    rng = np.random.default_rng(seed)
    log_w = np.linspace(skill_spread, 0.0, len(SYNTH_DRIVERS))
    rows: list[dict[str, Any]] = []
    for season in seasons:
        for round_ in range(1, rounds + 1):
            sessions = ("Q", *(("S",) if round_ in sprint_rounds else ()), "R")
            start = datetime(season, 3, 1, 15, 0) + timedelta(days=14 * (round_ - 1))
            rows += _weekend(rng, log_w, (season, round_, start), sessions, dnf_rate)
    return pd.DataFrame(rows)


def _weekend(
    rng: np.random.Generator,
    log_w: Any,
    when: tuple[int, int, datetime],
    sessions: tuple[str, ...],
    dnf_rate: float,
) -> list[dict[str, Any]]:
    season, round_, start = when
    rows: list[dict[str, Any]] = []
    grid: dict[int, int] = {}
    for offset, session in enumerate(sessions):
        order: list[int] = np.argsort(-(log_w + rng.gumbel(size=len(log_w)))).tolist()
        out = (rng.random(len(log_w)) < dnf_rate) & (session != "Q")
        base = {
            "season": season,
            "round": round_,
            "session": session,
            "available_at": start + timedelta(hours=offset * 24),
        }
        classified = [i for i in order if not out[i]]
        for idx in order:
            pos = classified.index(idx) + 1 if idx in classified else None
            rows.append(base | _driver_fields(idx, grid.get(idx, 0), pos))
        if session == "Q":
            grid = {idx: n for n, idx in enumerate(order, start=1)}
    return rows


def _driver_fields(idx: int, grid: int, position: int | None) -> dict[str, Any]:
    return {
        "driver_id": SYNTH_DRIVERS[idx],
        "driver_code": f"D{idx:02d}",
        "driver_number": idx + 1,
        "team_id": f"team_{idx // 2}",
        "team_name": f"Team {idx // 2}",
        **classification(grid, position),
        "laps": 57,
    }


def classification(grid: int, position: int | None) -> dict[str, Any]:
    """Result columns that follow from grid slot and finishing position (None = retired)."""
    return {
        "grid": grid,
        "position": position,
        "position_text": "R" if position is None else str(position),
        "status": "Retired" if position is None else "Finished",
        "points": 0.0,
        "time_s": None,
        "q1_s": None,
        "q2_s": None,
        "q3_s": None,
    }


class UniformModel:
    """Every position equally likely, with a fixed retirement rate; the no-skill yardstick."""

    name = "uniform"

    def __init__(self, seed: int, p_dnf: float = 0.08) -> None:
        """Create the model; the seed is kept only to match the factory signature."""
        self.seed = seed
        self.p_dnf = p_dnf

    def fit(self, reader: DataReader, *, snapshot: SnapshotId, cutoff: datetime) -> None:
        """Read the history once, as a real model would, and learn nothing from it."""
        reader.read("results", snapshot=snapshot, as_of=cutoff)

    def predict(self, event: EventKey, entries: Sequence[RaceEntry]) -> PositionMatrix:
        """Give every entry the same chance of every position."""
        del event
        n = len(entries)
        ids = tuple(e.driver_id for e in entries)
        finish = (1.0 - self.p_dnf) / n
        return PositionMatrix(ids, np.full((n, n), finish), np.full(n, self.p_dnf))
