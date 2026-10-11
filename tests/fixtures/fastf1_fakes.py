"""Fake FastF1 sessions with the same column names and dtypes as fastf1 3.x objects."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from f1sim.adapters.sources.fastf1_adapter import ScheduleLoader, SessionLoader

DRIVERS = [("VER", "max_verstappen", "red_bull"), ("NOR", "norris", "mclaren")]


@dataclass
class FakeSession:
    """Stands in for a loaded ``fastf1.core.Session`` with the attributes the adapter reads."""

    laps: pd.DataFrame
    results: pd.DataFrame
    weather_data: pd.DataFrame
    track_status: pd.DataFrame
    date: pd.Timestamp
    t0_date: pd.Timestamp


def fake_session(
    start: datetime = datetime(2024, 3, 2, 15, 0),
    n_laps: int = 6,
    *,
    safety_car_lap: int | None = 4,
    red_flag_lap: int | None = None,
    drop_driver_id: bool = False,
) -> FakeSession:
    """Build a two-driver session with a pit stop, and optionally a safety car or red flag."""
    t0 = pd.Timestamp(start) - pd.Timedelta(minutes=55)
    rows: list[dict[str, Any]] = []
    for code, _, team in DRIVERS:
        for lap in range(1, n_laps + 1):
            lap_time = pd.Timedelta(seconds=95.0 + lap * 0.05)
            if lap == red_flag_lap:
                lap_time = pd.Timedelta(minutes=35)  # red-flag lap: genuine but implausible pace
            status = "1"
            if lap == safety_car_lap:
                status = "14"
            rows.append(
                {
                    "Time": pd.Timedelta(minutes=55) + lap * pd.Timedelta(seconds=95),
                    "Driver": code,
                    "DriverNumber": "1" if code == "VER" else "4",
                    "LapTime": lap_time,
                    "LapNumber": float(lap),
                    "Stint": 1.0 if lap <= 3 else 2.0,
                    "PitOutTime": pd.Timedelta(minutes=60) if lap == 4 else pd.NaT,
                    "PitInTime": pd.Timedelta(minutes=59) if lap == 3 else pd.NaT,
                    "Sector1Time": pd.Timedelta(seconds=31) if lap > 1 else pd.NaT,
                    "Sector2Time": pd.Timedelta(seconds=33),
                    "Sector3Time": pd.Timedelta(seconds=31),
                    "Compound": "SOFT" if lap <= 3 else "HARD",
                    "TyreLife": float(lap if lap <= 3 else lap - 3),
                    "FreshTyre": True,
                    "Team": team,
                    "LapStartTime": pd.Timedelta(minutes=55) + (lap - 1) * pd.Timedelta(seconds=95),
                    "LapStartDate": t0
                    + pd.Timedelta(minutes=55)
                    + (lap - 1) * pd.Timedelta(seconds=95),
                    "TrackStatus": status,
                    "Position": 1.0 if code == "VER" else 2.0,
                    "Deleted": None,
                    "IsAccurate": lap not in (1,),
                }
            )
    laps = pd.DataFrame(rows)
    results = pd.DataFrame(
        [
            {
                "DriverNumber": "1" if code == "VER" else "4",
                "Abbreviation": code,
                "DriverId": "" if drop_driver_id and code == "NOR" else driver_id,
                "TeamName": team,
                "TeamId": team,
            }
            for code, driver_id, team in DRIVERS
        ]
    )
    weather = pd.DataFrame(
        {
            "Time": pd.to_timedelta(np.arange(0, 6) * 60, unit="s"),
            "AirTemp": 18.5,
            "Humidity": 46.0,
            "Pressure": 1017.0,
            "Rainfall": False,
            "TrackTemp": 27.1,
            "WindDirection": 270,
            "WindSpeed": 1.2,
        }
    )
    track_status = pd.DataFrame(
        {
            "Time": [pd.Timedelta(minutes=50), pd.Timedelta(minutes=60), pd.Timedelta(minutes=62)],
            "Status": ["1", "4", "1"],
            "Message": ["AllClear", "SCDeployed", "AllClear"],
        }
    )
    return FakeSession(laps, results, weather, track_status, pd.Timestamp(start), t0)


def fake_schedule(season: int = 2024) -> pd.DataFrame:
    """The 2024 Bahrain (round 1) and Saudi Arabian (round 2) timetables, in UTC."""
    names = ["Practice 1", "Practice 2", "Practice 3", "Qualifying", "Race"]

    def event(round_: int, name: str, starts: list[datetime]) -> dict[str, Any]:
        row: dict[str, Any] = {
            "RoundNumber": round_,
            "EventName": name,
            "EventFormat": "conventional",
        }
        for i, (session, start) in enumerate(zip(names, starts, strict=True), 1):
            row[f"Session{i}"] = session
            row[f"Session{i}DateUtc"] = pd.Timestamp(start)
        return row

    return pd.DataFrame(
        [
            event(
                1,
                "Bahrain Grand Prix",
                [
                    datetime(season, 2, 29, 11, 30),
                    datetime(season, 2, 29, 15, 0),
                    datetime(season, 3, 1, 12, 30),
                    datetime(season, 3, 1, 16, 0),
                    datetime(season, 3, 2, 15, 0),
                ],
            ),
            event(
                2,
                "Saudi Arabian Grand Prix",
                [
                    datetime(season, 3, 7, 13, 30),
                    datetime(season, 3, 7, 17, 0),
                    datetime(season, 3, 8, 13, 30),
                    datetime(season, 3, 8, 17, 0),
                    datetime(season, 3, 9, 17, 0),
                ],
            ),
        ]
    )


def loaders(
    sessions: dict[tuple[int, int, str], FakeSession], schedule: pd.DataFrame | None = None
) -> tuple[SessionLoader, ScheduleLoader]:
    """Loaders for ``FastF1Adapter`` that serve ``sessions`` and fail like FastF1 otherwise."""

    def load_session(season: int, round_: int, identifier: str) -> FakeSession:
        if (season, round_, identifier) not in sessions:
            raise RuntimeError("Session data not available")  # mimics fastf1 errors
        return sessions[(season, round_, identifier)]

    def load_schedule(season: int) -> pd.DataFrame:
        return fake_schedule(season) if schedule is None else schedule

    return load_session, load_schedule
