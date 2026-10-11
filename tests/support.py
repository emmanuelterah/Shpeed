"""Small builders for valid canonical rows, shared by tests."""

from datetime import datetime
from typing import Any

import pandas as pd

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
