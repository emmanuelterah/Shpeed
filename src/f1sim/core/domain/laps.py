"""Lap classification (P1-08).

A *clean* lap is one whose time reflects a car's real pace: not lap 1, not an in- or
out-lap, timed accurately, not deleted, and run entirely without a safety car, virtual
safety car or red flag. Yellow flags are allowed: they are local and short.
"""

import pandas as pd

# FastF1 track status codes: 1 green, 2 yellow, 4 safety car, 5 red flag,
# 6 VSC deployed, 7 VSC ending.
NEUTRALISED_STATUS = frozenset("4567")


def mark_clean_laps(laps: pd.DataFrame) -> "pd.Series[bool]":
    """Return a boolean Series, aligned with ``laps``, marking clean laps."""
    status = laps["track_status"].fillna("").astype(str)
    neutralised = status.map(lambda s: any(ch in NEUTRALISED_STATUS for ch in s))
    clean = (
        (laps["lap_number"] > 1)
        & ~laps["pit_in"].fillna(False).astype(bool)
        & ~laps["pit_out"].fillna(False).astype(bool)
        & laps["lap_time_s"].notna()
        & laps["is_accurate"].fillna(False).astype(bool)
        & ~laps["deleted"].fillna(False).astype(bool)
        & ~neutralised.astype(bool)
    )
    return clean.astype(bool)
