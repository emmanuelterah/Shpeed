"""Small parsers shared by source adapters."""

from datetime import datetime, timedelta
from typing import Any

# How long after a session's scheduled start its data can be treated as available.
SESSION_DURATION: dict[str, timedelta] = {
    "FP1": timedelta(hours=1, minutes=30),
    "FP2": timedelta(hours=1, minutes=30),
    "FP3": timedelta(hours=1, minutes=30),
    "SQ": timedelta(hours=1, minutes=15),
    "S": timedelta(hours=1, minutes=30),
    "Q": timedelta(hours=1, minutes=30),
    "R": timedelta(hours=3),
}


def parse_time_s(text: str | None) -> float | None:
    """'1:29.179' -> 89.179, '22.818' -> 22.818, '1:02:03.5' -> 3723.5; blank -> None."""
    if text is None or not str(text).strip():
        return None
    total = 0.0
    try:
        for part in str(text).strip().split(":"):
            total = total * 60 + float(part)
    except ValueError:
        return None
    return total


def parse_int(text: Any) -> int | None:
    try:
        return int(str(text))
    except (TypeError, ValueError):
        return None


def parse_float(text: Any) -> float | None:
    try:
        return float(str(text))
    except (TypeError, ValueError):
        return None


def parse_utc(
    date: str | None, time: str | None, *, default_time: str = "14:00:00"
) -> datetime | None:
    """Combine Ergast-style date and time ('2024-03-02', '15:00:00Z') into naive UTC."""
    if not date:
        return None
    clock = (time or default_time).rstrip("Z")
    return datetime.fromisoformat(f"{date}T{clock}")
