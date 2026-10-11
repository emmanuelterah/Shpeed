"""Small, forgiving parsers shared by the source adapters."""

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
    """Parse a lap or stop time into seconds.

    Args:
        text: A time such as ``"1:29.179"`` (89.179), ``"22.818"`` or ``"1:02:03.5"``.

    Returns:
        The time in seconds, or None if ``text`` is blank or not a time.
    """
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
    """Parse an integer, tolerating missing or malformed values.

    Args:
        text: Any value, typically a string from an API response.

    Returns:
        The integer, or None if ``text`` is not one.
    """
    try:
        return int(str(text))
    except (TypeError, ValueError):
        return None


def parse_float(text: Any) -> float | None:
    """Parse a number, tolerating missing or malformed values.

    Args:
        text: Any value, typically a string from an API response.

    Returns:
        The number, or None if ``text`` is not one.
    """
    try:
        return float(str(text))
    except (TypeError, ValueError):
        return None


def parse_utc(
    date: str | None, time: str | None, *, default_time: str = "14:00:00"
) -> datetime | None:
    """Combine an Ergast-style date and time into a naive UTC datetime.

    Args:
        date: A date such as ``"2024-03-02"``.
        time: A UTC time such as ``"15:00:00Z"``; None uses ``default_time``.
        default_time: Time to assume when ``time`` is missing (older seasons).

    Returns:
        The combined datetime without a timezone, or None if ``date`` is missing.
    """
    if not date:
        return None
    clock = (time or default_time).rstrip("Z")
    return datetime.fromisoformat(f"{date}T{clock}")
