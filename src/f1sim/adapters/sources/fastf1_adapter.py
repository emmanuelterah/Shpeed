"""FastF1 source adapter (P1-02): laps, stints, weather, track status and race control.

It owns the ``laps``, ``stints``, ``weather``, ``track_status`` and ``race_control``
tables. Drivers and teams are keyed by the Jolpica/Ergast IDs that FastF1 includes in
session results (P1-05).

FastF1 is reached through two injected callables (see ``fastf1_loaders``), so tests can
supply fake sessions without downloading anything.
"""

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import fastf1
import pandas as pd
import structlog

from f1sim.adapters.sources.parsing import SESSION_DURATION
from f1sim.core.domain.entities import EventKey, SessionKey, SessionType
from f1sim.core.domain.laps import mark_clean_laps
from f1sim.core.errors import DataError

log = structlog.get_logger(__name__)

FIRST_SEASON = 2018  # FastF1 timing data starts here

SESSION_NAMES: dict[str, SessionType] = {
    "Practice 1": SessionType.FP1,
    "Practice 2": SessionType.FP2,
    "Practice 3": SessionType.FP3,
    "Sprint Qualifying": SessionType.SPRINT_QUALIFYING,
    "Sprint Shootout": SessionType.SPRINT_QUALIFYING,
    "Sprint": SessionType.SPRINT,
    "Qualifying": SessionType.QUALIFYING,
    "Race": SessionType.RACE,
}
_IDENTIFIER = {v: v.value for v in SessionType}

LAP_TIME_RANGE = (40.0, 400.0)


class SessionLike(Protocol):
    """The parts of a loaded ``fastf1.core.Session`` this adapter reads."""

    @property
    def laps(self) -> pd.DataFrame | None:
        """Every lap of the session, one row per driver per lap."""
        ...

    @property
    def results(self) -> pd.DataFrame | None:
        """Classification, including Jolpica ``DriverId`` and ``TeamId``."""
        ...

    @property
    def weather_data(self) -> pd.DataFrame | None:
        """Weather samples, roughly one a minute."""
        ...

    @property
    def track_status(self) -> pd.DataFrame | None:
        """Track status changes (green, yellow, safety car and so on)."""
        ...

    @property
    def race_control_messages(self) -> pd.DataFrame | None:
        """Race-control messages (flags, safety car, investigations and so on)."""
        ...

    @property
    def date(self) -> pd.Timestamp:
        """Scheduled session start, in UTC."""
        ...


SessionLoader = Callable[[int, int, str], SessionLike]
ScheduleLoader = Callable[[int], pd.DataFrame]


def fastf1_loaders(cache_dir: Path) -> tuple[SessionLoader, ScheduleLoader]:
    """Build the real loaders, backed by the fastf1 package and its on-disk cache.

    Args:
        cache_dir: FastF1's cache directory; created if missing.

    Returns:
        A session loader and a schedule loader for ``FastF1Adapter``.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))

    def _load_session(season: int, round_: int, identifier: str) -> SessionLike:
        session = fastf1.get_session(season, round_, identifier)
        session.load(laps=True, telemetry=False, weather=True, messages=True)
        return session

    def _load_schedule(season: int) -> pd.DataFrame:
        return fastf1.get_event_schedule(season, include_testing=False)

    return _load_session, _load_schedule


class FastF1Adapter:
    """The ``SourceAdapter`` for FastF1.

    Attributes:
        name: Source name in the sync ledger.
    """

    name = "fastf1"

    def __init__(
        self,
        *,
        load_session: SessionLoader,
        load_schedule: ScheduleLoader,
        clock: Callable[[], datetime],
    ) -> None:
        """Create the adapter.

        Args:
            load_session: Loads one session by season, round and session code.
            load_schedule: Loads a season's event schedule.
            clock: Returns the current naive UTC time; decides which sessions have finished.
        """
        self._load_session = load_session
        self._load_schedule = load_schedule
        self._clock = clock

    # ------------------------------------------------------------------ SourceAdapter
    def sessions(self, season: int) -> Sequence[SessionKey]:
        """List every finished session of a season on FastF1's schedule.

        Args:
            season: Championship year; seasons before 2018 have no FastF1 timing data.

        Returns:
            Finished sessions, in schedule order.
        """
        if season < FIRST_SEASON:
            return []
        now = self._clock()
        found: list[SessionKey] = []
        schedule = self._load_schedule(season)
        for _, event in schedule.iterrows():
            round_ = int(event["RoundNumber"])
            if round_ < 1:
                continue
            for i in range(1, 6):
                session = SESSION_NAMES.get(str(event.get(f"Session{i}", "")))
                start = event.get(f"Session{i}DateUtc")
                if session is None or start is None or pd.isna(start):
                    continue
                if pd.Timestamp(start).to_pydatetime() + SESSION_DURATION[session.value] <= now:
                    found.append(SessionKey(EventKey(season, round_), session))
        return found

    def fetch(self, session: SessionKey) -> Mapping[str, pd.DataFrame]:
        """Load one session and convert it to canonical tables.

        Args:
            session: Any session of a weekend.

        Returns:
            ``laps`` (with the ``clean_lap`` flag) and the ``stints`` derived from them,
            ``weather``, ``track_status`` and ``race_control``.

        Raises:
            DataError: If FastF1 cannot load the session or it has no laps.
        """
        e = session.event
        # FastF1 raises many exception types; every one of them means "no usable data"
        try:
            s = self._load_session(e.season, e.round, _IDENTIFIER[session.session])
            laps_raw = s.laps
        except Exception as exc:
            raise DataError(f"FastF1 could not load {session}: {exc}") from exc
        if laps_raw is None or len(laps_raw) == 0:
            raise DataError(f"FastF1 has no laps for {session}")
        start = _naive_utc(s.date)
        available_at = start + SESSION_DURATION[session.session.value]
        t0 = _session_zero(laps_raw)
        base = {"season": e.season, "round": e.round, "session": session.session.value}
        return {
            "laps": _laps(laps_raw, s.results, base, available_at, session),
            "stints": _stints(laps_raw, s.results, base, available_at),
            "weather": _weather(s.weather_data, t0, base, available_at),
            "track_status": _track_status(s.track_status, t0, base, available_at),
            "race_control": _race_control(s.race_control_messages, t0, base, available_at),
        }


def _seconds(series: "pd.Series[Any]") -> "pd.Series[float]":
    return pd.to_timedelta(series).dt.total_seconds()


def _naive_utc(stamp: Any) -> datetime:
    """Convert a date or timestamp to a naive UTC datetime."""
    ts = pd.Timestamp(stamp)
    return (ts.tz_convert(UTC) if ts.tzinfo is not None else ts).tz_localize(None).to_pydatetime()


def _session_zero(laps: pd.DataFrame) -> pd.Timestamp:
    """Wall-clock time of FastF1's session time zero, from the lap start stamps.

    FastF1 only exposes ``Session.t0_date`` after loading telemetry; deriving it from the
    laps avoids that heavy download.
    """
    starts = laps[["LapStartDate", "LapStartTime"]].dropna()
    if len(starts) == 0:
        return pd.Timestamp(0)
    offsets = starts["LapStartDate"] - pd.to_timedelta(starts["LapStartTime"])
    return pd.Timestamp(offsets.max())


def _utc_times(times: "pd.Series[Any]", t0: pd.Timestamp) -> "pd.Series[Any]":
    """Turn a session-relative ``Time`` column into naive UTC timestamps."""
    shifted = t0 + pd.to_timedelta(times)
    return shifted.dt.tz_localize(None) if t0.tzinfo is not None else shifted


def _clean_id(value: Any) -> str | None:
    """A usable canonical ID, or None for missing, empty or ``"nan"`` values."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    return None if not text or text.lower() == "nan" else text


def _driver_keys(results: pd.DataFrame | None) -> tuple[dict[str, str], dict[str, str]]:
    ids: dict[str, str] = {}
    teams: dict[str, str] = {}
    if results is None or len(results) == 0:
        return ids, teams
    for _, r in results.iterrows():
        code = _clean_id(r.get("Abbreviation"))
        driver_id = _clean_id(r.get("DriverId"))
        team_id = _clean_id(r.get("TeamId"))
        if code and driver_id:
            ids[code] = driver_id
        if code and team_id:
            teams[code] = team_id
    return ids, teams


def _lap_times(raw: pd.DataFrame, session: SessionKey) -> tuple["pd.Series[float]", Any]:
    """Lap times in seconds, and a mask of implausible ones (such as red-flag laps)."""
    lap_time = _seconds(raw["LapTime"])
    lo, hi = LAP_TIME_RANGE
    implausible = lap_time.notna() & ((lap_time < lo) | (lap_time > hi))
    if implausible.any():
        log.info(
            "fastf1.implausible_lap_times_nulled", session=str(session), n=int(implausible.sum())
        )
    return lap_time, implausible


def _laps(
    raw: pd.DataFrame,
    results: pd.DataFrame | None,
    base: dict[str, Any],
    available_at: datetime,
    session: SessionKey,
) -> pd.DataFrame:
    raw = raw[raw["LapNumber"].notna() & raw["Driver"].notna()]
    ids, teams = _driver_keys(results)
    codes = raw["Driver"].astype(str)
    missing = sorted(set(codes) - set(ids))
    if missing:
        log.warning("fastf1.driver_id_fallback", session=str(session), codes=missing)

    lap_time, implausible = _lap_times(raw, session)
    df = pd.DataFrame(
        {
            **base,
            "driver_id": codes.map(lambda c: ids.get(c, f"code:{c.lower()}")),
            "driver_code": codes,
            "team_id": codes.map(teams.get),
            "lap_number": raw["LapNumber"].astype(int),
            "lap_time_s": lap_time.mask(implausible),
            "sector1_s": _seconds(raw["Sector1Time"]),
            "sector2_s": _seconds(raw["Sector2Time"]),
            "sector3_s": _seconds(raw["Sector3Time"]),
            "stint": raw["Stint"],
            "compound": raw["Compound"].astype("string"),
            "tyre_life": raw["TyreLife"],
            "fresh_tyre": raw["FreshTyre"].astype("boolean"),
            "pit_in": raw["PitInTime"].notna(),
            "pit_out": raw["PitOutTime"].notna(),
            "track_status": raw["TrackStatus"].astype("string"),
            "position": raw["Position"],
            "is_accurate": raw["IsAccurate"].astype("boolean") & ~implausible,
            "deleted": raw["Deleted"].astype("boolean").fillna(False)
            if "Deleted" in raw
            else False,
            "lap_start_utc": pd.to_datetime(raw["LapStartDate"], utc=True).dt.tz_localize(None),
            "available_at": available_at,
        }
    ).reset_index(drop=True)
    for col in ("sector1_s", "sector2_s", "sector3_s"):
        df[col] = df[col].where(df[col] > 0)
    before = len(df)
    df = df.drop_duplicates(subset=["driver_id", "lap_number"], keep="first")
    if len(df) < before:
        log.warning("fastf1.duplicate_laps_dropped", session=str(session), n=before - len(df))
    df["clean_lap"] = mark_clean_laps(df)
    return df.reset_index(drop=True)


def _stints(
    raw: pd.DataFrame,
    results: pd.DataFrame | None,
    base: dict[str, Any],
    available_at: datetime,
) -> pd.DataFrame:
    """One row per driver's stint, derived from the session's laps."""
    laps = raw.loc[raw["LapNumber"].notna() & raw["Stint"].notna() & raw["Driver"].notna()]
    ids, _ = _driver_keys(results)
    codes = laps["Driver"].astype(str)
    grouped = (
        laps.assign(
            driver_id=codes.map(lambda c: ids.get(c, f"code:{c.lower()}")),
            lap=laps["LapNumber"].astype(int),
            stint=laps["Stint"].astype(int),
        )
        .groupby(["driver_id", "stint"], sort=True)
        .agg(start_lap=("lap", "min"), end_lap=("lap", "max"), compound=("Compound", "first"))
        .reset_index()
    )
    return pd.DataFrame(
        {
            **base,
            "driver_id": grouped["driver_id"],
            "stint": grouped["stint"].astype(int),
            "compound": grouped["compound"].astype("string"),
            "start_lap": grouped["start_lap"].astype(int),
            "end_lap": grouped["end_lap"].astype(int),
            "available_at": available_at,
        }
    ).reset_index(drop=True)


def _optional_text(raw: pd.DataFrame, column: str) -> "pd.Series[Any]":
    if column not in raw:
        return pd.Series(pd.NA, index=raw.index, dtype="string")
    return raw[column].astype("string")


def _optional_numeric(raw: pd.DataFrame, column: str) -> "pd.Series[Any]":
    if column not in raw:
        return pd.Series(pd.NA, index=raw.index, dtype="Float64")
    return pd.to_numeric(raw[column], errors="coerce")


def _race_control(
    raw: pd.DataFrame | None, _t0: pd.Timestamp, base: dict[str, Any], available_at: datetime
) -> pd.DataFrame:
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    # Race-control messages carry a wall-clock UTC time, not a session-relative one.
    number = "RacingNumber" if "RacingNumber" in raw else "DriverNumber"
    df = pd.DataFrame(
        {
            **base,
            "time_utc": pd.to_datetime(raw["Time"], utc=True).dt.tz_localize(None),
            "category": raw["Category"].astype("string"),
            "message": raw["Message"].astype("string"),
            "flag": _optional_text(raw, "Flag"),
            "scope": _optional_text(raw, "Scope"),
            "sector": _optional_numeric(raw, "Sector"),
            "driver_number": _optional_numeric(raw, number),
            "lap": _optional_numeric(raw, "Lap"),
            "available_at": available_at,
        }
    )
    return df.drop_duplicates(subset=["time_utc", "message"], keep="first").reset_index(drop=True)


def _weather(
    raw: pd.DataFrame | None, t0: pd.Timestamp, base: dict[str, Any], available_at: datetime
) -> pd.DataFrame:
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(
        {
            **base,
            "time_utc": _utc_times(raw["Time"], t0),
            "air_temp_c": raw["AirTemp"],
            "track_temp_c": raw["TrackTemp"],
            "humidity_pct": raw["Humidity"],
            "pressure_mbar": raw["Pressure"],
            "rainfall": raw["Rainfall"].astype("boolean"),
            "wind_speed_ms": raw["WindSpeed"],
            "wind_direction_deg": raw["WindDirection"],
            "available_at": available_at,
        }
    )
    return df.drop_duplicates(subset=["time_utc"]).reset_index(drop=True)


def _track_status(
    raw: pd.DataFrame | None, t0: pd.Timestamp, base: dict[str, Any], available_at: datetime
) -> pd.DataFrame:
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(
        {
            **base,
            "time_utc": _utc_times(raw["Time"], t0),
            "status": raw["Status"].astype(str),
            "message": raw["Message"].astype("string"),
            "available_at": available_at,
        }
    )
    return df.drop_duplicates(subset=["time_utc", "status"]).reset_index(drop=True)
