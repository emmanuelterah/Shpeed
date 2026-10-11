"""FastF1 source adapter (P1-02): laps, tyre stints, weather and track status, 2018 onwards.

It owns the ``laps``, ``weather`` and ``track_status`` tables. Drivers and teams are keyed
by the Jolpica/Ergast IDs that FastF1 includes in session results (P1-05).

FastF1 is reached through two injected callables, so tests can supply fake sessions and
the module imports without the ``data`` extra installed.
"""

import importlib
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

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
    """The parts of ``fastf1.core.Session`` this adapter reads."""

    laps: pd.DataFrame | None
    results: pd.DataFrame | None
    weather_data: pd.DataFrame | None
    track_status: pd.DataFrame | None
    date: Any
    t0_date: Any


SessionLoader = Callable[[int, int, str], SessionLike]
ScheduleLoader = Callable[[int], pd.DataFrame]


def fastf1_loaders(cache_dir: Path) -> tuple[SessionLoader, ScheduleLoader]:
    """Real loaders backed by the fastf1 package and its on-disk cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    def module() -> Any:
        ff1: Any = importlib.import_module("fastf1")
        ff1.Cache.enable_cache(str(cache_dir))
        return ff1

    def load_session(season: int, round_: int, identifier: str) -> SessionLike:
        session: Any = module().get_session(season, round_, identifier)
        session.load(laps=True, telemetry=False, weather=True, messages=True)
        return session  # type: ignore[no-any-return]

    def load_schedule(season: int) -> pd.DataFrame:
        schedule: pd.DataFrame = module().get_event_schedule(season, include_testing=False)
        return schedule

    return load_session, load_schedule


class FastF1Adapter:
    name = "fastf1"

    def __init__(
        self,
        *,
        load_session: SessionLoader,
        load_schedule: ScheduleLoader,
        clock: Callable[[], datetime],
    ) -> None:
        self._load_session = load_session
        self._load_schedule = load_schedule
        self._clock = clock

    # ------------------------------------------------------------------ SourceAdapter
    def sessions(self, season: int) -> Sequence[SessionKey]:
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
        e = session.event
        try:
            s = self._load_session(e.season, e.round, _IDENTIFIER[session.session])
            laps_raw = s.laps
        except Exception as exc:  # FastF1 raises many types; all mean "no usable data"
            raise DataError(f"FastF1 could not load {session}: {exc}") from exc
        if laps_raw is None or len(laps_raw) == 0:
            raise DataError(f"FastF1 has no laps for {session}")
        start = pd.Timestamp(s.date).to_pydatetime().replace(tzinfo=None)
        available_at = start + SESSION_DURATION[session.session.value]
        t0 = pd.Timestamp(s.t0_date)
        base = {"season": e.season, "round": e.round, "session": session.session.value}
        return {
            "laps": _laps(laps_raw, s.results, base, available_at, session),
            "weather": _weather(s.weather_data, t0, base, available_at),
            "track_status": _track_status(s.track_status, t0, base, available_at),
        }


def _seconds(series: "pd.Series[Any]") -> "pd.Series[float]":
    return pd.to_timedelta(series).dt.total_seconds()


def _driver_keys(results: pd.DataFrame | None) -> tuple[dict[str, str], dict[str, str]]:
    ids: dict[str, str] = {}
    teams: dict[str, str] = {}
    if results is None or len(results) == 0:
        return ids, teams
    for _, r in results.iterrows():
        code = str(r.get("Abbreviation", ""))
        driver_id = r.get("DriverId")
        team_id = r.get("TeamId")
        if code and isinstance(driver_id, str) and driver_id:
            ids[code] = driver_id
        if code and isinstance(team_id, str) and team_id:
            teams[code] = team_id
    return ids, teams


def _laps(
    raw: pd.DataFrame,
    results: pd.DataFrame | None,
    base: dict[str, Any],
    available_at: datetime,
    session: SessionKey,
) -> pd.DataFrame:
    raw = raw[raw["LapNumber"].notna()]
    ids, teams = _driver_keys(results)
    codes = raw["Driver"].astype(str)
    missing = sorted(set(codes) - set(ids))
    if missing:
        log.warning("fastf1.driver_id_fallback", session=str(session), codes=missing)

    lap_time = _seconds(raw["LapTime"])
    lo, hi = LAP_TIME_RANGE
    implausible = lap_time.notna() & ((lap_time < lo) | (lap_time > hi))
    if implausible.any():
        log.info(
            "fastf1.implausible_lap_times_nulled", session=str(session), n=int(implausible.sum())
        )

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
            "lap_start_utc": pd.to_datetime(raw["LapStartDate"]),
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


def _weather(
    raw: pd.DataFrame | None, t0: pd.Timestamp, base: dict[str, Any], available_at: datetime
) -> pd.DataFrame:
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(
        {
            **base,
            "time_utc": (t0 + pd.to_timedelta(raw["Time"])).dt.tz_localize(None)
            if getattr(t0, "tzinfo", None)
            else t0 + pd.to_timedelta(raw["Time"]),
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
            "time_utc": t0 + pd.to_timedelta(raw["Time"]),
            "status": raw["Status"].astype(str),
            "message": raw["Message"].astype("string"),
            "available_at": available_at,
        }
    )
    return df.drop_duplicates(subset=["time_utc", "status"]).reset_index(drop=True)
