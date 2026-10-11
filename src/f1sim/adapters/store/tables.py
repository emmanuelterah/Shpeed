"""Canonical tables (P1-01): one registry that the store, validation and reports all use.

Every data table carries ``available_at``: the UTC time the data became knowable (normally
the end of its session). Readers only return rows with ``available_at <= as_of``, which is
what stops a forecast or backtest from seeing the future.

Ownership is explicit so two sources never write the same rows:
  jolpica -> events, results, pit_stops
  fastf1  -> laps, weather, track_status
"""

from dataclasses import dataclass

# DuckDB column types; pandas dtypes are derived from these in schemas.py
INT, FLOAT, TEXT, BOOL, TS = "INTEGER", "DOUBLE", "VARCHAR", "BOOLEAN", "TIMESTAMP"


@dataclass(frozen=True)
class TableSpec:
    name: str
    columns: dict[str, str]
    replace_key: tuple[str, ...]
    """A commit replaces all existing rows matching these columns, so re-syncs are idempotent."""
    unique: tuple[str, ...]
    """Columns that identify one row; checked inside the transaction before COMMIT."""
    time_filtered: bool = True
    """Whether reads filter on ``available_at`` (internal tables are not filtered)."""

    def ddl(self, catalog: str) -> str:
        cols = ", ".join(f"{name} {kind}" for name, kind in self.columns.items())
        return f"CREATE TABLE IF NOT EXISTS {catalog}.{self.name} ({cols})"


_SESSION = {"season": INT, "round": INT, "session": TEXT}

TABLES: dict[str, TableSpec] = {
    spec.name: spec
    for spec in [
        TableSpec(
            "events",
            {
                "season": INT,
                "round": INT,
                "event_name": TEXT,
                "circuit_id": TEXT,
                "locality": TEXT,
                "country": TEXT,
                "event_date": TS,
                "event_format": TEXT,
                "available_at": TS,
            },
            replace_key=("season", "round"),
            unique=("season", "round"),
        ),
        TableSpec(
            "results",
            {
                **_SESSION,
                "driver_id": TEXT,
                "driver_code": TEXT,
                "driver_number": INT,
                "team_id": TEXT,
                "team_name": TEXT,
                "grid": INT,
                "position": INT,
                "position_text": TEXT,
                "status": TEXT,
                "points": FLOAT,
                "laps": INT,
                "time_s": FLOAT,
                "q1_s": FLOAT,
                "q2_s": FLOAT,
                "q3_s": FLOAT,
                "available_at": TS,
            },
            replace_key=("season", "round", "session"),
            unique=("season", "round", "session", "driver_id"),
        ),
        TableSpec(
            "pit_stops",
            {
                **_SESSION,
                "driver_id": TEXT,
                "stop": INT,
                "lap": INT,
                "duration_s": FLOAT,
                "available_at": TS,
            },
            replace_key=("season", "round", "session"),
            unique=("season", "round", "session", "driver_id", "stop"),
        ),
        TableSpec(
            "laps",
            {
                **_SESSION,
                "driver_id": TEXT,
                "driver_code": TEXT,
                "team_id": TEXT,
                "lap_number": INT,
                "lap_time_s": FLOAT,
                "sector1_s": FLOAT,
                "sector2_s": FLOAT,
                "sector3_s": FLOAT,
                "stint": INT,
                "compound": TEXT,
                "tyre_life": FLOAT,
                "fresh_tyre": BOOL,
                "pit_in": BOOL,
                "pit_out": BOOL,
                "track_status": TEXT,
                "position": INT,
                "is_accurate": BOOL,
                "deleted": BOOL,
                "lap_start_utc": TS,
                "clean_lap": BOOL,
                "available_at": TS,
            },
            replace_key=("season", "round", "session"),
            unique=("season", "round", "session", "driver_id", "lap_number"),
        ),
        TableSpec(
            "weather",
            {
                **_SESSION,
                "time_utc": TS,
                "air_temp_c": FLOAT,
                "track_temp_c": FLOAT,
                "humidity_pct": FLOAT,
                "pressure_mbar": FLOAT,
                "rainfall": BOOL,
                "wind_speed_ms": FLOAT,
                "wind_direction_deg": FLOAT,
                "available_at": TS,
            },
            replace_key=("season", "round", "session"),
            unique=("season", "round", "session", "time_utc"),
        ),
        TableSpec(
            "track_status",
            {
                **_SESSION,
                "time_utc": TS,
                "status": TEXT,
                "message": TEXT,
                "available_at": TS,
            },
            replace_key=("season", "round", "session"),
            unique=("season", "round", "session", "time_utc", "status"),
        ),
        TableSpec(
            "_ingested",
            {
                "source": TEXT,
                **_SESSION,
                "run_id": TEXT,
                "ingested_at": TS,
            },
            replace_key=("source", "season", "round", "session"),
            unique=("source", "season", "round", "session"),
            time_filtered=False,
        ),
    ]
}

DATA_TABLES = tuple(name for name, spec in TABLES.items() if spec.time_filtered)
