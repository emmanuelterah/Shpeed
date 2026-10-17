"""Validation for every table (P1-07), run before any write commits (ACID consistency).

Each schema checks types, required values, plausible ranges, uniqueness and, for laps,
that stints join up. A failure raises ``ValidationError`` listing the first problems.
"""

from typing import Any, cast

import pandas as pd
import pandera.pandas as pa

from f1sim.adapters.store.tables import BOOL, FLOAT, INT, TABLES, TEXT, TS, TableSpec
from f1sim.core.errors import ValidationError

# pandera's Check constructors are untyped; one Any-typed alias keeps strict pyright happy
# without a pragma comment on every call.
_CHECK: Any = pa.Check

_DTYPES: dict[str, Any] = {
    INT: "Int64",
    FLOAT: "float64",
    TEXT: pd.StringDtype(),
    BOOL: "boolean",
    TS: "datetime64[ns]",
}

_REQUIRED = {
    "season",
    "round",
    "session",
    "source",
    "driver_id",
    "available_at",
    "lap_number",
    "time_utc",
    "stop",
    "run_id",
    "ingested_at",
    "start_lap",
    "end_lap",
    "subject_id",
}

_RANGES: dict[str, dict[str, Any]] = {
    "season": {"ge": 1950, "le": 2100},
    "round": {"ge": 1, "le": 40},
    "lap_number": {"ge": 1, "le": 120},
    "lap_time_s": {"ge": 40.0, "le": 400.0},
    "sector1_s": {"gt": 0.0, "le": 300.0},
    "sector2_s": {"gt": 0.0, "le": 300.0},
    "sector3_s": {"gt": 0.0, "le": 300.0},
    "stint": {"ge": 1, "le": 12},
    "tyre_life": {"ge": 0, "le": 120},
    "position": {"ge": 1, "le": 30},
    "grid": {"ge": 0, "le": 30},
    "points": {"ge": 0},
    "wins": {"ge": 0},
    "laps": {"ge": 0, "le": 120},
    "stop": {"ge": 1, "le": 12},
    "lap": {"ge": 1, "le": 120},
    "start_lap": {"ge": 1, "le": 120},
    "end_lap": {"ge": 1, "le": 120},
    "sector": {"ge": 0, "le": 99},
    "driver_number": {"ge": 1, "le": 120},
    "duration_s": {"gt": 0.0, "le": 7200.0},
    "air_temp_c": {"ge": -10.0, "le": 60.0},
    "track_temp_c": {"ge": -10.0, "le": 80.0},
    "humidity_pct": {"ge": 0.0, "le": 100.0},
    "wind_speed_ms": {"ge": 0.0, "le": 60.0},
}

_ALLOWED: dict[str, list[str]] = {
    "session": ["FP1", "FP2", "FP3", "SQ", "S", "Q", "R"],
    "event_format": ["conventional", "sprint", "sprint_qualifying", "sprint_shootout"],
    "status": ["1", "2", "3", "4", "5", "6", "7"],
    "scope": ["driver", "constructor"],
}


def _checks(table: str, column: str) -> list[pa.Check]:
    checks: list[pa.Check] = []
    if column in _RANGES:
        bounds = _RANGES[column]
        if "ge" in bounds and "le" in bounds:
            checks.append(_CHECK.in_range(bounds["ge"], bounds["le"]))
        elif "gt" in bounds and "le" in bounds:
            checks.append(_CHECK.in_range(bounds["gt"], bounds["le"], include_min=False))
        elif "ge" in bounds:
            checks.append(_CHECK.ge(bounds["ge"]))
    allowed = _ALLOWED.get(column)
    # ``scope`` and ``status`` mean different things per table, so only constrain the
    # table that owns the enum.
    skipped = (table == "results" and column == "status") or (
        table == "race_control" and column == "scope"
    )
    if allowed and not skipped:
        checks.append(_CHECK.isin(allowed))
    return checks


def _required(table: str, column: str) -> bool:
    if table == "race_control" and column in {"category", "message"}:
        return True
    if table == "standings" and column == "scope":
        return True
    return column in _REQUIRED or (table == "track_status" and column == "status")


def _stints_join_up(df: pd.DataFrame) -> bool:
    """Within each driver's session, stint numbers never go backwards as laps increase."""
    if df.empty or df["stint"].isna().all():
        return True
    ordered = df.dropna(subset=["stint"]).sort_values(["lap_number"])
    keys = ["season", "round", "session", "driver_id"]
    steps = ordered.groupby(keys, sort=False)["stint"].diff().dropna()
    return bool((steps >= 0).all())


def build_schema(spec: TableSpec) -> pa.DataFrameSchema:
    """Build the pandera schema for one canonical table.

    Args:
        spec: The table's registry entry.

    Returns:
        A strict schema that coerces types and checks ranges, nulls and uniqueness.
    """
    columns = {
        name: pa.Column(
            _DTYPES[kind],
            checks=_checks(spec.name, name),
            nullable=not _required(spec.name, name),
            coerce=True,
        )
        for name, kind in spec.columns.items()
    }
    checks: list[pa.Check] = []
    if spec.name == "laps":
        checks.append(
            pa.Check(_stints_join_up, error="stints must join up within a driver's session")
        )
    return pa.DataFrameSchema(
        columns, checks=checks, strict=True, unique=list(spec.unique), ordered=False
    )


SCHEMAS: dict[str, pa.DataFrameSchema] = {name: build_schema(s) for name, s in TABLES.items()}


def validate(table: str, df: pd.DataFrame) -> pd.DataFrame:
    """Validate rows for one table and coerce them to the canonical types.

    Args:
        table: Canonical table name.
        df: Rows to check; column order does not matter.

    Returns:
        The rows with canonical dtypes, in canonical column order, with a fresh index.

    Raises:
        ValidationError: If the table is unknown, columns are missing, or any check fails.
    """
    if table not in SCHEMAS:
        raise ValidationError(f"Unknown table {table!r}; known tables: {sorted(SCHEMAS)}")
    missing = [c for c in TABLES[table].columns if c not in df.columns]
    if missing:
        raise ValidationError(f"{table}: missing columns {missing}")
    try:
        checked = SCHEMAS[table].validate(df.reset_index(drop=True), lazy=True)
    except pa.errors.SchemaErrors as exc:
        errors: Any = exc
        failures = cast(pd.DataFrame, errors.failure_cases)
        cases = failures.head(10).to_string(index=False)
        raise ValidationError(f"{table}: {len(failures)} failed checks\n{cases}") from exc
    return checked[list(TABLES[table].columns)].reset_index(drop=True)
