"""Validation for every table (P1-07), run before any write commits (ACID consistency).

Each schema checks types, required values, plausible ranges, uniqueness and, for laps,
that stints join up. A failure raises ``ValidationError`` listing the first problems.
"""

from typing import Any, cast

import pandas as pd
import pandera.pandas as pa

from f1sim.adapters.store.tables import BOOL, FLOAT, INT, TABLES, TEXT, TS, TableSpec
from f1sim.core.errors import ValidationError

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
    "laps": {"ge": 0, "le": 120},
    "stop": {"ge": 1, "le": 12},
    "lap": {"ge": 1, "le": 120},
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
}


def _checks(table: str, column: str) -> list[pa.Check]:
    checks: list[pa.Check] = []
    if column in _RANGES:
        bounds = _RANGES[column]
        if "ge" in bounds and "le" in bounds:
            checks.append(pa.Check.in_range(bounds["ge"], bounds["le"]))  # pyright: ignore[reportUnknownMemberType]
        elif "gt" in bounds and "le" in bounds:
            checks.append(pa.Check.in_range(bounds["gt"], bounds["le"], include_min=False))  # pyright: ignore[reportUnknownMemberType]
        elif "ge" in bounds:
            checks.append(pa.Check.ge(bounds["ge"]))  # pyright: ignore[reportUnknownMemberType]
    allowed = _ALLOWED.get(column)
    if allowed and not (table == "results" and column == "status"):
        checks.append(pa.Check.isin(allowed))  # pyright: ignore[reportUnknownMemberType]
    return checks


def _required(table: str, column: str) -> bool:
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
    """Return ``df`` coerced to the canonical types, in canonical column order.

    Raises ``ValidationError`` for an unknown table or any failed check.
    """
    if table not in SCHEMAS:
        raise ValidationError(f"Unknown table {table!r}; known tables: {sorted(SCHEMAS)}")
    missing = [c for c in TABLES[table].columns if c not in df.columns]
    if missing:
        raise ValidationError(f"{table}: missing columns {missing}")
    try:
        checked = SCHEMAS[table].validate(df.reset_index(drop=True), lazy=True)
    except pa.errors.SchemaErrors as exc:
        failures = cast(pd.DataFrame, exc.failure_cases)  # pyright: ignore[reportUnknownMemberType]
        cases = failures.head(10).to_string(index=False)
        raise ValidationError(f"{table}: {len(failures)} failed checks\n{cases}") from exc
    return checked[list(TABLES[table].columns)].reset_index(drop=True)
