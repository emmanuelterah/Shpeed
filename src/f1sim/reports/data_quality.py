"""Per-session data-quality report (P1-16), written as Markdown."""

from datetime import datetime

import pandas as pd

from f1sim.core.domain.entities import SnapshotId
from f1sim.core.ports.store import DataReader

FAR_FUTURE = datetime(2100, 1, 1)
LOW_CLEAN_PCT = 40.0


def _only(
    df: pd.DataFrame, season: int, round_: int | None = None, session: str | None = None
) -> pd.DataFrame:
    mask = df["season"] == season
    if round_ is not None:
        mask &= df["round"] == round_
    if session is not None:
        mask &= df["session"] == session
    selected: pd.DataFrame = df.loc[mask]
    return selected


def _pct(flags: list[bool]) -> float | None:
    return round(100 * sum(flags) / len(flags), 1) if flags else None


def _session_row(
    laps: pd.DataFrame, results: pd.DataFrame, weather: pd.DataFrame
) -> dict[str, object]:
    """Quality figures for one session, given that session's rows of each table."""
    first = laps if len(laps) else results
    driver_ids: list[object] = first["driver_id"].tolist()
    lap_ids: list[object] = laps["driver_id"].tolist()
    return {
        "round": int(first["round"].iloc[0]),
        "session": str(first["session"].iloc[0]),
        "drivers": len(set(driver_ids)),
        "laps": len(laps),
        "clean_lap_pct": _pct([bool(v) for v in laps["clean_lap"].tolist()]),
        "missing_lap_time_pct": _pct([bool(v) for v in laps["lap_time_s"].isna().tolist()]),
        "id_fallbacks": sum(str(d).startswith("code:") for d in lap_ids),
        "weather_rows": len(weather),
        "results_rows": len(results),
    }


def session_quality(reader: DataReader, season: int, snapshot: SnapshotId) -> pd.DataFrame:
    """Summarise data quality per session.

    Args:
        reader: Read access to the store.
        season: Season to summarise.
        snapshot: Snapshot to read.

    Returns:
        One row per session: drivers, laps, clean-lap share, missing lap times, driver ID
        fallbacks, and weather and results row counts.
    """
    laps = _only(reader.read("laps", snapshot=snapshot, as_of=FAR_FUTURE), season)
    results = _only(reader.read("results", snapshot=snapshot, as_of=FAR_FUTURE), season)
    weather = _only(reader.read("weather", snapshot=snapshot, as_of=FAR_FUTURE), season)

    keys: set[tuple[int, str]] = set()
    for frame in (laps, results):
        for r, s in zip(frame["round"].tolist(), frame["session"].tolist(), strict=True):
            keys.add((int(r), str(s)))

    rows = [
        _session_row(
            _only(laps, season, round_, session),
            _only(results, season, round_, session),
            _only(weather, season, round_, session),
        )
        for round_, session in sorted(keys)
    ]
    return pd.DataFrame(rows)


def render_markdown(df: pd.DataFrame, season: int, snapshot: SnapshotId) -> str:
    """Render the quality summary as a Markdown report.

    Args:
        df: Output of ``session_quality``.
        season: Season shown in the title.
        snapshot: Snapshot the summary was read from.

    Returns:
        The report, with a "Needs a look" list of sessions with few clean laps or ID gaps.
    """
    lines = [
        f"# Data quality: {season} season",
        "",
        f"Snapshot {int(snapshot)}. Clean laps exclude lap 1, in/out laps, inaccurate or deleted "
        "laps, and laps under safety car, VSC or red flag. `id_fallbacks` counts laps whose "
        "driver had no Jolpica ID in FastF1's results.",
        "",
    ]
    if df.empty:
        lines.append("No sessions stored for this season yet. Run `f1sim data sync`.")
        return "\n".join(lines) + "\n"
    columns = [str(c) for c in df.columns]
    lines += ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    flagged: list[str] = []
    for record in df.to_dict(orient="records"):
        cells = ["" if pd.isna(record[c]) else str(record[c]) for c in columns]
        lines.append("| " + " | ".join(cells) + " |")
        clean = record["clean_lap_pct"]
        low_clean = isinstance(clean, float) and clean < LOW_CLEAN_PCT
        if record["laps"] and (low_clean or record["id_fallbacks"]):
            flagged.append(
                f"- Round {record['round']} {record['session']}: {clean}% clean, "
                f"{record['id_fallbacks']} ID fallbacks"
            )
    if flagged:
        lines += ["", "## Needs a look", "", *flagged]
    return "\n".join(lines) + "\n"
