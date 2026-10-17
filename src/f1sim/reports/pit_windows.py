"""Pit-window calibration report (P4-10), written as Markdown.

The report checks the strategy chooser the way it will be used: the tyre model and team
tendencies are fitted on training seasons, then for each held-out race the predicted pit
windows are compared with the laps cars actually stopped on. The window is the chooser's mean
stop lap widened by a stop-lap spread learned from training residuals, because a window around
the raw optimum is far too narrow (see :class:`~f1sim.adapters.models.strategy.StopLapSpread`).
The headline number is the share of actual stops that fall inside their predicted window.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from f1sim.adapters.config_files import load_season_rules
from f1sim.adapters.models.pit_stops import race_stops
from f1sim.adapters.models.strategy import (
    DEFAULT_STOP_SPREAD,
    PitWindow,
    StopLapSpread,
    StrategyChooser,
    StrategyGenerator,
    TeamTendencies,
    fit_team_tendencies,
)
from f1sim.adapters.models.tyres import TyreModel, fit_tyre_model
from f1sim.core.domain.entities import SnapshotId
from f1sim.core.domain.rules import SeasonRules
from f1sim.core.errors import ShpeedError
from f1sim.core.ports.store import DataReader

FAR_FUTURE = datetime(2100, 1, 1)

MIN_SPREAD_SAMPLES = 5
"""Fewer training residuals than this and a stop index falls back to the pooled spread."""

LOW_QUANTILE = 0.1
HIGH_QUANTILE = 0.9


@dataclass(frozen=True)
class WindowRow:
    """One stop checked against its window.

    Attributes:
        circuit: Circuit ID.
        driver: Driver ID.
        stop_index: Which stop, 0-based.
        low_lap: Lowest lap of the predicted window.
        high_lap: Highest lap of the predicted window.
        actual_lap: Lap the car actually stopped on.
        hit: Whether the actual lap was inside the window.
    """

    circuit: str
    driver: str
    stop_index: int
    low_lap: int
    high_lap: int
    actual_lap: int
    hit: bool


@dataclass
class PitWindowCalibration:
    """The calibration result (P4-10).

    Attributes:
        rows: One row per checked stop.
        train_seasons: Seasons the models were fitted on.
        holdout_seasons: Seasons the windows were measured on.
    """

    rows: list[WindowRow] = field(default_factory=list[WindowRow])
    train_seasons: tuple[int, ...] = ()
    holdout_seasons: tuple[int, ...] = ()

    @property
    def stops(self) -> int:
        """Number of stops checked.

        Returns:
            The count.
        """
        return len(self.rows)

    @property
    def hits(self) -> int:
        """Number of stops inside their window.

        Returns:
            The count.
        """
        return sum(row.hit for row in self.rows)

    @property
    def share(self) -> float:
        """Share of stops inside their window.

        Returns:
            The share, or 0 when nothing was checked.
        """
        return self.hits / self.stops if self.stops else 0.0

    def by_circuit(self) -> dict[str, tuple[int, int]]:
        """Hit and total counts per circuit.

        Returns:
            Circuit ID to ``(hits, stops)``.
        """
        counts: dict[str, tuple[int, int]] = {}
        for row in self.rows:
            hits, stops = counts.get(row.circuit, (0, 0))
            counts[row.circuit] = (hits + int(row.hit), stops + 1)
        return counts


def window_hits(
    predicted: Mapping[str, Sequence[PitWindow]], actual: Mapping[str, Sequence[int]]
) -> tuple[int, int]:
    """Count how many actual stops fall inside their predicted window.

    Args:
        predicted: Driver ID to that driver's windows, in stop order.
        actual: Driver ID to the laps the driver stopped on, in stop order.

    Returns:
        ``(hits, stops)``.
    """
    hits = stops = 0
    for driver, windows in predicted.items():
        laps = actual.get(driver, ())
        for window in windows:
            index = window.stop_index
            if index >= len(laps):
                continue
            stops += 1
            hits += int(window.contains(laps[index]))
    return hits, stops


def calibrate_pit_windows(  # pylint: disable=too-many-locals
    reader: DataReader,
    snapshot: SnapshotId,
    *,
    train_seasons: Sequence[int],
    holdout_seasons: Sequence[int],
    rules_dir: Path | None = None,
) -> PitWindowCalibration:
    """Fit on training seasons and measure pit windows on held-out races (P4-10).

    Args:
        reader: Read access to the store.
        snapshot: Snapshot to read.
        train_seasons: Seasons the tyre model and tendencies are fitted on.
        holdout_seasons: Seasons whose races are measured.
        rules_dir: Folder holding the season-rules files.

    Returns:
        The calibration result.
    """
    laps = reader.read("laps", snapshot=snapshot, as_of=FAR_FUTURE)
    events = reader.read("events", snapshot=snapshot, as_of=FAR_FUTURE)
    pit_stops = reader.read("pit_stops", snapshot=snapshot, as_of=FAR_FUTURE)
    train = laps.loc[laps["season"].isin(list(train_seasons))]
    train_stops = pit_stops.loc[pit_stops["season"].isin(list(train_seasons))]
    tyre = fit_tyre_model(train, events, train_stops)
    tendencies = fit_team_tendencies(train_stops, train)
    circuit_of = {
        (int(r["season"]), int(r["round"])): str(r["circuit_id"]) for r in _records(events)
    }
    spread = _fit_spread(
        train, train_stops, circuit_of, tyre=tyre, tendencies=tendencies, rules_dir=rules_dir
    )
    holdout = laps.loc[laps["season"].isin(list(holdout_seasons))]
    rows = _measure_races(
        holdout,
        pit_stops,
        circuit_of,
        tyre=tyre,
        tendencies=tendencies,
        rules_dir=rules_dir,
        spread=spread,
    )
    return PitWindowCalibration(
        rows=rows,
        train_seasons=tuple(train_seasons),
        holdout_seasons=tuple(holdout_seasons),
    )


def render_pit_windows(result: PitWindowCalibration) -> str:
    """Render the calibration as a Markdown report.

    Args:
        result: Output of ``calibrate_pit_windows``.

    Returns:
        The report.
    """
    train = ", ".join(str(s) for s in result.train_seasons) or "none"
    holdout = ", ".join(str(s) for s in result.holdout_seasons) or "none"
    lines = [
        "# Pit-window calibration (P4-10)",
        "",
        f"Tyre model and team tendencies fitted on {train}; windows measured on {holdout}.",
        f"Actual stops inside their calibrated window (10th-90th percentile): "
        f"{result.share:.0%} of {result.stops} stops.",
        "",
        "| Circuit | Stops | Inside | Share |",
        "|---|---:|---:|---:|",
    ]
    for circuit, (hits, stops) in sorted(result.by_circuit().items()):
        share = hits / stops if stops else 0.0
        lines.append(f"| {circuit} | {stops} | {hits} | {share:.0%} |")
    return "\n".join(lines) + "\n"


def _measure_races(  # pylint: disable=too-many-locals
    laps: pd.DataFrame,
    pit_stops: pd.DataFrame,
    circuit_of: Mapping[tuple[int, int], str],
    *,
    tyre: TyreModel,
    tendencies: TeamTendencies,
    rules_dir: Path | None,
    spread: StopLapSpread,
) -> list[WindowRow]:
    race = laps.loc[laps["session"] == "R"]
    actual = _actual_stops(pit_stops)
    rows: list[WindowRow] = []
    for (season, round_), group in race.groupby(["season", "round"], sort=True):
        circuit = circuit_of.get((_as_int(season), _as_int(round_)))
        if circuit is None:
            continue
        try:
            rules = load_season_rules(_as_int(season), rules_dir)
        except ShpeedError:
            continue
        total_laps = int(group["lap_number"].max())
        base_lap_s = _base_lap(group)
        for driver, driver_laps in group.groupby("driver_id", sort=True):
            key = (_as_int(season), _as_int(round_), str(driver))
            rows.extend(
                _driver_windows(
                    str(driver),
                    circuit,
                    key,
                    driver_laps=driver_laps,
                    total_laps=total_laps,
                    base_lap_s=base_lap_s,
                    rules=rules,
                    tyre=tyre,
                    tendencies=tendencies,
                    actual=actual,
                    spread=spread,
                )
            )
    return rows


def _fit_spread(  # pylint: disable=too-many-locals
    laps: pd.DataFrame,
    pit_stops: pd.DataFrame,
    circuit_of: Mapping[tuple[int, int], str],
    *,
    tyre: TyreModel,
    tendencies: TeamTendencies,
    rules_dir: Path | None,
) -> StopLapSpread:
    race = laps.loc[laps["session"] == "R"]
    actual = _actual_stops(pit_stops)
    fractions: dict[int, list[float]] = {}
    for (season, round_), group in race.groupby(["season", "round"], sort=True):
        circuit = circuit_of.get((_as_int(season), _as_int(round_)))
        if circuit is None:
            continue
        try:
            rules = load_season_rules(_as_int(season), rules_dir)
        except ShpeedError:
            continue
        total_laps = int(group["lap_number"].max())
        base_lap_s = _base_lap(group)
        for driver, driver_laps in group.groupby("driver_id", sort=True):
            key = (_as_int(season), _as_int(round_), str(driver))
            stops = actual.get(key, [])
            if not stops:
                continue
            chooser = _chooser(
                driver_laps,
                total_laps=total_laps,
                base_lap_s=base_lap_s,
                rules=rules,
                tyre=tyre,
                tendencies=tendencies,
                circuit=circuit,
            )
            if chooser is None:
                continue
            team = _team_of(driver_laps)
            centres = chooser.stop_lap_mean(team, _start_compound(driver_laps))
            for index, lap in enumerate(stops):
                if index in centres:
                    fractions.setdefault(index, []).append((lap - centres[index]) / total_laps)
    offsets = {
        index: _quantiles(values)
        for index, values in fractions.items()
        if len(values) >= MIN_SPREAD_SAMPLES
    }
    pooled = [fraction for values in fractions.values() for fraction in values]
    default = _quantiles(pooled) if pooled else DEFAULT_STOP_SPREAD
    return StopLapSpread(offsets, default)


def _quantiles(values: Sequence[float]) -> tuple[float, float]:
    return (float(np.quantile(values, LOW_QUANTILE)), float(np.quantile(values, HIGH_QUANTILE)))


def _chooser(  # pylint: disable=too-many-arguments
    driver_laps: pd.DataFrame,
    *,
    total_laps: int,
    base_lap_s: float,
    rules: SeasonRules,
    tyre: TyreModel,
    tendencies: TeamTendencies,
    circuit: str,
) -> StrategyChooser | None:
    try:
        generator = StrategyGenerator(
            tyre,
            rules,
            total_laps=total_laps,
            base_lap_s=base_lap_s,
            circuit=circuit,
            team=_team_of(driver_laps),
            start_compound=_start_compound(driver_laps),
        )
        return StrategyChooser(generator.plans(), tendencies)
    except ShpeedError:
        return None


def _team_of(driver_laps: pd.DataFrame) -> str:
    first = driver_laps.sort_values("lap_number")
    return str(first["team_id"].iloc[0])


def _start_compound(driver_laps: pd.DataFrame) -> str:
    first = driver_laps.sort_values("lap_number")
    return str(first["compound"].iloc[0]).upper()


def _driver_windows(  # pylint: disable=too-many-arguments,too-many-locals
    driver: str,
    circuit: str,
    key: tuple[int, int, str],
    *,
    driver_laps: pd.DataFrame,
    total_laps: int,
    base_lap_s: float,
    rules: SeasonRules,
    tyre: TyreModel,
    tendencies: TeamTendencies,
    actual: Mapping[tuple[int, int, str], list[int]],
    spread: StopLapSpread,
) -> list[WindowRow]:
    stops = actual.get(key, [])
    if not stops:
        return []
    chooser = _chooser(
        driver_laps,
        total_laps=total_laps,
        base_lap_s=base_lap_s,
        rules=rules,
        tyre=tyre,
        tendencies=tendencies,
        circuit=circuit,
    )
    if chooser is None:
        return []
    windows = chooser.windows(
        _team_of(driver_laps),
        _start_compound(driver_laps),
        spread=spread,
        total_laps=total_laps,
    )
    rows: list[WindowRow] = []
    for window in windows:
        index = window.stop_index
        if index >= len(stops):
            continue
        rows.append(
            WindowRow(
                circuit=circuit,
                driver=driver,
                stop_index=index,
                low_lap=window.low_lap,
                high_lap=window.high_lap,
                actual_lap=stops[index],
                hit=window.contains(stops[index]),
            )
        )
    return rows


def _actual_stops(pit_stops: pd.DataFrame) -> dict[tuple[int, int, str], list[int]]:
    stops: dict[tuple[int, int, str], list[int]] = {}
    for row in _records(race_stops(pit_stops)):
        key = (int(row["season"]), int(row["round"]), str(row["driver_id"]))
        stops.setdefault(key, []).append(int(row["lap"]))
    for value in stops.values():
        value.sort()
    return stops


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], frame.to_dict("records"))


def _as_int(value: object) -> int:
    return int(cast(Any, value))


def _base_lap(group: pd.DataFrame) -> float:
    clean = group["clean_lap"].fillna(False).to_numpy(dtype=bool)
    times = group["lap_time_s"].to_numpy(dtype=np.float64)
    selected = times[clean & np.isfinite(times)]
    if len(selected) == 0:
        selected = times[np.isfinite(times)]
    return float(np.median(selected))
