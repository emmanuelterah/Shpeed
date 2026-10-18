"""Event hazards for the race simulator: safety cars, retirements, lap 1 and penalties (P6).

Real races are decided as much by events as by pace. This module estimates the chance of each
event from the stored history, always shrinking a thin sample toward the league average so a
single bad weekend cannot dominate (the same idea as the pace model's pooling):

- **Safety car and VSC (P6-01).** The race-lap ``track_status`` records when the field was
  neutralised. Counting the transitions into status ``4`` (safety car) or ``6``/``7`` (VSC) gives
  deployments per race lap, pooled per circuit; the red flags are the races whose status reaches
  ``5`` (P6-02).
- **Reliability (P6-03) and crashes (P6-04).** A race result with no finishing position is a
  retirement. Its ``status`` text says whether it was mechanical or a collision, so mechanical
  hazards are pooled per team and crash hazards per driver, both shrunk toward the field.
- **Lap 1 (P6-05).** Each driver's launch is the places gained from the grid to the end of lap 1,
  and the first-corner incident rate is the share of starters involved in a lap-1 collision.
- **Penalties (P6-06).** The timestamped race-control feed is not part of the model's read set,
  so the in-race penalty hazard is a fixed prior rather than a fit.

Every model has a sensible default, so an early-season forecast runs before there is enough
history to fit anything, and the ablation runner (P6-09) can still switch each piece off.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd

RACE = "R"
"""Race session code; only race data feeds the hazards."""

SAFETY_CAR_CODE = "4"
VSC_CODES = frozenset({"6", "7"})
RED_FLAG_CODE = "5"
NEUTRAL_CODES = frozenset({SAFETY_CAR_CODE, *VSC_CODES})
"""FastF1 track-status codes: green (1), yellow (2), safety car (4), VSC (6, 7) and red (5).

FastF1 stores a lap's status as the codes that applied during it concatenated into one string
(for example ``"1264"`` for a lap that went green, yellow, then safety car), so a lap is
neutralised when *any* of its digits is a neutralisation code, not when the string equals one.
"""

CRASH_KEYWORDS = ("accident", "collision", "crash", "spun", "damage")
"""Result-status substrings that mark a retirement as an accident rather than a failure."""

SC_PRIOR_PER_LAP = 0.010
"""League prior for the per-lap chance a safety car or VSC is deployed."""
SC_PRIOR_LAPS = 300.0
"""Pseudo-laps behind the prior, so the observed rate dominates after a full season."""
SC_DURATION_PRIOR_LAPS = 4.0
"""Prior length of a neutralised period, in laps."""
RED_FLAG_PRIOR_PER_RACE = 0.05
RETIREMENT_PRIOR_PER_RACE = 0.08
CRASH_PRIOR_PER_RACE = 0.02
RETIREMENT_PRIOR_STARTS = 40.0
"""Pseudo-starts behind the reliability and crash priors."""
LAP1_INCIDENT_PRIOR = 0.03
"""Prior chance a car is caught in a first-corner incident."""
LAP1_LAUNCH_PRIOR_STRENGTH = 12.0
"""Pseudo-races behind a driver's launch estimate (the prior mean is zero places)."""
PENALTY_PRIOR_PER_RACE = 0.15
"""Prior chance a car picks up an in-race time penalty (P6-06); not fitted."""
PENALTY_FIVE_SECOND_SHARE = 0.8
"""Share of penalties that are five rather than ten seconds."""


@dataclass(frozen=True)
class SafetyCarHazard:
    """Chance of a safety car, VSC or red flag, by circuit (P6-01, P6-02).

    Attributes:
        deploy_per_lap: League-wide chance a neutralisation is deployed on a race lap.
        circuits: Circuit ID to its shrunk per-lap deployment chance.
        duration_laps: Mean number of laps a neutralisation lasts.
        red_flag_per_race: Chance a race sees a red flag.
    """

    deploy_per_lap: float
    circuits: Mapping[str, float]
    duration_laps: float
    red_flag_per_race: float

    def deploy_rate(self, circuit: str | None) -> float:
        """Per-lap neutralisation chance at a circuit.

        Args:
            circuit: Circuit ID; None or an unknown circuit uses the league rate.

        Returns:
            The chance, in ``[0, 1]``.
        """
        if circuit is None:
            return self.deploy_per_lap
        return float(self.circuits.get(circuit, self.deploy_per_lap))


@dataclass(frozen=True)
class RetirementHazard:
    """Chance that a car retires, per team and per driver (P6-03, P6-04).

    Attributes:
        mechanical: Team ID to its shrunk per-race mechanical-failure chance.
        crash: Driver ID to their shrunk per-race accident chance.
        field_mechanical: League-wide mechanical chance, for an unknown team.
        field_crash: League-wide crash chance, for an unknown driver.
    """

    mechanical: Mapping[str, float] = field(default_factory=dict[str, float])
    crash: Mapping[str, float] = field(default_factory=dict[str, float])
    field_mechanical: float = RETIREMENT_PRIOR_PER_RACE
    field_crash: float = CRASH_PRIOR_PER_RACE

    def p_mechanical(self, team: str) -> float:
        """Chance a car of a team fails mechanically in a race.

        Args:
            team: Team ID.

        Returns:
            The chance, in ``[0, 1]``.
        """
        return float(self.mechanical.get(team, self.field_mechanical))

    def p_crash(self, driver: str) -> float:
        """Chance a driver is eliminated in an accident in a race.

        Args:
            driver: Driver ID.

        Returns:
            The chance, in ``[0, 1]``.
        """
        return float(self.crash.get(driver, self.field_crash))

    def p_retire(self, team: str, driver: str) -> float:
        """Chance a car retires for either reason, capped below certainty.

        Args:
            team: Team ID.
            driver: Driver ID.

        Returns:
            The combined chance, clipped to ``[0, 0.9]``.
        """
        return float(np.clip(self.p_mechanical(team) + self.p_crash(driver), 0.0, 0.9))


@dataclass(frozen=True)
class Lap1Hazard:
    """Launches and first-corner incidents (P6-05).

    Attributes:
        launch_places: Driver ID to their shrunk mean places gained on lap 1 (positive is
            gained; negative is lost).
        incident: Chance a car is eliminated in a first-corner incident.
    """

    launch_places: Mapping[str, float] = field(default_factory=dict[str, float])
    incident: float = LAP1_INCIDENT_PRIOR

    def launch(self, driver: str) -> float:
        """A driver's expected change in places over the start.

        Args:
            driver: Driver ID.

        Returns:
            Places gained on lap 1 (negative when places are lost).
        """
        return float(self.launch_places.get(driver, 0.0))


@dataclass(frozen=True)
class PenaltyHazard:
    """In-race time penalties (P6-06).

    Attributes:
        per_race: Chance a car picks up a time penalty in a race.
        five_second_share: Share of penalties that are five rather than ten seconds.
    """

    per_race: float = PENALTY_PRIOR_PER_RACE
    five_second_share: float = PENALTY_FIVE_SECOND_SHARE


@dataclass(frozen=True)
class HazardModel:
    """Every event hazard the simulator needs, fitted together (P6).

    Attributes:
        safety_car: Safety-car, VSC and red-flag hazards.
        retirement: Mechanical and accident retirement hazards.
        lap1: Launch and first-corner incident hazards.
        penalty: In-race penalty hazard.
    """

    safety_car: SafetyCarHazard
    retirement: RetirementHazard
    lap1: Lap1Hazard
    penalty: PenaltyHazard


def default_hazards() -> HazardModel:
    """The prior-only hazard model used when there is too little history.

    Returns:
        Every hazard at its league prior, with no per-team, per-driver or per-circuit terms.
    """
    return HazardModel(
        safety_car=SafetyCarHazard(
            deploy_per_lap=SC_PRIOR_PER_LAP,
            circuits={},
            duration_laps=SC_DURATION_PRIOR_LAPS,
            red_flag_per_race=RED_FLAG_PRIOR_PER_RACE,
        ),
        retirement=RetirementHazard(),
        lap1=Lap1Hazard(),
        penalty=PenaltyHazard(),
    )


def fit_hazards(laps: pd.DataFrame, events: pd.DataFrame, results: pd.DataFrame) -> HazardModel:
    """Fit every event hazard from the stored history.

    Args:
        laps: Canonical ``laps`` rows, for track status and lap-1 positions.
        events: Canonical ``events`` rows, for ``circuit_id``.
        results: Canonical ``results`` rows, for retirements and grids.

    Returns:
        The fitted hazards; each part falls back to its prior when the data is too thin.
    """
    circuits = {
        (int(r["season"]), int(r["round"])): str(r["circuit_id"])
        for r in _records(events)
        if not pd.isna(r.get("circuit_id"))
    }
    return HazardModel(
        safety_car=fit_safety_car(laps, circuits),
        retirement=fit_retirements(results),
        lap1=fit_lap1(laps, results),
        penalty=PenaltyHazard(),
    )


def fit_safety_car(  # pylint: disable=too-many-locals
    laps: pd.DataFrame, circuits: Mapping[tuple[int, int], str] | None = None
) -> SafetyCarHazard:
    """Fit safety-car, VSC and red-flag hazards from race track status (P6-01, P6-02).

    Args:
        laps: Canonical ``laps`` rows.
        circuits: ``(season, round)`` to ``circuit_id``, for the per-circuit pool.

    Returns:
        The fitted hazard; the league prior when there is no race status data.
    """
    mapping = dict(circuits or {})
    race = laps.loc[laps["session"].astype(str) == RACE]
    if race.empty or "track_status" not in race.columns:
        return default_hazards().safety_car
    per_circuit: dict[str, list[int]] = {}
    deployments = 0
    laps_seen = 0
    neutral_laps = 0
    races = 0
    red_races = 0
    for (season, round_), group in race.groupby(["season", "round"], sort=True):
        statuses = _race_statuses(group)
        if not statuses:
            continue
        race_laps = len(statuses)
        races += 1
        laps_seen += race_laps
        episodes, red = _count_episodes(statuses)
        deployments += episodes
        neutral_laps += sum(1 for status in statuses if _codes(status) & NEUTRAL_CODES)
        red_races += int(red)
        circuit = mapping.get((_as_int(season), _as_int(round_)))
        if circuit is not None:
            tally = per_circuit.setdefault(circuit, [0, 0])
            tally[0] += episodes
            tally[1] += race_laps
    if races == 0:
        return default_hazards().safety_car
    league = _shrunk(deployments, laps_seen, SC_PRIOR_PER_LAP, SC_PRIOR_LAPS)
    circuit_rates = {
        circuit: _shrunk(episodes, total, league, SC_PRIOR_LAPS)
        for circuit, (episodes, total) in per_circuit.items()
    }
    duration = neutral_laps / deployments if deployments else SC_DURATION_PRIOR_LAPS
    red = _shrunk(red_races, races, RED_FLAG_PRIOR_PER_RACE, 30.0)
    return SafetyCarHazard(league, circuit_rates, float(duration), red)


def fit_retirements(  # pylint: disable=too-many-locals
    results: pd.DataFrame,
) -> RetirementHazard:
    """Fit mechanical (per team) and accident (per driver) hazards (P6-03, P6-04).

    Args:
        results: Canonical ``results`` rows.

    Returns:
        The fitted hazard; the field priors when there are no races.
    """
    race = results.loc[results["session"].astype(str) == RACE]
    if race.empty:
        return RetirementHazard()
    team_starts: dict[str, int] = {}
    team_failures: dict[str, int] = {}
    driver_starts: dict[str, int] = {}
    driver_crashes: dict[str, int] = {}
    total_starts = 0
    total_failures = 0
    total_crashes = 0
    for row in _records(race):
        team = str(row["team_id"])
        driver = str(row["driver_id"])
        team_starts[team] = team_starts.get(team, 0) + 1
        driver_starts[driver] = driver_starts.get(driver, 0) + 1
        total_starts += 1
        if pd.notna(row.get("position")):
            continue
        if _is_crash(row.get("status")):
            driver_crashes[driver] = driver_crashes.get(driver, 0) + 1
            total_crashes += 1
        else:
            team_failures[team] = team_failures.get(team, 0) + 1
            total_failures += 1
    field_mechanical = _shrunk(
        total_failures, total_starts, RETIREMENT_PRIOR_PER_RACE, RETIREMENT_PRIOR_STARTS
    )
    field_crash = _shrunk(
        total_crashes, total_starts, CRASH_PRIOR_PER_RACE, RETIREMENT_PRIOR_STARTS
    )
    mechanical = {
        team: _shrunk(team_failures.get(team, 0), starts, field_mechanical, RETIREMENT_PRIOR_STARTS)
        for team, starts in team_starts.items()
    }
    crash = {
        driver: _shrunk(driver_crashes.get(driver, 0), starts, field_crash, RETIREMENT_PRIOR_STARTS)
        for driver, starts in driver_starts.items()
    }
    return RetirementHazard(mechanical, crash, field_mechanical, field_crash)


def fit_lap1(  # pylint: disable=too-many-locals
    laps: pd.DataFrame, results: pd.DataFrame
) -> Lap1Hazard:
    """Fit launches and the first-corner incident rate (P6-05).

    Args:
        laps: Canonical ``laps`` rows, for the end-of-lap-1 position.
        results: Canonical ``results`` rows, for the grid and lap-1 collisions.

    Returns:
        The fitted hazard; the priors when there is too little data.
    """
    grid = results.loc[results["session"].astype(str) == RACE, ["season", "round", "driver_id"]]
    grid = grid.assign(grid=_int_column(results.loc[grid.index, "grid"]))
    first = laps.loc[
        (laps["session"].astype(str) == RACE) & (_int_column(laps["lap_number"]) == 1),
        ["season", "round", "driver_id"],
    ]
    first = first.assign(lap1=_int_column(laps.loc[first.index, "position"]))
    merged = grid.merge(first, on=["season", "round", "driver_id"], how="inner")
    merged = merged.loc[merged["grid"].notna() & merged["lap1"].notna()]
    gained = merged["grid"] - merged["lap1"]
    by_driver: dict[str, list[float]] = {}
    for driver, value in zip(merged["driver_id"].tolist(), gained.tolist(), strict=True):
        by_driver.setdefault(str(driver), []).append(float(value))
    launch = {
        driver: _shrunk_mean(float(np.mean(values)), len(values), LAP1_LAUNCH_PRIOR_STRENGTH)
        for driver, values in by_driver.items()
    }
    race = results.loc[results["session"].astype(str) == RACE]
    starts = len(race)
    incidents = sum(
        1
        for row in _records(race)
        if pd.isna(row.get("position"))
        and _is_crash(row.get("status"))
        and _laps_completed(row.get("laps")) <= 1
    )
    incident = (
        _shrunk(incidents, starts, LAP1_INCIDENT_PRIOR, RETIREMENT_PRIOR_STARTS)
        if starts
        else LAP1_INCIDENT_PRIOR
    )
    return Lap1Hazard(launch, incident)


def _codes(status: str) -> frozenset[str]:
    """The set of FastF1 status-code digits in a concatenated per-lap status string."""
    return frozenset(status)


def _race_statuses(group: pd.DataFrame) -> list[str]:
    """One concatenated status string per lap, in lap order, pooling every car on the lap."""
    ordered = group.sort_values("lap_number")
    per_lap = ordered.groupby("lap_number", sort=True)["track_status"].agg(
        lambda series: "".join(sorted({code for value in series.dropna() for code in str(value)}))
    )
    return [str(status) for status in per_lap.tolist()]


def _count_episodes(statuses: Sequence[str]) -> tuple[int, bool]:
    episodes = 0
    neutral = False
    red = False
    for status in statuses:
        codes = _codes(status)
        now = bool(codes & NEUTRAL_CODES)
        if now and not neutral:
            episodes += 1
        neutral = now
        red = red or RED_FLAG_CODE in codes
    return episodes, red


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], frame.to_dict("records"))


def _is_crash(status: object) -> bool:
    if status is None or pd.isna(cast(Any, status)):
        return False
    text = str(status).lower()
    return any(keyword in text for keyword in CRASH_KEYWORDS)


def _laps_completed(value: object) -> int:
    return 0 if value is None or pd.isna(cast(Any, value)) else int(cast(Any, value))


def _int_column(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Float64")


def _shrunk(successes: int, trials: int, prior: float, strength: float) -> float:
    return float((successes + prior * strength) / (trials + strength))


def _shrunk_mean(mean: float, count: int, strength: float) -> float:
    return float((mean * count) / (count + strength))


def _as_int(value: object) -> int:
    return int(cast(Any, value))
