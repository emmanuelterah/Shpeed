"""Pit-lane loss and stop duration (P4-04, P4-05).

Pit loss (P4-04) is the extra time of a stop's in- and out-laps over the driver's clean-lap
baseline, measured from timing data and pooled per circuit. It already includes the time spent
stationary, because the in- and out-laps span the whole stop.

The stop duration (P4-05) is the station's figure, ``duration_s`` in the canonical store:
the recorded time in the pit lane from entry line to exit line. It is modelled per team as a
distribution with a slow-stop tail. Because the pit loss already contains the stop, the engine
adds only the team's *deviation* from the field mean (``StopTimeModel.offset``) to the circuit
pit loss, so a team's slow wheel-guns are felt without double counting the whole stop.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from f1sim.adapters.models.pace_corrections import MAD_TO_SD, Estimate, pool

PIT_LOSS_PRIOR = (20.0, 5.0)
PIT_LOSS_MIN_S = 0.0
PIT_LOSS_MAX_S = 60.0
"""Weak prior and plausibility bounds for the pit loss, in seconds."""

MIN_STOP_S = 8.0
MAX_STOP_S = 60.0
SLOW_MADS = 2.0
MIN_STOP_SAMPLES = 2


def race_stops(pit_stops: pd.DataFrame) -> pd.DataFrame:
    """Keep only race pit stops.

    The canonical store may hold stops from practice and qualifying, but strategy is about the
    race, so only the ``R`` session is used.

    Args:
        pit_stops: Canonical ``pit_stops`` rows.

    Returns:
        The race sessions' rows.
    """
    return pit_stops.loc[pit_stops["session"].astype(str) == "R"]


@dataclass(frozen=True)
class StopTimeDistribution:
    """A team's pit-stop duration: a normal core with a slow-stop tail (P4-05).

    Attributes:
        mu: Mean of a normal stop, seconds.
        sigma: Standard deviation of a normal stop, seconds.
        slow_fraction: Probability that a stop is slow.
        slow_scale: Mean excess of a slow stop, seconds.
    """

    mu: float
    sigma: float
    slow_fraction: float
    slow_scale: float

    def mean(self) -> float:
        """Mean stop duration.

        Returns:
            Seconds, adding the slow-stop tail's expected contribution.
        """
        return (1.0 - self.slow_fraction) * self.mu + self.slow_fraction * (
            self.mu + self.slow_scale
        )

    def sample(self, rng: np.random.Generator, size: int) -> NDArray[np.float64]:
        """Draw stop durations.

        Args:
            rng: Random generator.
            size: Number of draws.

        Returns:
            Stop durations in seconds, at least ``MIN_STOP_S``.
        """
        base = rng.normal(self.mu, self.sigma, size)
        slow = rng.random(size) < self.slow_fraction
        extra = rng.exponential(max(self.slow_scale, 1e-3), size)
        values = np.maximum(base + np.where(slow, extra, 0.0), MIN_STOP_S)
        return values.astype(np.float64)

    def to_dict(self) -> dict[str, float]:
        """Serialise for the state store.

        Returns:
            The four distribution parameters.
        """
        return {
            "mu": self.mu,
            "sigma": self.sigma,
            "slow_fraction": self.slow_fraction,
            "slow_scale": self.slow_scale,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StopTimeDistribution":
        """Rebuild a distribution saved with ``to_dict``.

        Args:
            data: The saved mapping.

        Returns:
            The distribution.
        """
        return cls(
            mu=float(data["mu"]),
            sigma=float(data["sigma"]),
            slow_fraction=float(data["slow_fraction"]),
            slow_scale=float(data["slow_scale"]),
        )


@dataclass
class StopTimeModel:
    """Stop duration by team, with a league-wide default (P4-05).

    Attributes:
        teams: Team ID to its stop-duration distribution.
        default: Distribution for a team with too few stops.
    """

    teams: dict[str, StopTimeDistribution]
    default: StopTimeDistribution

    def distribution(self, team: str) -> StopTimeDistribution:
        """Find a team's stop-duration distribution.

        Args:
            team: Team ID.

        Returns:
            The team's distribution, or the default.
        """
        return self.teams.get(team, self.default)

    def sample(self, rng: np.random.Generator, team: str, size: int) -> NDArray[np.float64]:
        """Draw stop durations for a team.

        Args:
            rng: Random generator.
            team: Team ID.
            size: Number of draws.

        Returns:
            Stop durations in seconds.
        """
        return self.distribution(team).sample(rng, size)

    def offset(self, rng: np.random.Generator, team: str, size: int) -> NDArray[np.float64]:
        """Draw a team's stop-duration deviation from the field mean.

        The pit loss (P4-04) already includes the stop, so the engine adds this offset rather
        than a whole stop duration, giving a team's slow wheel-guns without double counting.

        Args:
            rng: Random generator.
            team: Team ID.
            size: Number of draws.

        Returns:
            Deviations in seconds, mean-centred on the league-wide default.
        """
        return self.sample(rng, team, size) - self.default.mean()

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the state store.

        Returns:
            Team to distribution, plus the default.
        """
        return {
            "teams": {team: dist.to_dict() for team, dist in self.teams.items()},
            "default": self.default.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StopTimeModel":
        """Rebuild a model saved with ``to_dict``.

        Args:
            data: The saved mapping.

        Returns:
            The stop-time model.
        """
        teams = cast(Mapping[str, Any], data["teams"])
        return cls(
            teams={str(t): StopTimeDistribution.from_dict(v) for t, v in teams.items()},
            default=StopTimeDistribution.from_dict(cast(Mapping[str, Any], data["default"])),
        )


def measure_pit_loss(laps: pd.DataFrame, events: pd.DataFrame) -> dict[str, Estimate]:
    """Measure pit loss per circuit (P4-04).

    The loss is the extra time of a stop's in- and out-laps over the driver's clean-lap
    baseline. It already covers the time stationary, because the two laps span the whole stop.

    Args:
        laps: Canonical ``laps`` rows.
        events: Canonical ``events`` rows, for ``circuit_id``.

    Returns:
        Circuit ID to the pooled pit loss.
    """
    circuit_of = {
        (int(r["season"]), int(r["round"])): str(r["circuit_id"]) for r in _records(events)
    }
    per_circuit: dict[str, list[Estimate]] = {}
    race = laps.loc[laps["session"] == "R"]
    for (season, round_), group in race.groupby(["season", "round"], sort=True):
        key = (_as_int(season), _as_int(round_))
        circuit = circuit_of.get(key)
        if circuit is None:
            continue
        estimate = _race_pit_loss(group)
        if estimate is not None:
            per_circuit.setdefault(circuit, []).append(estimate)
    return {c: pool(items, PIT_LOSS_PRIOR) for c, items in per_circuit.items()}


def fit_stop_times(pit_stops: pd.DataFrame, laps: pd.DataFrame) -> StopTimeModel:
    """Fit stop-duration distributions per team (P4-05).

    Only race stops are used.

    Args:
        pit_stops: Canonical ``pit_stops`` rows.
        laps: Canonical ``laps`` rows, to map drivers to teams.

    Returns:
        The stop-duration model.
    """
    team_of = {
        (int(r["season"]), int(r["round"]), str(r["driver_id"])): str(r["team_id"])
        for r in _records(laps)
        if r["team_id"] is not None
    }
    by_team: dict[str, list[float]] = {}
    everything: list[float] = []
    for row in _records(race_stops(pit_stops)):
        if row["duration_s"] is None or pd.isna(row["duration_s"]):
            continue
        duration = float(row["duration_s"])
        everything.append(duration)
        team = team_of.get((int(row["season"]), int(row["round"]), str(row["driver_id"])))
        if team is not None:
            by_team.setdefault(team, []).append(duration)
    default = _fit_distribution(everything) or _default_distribution()
    teams = {
        team: fitted
        for team, values in by_team.items()
        if (fitted := _fit_distribution(values)) is not None
    }
    return StopTimeModel(teams, default)


def _race_pit_loss(group: pd.DataFrame) -> Estimate | None:  # pylint: disable=too-many-locals
    losses: list[float] = []
    for _driver, sub in group.groupby("driver_id", sort=True):
        baseline = _baseline(sub)
        if baseline is None:
            continue
        lap = sub["lap_number"].to_numpy(dtype=np.int64)
        time_s = sub["lap_time_s"].to_numpy(dtype=np.float64)
        pit_in = sub["pit_in"].fillna(False).to_numpy(dtype=bool)
        pit_out = sub["pit_out"].fillna(False).to_numpy(dtype=bool)
        times = dict(zip(lap.tolist(), time_s.tolist(), strict=True))
        out_laps = set(lap[pit_out].tolist())
        for i in np.flatnonzero(pit_in).tolist():
            entry = int(lap[i])
            if entry + 1 not in out_laps or entry + 1 not in times:
                continue
            loss = times[entry] + times[entry + 1] - 2 * baseline
            if np.isfinite(loss):
                losses.append(float(loss))
    if not losses:
        return None
    values = np.asarray(losses, dtype=np.float64)
    values = values[(values > PIT_LOSS_MIN_S) & (values < PIT_LOSS_MAX_S)]
    if len(values) == 0:
        return None
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    sem = max(MAD_TO_SD * mad / np.sqrt(len(values)), 1e-3)
    return Estimate(median, sem)


def _baseline(sub: pd.DataFrame) -> float | None:
    clean = sub["clean_lap"].fillna(False).to_numpy(dtype=bool)
    times = sub["lap_time_s"].to_numpy(dtype=np.float64)
    selected = times[clean & np.isfinite(times)]
    if len(selected) == 0:
        return None
    return float(np.median(selected))


def _fit_distribution(values: Sequence[float]) -> StopTimeDistribution | None:
    array = np.asarray([v for v in values if MIN_STOP_S <= v <= MAX_STOP_S], dtype=np.float64)
    if len(array) < MIN_STOP_SAMPLES:
        return None
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    sigma = max(MAD_TO_SD * mad, 0.05)
    threshold = median + SLOW_MADS * sigma
    fast = array[array <= threshold]
    slow = array[array > threshold]
    mu = float(np.mean(fast))
    fast_sigma = max(float(np.std(fast)), 0.02)
    fraction = float(len(slow) / len(array))
    scale = max(float(np.mean(slow - mu)), 1e-3) if len(slow) else 1e-3
    return StopTimeDistribution(mu, fast_sigma, fraction, scale)


def _default_distribution() -> StopTimeDistribution:
    return StopTimeDistribution(mu=24.0, sigma=1.5, slow_fraction=0.05, slow_scale=8.0)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], frame.to_dict("records"))


def _as_int(value: object) -> int:
    return int(cast(Any, value))
