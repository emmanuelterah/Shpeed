"""Pass-probability model fitted on historical position changes (P5-04).

A pass is modelled as a Bernoulli event whose chance rises with the follower's pace advantage over
the car ahead, via a logistic link. The fit takes the running order from each race lap's end-of-lap
position, and the gap from the times the two cars crossed the line (lap start plus lap time). Two
adjacent cars within ``CLOSE_GAP_S`` of each other are a chance to pass, and a swap by the end of
the next lap is a pass. Only green-flag laps on which neither car entered or left the pits count,
since a stop or a safety car reorders the field without anyone passing.

The covariate is the follower's pace advantage over the race, the difference between the two
drivers' median clean lap times: a single lap's difference is mostly noise, and conditioning on a
close pair makes it regress to the mean, which biases the slope toward zero or below. Conditioning
on close pairs makes the probability a per-lap chance *given the car is in range*, which is how the
engine's overtake step uses it. Because position changes alone cannot tell a DRS pass from an
Overtake Mode pass, the era's aid enters as a fixed log-odds bonus supplied by the caller, and each
circuit gets an empirical difficulty offset on top of the league-wide fit.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from f1sim.adapters.models.tyres import race_laps

RIDGE = 1e-3
"""L2 penalty on the logistic coefficients, so a degenerate design still solves."""
MAX_ITERATIONS = 50
CONVERGENCE = 1e-9
MIN_OBSERVATIONS = 50
MIN_CIRCUIT_OBSERVATIONS = 20
DIFFICULTY_BOUND = 2.0
"""A circuit's log-odds offset is clipped to this, so a thin sample cannot dominate."""

GREEN = "1"
"""Track status of a green-flag lap; only green laps without a stop count as a chance to pass."""
CLOSE_GAP_S = 1.0
"""Only pairs this close at the end of a lap count as a pass opportunity, seconds."""

DEFAULT_INTERCEPT = -1.6
DEFAULT_PACE_COEF = 0.8
"""Baseline log-odds and the log-odds gained per second per lap of pace advantage."""

DEFAULT_AID_BONUS: Mapping[str, float] = {"drs": 0.5, "overtake_mode": 0.8}
"""Fixed log-odds bonus per overtake aid; not identifiable from position changes alone."""


@dataclass(frozen=True)
class OvertakingModel:
    """Chance that the car behind completes a pass (P5-04).

    Attributes:
        intercept: Log-odds of a pass at zero pace advantage.
        pace_coef: Log-odds gained per second per lap of the follower's advantage.
        difficulty: Circuit ID to its log-odds offset from the league-wide fit.
        aid_bonus: Overtake aid to its fixed log-odds bonus.
    """

    intercept: float
    pace_coef: float
    difficulty: Mapping[str, float]
    aid_bonus: Mapping[str, float]

    def pass_probability(
        self,
        pace_gap_s: NDArray[np.float64] | float,
        *,
        circuit: str | None = None,
        aid: str | None = None,
    ) -> NDArray[np.float64]:
        """Probability that the car behind passes, given its per-lap pace advantage.

        Args:
            pace_gap_s: The follower's pace advantage in seconds per lap, positive when faster.
            circuit: Circuit ID for its difficulty offset; None ignores it.
            aid: ``"drs"`` or ``"overtake_mode"``, or None for neither.

        Returns:
            The pass probability, same shape as ``pace_gap_s``.
        """
        logit = self.intercept + self.pace_coef * np.asarray(pace_gap_s, dtype=np.float64)
        if circuit is not None:
            logit = logit + self.difficulty.get(circuit, 0.0)
        if aid is not None:
            logit = logit + self.aid_bonus.get(aid, 0.0)
        return _sigmoid(logit)


def fit_overtaking(laps: pd.DataFrame, events: pd.DataFrame) -> OvertakingModel:
    """Fit the pass model from adjacent position changes in race laps (P5-04).

    Args:
        laps: Canonical ``laps`` rows.
        events: Canonical ``events`` rows, for ``circuit_id``.

    Returns:
        The fitted model; a sensible default when there is too little history.
    """
    circuit_of = {
        (int(r["season"]), int(r["round"])): str(r["circuit_id"]) for r in _records(events)
    }
    advantage, passed, circuits = _observations(laps, circuit_of)
    if len(advantage) < MIN_OBSERVATIONS or np.all(passed) or not np.any(passed):
        return default_overtaking()
    intercept, coef = _logistic_fit(advantage, passed)
    model = OvertakingModel(intercept, coef, {}, DEFAULT_AID_BONUS)
    return replace(model, difficulty=_circuit_difficulty(advantage, passed, circuits, model))


def default_overtaking() -> OvertakingModel:
    """A weakly informed model for when nothing can be fitted.

    Returns:
        The default intercept and pace coefficient, with no circuit offsets.
    """
    return OvertakingModel(DEFAULT_INTERCEPT, DEFAULT_PACE_COEF, {}, dict(DEFAULT_AID_BONUS))


def _observations(
    laps: pd.DataFrame, circuit_of: Mapping[tuple[int, int], str]
) -> tuple[NDArray[np.float64], NDArray[np.bool_], list[str]]:
    frame = _timed_race_laps(laps)
    if frame.empty:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=bool), []
    keys = ["season", "round", "lap_number"]
    frame["ahead_position"] = frame["position"] - 1
    pairs = frame.merge(
        frame[[*keys, "position", "driver_id", "cross_s", "open"]],
        left_on=[*keys, "ahead_position"],
        right_on=[*keys, "position"],
        suffixes=("", "_ahead"),
    )
    pairs["gap"] = pairs["cross_s"] - pairs["cross_s_ahead"]
    pairs = pairs.loc[
        pairs["open"] & pairs["open_ahead"] & (pairs["gap"] >= 0.0) & (pairs["gap"] < CLOSE_GAP_S)
    ]
    following = frame[[*keys, "driver_id", "position", "open"]].assign(
        lap_number=frame["lap_number"] - 1
    )
    pairs = pairs.merge(following, on=[*keys, "driver_id"], suffixes=("", "_next")).merge(
        following.rename(columns={"driver_id": "driver_id_ahead"}),
        on=[*keys, "driver_id_ahead"],
        suffixes=("", "_ahead_next"),
    )
    pairs = pairs.loc[pairs["open_next"] & pairs["open_ahead_next"]]
    pace = _race_pace(laps)
    pairs = pairs.merge(pace, on=["season", "round", "driver_id"]).merge(
        pace.rename(columns={"driver_id": "driver_id_ahead", "pace_s": "pace_s_ahead"}),
        on=["season", "round", "driver_id_ahead"],
    )
    pairs["circuit"] = [
        circuit_of.get((int(season), int(round_)), "")
        for season, round_ in zip(pairs["season"].tolist(), pairs["round"].tolist(), strict=True)
    ]
    pairs = pairs.loc[pairs["circuit"] != ""]
    advantage = (pairs["pace_s_ahead"] - pairs["pace_s"]).to_numpy(dtype=np.float64)
    passed = (pairs["position_next"] < pairs["position_ahead_next"]).to_numpy(dtype=bool)
    return advantage, passed, [str(circuit) for circuit in pairs["circuit"].tolist()]


def _timed_race_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Race laps with the running position and the time each car crossed the line."""
    columns = [
        "season",
        "round",
        "driver_id",
        "lap_number",
        "position",
        "lap_time_s",
        "lap_start_utc",
        "track_status",
        "pit_in",
        "pit_out",
    ]
    race = laps.loc[laps["session"] == "R", columns].dropna(
        subset=["position", "lap_time_s", "lap_start_utc"]
    )
    if race.empty:
        return race
    start = pd.to_datetime(race["lap_start_utc"])
    frame = race.assign(
        position=race["position"].astype(np.int64),
        cross_s=(start - start.min()).dt.total_seconds() + race["lap_time_s"].astype(float),
        open=(race["track_status"].astype(str) == GREEN)
        & ~race["pit_in"].fillna(False).astype(bool)
        & ~race["pit_out"].fillna(False).astype(bool),
    )
    return frame.drop(columns=["lap_start_utc", "track_status", "pit_in", "pit_out"])


def _race_pace(laps: pd.DataFrame) -> pd.DataFrame:
    """Each driver's median clean race lap, per race."""
    clean = race_laps(laps)
    grouped = clean.groupby(["season", "round", "driver_id"])["lap_time_s"].median()
    return grouped.rename("pace_s").reset_index()


def _logistic_fit(advantage: NDArray[np.float64], passed: NDArray[np.bool_]) -> tuple[float, float]:
    design = np.column_stack([np.ones_like(advantage), advantage])
    target = passed.astype(np.float64)
    beta = np.zeros(2, dtype=np.float64)
    for _ in range(MAX_ITERATIONS):
        probability = _sigmoid(design @ beta)
        weight = np.maximum(probability * (1.0 - probability), 1e-6)
        working = (design @ beta) + (target - probability) / weight
        xtw = design.T * weight
        matrix = xtw @ design + RIDGE * np.eye(2)
        updated = np.linalg.solve(matrix, xtw @ working)
        if np.max(np.abs(updated - beta)) < CONVERGENCE:
            beta = updated
            break
        beta = updated
    return float(beta[0]), float(beta[1])


def _circuit_difficulty(
    advantage: NDArray[np.float64],
    passed: NDArray[np.bool_],
    circuits: Sequence[str],
    model: OvertakingModel,
) -> dict[str, float]:
    labels = np.asarray(list(circuits), dtype=object)
    difficulty: dict[str, float] = {}
    for circuit in sorted(set(circuits)):
        mask = labels == circuit
        if int(mask.sum()) < MIN_CIRCUIT_OBSERVATIONS:
            continue
        rate = float(passed[mask].mean())
        predicted = float(np.mean(model.pass_probability(advantage[mask], circuit=circuit)))
        offset = _logit(_clip(rate)) - _logit(_clip(predicted))
        difficulty[circuit] = float(np.clip(offset, -DIFFICULTY_BOUND, DIFFICULTY_BOUND))
    return difficulty


def _sigmoid(value: NDArray[np.float64] | float) -> NDArray[np.float64]:
    clipped = np.clip(value, -30.0, 30.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float64)


def _logit(probability: float) -> float:
    return float(np.log(probability / (1.0 - probability)))


def _clip(probability: float) -> float:
    return min(max(probability, 1e-6), 1.0 - 1e-6)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], frame.to_dict("records"))
