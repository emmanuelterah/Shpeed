"""Helpers shared by the baseline predictors: history, DNF rates and order sampling.

Every baseline turns a per-driver score into a ``PositionMatrix`` the same way: add random
noise to the scores, mark some drivers as retired, rank the rest, and count positions over
many simulated races.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from f1sim.core.domain.entities import SnapshotId
from f1sim.core.domain.prediction import PositionMatrix
from f1sim.core.domain.race import RaceEntry
from f1sim.core.ports.store import DataReader
from f1sim.core.usecases.readers import as_naive_utc

N_SIMS = 4000
"""Simulated races per forecast; enough for position probabilities to about 0.01."""

SMOOTHING = 1e-3
"""Weight of the uniform forecast mixed into every matrix, so no outcome is impossible."""

DNF_PRIOR_RATE = 0.10
DNF_PRIOR_STRENGTH = 50.0
"""Pseudo-starts behind the DNF prior; the observed rate dominates after a few races."""


def race_history(reader: DataReader, snapshot: SnapshotId, cutoff: datetime) -> pd.DataFrame:
    """Read every results row a model may use.

    Args:
        reader: The data store, usually a ``CutoffReader``.
        snapshot: Snapshot to read.
        cutoff: Latest moment whose data may be used.

    Returns:
        All results rows available at ``cutoff``, sorted by season, round and session.
    """
    rows = reader.read("results", snapshot=snapshot, as_of=cutoff)
    return rows.sort_values(["season", "round", "session"], kind="stable")


def dnf_rate(history: pd.DataFrame) -> float:
    """Estimate the chance that a starter does not finish, shrunk toward a prior.

    Args:
        history: Results rows; only race (``R``) rows are used.

    Returns:
        ``(retirements + prior) / (starts + strength)``.
    """
    races = history.loc[history["session"] == "R"]
    starts = float(len(races))
    retired = float(races["position"].isna().sum())
    prior = DNF_PRIOR_RATE * DNF_PRIOR_STRENGTH
    return (retired + prior) / (starts + DNF_PRIOR_STRENGTH)


@dataclass(frozen=True)
class SessionOrder:
    """The classified order of one session.

    Attributes:
        season: Season of the session.
        round: Round of the session.
        session: Session code, such as ``R``.
        drivers: Classified driver IDs, best first.
        teams: Their constructor IDs, aligned with ``drivers``.
        available_at: When the result became known.
    """

    season: int
    round: int
    session: str
    drivers: tuple[str, ...]
    teams: tuple[str, ...]
    available_at: datetime


def rankings(history: pd.DataFrame, session: str) -> list[SessionOrder]:
    """List each session's classified drivers in finishing order.

    Args:
        history: Results rows.
        session: Session code such as ``R``, ``Q`` or ``S``.

    Returns:
        One ``SessionOrder`` per session, in calendar order.
    """
    rows = history.loc[(history["session"] == session) & history["position"].notna()]
    rows = rows.sort_values(["season", "round", "position"], kind="stable")
    seasons = [int(v) for v in rows["season"].tolist()]
    rounds = [int(v) for v in rows["round"].tolist()]
    drivers = [str(v) for v in rows["driver_id"].tolist()]
    teams = [str(v) for v in rows["team_id"].tolist()]
    moments = [as_naive_utc(pd.Timestamp(v).to_pydatetime()) for v in rows["available_at"]]
    grouped: dict[tuple[int, int], list[int]] = {}
    for i, key in enumerate(zip(seasons, rounds, strict=True)):
        grouped.setdefault(key, []).append(i)
    return [
        SessionOrder(
            season=season,
            round=round_,
            session=session,
            drivers=tuple(drivers[i] for i in idx),
            teams=tuple(teams[i] for i in idx),
            available_at=max(moments[i] for i in idx),
        )
        for (season, round_), idx in sorted(grouped.items())
    ]


def finishing_orders(scores: NDArray[np.float64], retired: NDArray[np.bool_]) -> NDArray[np.int_]:
    """Rank simulated races: higher score finishes ahead; retired drivers get 0.

    Args:
        scores: Array of shape ``(n_sims, n_drivers)``; scores with noise already added.
        retired: Same shape; ``True`` where the driver did not finish.

    Returns:
        Array of the same shape: positions from 1 among finishers, 0 for a DNF.
    """
    masked = np.where(retired, -np.inf, scores)
    order = np.argsort(-masked, axis=1, kind="stable")
    ranks = np.empty_like(order)
    sims = np.arange(scores.shape[0], dtype=np.int64).reshape(-1, 1)
    ranks[sims, order] = np.arange(1, scores.shape[1] + 1)
    orders: NDArray[np.int_] = np.where(retired, 0, ranks).astype(np.int64)
    return orders


def simulate(
    entries: Sequence[RaceEntry],
    utility: NDArray[np.float64],
    noise: NDArray[np.float64],
    p_dnf: float,
    rng: np.random.Generator,
) -> PositionMatrix:
    """Turn scores and noise into a smoothed position matrix.

    Args:
        entries: The starters; rows of the result follow their order.
        utility: One score per entry; higher is faster.
        noise: Array of shape ``(n_sims, n_entries)`` added to ``utility``.
        p_dnf: Chance that each driver retires.
        rng: Random source for retirements.

    Returns:
        The forecast, mixed with ``SMOOTHING`` of the uniform forecast.
    """
    retired = rng.random(noise.shape) < p_dnf
    orders = finishing_orders(utility[np.newaxis, :] + noise, retired)
    drivers = tuple(e.driver_id for e in entries)
    return PositionMatrix.from_finishing_orders(drivers, orders).smoothed(SMOOTHING)
