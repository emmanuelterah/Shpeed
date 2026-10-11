"""Grid-with-noise, Elo and Plackett-Luce predictors (P2-08 to P2-10).

These are the yardsticks: the lap-by-lap simulator has to beat all three in the
walk-forward backtest before anyone should trust it. Each one learns only from the
``results`` table, read through the reader it is given, so in a backtest it cannot see
past the cutoff.
"""

import math
from collections.abc import Sequence
from datetime import datetime

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from f1sim.adapters.models.common import (
    DNF_PRIOR_RATE,
    N_SIMS,
    dnf_rate,
    race_history,
    rankings,
    simulate,
)
from f1sim.adapters.models.ranking_fit import fit_plackett_luce
from f1sim.core.domain.entities import EventKey, SnapshotId
from f1sim.core.domain.lineage import TeamLineage
from f1sim.core.domain.prediction import PositionMatrix
from f1sim.core.domain.race import RaceEntry
from f1sim.core.errors import InvariantError
from f1sim.core.ports.store import DataReader
from f1sim.core.usecases.readers import as_naive_utc

MIN_SIGMA = 0.5
MAX_SIGMA = 60.0
SPREAD_CANDIDATES = 40
SPREAD_SIMS = 2000
DEFAULT_SIGMA = 4.0
MIN_GAINS_FOR_SIGMA = 20
"""Finishers needed before the grid-noise spread is fitted rather than assumed."""

ELO_SCALE = 400.0
ELO_START = 1500.0
MIN_FINISHERS = 2
"""A session needs two classified drivers before anyone can be compared."""

SECONDS_PER_DAY = 86_400.0
SESSION_WEIGHTS: dict[str, float] = {"R": 1.0, "S": 0.5, "Q": 0.5}
"""How much each session's order counts toward race pace."""


class GridNoise:
    """Finish where you start, give or take some noise (P2-08).

    The noise is normal, with its spread fitted so that simulated finishers move from
    their grid slot as much as real ones have. Retirements happen at the historical rate.
    """

    name = "grid_noise"

    def __init__(self, seed: int, *, n_sims: int = N_SIMS) -> None:
        """Create an unfitted model.

        Args:
            seed: Random seed for the simulated races.
            n_sims: Simulated races per forecast.
        """
        self._rng = np.random.default_rng(seed)
        self._n_sims = n_sims
        self.sigma = DEFAULT_SIGMA
        self.p_dnf = DNF_PRIOR_RATE

    def fit(self, reader: DataReader, *, snapshot: SnapshotId, cutoff: datetime) -> None:
        """Fit the noise spread and the retirement rate.

        Args:
            reader: The data store.
            snapshot: Snapshot to read.
            cutoff: Latest moment whose data may be used.
        """
        history = race_history(reader, snapshot, cutoff)
        races = history.loc[(history["session"] == "R") & history["position"].notna()]
        grid = pd.to_numeric(races["grid"])
        gains = (pd.to_numeric(races["position"]) - grid).loc[grid > 0]
        if len(gains) >= MIN_GAINS_FOR_SIGMA:
            field = int(races.groupby(["season", "round"]).size().median())
            self.sigma = match_spread(float(gains.std()), field, self._rng)
        self.p_dnf = dnf_rate(history)

    def predict(self, event: EventKey, entries: Sequence[RaceEntry]) -> PositionMatrix:
        """Forecast a race from the grid alone.

        Args:
            event: The race weekend (unused; the grid says it all).
            entries: The starters with their grid positions.

        Returns:
            Finishing-position probabilities in ``entries`` order.
        """
        del event
        utility = -np.array([e.grid for e in entries], dtype=np.float64)
        noise = self._rng.normal(0.0, self.sigma, size=(self._n_sims, len(entries)))
        return simulate(entries, utility, noise, self.p_dnf, self._rng)


def match_spread(target: float, field: int, rng: np.random.Generator) -> float:
    """Find the grid noise whose simulated races move drivers as much as real ones did.

    Ranking ``-grid + noise`` moves drivers less than the noise itself, because places
    are bounded and neighbours move too, so the noise is chosen by simulation: the
    candidate whose finish-minus-grid spread is closest to the observed one.

    Args:
        target: Observed standard deviation of finish position minus grid position.
        field: Typical number of starters.
        rng: Random source; the same draws are reused for every candidate.

    Returns:
        The noise standard deviation, in grid places.
    """
    grid = np.arange(1, field + 1, dtype=np.float64)
    z = rng.standard_normal((SPREAD_SIMS, field))
    candidates = np.geomspace(MIN_SIGMA, MAX_SIGMA, SPREAD_CANDIDATES)
    spreads: list[float] = []
    for sigma in candidates:
        ranks = np.argsort(np.argsort(grid + sigma * z, axis=1), axis=1) + 1
        spreads.append(float(np.std(ranks - grid)))
    return float(candidates[int(np.argmin(np.abs(np.array(spreads) - target)))])


class Elo:
    """Multiplayer Elo with separate driver and team ratings (P2-09).

    A driver's strength is their rating plus their team's. After each race, every pair of
    classified finishers is scored as a head-to-head result. Team ratings follow lineage,
    so a rebranded team keeps its rating. Ratings regress toward the mean between seasons.
    Forecasts sample orders from the Plackett-Luce model whose head-to-head probabilities
    are exactly the Elo ones.
    """

    name = "elo"

    def __init__(
        self,
        seed: int,
        *,
        lineage: TeamLineage,
        k_driver: float = 24.0,
        k_team: float = 12.0,
        season_regression: float = 0.3,
        n_sims: int = N_SIMS,
    ) -> None:
        """Create an unrated model.

        Args:
            seed: Random seed for the simulated races.
            lineage: Maps constructor IDs to continuing teams.
            k_driver: Driver rating step per race.
            k_team: Team rating step per race.
            season_regression: Share of each rating's distance from the mean removed
                at the start of every season, between 0 and 1.
            n_sims: Simulated races per forecast.

        Raises:
            InvariantError: If ``season_regression`` is outside [0, 1].
        """
        if not 0.0 <= season_regression <= 1.0:
            raise InvariantError("season_regression must be between 0 and 1")
        self._rng = np.random.default_rng(seed)
        self._lineage = lineage
        self._k = (k_driver, k_team)
        self._regression = season_regression
        self._n_sims = n_sims
        self.drivers: dict[str, float] = {}
        self.teams: dict[str, float] = {}
        self.p_dnf = DNF_PRIOR_RATE

    def fit(self, reader: DataReader, *, snapshot: SnapshotId, cutoff: datetime) -> None:
        """Replay every past race in order, updating ratings.

        Args:
            reader: The data store.
            snapshot: Snapshot to read.
            cutoff: Latest moment whose data may be used.
        """
        history = race_history(reader, snapshot, cutoff)
        season: int | None = None
        for race in rankings(history, "R"):
            if season is not None and race.season != season:
                self._regress()
            season = race.season
            teams = [self._lineage.lineage_of(t) for t in race.teams]
            self._update(list(race.drivers), teams)
        self.p_dnf = dnf_rate(history)

    def strength(self, entry: RaceEntry) -> float:
        """Combined driver and team rating, relative to two start ratings.

        Args:
            entry: A starter.

        Returns:
            ``driver + team - 2 * ELO_START``; 0 for a new driver in a new team.
        """
        team = self._lineage.lineage_of(entry.team_id)
        driver = self.drivers.get(entry.driver_id, ELO_START)
        return driver + self.teams.get(team, ELO_START) - 2 * ELO_START

    def predict(self, event: EventKey, entries: Sequence[RaceEntry]) -> PositionMatrix:
        """Forecast a race from the current ratings.

        Args:
            event: The race weekend (unused).
            entries: The starters.

        Returns:
            Finishing-position probabilities in ``entries`` order.
        """
        del event
        ratings = np.array([self.strength(e) for e in entries], dtype=np.float64)
        utility = ratings * math.log(10.0) / ELO_SCALE
        noise = self._rng.gumbel(size=(self._n_sims, len(entries)))
        return simulate(entries, utility, noise, self.p_dnf, self._rng)

    def _update(self, drivers: list[str], teams: list[str]) -> None:
        if len(drivers) < MIN_FINISHERS:
            return
        driver_r = np.array([self.drivers.get(d, ELO_START) for d in drivers])
        team_r = np.array([self.teams.get(t, ELO_START) for t in teams])
        surprise = elo_surprise(driver_r + team_r)
        k_driver, k_team = self._k
        updated = driver_r + k_driver * surprise
        self.drivers.update(zip(drivers, updated.tolist(), strict=True))
        by_team: dict[str, list[float]] = {}
        for team, s in zip(teams, surprise.tolist(), strict=True):
            by_team.setdefault(team, []).append(s)
        for team, values in by_team.items():
            self.teams[team] = self.teams.get(team, ELO_START) + k_team * float(np.mean(values))

    def _regress(self) -> None:
        for ratings in (self.drivers, self.teams):
            if ratings:
                mean = float(np.mean(list(ratings.values())))
                for key, value in ratings.items():
                    ratings[key] = mean + (value - mean) * (1.0 - self._regression)


def elo_surprise(strength: NDArray[np.float64]) -> NDArray[np.float64]:
    """Each finisher's head-to-head wins minus expected wins, averaged over opponents.

    Args:
        strength: Ratings of the classified finishers, in finishing order.

    Returns:
        One value per finisher between -1 and 1; they sum to 0.
    """
    n = len(strength)
    gap = strength[np.newaxis, :] - strength[:, np.newaxis]
    expected = 1.0 / (1.0 + 10.0 ** (gap / ELO_SCALE))
    np.fill_diagonal(expected, 0.0)
    beat = np.triu(np.ones((n, n)), k=1)
    surprise: NDArray[np.float64] = (beat - expected).sum(axis=1) / (n - 1)
    return surprise


class PlackettLuce:
    """Plackett-Luce strengths fitted to recent race, sprint and qualifying orders (P2-10).

    Each driver has a strength ``w``; a race is a sequence of choices in which the next
    finisher is drawn from the drivers left with probability proportional to ``w``. The
    strengths are the posterior mode under a gamma prior (see ``ranking_fit``), and older
    sessions count less, halving every ``half_life_days``. The weekend's own qualifying
    is known at the cutoff, so it counts at full freshness.
    """

    name = "plackett_luce"

    def __init__(
        self,
        seed: int,
        *,
        half_life_days: float = 120.0,
        prior_strength: float = 1.0,
        n_sims: int = N_SIMS,
    ) -> None:
        """Create an unfitted model.

        Args:
            seed: Random seed for the simulated races.
            half_life_days: Age at which a session counts half as much as a fresh one.
            prior_strength: Pseudo-observations behind the prior strength of 1.
            n_sims: Simulated races per forecast.

        Raises:
            InvariantError: If ``half_life_days`` or ``prior_strength`` is not positive.
        """
        if half_life_days <= 0 or prior_strength <= 0:
            raise InvariantError("half_life_days and prior_strength must be positive")
        self._rng = np.random.default_rng(seed)
        self._half_life = half_life_days
        self._prior = prior_strength
        self._n_sims = n_sims
        self.strengths: dict[str, float] = {}
        self.p_dnf = DNF_PRIOR_RATE

    def fit(self, reader: DataReader, *, snapshot: SnapshotId, cutoff: datetime) -> None:
        """Fit driver strengths to every classified order before the cutoff.

        Args:
            reader: The data store.
            snapshot: Snapshot to read.
            cutoff: Latest moment whose data may be used.
        """
        history = race_history(reader, snapshot, cutoff)
        self.p_dnf = dnf_rate(history)
        end = as_naive_utc(cutoff)
        orders: list[tuple[str, ...]] = []
        weights: list[float] = []
        for session, session_weight in SESSION_WEIGHTS.items():
            for ranked in rankings(history, session):
                if len(ranked.drivers) >= MIN_FINISHERS:
                    age = (end - ranked.available_at).total_seconds() / SECONDS_PER_DAY
                    orders.append(ranked.drivers)
                    weights.append(session_weight * 0.5 ** (max(age, 0.0) / self._half_life))
        names = sorted({d for order in orders for d in order})
        index = {d: i for i, d in enumerate(names)}
        w = fit_plackett_luce(
            [[index[d] for d in order] for order in orders],
            weights,
            n_items=len(names),
            prior_strength=self._prior,
        )
        self.strengths = dict(zip(names, w.tolist(), strict=True))

    def predict(self, event: EventKey, entries: Sequence[RaceEntry]) -> PositionMatrix:
        """Forecast a race by sampling Plackett-Luce orders.

        Args:
            event: The race weekend (unused).
            entries: The starters; unknown drivers get the prior strength of 1.

        Returns:
            Finishing-position probabilities in ``entries`` order.
        """
        del event
        w = np.array([self.strengths.get(e.driver_id, 1.0) for e in entries])
        noise = self._rng.gumbel(size=(self._n_sims, len(entries)))
        return simulate(entries, np.log(w), noise, self.p_dnf, self._rng)
