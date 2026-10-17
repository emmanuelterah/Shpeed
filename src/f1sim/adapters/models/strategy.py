"""Strategy generator and chooser (P4-07 to P4-09).

A stop plan is a compound for every stint and the lap of every stop. The generator enumerates
the legal plans for a race and scores each one with the expected race time: fuel, degradation
and warm-up loss per lap, plus the pit loss per stop (which already includes the stop's time, so
only the team's deviation from the field is added). The chooser turns those times into a
probability per plan, mixing the time-optimal plan with the team's own tendency (how many stops
it runs and which compound it starts on), so a team is rarely modelled as taking the exact
optimum (P4-08). The number of sets a driver has left limits the plans that are legal (P4-09),
and the chooser's stop-lap distribution gives the pit windows P4-10 measures.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from f1sim.adapters.models.pit_stops import race_stops
from f1sim.adapters.models.tyres import TyreModel
from f1sim.core.domain.rules import SeasonRules
from f1sim.core.errors import InvariantError

DRY_COMPOUNDS: tuple[str, ...] = ("SOFT", "MEDIUM", "HARD")
"""Canonical dry-compound names, fastest to slowest."""

MIN_STINT_LAPS = 4
"""A stint shorter than this is not raced; it also keeps the degradation region in view."""

MAX_STOPS = 3
MIN_MANDATORY_COMPOUNDS = 2
STOP_COUNT_SD_MIN = 0.6
STOP_COUNT_PRIOR = 1.5
START_PRIOR = 0.5
CHOOSER_TEMPERATURE = 8.0
"""Seconds of expected race time that a plan can give away for a factor of e in probability."""

DEFAULT_STOP_SPREAD = (-0.05, 0.35)
"""Fallback stop-lap offsets, as a fraction of race distance, when nothing was measured."""


@dataclass(frozen=True)
class StopPlan:
    """One legal race plan.

    Attributes:
        compounds: Compound of each stint, in order.
        stop_laps: Lap of each stop, in order.
        total_time_s: Expected race time, seconds.
    """

    compounds: tuple[str, ...]
    stop_laps: tuple[int, ...]
    total_time_s: float

    @property
    def n_stops(self) -> int:
        """Number of stops in the plan.

        Returns:
            The stop count.
        """
        return len(self.stop_laps)


@dataclass(frozen=True)
class PitWindow:
    """A predicted lap range for one stop.

    Attributes:
        stop_index: Which stop, 0-based.
        low_lap: Lowest lap of the window.
        high_lap: Highest lap of the window.
    """

    stop_index: int
    low_lap: int
    high_lap: int

    def contains(self, lap: int) -> bool:
        """Whether a lap falls in the window.

        Args:
            lap: A stop lap.

        Returns:
            True if the lap is inside the closed window.
        """
        return self.low_lap <= lap <= self.high_lap


@dataclass(frozen=True)
class StopLapSpread:
    """How far a stop's lap strays from the modelled lap (P4-10 calibration).

    The chooser predicts the optimal lap for each stop; real crews stop earlier or later, often
    by many laps, so a raw optimal-lap window is far too narrow. The spread records, per stop
    index, the lower and upper quantiles of the observed error as a fraction of race distance.
    A window is the predicted lap plus these offsets, which is what makes the calibration score
    meaningful.

    Attributes:
        offsets: Stop index to ``(low_fraction, high_fraction)`` of race distance.
        default: Offsets used for a stop index with too few measurements.
    """

    offsets: Mapping[int, tuple[float, float]]
    default: tuple[float, float]

    def range(self, stop_index: int, centre_lap: float, total_laps: int) -> tuple[int, int]:
        """Window around a predicted stop lap.

        Args:
            stop_index: Which stop, 0-based.
            centre_lap: Predicted lap of the stop.
            total_laps: Race distance in laps.

        Returns:
            ``(low_lap, high_lap)``, clipped to the race.
        """
        low_fraction, high_fraction = self.offsets.get(stop_index, self.default)
        low = round(centre_lap + low_fraction * total_laps)
        high = round(centre_lap + high_fraction * total_laps)
        low = min(max(low, 1), total_laps)
        high = min(max(high, 1), total_laps)
        if low > high:
            low, high = high, low
        return low, high


@dataclass(frozen=True)
class TeamTendency:
    """A team's historical strategy habits (P4-08).

    Attributes:
        mean_stops: Average stops per race.
        stop_sd: Spread of the stop count.
        start_compounds: Compound to the probability of starting on it.
    """

    mean_stops: float
    stop_sd: float
    start_compounds: Mapping[str, float]

    def start_prob(self, compound: str) -> float:
        """Probability that the team starts a race on a compound.

        Args:
            compound: Compound name.

        Returns:
            The probability; zero for a compound the team has never started on.
        """
        return self.start_compounds.get(compound, 0.0)


@dataclass
class TeamTendencies:
    """Team tendencies with a league-wide default (P4-08).

    Attributes:
        teams: Team ID to its tendency.
        default: Tendency for an unknown team.
    """

    teams: dict[str, TeamTendency]
    default: TeamTendency

    def for_team(self, team: str) -> TeamTendency:
        """Find a team's tendency.

        Args:
            team: Team ID.

        Returns:
            The team's tendency, or the default.
        """
        return self.teams.get(team, self.default)


class StrategyGenerator:  # pylint: disable=too-many-instance-attributes
    """Enumerate and score a race's legal stop plans (P4-07, P4-09)."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        tyre: TyreModel,
        rules: SeasonRules,
        *,
        total_laps: int,
        base_lap_s: float,
        circuit: str | None = None,
        team: str = "",
        start_compound: str | None = None,
        sets: Mapping[str, int] | None = None,
        compounds: Sequence[str] | None = None,
        max_stops: int = MAX_STOPS,
    ) -> None:
        """Create the generator.

        Args:
            tyre: Fitted tyre model.
            rules: The season's rules.
            total_laps: Race distance in laps.
            base_lap_s: Reference lap time in seconds; offsets and losses are added to it.
            circuit: Circuit ID, for the circuit's degradation and pit loss.
            team: Team ID, for its stationary time.
            start_compound: The compound the driver starts on, or None to consider all.
            sets: Remaining sets per compound, limiting the plan (P4-09), or None to ignore.
            compounds: Compounds to consider; defaults to the season's dry compounds.
            max_stops: Largest number of stops to consider.
        """
        self.tyre = tyre
        self.rules = rules
        self.total_laps = total_laps
        self.base_lap_s = base_lap_s
        self.circuit = circuit
        self.team = team
        self.start_compound = start_compound
        self.sets = sets
        self.compounds = tuple(compounds) if compounds is not None else self._default_compounds()
        self.max_stops = max_stops
        if total_laps < MIN_STINT_LAPS:
            raise InvariantError(f"race of {total_laps} laps is shorter than a stint")
        self._stop_cost = self.tyre.pit_loss_s(circuit) + self.tyre.stop_offset_s(team)
        self._costs = _StintCosts(tyre, circuit, total_laps, base_lap_s, self.compounds)

    def plans(self) -> list[StopPlan]:
        """Enumerate the legal plans, best first.

        Returns:
            Every legal plan, sorted by expected race time.

        Raises:
            InvariantError: If no plan is legal.
        """
        found: dict[tuple[tuple[str, ...], tuple[int, ...]], StopPlan] = {}
        for n_stops in range(1, self.max_stops + 1):
            for compounds in product(self.compounds, repeat=n_stops + 1):
                if not self._legal_compounds(compounds):
                    continue
                plan = self._best_plan(compounds)
                if plan is None:
                    continue
                found[(plan.compounds, plan.stop_laps)] = plan
                for variant in self._variants(plan):
                    found[(variant.compounds, variant.stop_laps)] = variant
        if not found:
            raise InvariantError("no legal strategy fits this race")
        return sorted(found.values(), key=lambda p: p.total_time_s)

    def _default_compounds(self) -> tuple[str, ...]:
        count = max(1, min(self.rules.dry_compounds_per_event, len(DRY_COMPOUNDS)))
        return DRY_COMPOUNDS[:count]

    def _legal_compounds(self, compounds: tuple[str, ...]) -> bool:
        if self.start_compound is not None and compounds[0] != self.start_compound:
            return False
        if self.rules.mandatory_compound_change and len(set(compounds)) < MIN_MANDATORY_COMPOUNDS:
            return False
        if self.sets is None:
            return True
        used = Counter(compounds)
        return all(count <= self.sets.get(compound, 0) for compound, count in used.items())

    def _best_plan(  # pylint: disable=too-many-locals
        self, compounds: tuple[str, ...]
    ) -> StopPlan | None:
        n_stints = len(compounds)
        min_stint = MIN_STINT_LAPS
        if self.total_laps < n_stints * min_stint:
            return None
        dp = np.full((n_stints, self.total_laps + 1), np.inf, dtype=np.float64)
        choice = np.full((n_stints, self.total_laps + 1), -1, dtype=np.int64)
        for stint in range(n_stints):
            remaining = n_stints - 1 - stint
            first_ok = (stint + 1) * min_stint
            last_ok = self.total_laps - remaining * min_stint
            table = self._costs.table(compounds[stint])
            if stint == 0:
                dp[0, first_ok : last_ok + 1] = table[1, first_ok : last_ok + 1]
                continue
            low = stint * min_stint
            for last in range(first_ok, last_ok + 1):
                # Stops after laps low..last - min_stint; the stint then runs prev + 1..last.
                high = last - min_stint
                candidates = (
                    dp[stint - 1, low : high + 1]
                    + self._stop_cost
                    + table[low + 1 : high + 2, last]
                )
                if not np.isfinite(candidates).any():
                    continue
                best = int(np.argmin(candidates))
                dp[stint, last] = candidates[best]
                choice[stint, last] = low + best
        if not np.isfinite(dp[n_stints - 1, self.total_laps]):
            return None
        stop_laps = self._trace(choice, n_stints)
        return StopPlan(compounds, stop_laps, float(dp[n_stints - 1, self.total_laps]))

    def _trace(self, choice: NDArray[np.int64], n_stints: int) -> tuple[int, ...]:
        stop_laps: list[int] = []
        last = self.total_laps
        for stint in range(n_stints - 1, 0, -1):
            last = int(choice[stint, last])
            stop_laps.append(last)
        stop_laps.reverse()
        return tuple(stop_laps)

    def _variants(self, plan: StopPlan) -> list[StopPlan]:
        variants: list[StopPlan] = []
        for index in range(plan.n_stops):
            for delta in (-2, -1, 1, 2):
                laps = list(plan.stop_laps)
                laps[index] += delta
                if not _legal_stints(tuple(laps), self.total_laps, MIN_STINT_LAPS):
                    continue
                cost = _plan_cost(
                    self._costs,
                    plan.compounds,
                    tuple(laps),
                    self._stop_cost,
                    self.total_laps,
                )
                variants.append(StopPlan(plan.compounds, tuple(laps), cost))
        return variants


class StrategyChooser:
    """Turn a race's plans into a probability per plan and per stop lap (P4-08)."""

    def __init__(
        self,
        plans: Sequence[StopPlan],
        tendencies: TeamTendencies,
        *,
        temperature: float = CHOOSER_TEMPERATURE,
    ) -> None:
        """Create the chooser.

        Args:
            plans: Plans from a :class:`StrategyGenerator`, best first.
            tendencies: Team tendencies.
            temperature: Softmax scale in seconds.

        Raises:
            InvariantError: If no plans are given.
        """
        if not plans:
            raise InvariantError("chooser needs at least one plan")
        self.plans = list(plans)
        self.tendencies = tendencies
        self.temperature = temperature

    def distribution(self, team: str, start_compound: str | None = None) -> list[float]:
        """Probability of each plan for a team.

        Args:
            team: Team ID.
            start_compound: A compound the driver must start on, or None.

        Returns:
            A probability per plan, summing to one.
        """
        tendency = self.tendencies.for_team(team)
        best = min(plan.total_time_s for plan in self.plans)
        weights = [self._weight(plan, tendency, best, start_compound) for plan in self.plans]
        total = float(np.sum(weights))
        if total <= 0.0:
            return [1.0 / len(self.plans)] * len(self.plans)
        return [float(w / total) for w in weights]

    def _weight(
        self,
        plan: StopPlan,
        tendency: TeamTendency,
        best: float,
        start_compound: str | None,
    ) -> float:
        weight = float(np.exp(-(plan.total_time_s - best) / self.temperature))
        weight *= max(tendency.start_prob(plan.compounds[0]), 1e-9)
        weight *= float(
            np.exp(-0.5 * ((plan.n_stops - tendency.mean_stops) / tendency.stop_sd) ** 2)
        )
        if start_compound is not None and plan.compounds[0] != start_compound:
            weight *= 1e-6
        return weight

    def stop_lap_distribution(
        self, team: str, stop_index: int, start_compound: str | None = None
    ) -> dict[int, float]:
        """Weighted distribution of one stop's lap.

        Args:
            team: Team ID.
            stop_index: Which stop, 0-based.
            start_compound: A compound the driver must start on, or None.

        Returns:
            Lap to probability, for plans that have that stop.
        """
        probabilities = self.distribution(team, start_compound)
        laps: dict[int, float] = {}
        for plan, probability in zip(self.plans, probabilities, strict=True):
            if stop_index < plan.n_stops:
                lap = plan.stop_laps[stop_index]
                laps[lap] = laps.get(lap, 0.0) + probability
        return laps

    def stop_lap_mean(self, team: str, start_compound: str | None = None) -> dict[int, float]:
        """Weighted mean lap of each stop across the plans (P4-08).

        Args:
            team: Team ID.
            start_compound: A compound the driver must start on, or None.

        Returns:
            Stop index to the probability-weighted mean lap.
        """
        probabilities = self.distribution(team, start_compound)
        means: dict[int, float] = {}
        for plan, probability in zip(self.plans, probabilities, strict=True):
            for index, lap in enumerate(plan.stop_laps):
                means[index] = means.get(index, 0.0) + probability * lap
        return means

    def windows(  # pylint: disable=too-many-locals
        self,
        team: str,
        start_compound: str | None = None,
        *,
        low: float = 0.1,
        high: float = 0.9,
        spread: StopLapSpread | None = None,
        total_laps: int | None = None,
    ) -> list[PitWindow]:
        """Pit windows from the stop-lap distributions.

        Args:
            team: Team ID.
            start_compound: A compound the driver must start on, or None.
            low: Lower quantile, used when no spread is given.
            high: Upper quantile, used when no spread is given.
            spread: Calibrated stop-lap spread; when given the window is the predicted mean lap
                plus the spread's quantile offsets (P4-10).
            total_laps: Race distance in laps; required together with ``spread``.

        Returns:
            One window per stop index that some plan has.

        Raises:
            InvariantError: If a spread is given without the race distance.
        """
        n_stops = max((plan.n_stops for plan in self.plans), default=0)
        windows: list[PitWindow] = []
        for index in range(n_stops):
            distribution = self.stop_lap_distribution(team, index, start_compound)
            if not distribution:
                continue
            if spread is not None:
                if total_laps is None:
                    raise InvariantError("a spread needs the race distance")
                centre = sum(lap * probability for lap, probability in distribution.items())
                low_lap, high_lap = spread.range(index, centre, total_laps)
                windows.append(PitWindow(index, low_lap, high_lap))
                continue
            laps = sorted(distribution)
            cumulative = np.cumsum([distribution[lap] for lap in laps])
            if cumulative[-1] <= 0.0:
                continue
            low_index = int(min(np.searchsorted(cumulative, low), len(laps) - 1))
            high_index = int(min(np.searchsorted(cumulative, high), len(laps) - 1))
            windows.append(PitWindow(index, laps[low_index], laps[high_index]))
        return windows


def fit_team_tendencies(pit_stops: pd.DataFrame, laps: pd.DataFrame) -> TeamTendencies:
    """Measure each team's stop count and starting compound (P4-08).

    Only race stops count toward a team's stop tendency.

    Args:
        pit_stops: Canonical ``pit_stops`` rows.
        laps: Canonical ``laps`` rows.

    Returns:
        Team tendencies with a league-wide default.
    """
    team_of = {
        (int(r["season"]), int(r["round"]), str(r["driver_id"])): str(r["team_id"])
        for r in _records(laps)
        if r["team_id"] is not None
    }
    counts: dict[str, list[int]] = {}
    for (season, round_, driver), group in race_stops(pit_stops).groupby(
        ["season", "round", "driver_id"]
    ):
        team = team_of.get((_as_int(season), _as_int(round_), str(driver)))
        if team is not None:
            counts.setdefault(team, []).append(len(group))
    starters: dict[str, Counter[str]] = {}
    for key, compound in _first_compounds(laps).items():
        team = team_of.get(key)
        if team is not None and compound in DRY_COMPOUNDS:
            starters.setdefault(team, Counter())[compound] += 1
    teams = {
        team: TeamTendency(*_stop_stats(counts.get(team, [])), _start_probs(starters.get(team)))
        for team in sorted(set(counts) | set(starters))
    }
    default = TeamTendency(
        *_stop_stats([c for values in counts.values() for c in values]),
        _start_probs(_merge(starters)),
    )
    return TeamTendencies(teams, default)


class _StintCosts:
    """Inclusive cost of running one compound from a first lap to a last lap."""

    def __init__(
        self,
        tyre: TyreModel,
        circuit: str | None,
        total_laps: int,
        base_lap_s: float,
        compounds: Sequence[str],
    ) -> None:
        self.total_laps = total_laps
        self._tables: dict[str, NDArray[np.float64]] = {}
        laps = np.arange(total_laps + 1, dtype=np.float64)
        fuel = tyre.fuel_s_per_lap * (total_laps - laps)
        for compound in compounds:
            offset = tyre.compound_offset(compound)
            loss = np.array(
                [tyre.loss(compound, circuit, float(age)) for age in range(total_laps + 1)],
                dtype=np.float64,
            )
            table = np.full((total_laps + 1, total_laps + 1), np.inf, dtype=np.float64)
            for first in range(1, total_laps + 1):
                ages = np.arange(1, total_laps - first + 2)
                per_lap = base_lap_s + offset + loss[ages] + fuel[first:]
                table[first, first:] = np.cumsum(per_lap)
            self._tables[compound] = table

    def table(self, compound: str) -> NDArray[np.float64]:
        """The whole cost table of a compound, ``table[first, last]``.

        Args:
            compound: Compound name.

        Returns:
            Array of shape ``(total_laps + 1, total_laps + 1)``; infinite where undefined.
        """
        missing = np.full((self.total_laps + 1, self.total_laps + 1), np.inf, dtype=np.float64)
        return self._tables.get(compound, missing)

    def __call__(self, compound: str, first: int, last: int) -> float:
        """Cost of one stint.

        Args:
            compound: Compound name.
            first: First lap of the stint.
            last: Last lap of the stint.

        Returns:
            Expected seconds; infinite if the stint is empty or the compound is unknown.
        """
        if first > last or compound not in self._tables:
            return np.inf
        return float(self._tables[compound][first, last])


def _plan_cost(
    costs: _StintCosts,
    compounds: tuple[str, ...],
    stop_laps: tuple[int, ...],
    stop_cost: float,
    total_laps: int,
) -> float:
    total = 0.0
    first = 1
    for stint, stop in enumerate(stop_laps):
        total += costs(compounds[stint], first, stop) + stop_cost
        first = stop + 1
    total += costs(compounds[len(stop_laps)], first, total_laps)
    return total


def _legal_stints(stop_laps: tuple[int, ...], total_laps: int, min_stint: int) -> bool:
    previous = 0
    for stop in stop_laps:
        if stop - previous < min_stint:
            return False
        previous = stop
    return total_laps - previous >= min_stint


def _stop_stats(counts: Sequence[int]) -> tuple[float, float]:
    if not counts:
        return STOP_COUNT_PRIOR, STOP_COUNT_SD_MIN + 0.2
    return float(np.mean(counts)), max(float(np.std(counts)), STOP_COUNT_SD_MIN)


def _start_probs(counter: Counter[str] | None) -> dict[str, float]:
    counts = dict.fromkeys(DRY_COMPOUNDS, START_PRIOR)
    if counter is not None:
        for compound, value in counter.items():
            counts[compound] += value
    total = sum(counts.values())
    return {compound: value / total for compound, value in counts.items()}


def _merge(counters: Mapping[str, Counter[str]]) -> Counter[str]:
    merged: Counter[str] = Counter()
    for counter in counters.values():
        merged.update(counter)
    return merged


def _first_compounds(laps: pd.DataFrame) -> dict[tuple[int, int, str], str]:
    race = laps.loc[laps["session"] == "R"]
    ordered = race.sort_values("lap_number")
    first: dict[tuple[int, int, str], str] = {}
    for row in _records(ordered):
        key = (int(row["season"]), int(row["round"]), str(row["driver_id"]))
        if key not in first and row["compound"] is not None:
            first[key] = str(row["compound"]).upper()
    return first


def _as_int(value: object) -> int:
    return int(cast(Any, value))


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], frame.to_dict("records"))
