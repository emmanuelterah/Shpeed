"""Fuel effect and track evolution (P3-01, P3-02).

Raw lap times mix a car's pace with things that change through a session: the car gets
lighter as fuel burns, tyres wear, and the track gets faster as rubber goes down. These
fits estimate those effects so that laps become comparable.

- **Races and sprints:** ``lap time ~ driver + compound + compound x tyre age + lap``.
  The lap-number slope is the fuel effect. It also absorbs track evolution during the
  race, because both move with the lap count and cannot be told apart; separating them
  would need an outside fuel figure. Tyre age resets at every stop while the lap count
  keeps going, which is what lets the fit tell fuel from tyre wear.
- **Qualifying and practice:** ``lap time ~ driver + compound + minutes into session`` on
  push laps. The minutes slope is track evolution.

One session is a small sample, so each estimate is pooled with the same circuit's earlier
sessions and a global prior, weighting each by its precision.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

WET_COMPOUNDS = frozenset({"INTERMEDIATE", "WET", "UNKNOWN", "TEST_UNKNOWN", "NAN", ""})
PUSH_LAP_MARGIN = 1.03
"""A qualifying or practice lap within 3% of the driver's best is treated as a push lap."""

OUTLIER_MADS = 3.5
MAD_TO_SD = 1.4826
MIN_FIT_LAPS = 30
"""Laps a session needs before its own fit is trusted."""

FUEL_PRIOR = (0.06, 0.03)
"""Global prior for the fuel effect: mean and standard deviation, seconds per lap."""

EVOLUTION_PRIOR = (-0.01, 0.02)
"""Global prior for track evolution: mean and standard deviation, seconds per minute."""

SECONDS_PER_MINUTE = 60.0


@dataclass(frozen=True)
class Estimate:
    """A slope with its standard error.

    Attributes:
        value: The estimate.
        se: Its standard error.
    """

    value: float
    se: float


@dataclass(frozen=True)
class SessionFit:
    """What one session's laps say about the effects to correct for.

    Attributes:
        drivers: Driver pace in seconds (the driver fixed effects), by driver ID.
        offsets: Pace offset of each dry compound against the reference compound, seconds.
        wear: Lap-time increase per lap of tyre age, by compound, seconds.
        slope: The lap-number slope (races) or minutes slope (qualifying, practice).
        residual_sd: Standard deviation of the residuals, seconds.
        n_laps: Laps used after outlier removal.
    """

    drivers: dict[str, float]
    offsets: dict[str, float]
    wear: dict[str, float]
    slope: Estimate | None
    residual_sd: float
    n_laps: int


def dry_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Keep the clean laps run on dry tyres.

    Args:
        laps: Canonical ``laps`` rows.

    Returns:
        Laps flagged ``clean_lap`` with a known dry compound and a lap time.
    """
    compound = laps["compound"].fillna("").astype(str).str.upper()
    mask = (
        laps["clean_lap"].fillna(False).astype(bool)
        & laps["lap_time_s"].notna()
        & ~compound.isin(list(WET_COMPOUNDS))
    )
    selected: pd.DataFrame = laps.loc[mask].assign(compound=compound.loc[mask])
    return selected


def push_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Keep each driver's push laps: dry, clean and close to their best of the session.

    Args:
        laps: One session's ``laps`` rows.

    Returns:
        The push laps, with ``minutes`` (since the session's first lap) added.
    """
    dry = dry_laps(laps)
    if dry.empty:
        return dry.assign(minutes=pd.Series(dtype="float64"))
    times = dry["lap_time_s"].to_numpy(dtype=np.float64)
    drivers = [str(v) for v in dry["driver_id"].tolist()]
    best: dict[str, float] = {}
    for driver, time_s in zip(drivers, times.tolist(), strict=True):
        best[driver] = min(best.get(driver, time_s), time_s)
    limit = np.array([best[d] * PUSH_LAP_MARGIN for d in drivers])
    pushed: pd.DataFrame = dry.iloc[np.flatnonzero(times <= limit)]
    starts = pd.to_datetime(laps["lap_start_utc"]).to_numpy(dtype="datetime64[ns]")
    stamps = pd.to_datetime(pushed["lap_start_utc"]).to_numpy(dtype="datetime64[ns]")
    first = np.nanmin(starts) if not np.isnat(starts).all() else np.datetime64("NaT")
    # A lap without a start time gets minute 0, so it is simply not corrected.
    minutes = np.nan_to_num((stamps - first) / np.timedelta64(1, "m"))
    selected: pd.DataFrame = pushed.assign(minutes=minutes.astype(np.float64))
    return selected


def fit_race_laps(laps: pd.DataFrame, fuel: float | None = None) -> SessionFit | None:
    """Fit a race or sprint: driver, compound, tyre wear and the lap-number slope.

    Args:
        laps: One session's ``laps`` rows.
        fuel: A known fuel effect (seconds per lap). When given, it is removed first and
            only driver, compound and wear are fitted; ``slope`` is then None.

    Returns:
        The fit, or None when there are too few clean dry laps. The fuel effect is
        ``-slope``, since lap times fall as fuel burns.
    """
    dry = dry_laps(laps)
    if len(dry) < MIN_FIT_LAPS:
        return None
    y = dry["lap_time_s"].to_numpy(dtype=np.float64)
    lap = dry["lap_number"].to_numpy(dtype=np.float64)
    extra: dict[str, NDArray[np.float64]] = {}
    if fuel is None:
        extra["slope"] = lap
    else:
        y = y + fuel * lap
    return _fit(dry, y, extra, with_wear=True)


def fit_push_laps(laps: pd.DataFrame, evolution: float | None = None) -> SessionFit | None:
    """Fit qualifying or practice push laps: driver, compound and track evolution.

    Args:
        laps: One session's ``laps`` rows.
        evolution: A known evolution slope (seconds per minute). When given, it is
            removed first and ``slope`` is None.

    Returns:
        The fit, or None when no driver has a push lap.
    """
    pushed = push_laps(laps)
    if pushed.empty:
        return None
    y = pushed["lap_time_s"].to_numpy(dtype=np.float64)
    minutes = pushed["minutes"].to_numpy(dtype=np.float64)
    extra: dict[str, NDArray[np.float64]] = {}
    if evolution is None and len(pushed) >= MIN_FIT_LAPS:
        extra["slope"] = minutes
    elif evolution is not None:
        y = y - evolution * minutes
    return _fit(pushed, y, extra, with_wear=False)


def pool(estimates: Sequence[Estimate], prior: tuple[float, float]) -> Estimate:
    """Combine estimates of one quantity with a prior, weighting by precision.

    Args:
        estimates: Independent estimates, such as one per earlier race at a circuit.
        prior: ``(mean, standard deviation)`` before any data.

    Returns:
        The pooled estimate.
    """
    precision = 1.0 / prior[1] ** 2
    weighted = prior[0] * precision
    for e in estimates:
        if e.se > 0 and np.isfinite(e.value):
            precision += 1.0 / e.se**2
            weighted += e.value / e.se**2
    return Estimate(weighted / precision, float(np.sqrt(1.0 / precision)))


@dataclass
class CircuitHistory:
    """Per-circuit fuel and evolution estimates from past sessions.

    Attributes:
        fuel: Circuit ID to fuel estimates (seconds per lap), one per past race.
        evolution: Circuit ID to evolution estimates (seconds per minute).
    """

    fuel: dict[str, list[Estimate]] = field(default_factory=dict[str, list[Estimate]])
    evolution: dict[str, list[Estimate]] = field(default_factory=dict[str, list[Estimate]])

    def add(self, circuit: str, *, fuel: Estimate | None, evolution: Estimate | None) -> None:
        """Record one session's estimates for a circuit.

        Args:
            circuit: Circuit ID.
            fuel: The session's fuel estimate, if it gave one.
            evolution: The session's evolution estimate, if it gave one.
        """
        if fuel is not None:
            self.fuel.setdefault(circuit, []).append(fuel)
        if evolution is not None:
            self.evolution.setdefault(circuit, []).append(evolution)

    def to_dict(self) -> dict[str, dict[str, list[list[float]]]]:
        """Serialise for the fitted-state store.

        Returns:
            ``fuel`` and ``evolution``: circuit to ``[value, se]`` pairs.
        """
        return {
            "fuel": {c: [[e.value, e.se] for e in v] for c, v in self.fuel.items()},
            "evolution": {c: [[e.value, e.se] for e in v] for c, v in self.evolution.items()},
        }

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Mapping[str, Sequence[Sequence[float]]]]
    ) -> "CircuitHistory":
        """Rebuild a history saved with ``to_dict``.

        Args:
            data: The saved dictionary.

        Returns:
            The history.
        """

        def _read(part: str) -> dict[str, list[Estimate]]:
            return {
                str(c): [Estimate(float(v), float(se)) for v, se in pairs]
                for c, pairs in data.get(part, {}).items()
            }

        return cls(fuel=_read("fuel"), evolution=_read("evolution"))

    def fuel_at(self, circuit: str) -> Estimate:
        """Pooled fuel effect for a circuit.

        Args:
            circuit: Circuit ID.

        Returns:
            The estimate from that circuit's past races and the global prior.
        """
        return pool(self.fuel.get(circuit, []), FUEL_PRIOR)

    def evolution_at(self, circuit: str) -> Estimate:
        """Pooled track evolution for a circuit.

        Args:
            circuit: Circuit ID.

        Returns:
            The estimate from that circuit's past sessions and the global prior.
        """
        return pool(self.evolution.get(circuit, []), EVOLUTION_PRIOR)


def _fit(
    laps: pd.DataFrame,
    y: NDArray[np.float64],
    extra: dict[str, NDArray[np.float64]],
    *,
    with_wear: bool,
) -> SessionFit | None:
    design = _Design(laps, extra, with_wear=with_wear)
    keep = np.ones(len(y), dtype=bool)
    beta = _least_squares(design.matrix, y)
    for _ in range(2):
        resid = y - design.matrix @ beta
        mad = float(np.median(np.abs(resid[keep] - np.median(resid[keep]))))
        keep = np.abs(resid) <= OUTLIER_MADS * MAD_TO_SD * max(mad, 1e-6)
        beta = _least_squares(design.matrix[keep], y[keep])
    x = design.matrix[keep]
    dof = int(keep.sum()) - x.shape[1]
    if dof <= 0 or np.linalg.matrix_rank(x) < x.shape[1]:
        return None
    resid = y[keep] - x @ beta
    sigma2 = float(resid @ resid) / dof
    return design.unpack(beta, np.linalg.inv(x.T @ x) * sigma2, sigma2, int(keep.sum()))


def _least_squares(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    solution: Any = np.linalg.lstsq(x, y, rcond=None)[0]
    return np.asarray(solution, dtype=np.float64)


class _Design:
    """Dummy-coded design matrix: drivers, compound offsets, wear slopes, extra columns."""

    def __init__(
        self, laps: pd.DataFrame, extra: dict[str, NDArray[np.float64]], *, with_wear: bool
    ) -> None:
        drivers = np.array([str(v) for v in laps["driver_id"].tolist()])
        compounds = np.array([str(v) for v in laps["compound"].tolist()])
        self.drivers: list[str] = sorted({str(v) for v in drivers})
        self.compounds: list[str] = sorted({str(v) for v in compounds})
        columns = [np.asarray(drivers == d, dtype=np.float64) for d in self.drivers]
        columns += [np.asarray(compounds == c, dtype=np.float64) for c in self.compounds[1:]]
        self.wear_compounds = self.compounds if with_wear else []
        life = laps["tyre_life"].fillna(0).to_numpy(dtype=np.float64) if with_wear else None
        if life is not None:
            columns += [np.where(compounds == c, life, 0.0) for c in self.compounds]
        self.extra = list(extra)
        columns += [extra[name] for name in self.extra]
        self.matrix: NDArray[np.float64] = np.column_stack(columns).astype(np.float64)

    def unpack(
        self, beta: NDArray[np.float64], cov: NDArray[np.float64], sigma2: float, n: int
    ) -> SessionFit:
        """Name the fitted coefficients.

        Args:
            beta: Coefficients, in design-column order.
            cov: Their covariance.
            sigma2: Residual variance.
            n: Laps used.

        Returns:
            The session fit.
        """
        n_d, n_c = len(self.drivers), len(self.compounds)
        drivers = dict(zip(self.drivers, beta[:n_d].tolist(), strict=True))
        offsets = {self.compounds[0]: 0.0}
        offsets |= dict(zip(self.compounds[1:], beta[n_d : n_d + n_c - 1].tolist(), strict=True))
        start = n_d + n_c - 1
        wear = dict(
            zip(
                self.wear_compounds,
                beta[start : start + len(self.wear_compounds)].tolist(),
                strict=True,
            )
        )
        slope = None
        if self.extra:
            idx = len(beta) - 1
            slope = Estimate(float(beta[idx]), float(np.sqrt(cov[idx, idx])))
        return SessionFit(drivers, offsets, wear, slope, float(np.sqrt(sigma2)), n)
