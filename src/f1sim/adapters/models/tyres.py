"""Tyre degradation, warm-up, pit loss and stop times (P4-01 to P4-05).

Raw race laps mix a car's pace with fuel and tyre age, like the pace model (P3-01) already
removes. What is left once drivers and fuel are corrected is lap-time loss against tyre age,
by compound. A truncated-power spline, with relative terms ``max(0, age - knot)``, can bend
late and so captures a cliff (P4-01). Each circuit's curve is pooled with the global curve
for that compound, so a rarely used track borrows strength (hierarchical pooling).

The first laps of a stint are held out of the degradation fit and measured separately as a
warm-up penalty (P4-02), because they are not a smooth function of age. Track temperature can
enter as an age interaction (P4-03) but is kept only when it improves held-out error.

Pit loss (P4-04) is the extra time of the stop's in- and out-laps over the driver's baseline,
measured from timing data and pooled per circuit; it already includes the time stationary. The
stop duration (P4-05) is the recorded pit-lane time, modelled per team with a slow-stop tail,
so the engine can add a team's deviation from the field mean without double counting the stop.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from f1sim.adapters.models.pace_corrections import (
    FUEL_PRIOR,
    MAD_TO_SD,
    OUTLIER_MADS,
    Estimate,
    dry_laps,
    pool,
)
from f1sim.adapters.models.pit_stops import (
    PIT_LOSS_PRIOR,
    StopTimeModel,
    fit_stop_times,
    measure_pit_loss,
)
from f1sim.core.errors import InvariantError

KNOTS = (5.0, 10.0, 15.0, 20.0, 25.0, 30.0)
"""Truncated-power spline knots; the last term is what lets a curve bend up into a cliff."""

MAX_TYRE_AGE = 40
AGE_GRID = np.arange(0.0, MAX_TYRE_AGE + 1.0, 1.0, dtype=np.float64)
"""Ages the pooled curves are stored and looked up on, in laps."""

WARMUP_LAPS = 3
"""Laps at the start of a stint (ages 1 and 2) are warm-up, not degradation."""

MIN_FIT_LAPS = 30
MAX_CHECK_AGE = 30
TEMPERATURE_REF_C = 40.0
TEMPERATURE_GAIN_MIN = 0.005
"""A temperature term must beat the plain fit by this many seconds to be kept (P4-03)."""

LOSS_PRIOR = (0.0, 0.5)
OFFSET_PRIOR = (0.0, 1.0)
TEMPERATURE_PRIOR = (0.0, 0.05)
"""Weak priors, in seconds (per lap for the temperature slope)."""

MIN_CHECK_AGES = 3
WET_SLACK_S = 5.0
"""A stint whose fastest median lap is this far off the driver's best is dropped as wet."""


@dataclass(frozen=True)
class CompoundCurve:
    """One compound's degradation curve.

    Attributes:
        compound: Compound name.
        values: Lap-time loss over ``AGE_GRID``, seconds, zero at age 0.
        se: Standard error of each value.
        offset: Compound pace level against the session median, seconds.
        offset_se: Standard error of the offset.
        temperature: Extra seconds per lap of age per degree Celsius away from the reference.
        temperature_se: Standard error of the temperature slope.
        n_laps: Laps behind the curve.
    """

    compound: str
    values: NDArray[np.float64]
    se: NDArray[np.float64]
    offset: float
    offset_se: float
    temperature: float
    temperature_se: float
    n_laps: int


@dataclass(frozen=True)
class RaceDegradation:
    """One race's degradation fit, before pooling.

    Attributes:
        circuit: Circuit the race was held at.
        curves: Compound to its fitted, uncollapsed curve.
        fuel_s: Fuel effect, seconds per lap.
        residual_sd: Residual standard deviation, seconds.
        n_laps: Laps used after outlier removal.
        warmup: ``(compound, age, residual)`` for the held-out warm-up laps.
    """

    circuit: str
    curves: dict[str, CompoundCurve]
    fuel_s: Estimate
    residual_sd: float
    n_laps: int
    warmup: tuple[tuple[str, int, float], ...]


@dataclass
class DegradationModel:
    """Pooled degradation curves, global and per circuit (P4-01).

    Attributes:
        compounds: Compounds with a global curve.
        global_curves: Compound to the all-circuits curve.
        circuit_curves: ``(circuit, compound)`` to the circuit's shrunk curve.
    """

    compounds: tuple[str, ...]
    global_curves: dict[str, CompoundCurve]
    circuit_curves: dict[tuple[str, str], CompoundCurve]

    def curve(self, compound: str, circuit: str | None = None) -> CompoundCurve:
        """Find the best curve for a compound and circuit.

        Args:
            compound: Compound name.
            circuit: Circuit ID; None or an unknown circuit uses the global curve.

        Returns:
            The circuit's curve if it has one, otherwise the global curve.

        Raises:
            InvariantError: If the compound has no curve at all.
        """
        if circuit is not None and (circuit, compound) in self.circuit_curves:
            return self.circuit_curves[(circuit, compound)]
        if compound in self.global_curves:
            return self.global_curves[compound]
        raise InvariantError(f"no degradation curve for {compound!r}")

    def loss(
        self, compound: str, circuit: str | None, age: float, track_temp: float | None
    ) -> float:
        """Lap-time loss of a tyre at one age.

        Args:
            compound: Compound name.
            circuit: Circuit ID.
            age: Tyre age in laps.
            track_temp: Track temperature in Celsius, or None to ignore it.

        Returns:
            Loss in seconds, zero at age 0.
        """
        curve = self.curve(compound, circuit)
        clamped = min(max(age, 0.0), float(MAX_TYRE_AGE))
        value = float(np.interp(clamped, AGE_GRID, curve.values))
        if track_temp is not None:
            value += curve.temperature * (track_temp - TEMPERATURE_REF_C) * clamped
        return value

    def offset(self, compound: str) -> float:
        """Pace level of a compound against the session median.

        Args:
            compound: Compound name.

        Returns:
            Offset in seconds; lower is faster.
        """
        return self.curve(compound, None).offset


@dataclass(frozen=True)
class WarmupModel:
    """Warm-up penalty per compound, for the first laps of a stint (P4-02).

    Attributes:
        penalties: Compound to its penalty at ages 1, 2, ... seconds.
    """

    penalties: dict[str, tuple[float, ...]]

    def penalty(self, compound: str, age: float) -> float:
        """Penalty of a compound at one warm-up age.

        Args:
            compound: Compound name.
            age: Tyre age in laps, below ``WARMUP_LAPS``.

        Returns:
            Seconds slower than the degradation curve, or 0 outside the warm-up ages.
        """
        index = int(age) - 1
        values = self.penalties.get(compound)
        if values is None or not 0 <= index < len(values):
            return 0.0
        return values[index]


@dataclass(frozen=True)
class SampledLoss:
    """Lap-time loss drawn per simulation, ready for the engine (P5-03).

    The strategy and pace models work with point estimates; the engine needs whole loss curves,
    one draw per simulated race, so that uncertainty in degradation is felt in the forecast.

    Attributes:
        compounds: Compound names; index ``k`` addresses ``values[k]``.
        values: Array of shape ``(n_compounds, n_sims, len(AGE_GRID))``, seconds, zero at age 0.
    """

    compounds: tuple[str, ...]
    values: NDArray[np.float64]

    def compound_index(self, compound: str) -> int:
        """Position of a compound in ``compounds``.

        Args:
            compound: Compound name.

        Returns:
            The index used to address ``values``.

        Raises:
            InvariantError: If the compound was not sampled.
        """
        try:
            return self.compounds.index(compound)
        except ValueError as error:
            raise InvariantError(f"compound {compound!r} was not sampled") from error


@dataclass
class TyreModel:
    """Everything the strategy and engine need about tyres and stops (P4-01 to P4-05).

    Attributes:
        degradation: Pooled degradation curves.
        warmup: Warm-up penalties.
        pit_loss: Circuit ID to pit loss.
        pit_loss_default: Pit loss for an unknown circuit.
        stops: Stop-duration distributions.
        fuel_s_per_lap: Fuel effect, seconds per lap.
    """

    degradation: DegradationModel
    warmup: WarmupModel
    pit_loss: dict[str, Estimate]
    pit_loss_default: Estimate
    stops: StopTimeModel
    fuel_s_per_lap: float

    def loss(
        self,
        compound: str | None,
        circuit: str | None,
        age: float,
        track_temp: float | None = None,
    ) -> float:
        """Lap-time loss at one age, including the warm-up penalty.

        Args:
            compound: Compound name, or None for no loss.
            circuit: Circuit ID.
            age: Tyre age in laps.
            track_temp: Track temperature in Celsius, or None.

        Returns:
            Loss in seconds.
        """
        if compound is None:
            return 0.0
        if age < WARMUP_LAPS:
            return self.warmup.penalty(compound, age)
        return self.degradation.loss(compound, circuit, age, track_temp)

    def pit_loss_s(self, circuit: str | None) -> float:
        """Pit loss at a circuit, including the time stationary.

        Args:
            circuit: Circuit ID.

        Returns:
            Loss in seconds.
        """
        if circuit is not None and circuit in self.pit_loss:
            return self.pit_loss[circuit].value
        return self.pit_loss_default.value

    def compound_offset(self, compound: str) -> float:
        """Pace level of a compound against the session median.

        Args:
            compound: Compound name.

        Returns:
            Offset in seconds; lower is faster.
        """
        return self.degradation.offset(compound)

    def stop_time_s(self, team: str) -> float:
        """Mean stop duration for a team.

        Args:
            team: Team ID.

        Returns:
            Seconds.
        """
        return self.stops.distribution(team).mean()

    def stop_offset_s(self, team: str) -> float:
        """A team's mean stop-duration deviation from the field.

        The pit loss already includes the stop, so strategy scoring adds this offset rather
        than a whole stop duration.

        Args:
            team: Team ID.

        Returns:
            Seconds; negative for a team that stops faster than the field.
        """
        return self.stops.distribution(team).mean() - self.stops.default.mean()

    def sample_loss(
        self,
        rng: np.random.Generator,
        n_sims: int,
        *,
        circuit: str | None = None,
        compounds: Sequence[str] | None = None,
    ) -> SampledLoss:
        """Draw whole loss curves, one per simulation, for the engine (P5-03).

        Each age's loss is its fitted value plus Gaussian noise scaled by the curve's standard
        error, so a compound whose curve is well measured barely moves. Warm-up ages are held
        out of the fit and so are drawn without noise, and no loss may go below zero.

        Args:
            rng: Random generator.
            n_sims: Number of draws per compound.
            circuit: Circuit ID for the circuit's curved curve; None uses the global curve.
            compounds: Compounds to sample; defaults to every compound with a global curve.

        Returns:
            The sampled loss curves.

        Raises:
            InvariantError: If a requested compound has no curve.
        """
        names = tuple(compounds) if compounds is not None else self.degradation.compounds
        values = np.empty((len(names), n_sims, AGE_GRID.shape[0]), dtype=np.float64)
        for k, compound in enumerate(names):
            curve = self.degradation.curve(compound, circuit)
            mean = np.array(
                [self.loss(compound, circuit, float(age)) for age in AGE_GRID], dtype=np.float64
            )
            se = curve.se.copy()
            se[:WARMUP_LAPS] = 0.0
            draws = mean[np.newaxis, :] + rng.standard_normal((n_sims, AGE_GRID.shape[0])) * se
            draws[:, 0] = 0.0
            values[k] = np.clip(draws, 0.0, None)
        return SampledLoss(names, values)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the fitted model for the state store.

        Returns:
            A JSON-friendly mapping.
        """
        return {
            "fuel_s_per_lap": self.fuel_s_per_lap,
            "degradation": {
                "compounds": list(self.degradation.compounds),
                "global": {c: _curve_dict(v) for c, v in self.degradation.global_curves.items()},
                "circuits": {
                    f"{circuit}|{compound}": _curve_dict(curve)
                    for (circuit, compound), curve in self.degradation.circuit_curves.items()
                },
            },
            "warmup": {c: list(v) for c, v in self.warmup.penalties.items()},
            "pit_loss": {c: [e.value, e.se] for c, e in self.pit_loss.items()},
            "pit_loss_default": [self.pit_loss_default.value, self.pit_loss_default.se],
            "stops": self.stops.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TyreModel":
        """Rebuild a model saved with ``to_dict``.

        Args:
            data: The saved mapping.

        Returns:
            The tyre model.
        """
        degradation = cast(Mapping[str, Any], data["degradation"])
        global_curves = {
            str(c): _curve_from_dict(str(c), v)
            for c, v in cast(Mapping[str, Any], degradation["global"]).items()
        }
        circuit_curves: dict[tuple[str, str], CompoundCurve] = {}
        for key, v in cast(Mapping[str, Any], degradation["circuits"]).items():
            circuit, _, compound = str(key).partition("|")
            circuit_curves[(circuit, compound)] = _curve_from_dict(compound, v)
        warmup = {
            str(c): tuple(float(x) for x in v)
            for c, v in cast(Mapping[str, Any], data["warmup"]).items()
        }
        return cls(
            degradation=DegradationModel(
                compounds=tuple(str(c) for c in cast(Sequence[Any], degradation["compounds"])),
                global_curves=global_curves,
                circuit_curves=circuit_curves,
            ),
            warmup=WarmupModel(warmup),
            pit_loss={
                str(c): Estimate(float(v[0]), float(v[1]))
                for c, v in cast(Mapping[str, Any], data["pit_loss"]).items()
            },
            pit_loss_default=_estimate(data["pit_loss_default"]),
            stops=StopTimeModel.from_dict(cast(Mapping[str, Any], data["stops"])),
            fuel_s_per_lap=float(data["fuel_s_per_lap"]),
        )


@dataclass(frozen=True)
class TemperatureCheck:
    """Whether a temperature interaction earns its place (P4-03).

    Attributes:
        rmse_without: Held-out stint RMSE without the term, seconds.
        rmse_with: Held-out stint RMSE with the term, seconds.
        warranted: Whether the improvement clears ``TEMPERATURE_GAIN_MIN``.
    """

    rmse_without: float
    rmse_with: float
    warranted: bool


def race_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Keep green-flag clean dry race laps with a known tyre age.

    Args:
        laps: Canonical ``laps`` rows.

    Returns:
        The race laps used to fit degradation.
    """
    race = laps.loc[laps["session"] == "R"].copy()
    dry = dry_laps(race)
    green = dry["track_status"].astype(str) == "1"
    accurate = dry["is_accurate"].fillna(False).astype(bool)
    keep = dry["tyre_life"].notna() & (dry["tyre_life"] >= 1.0) & green & accurate
    return dry.loc[keep]


def fit_race_degradation(  # pylint: disable=too-many-locals
    laps: pd.DataFrame, circuit: str, *, temperature: bool = False
) -> RaceDegradation | None:
    """Fit one race: driver and compound levels, fuel, and a spline per compound (P4-01).

    Args:
        laps: One race's ``laps`` rows.
        circuit: Circuit the race was held at.
        temperature: Whether to add a track-temperature age interaction (P4-03).

    Returns:
        The fit, or None when there are too few laps or the design is degenerate.
    """
    frame = race_laps(laps)
    if len(frame) < MIN_FIT_LAPS:
        return None
    ages = frame["tyre_life"].to_numpy(dtype=np.float64)
    fit_mask = ages >= WARMUP_LAPS
    if int(fit_mask.sum()) < MIN_FIT_LAPS:
        return None
    frame = _drop_unfitted(frame, fit_mask)
    if len(frame) < MIN_FIT_LAPS:
        return None
    ages = frame["tyre_life"].to_numpy(dtype=np.float64)
    fit_mask = ages >= WARMUP_LAPS
    y = frame["lap_time_s"].to_numpy(dtype=np.float64)
    y = y - float(np.median(y))
    design = _RaceDesign(frame, temperature=temperature)
    fit = _robust_fit(design.matrix, y, fit_mask)
    if fit is None:
        return None
    beta, cov, sigma2, n_used = fit
    residuals = y - design.matrix @ beta
    warmup = tuple(
        (design.row_compounds[i], int(ages[i]), float(residuals[i]))
        for i in np.flatnonzero(~fit_mask).tolist()
    )
    fuel_idx = design.n_driver_cols + len(design.compounds)
    fuel = Estimate(-float(beta[fuel_idx]), float(np.sqrt(cov[fuel_idx, fuel_idx])))
    return RaceDegradation(
        circuit=circuit,
        curves=design.unpack(beta, cov, n_used),
        fuel_s=fuel,
        residual_sd=float(np.sqrt(sigma2)),
        n_laps=n_used,
        warmup=warmup,
    )


def fit_degradation(races: Sequence[RaceDegradation]) -> DegradationModel:
    """Pool race curves into global and per-circuit curves (P4-01).

    Args:
        races: Fitted races.

    Returns:
        The pooled model.

    Raises:
        InvariantError: If there is nothing to pool.
    """
    compounds = sorted({c for r in races for c in r.curves})
    if not compounds:
        raise InvariantError("no degradation fits to pool")
    global_curves = {
        c: _pool_curve(c, [r.curves[c] for r in races if c in r.curves], _flat_prior(LOSS_PRIOR))
        for c in compounds
    }
    keys = sorted({(r.circuit, c) for r in races for c in r.curves})
    circuit_curves = {
        (circuit, c): _pool_curve(
            c,
            [r.curves[c] for r in races if r.circuit == circuit and c in r.curves],
            _global_prior(global_curves[c]),
        )
        for circuit, c in keys
    }
    return DegradationModel(tuple(compounds), global_curves, circuit_curves)


def fit_warmup(races: Sequence[RaceDegradation]) -> WarmupModel:
    """Pool the held-out warm-up residuals by compound and age (P4-02).

    Args:
        races: Fitted races.

    Returns:
        The warm-up penalties.
    """
    buckets: dict[str, dict[int, list[float]]] = {}
    for race in races:
        for compound, age, residual in race.warmup:
            buckets.setdefault(compound, {}).setdefault(age, []).append(residual)
    penalties = {
        compound: tuple(_median(by_age.get(age, [])) for age in range(1, WARMUP_LAPS))
        for compound, by_age in buckets.items()
    }
    return WarmupModel(penalties)


def fit_degradation_from_laps(
    laps: pd.DataFrame, events: pd.DataFrame, *, temperature: bool = False
) -> DegradationModel:
    """Fit the pooled degradation model from laps and events.

    Args:
        laps: Canonical ``laps`` rows.
        events: Canonical ``events`` rows, for ``circuit_id``.
        temperature: Whether to include a track-temperature interaction (P4-03).

    Returns:
        The pooled model.

    Raises:
        InvariantError: If no race gave a usable fit.
    """
    races = _race_degradations(laps, events, temperature=temperature)
    if not races:
        raise InvariantError("no race gave a usable degradation fit")
    return fit_degradation(races)


def fit_tyre_model(
    laps: pd.DataFrame,
    events: pd.DataFrame,
    pit_stops: pd.DataFrame,
    *,
    temperature: bool = False,
) -> TyreModel:
    """Fit degradation, warm-up, pit loss and stop times together (P4-01 to P4-05).

    Args:
        laps: Canonical ``laps`` rows.
        events: Canonical ``events`` rows.
        pit_stops: Canonical ``pit_stops`` rows.
        temperature: Whether to include a track-temperature interaction (P4-03).

    Returns:
        The fitted tyre model.

    Raises:
        InvariantError: If no race gave a usable degradation fit.
    """
    races = _race_degradations(laps, events, temperature=temperature)
    if not races:
        raise InvariantError("no race gave a usable degradation fit")
    pit_loss = measure_pit_loss(laps, events)
    fallback = pool(list(pit_loss.values()), PIT_LOSS_PRIOR)
    fuel = pool([r.fuel_s for r in races], FUEL_PRIOR)
    return TyreModel(
        degradation=fit_degradation(races),
        warmup=fit_warmup(races),
        pit_loss=pit_loss,
        pit_loss_default=fallback,
        stops=fit_stop_times(pit_stops, laps),
        fuel_s_per_lap=fuel.value,
    )


def held_out_stint_rmse(  # pylint: disable=too-many-locals
    laps: pd.DataFrame,
    events: pd.DataFrame,
    model: DegradationModel,
    seasons: Sequence[int],
    *,
    fuel_s_per_lap: float = 0.0,
) -> float:
    """Root-mean-square error of the model's degradation shape on some seasons (gate).

    Only green-flag dry laps are used. Each held-out driver stint is measured against its own
    youngest age, so driver and team pace cancel, and a stint whose fastest lap is far from the
    driver's best (a wet or interrupted stint) is dropped. The observed loss at each age is
    corrected for the fuel burnt since that reference, then compared with the model's own loss at
    that age measured against the same reference, so only the age shape is judged. Each stint's
    median track temperature is passed to the model, so a fitted temperature term is judged too.
    Errors beyond ``OUTLIER_MADS`` robust deviations are trimmed before the RMSE.

    Args:
        laps: Canonical ``laps`` rows. A ``track_temp_c`` column is used when present.
        events: Canonical ``events`` rows.
        model: A fitted degradation model.
        seasons: Seasons to measure.
        fuel_s_per_lap: Fuel effect, seconds per lap, removed from the observed loss.

    Returns:
        The RMSE in seconds; ``nan`` when there is no measurable stint (so the gate cannot pass
        on an empty holdout).
    """
    circuit_of = {
        (int(r["season"]), int(r["round"])): str(r["circuit_id"]) for r in _records(events)
    }
    frame = race_laps(laps)
    frame = frame.loc[frame["season"].isin(list(seasons)) & frame["stint"].notna()]
    best = frame.groupby(["season", "round", "driver_id"])["lap_time_s"].min()
    buckets: dict[tuple[int, int, str, int, str], dict[int, list[float]]] = {}
    temps: dict[tuple[int, int, str, int, str], list[float]] = {}
    for row in _records(frame):
        key = (
            int(row["season"]),
            int(row["round"]),
            str(row["driver_id"]),
            int(row["stint"]),
            str(row["compound"]).upper(),
        )
        buckets.setdefault(key, {}).setdefault(int(row["tyre_life"]), []).append(
            float(row["lap_time_s"])
        )
        temperature = row.get("track_temp_c")
        if temperature is not None and not pd.isna(temperature):
            temps.setdefault(key, []).append(float(temperature))
    errors: list[float] = []
    for key, by_age in buckets.items():
        season, round_, driver, _stint, compound = key
        circuit = circuit_of.get((season, round_))
        if circuit is None or compound not in model.compounds or len(by_age) < MIN_CHECK_AGES:
            continue
        medians = {age: float(np.median(times)) for age, times in by_age.items()}
        ref_age = min(medians)
        if medians[ref_age] > float(best.loc[(season, round_, driver)]) + WET_SLACK_S:
            continue
        stint_temperature = float(np.median(temps[key])) if temps.get(key) else None
        ref_model = model.loss(compound, circuit, float(ref_age), stint_temperature)
        for age, median in medians.items():
            if age < 1 or age > MAX_CHECK_AGE:
                continue
            observed = median - medians[ref_age] + fuel_s_per_lap * (age - ref_age)
            predicted = model.loss(compound, circuit, float(age), stint_temperature) - ref_model
            errors.append(predicted - observed)
    if not errors:
        return float("nan")
    return _robust_rmse(np.asarray(errors, dtype=np.float64))


def _robust_rmse(errors: NDArray[np.float64]) -> float:
    median = float(np.median(errors))
    mad = float(np.median(np.abs(errors - median)))
    keep = np.abs(errors - median) <= OUTLIER_MADS * MAD_TO_SD * mad
    if not keep.any():
        return float(np.sqrt(np.mean(np.square(errors))))
    return float(np.sqrt(np.mean(np.square(errors[keep]))))


def temperature_gain(
    laps: pd.DataFrame,
    events: pd.DataFrame,
    *,
    train_seasons: Sequence[int],
    holdout_seasons: Sequence[int],
) -> TemperatureCheck:
    """Compare held-out stint RMSE with and without the temperature term (P4-03).

    Args:
        laps: Canonical ``laps`` rows, including ``track_temp_c`` if it is to be tested.
        events: Canonical ``events`` rows.
        train_seasons: Seasons the models are fitted on.
        holdout_seasons: Seasons the models are measured on.

    Returns:
        The two RMSEs and whether the term clears ``TEMPERATURE_GAIN_MIN``.
    """
    train = laps.loc[laps["season"].isin(list(train_seasons))]
    plain = fit_degradation_from_laps(train, events, temperature=False)
    without = held_out_stint_rmse(laps, events, plain, holdout_seasons)
    warm = fit_degradation_from_laps(train, events, temperature=True)
    with_temp = held_out_stint_rmse(laps, events, warm, holdout_seasons)
    return TemperatureCheck(without, with_temp, without - with_temp > TEMPERATURE_GAIN_MIN)


def attach_track_temperature(laps: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Copy each lap's nearest track temperature onto it (P4-03).

    Laps and weather rows without a timestamp are left unmatched (a NaN temperature), because
    ``merge_asof`` needs non-null, sorted keys.

    Args:
        laps: Canonical ``laps`` rows with ``lap_start_utc``.
        weather: Canonical ``weather`` rows with ``time_utc`` and ``track_temp_c``.

    Returns:
        The laps with a ``track_temp_c`` column, matched backwards in time within a session.
    """
    left = laps.sort_values("lap_start_utc").reset_index(drop=True)
    left["track_temp_c"] = np.nan
    columns = ["season", "round", "session", "time_utc", "track_temp_c"]
    right = (
        weather.loc[weather["track_temp_c"].notna() & weather["time_utc"].notna(), columns]
        .sort_values("time_utc")
        .reset_index(drop=True)
    )
    if right.empty:
        return left
    keyed = left.loc[left["lap_start_utc"].notna()].drop(columns=["track_temp_c"]).copy()
    keyed["_row"] = keyed.index
    merged = pd.merge_asof(
        keyed,
        right,
        left_on="lap_start_utc",
        right_on="time_utc",
        by=["season", "round", "session"],
        direction="backward",
    )
    left.loc[merged["_row"], "track_temp_c"] = merged["track_temp_c"].to_numpy()
    return left


def _race_degradations(
    laps: pd.DataFrame, events: pd.DataFrame, *, temperature: bool
) -> list[RaceDegradation]:
    circuit_of = {
        (int(r["season"]), int(r["round"])): str(r["circuit_id"]) for r in _records(events)
    }
    races: list[RaceDegradation] = []
    race = laps.loc[laps["session"] == "R"]
    for (season, round_), group in race.groupby(["season", "round"], sort=True):
        circuit = circuit_of.get((_as_int(season), _as_int(round_)))
        if circuit is None:
            continue
        fit = fit_race_degradation(group, circuit, temperature=temperature)
        if fit is not None:
            races.append(fit)
    return races


def _drop_unfitted(frame: pd.DataFrame, fit_mask: NDArray[np.bool_]) -> pd.DataFrame:
    compounds = {
        str(c)
        for c, keep in zip(frame["compound"].tolist(), fit_mask.tolist(), strict=True)
        if keep
    }
    drivers = {
        str(d)
        for d, keep in zip(frame["driver_id"].tolist(), fit_mask.tolist(), strict=True)
        if keep
    }
    keep = [
        str(c) in compounds and str(d) in drivers
        for c, d in zip(frame["compound"].tolist(), frame["driver_id"].tolist(), strict=True)
    ]
    return frame.iloc[np.flatnonzero(np.asarray(keep, dtype=bool))]


def _basis(ages: NDArray[np.float64]) -> NDArray[np.float64]:
    columns = [ages, *(np.maximum(ages - knot, 0.0) for knot in KNOTS)]
    return np.column_stack(columns).astype(np.float64)


class _RaceDesign:
    """Dummy-coded design for one race: drivers, compound levels, fuel, splines, temperature."""

    def __init__(self, frame: pd.DataFrame, *, temperature: bool) -> None:
        self.drivers = sorted({str(d) for d in frame["driver_id"].tolist()})
        self.compounds = sorted({str(c) for c in frame["compound"].tolist()})
        ages = frame["tyre_life"].to_numpy(dtype=np.float64)
        lap = frame["lap_number"].to_numpy(dtype=np.float64)
        comp = np.array([str(c) for c in frame["compound"].tolist()], dtype=object)
        drv = np.array([str(d) for d in frame["driver_id"].tolist()], dtype=object)
        self.row_compounds = [str(c) for c in frame["compound"].tolist()]
        self.n_basis = 1 + len(KNOTS)
        basis = _basis(ages)
        self.n_driver_cols = len(self.drivers) - 1
        columns = [np.asarray(drv == d, dtype=np.float64) for d in self.drivers[1:]]
        columns += [np.asarray(comp == c, dtype=np.float64) for c in self.compounds]
        columns.append(lap)
        self.spline_start = len(columns)
        for compound in self.compounds:
            block = np.where((comp == compound)[:, None], basis, 0.0)
            columns.extend(block[:, k] for k in range(self.n_basis))
        self.temperature_start = len(columns)
        self.has_temperature = (
            temperature
            and "track_temp_c" in frame.columns
            and bool(frame["track_temp_c"].notna().any())
        )
        if self.has_temperature:
            centered = (
                frame["track_temp_c"].fillna(TEMPERATURE_REF_C).to_numpy(dtype=np.float64)
                - TEMPERATURE_REF_C
            )
            for compound in self.compounds:
                columns.append(np.where(comp == compound, centered * ages, 0.0))
        self.matrix = np.column_stack(columns).astype(np.float64)

    def unpack(  # pylint: disable=too-many-locals
        self, beta: NDArray[np.float64], cov: NDArray[np.float64], n_laps: int
    ) -> dict[str, CompoundCurve]:
        """Turn the fitted coefficients into a curve per compound.

        Args:
            beta: Fitted coefficients.
            cov: Their covariance.
            n_laps: Laps used, recorded on each curve.

        Returns:
            Compound to its curve on ``AGE_GRID``.
        """
        grid = _basis(AGE_GRID)
        curves: dict[str, CompoundCurve] = {}
        n_d = self.n_driver_cols
        for i, compound in enumerate(self.compounds):
            start = self.spline_start + i * self.n_basis
            idx = np.arange(start, start + self.n_basis)
            block = cov[np.ix_(idx, idx)]
            values = grid @ beta[idx]
            variance = np.einsum("ij,jk,ik->i", grid, block, grid)
            temperature, temperature_se = 0.0, 0.0
            if self.has_temperature:
                j = self.temperature_start + i
                temperature = float(beta[j])
                temperature_se = float(np.sqrt(max(cov[j, j], 0.0)))
            curves[compound] = CompoundCurve(
                compound=compound,
                values=values.astype(np.float64),
                se=np.sqrt(np.clip(variance, 0.0, None)).astype(np.float64),
                offset=float(beta[n_d + i]),
                offset_se=float(np.sqrt(max(cov[n_d + i, n_d + i], 0.0))),
                temperature=temperature,
                temperature_se=temperature_se,
                n_laps=n_laps,
            )
        return curves


def _robust_fit(
    matrix: NDArray[np.float64], y: NDArray[np.float64], mask: NDArray[np.bool_]
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, int] | None:
    x = matrix[mask]
    target = y[mask]
    if x.shape[0] <= x.shape[1] or np.linalg.matrix_rank(x) < x.shape[1]:
        return None
    beta = _least_squares(x, target)
    keep = np.ones(len(target), dtype=bool)
    for _ in range(2):
        resid = target - x @ beta
        mad = float(np.median(np.abs(resid[keep] - np.median(resid[keep]))))
        keep = np.abs(resid) <= OUTLIER_MADS * MAD_TO_SD * max(mad, 1e-6)
        beta = _least_squares(x[keep], target[keep])
    xk = x[keep]
    dof = int(keep.sum()) - xk.shape[1]
    if dof <= 0 or np.linalg.matrix_rank(xk) < xk.shape[1]:
        return None
    resid = target[keep] - xk @ beta
    sigma2 = float(resid @ resid) / dof
    cov = np.linalg.inv(xk.T @ xk) * sigma2
    return beta, cov, sigma2, int(keep.sum())


def _least_squares(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    solution: Any = np.linalg.lstsq(x, y, rcond=None)[0]
    return np.asarray(solution, dtype=np.float64)


def _pool_curve(
    compound: str, items: Sequence[CompoundCurve], prior_of: Callable[[int], tuple[float, float]]
) -> CompoundCurve:
    values = np.empty(AGE_GRID.shape[0], dtype=np.float64)
    se = np.empty(AGE_GRID.shape[0], dtype=np.float64)
    for g in range(AGE_GRID.shape[0]):
        estimate = pool(
            [Estimate(float(it.values[g]), float(it.se[g])) for it in items], prior_of(g)
        )
        values[g], se[g] = estimate.value, estimate.se
    values[0], se[0] = 0.0, 0.0
    offset = pool([Estimate(it.offset, it.offset_se) for it in items], OFFSET_PRIOR)
    temperature = pool(
        [Estimate(it.temperature, it.temperature_se) for it in items], TEMPERATURE_PRIOR
    )
    return CompoundCurve(
        compound=compound,
        values=values,
        se=se,
        offset=offset.value,
        offset_se=offset.se,
        temperature=temperature.value,
        temperature_se=temperature.se,
        n_laps=sum(it.n_laps for it in items),
    )


def _flat_prior(prior: tuple[float, float]) -> Callable[[int], tuple[float, float]]:
    def _prior_of(_g: int) -> tuple[float, float]:
        return prior

    return _prior_of


def _global_prior(global_curve: CompoundCurve) -> Callable[[int], tuple[float, float]]:
    def _prior_of(g: int) -> tuple[float, float]:
        return float(global_curve.values[g]), max(float(global_curve.se[g]), 1e-3)

    return _prior_of


def _median(values: Sequence[float]) -> float:
    return float(np.median(values)) if values else 0.0


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], frame.to_dict("records"))


def _as_int(value: object) -> int:
    return int(cast(Any, value))


def _curve_dict(curve: CompoundCurve) -> dict[str, Any]:
    return {
        "values": curve.values.tolist(),
        "se": curve.se.tolist(),
        "offset": curve.offset,
        "offset_se": curve.offset_se,
        "temperature": curve.temperature,
        "temperature_se": curve.temperature_se,
        "n_laps": curve.n_laps,
    }


def _curve_from_dict(compound: str, data: Any) -> CompoundCurve:
    record = cast(Mapping[str, Any], data)
    return CompoundCurve(
        compound=compound,
        values=np.asarray(record["values"], dtype=np.float64),
        se=np.asarray(record["se"], dtype=np.float64),
        offset=float(record["offset"]),
        offset_se=float(record["offset_se"]),
        temperature=float(record["temperature"]),
        temperature_se=float(record["temperature_se"]),
        n_laps=int(record["n_laps"]),
    )


def _estimate(data: Any) -> Estimate:
    values = cast(Sequence[Any], data)
    return Estimate(float(values[0]), float(values[1]))
