"""A prior-only tyre model, for when there is too little history to fit one (P5-03).

Early in a backtest, or on a store with few clean race stints, no race gives a usable degradation
fit, but the simulator must still forecast. It then runs on linear wear per compound, the
league-wide pit-loss prior and default stop times, so strategy and track position still play a
part and every ``Predictor`` passes the same contract suite.
"""

from collections.abc import Mapping

import numpy as np

from f1sim.adapters.models.pace_corrections import FUEL_PRIOR, Estimate
from f1sim.adapters.models.pit_stops import PIT_LOSS_PRIOR, default_stop_times
from f1sim.adapters.models.tyres import (
    AGE_GRID,
    OFFSET_PRIOR,
    TEMPERATURE_PRIOR,
    CompoundCurve,
    DegradationModel,
    TyreModel,
    WarmupModel,
)

PRIOR_COMPOUNDS: Mapping[str, tuple[float, float]] = {
    "SOFT": (-0.6, 0.09),
    "MEDIUM": (0.0, 0.05),
    "HARD": (0.4, 0.03),
}
"""Compound to (pace offset, wear in seconds per lap of age) used when nothing can be fitted."""
PRIOR_WEAR_SE = 0.02
"""Standard error of the prior wear rate, seconds per lap of age."""


def default_tyre_model() -> TyreModel:
    """Build the prior-only tyre model.

    Returns:
        A tyre model built from ``PRIOR_COMPOUNDS``, ``PIT_LOSS_PRIOR`` and ``FUEL_PRIOR``.
    """
    curves = {
        compound: CompoundCurve(
            compound=compound,
            values=(rate * AGE_GRID).astype(np.float64),
            se=(PRIOR_WEAR_SE * AGE_GRID).astype(np.float64),
            offset=offset,
            offset_se=OFFSET_PRIOR[1],
            temperature=0.0,
            temperature_se=TEMPERATURE_PRIOR[1],
            n_laps=0,
        )
        for compound, (offset, rate) in PRIOR_COMPOUNDS.items()
    }
    return TyreModel(
        degradation=DegradationModel(tuple(curves), curves, {}),
        warmup=WarmupModel({}),
        pit_loss={},
        pit_loss_default=Estimate(*PIT_LOSS_PRIOR),
        stops=default_stop_times(),
        fuel_s_per_lap=FUEL_PRIOR[0],
    )
