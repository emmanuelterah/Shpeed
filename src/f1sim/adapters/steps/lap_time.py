"""Sample lap times from posterior draws (P5-03).

A lap time is four additive pieces: the car's clean pace, its compound's pace offset, its tyre loss
at the current age, and a fuel term, plus lap-to-lap noise. Pace and the tyre loss curves are drawn
once per simulation (so the whole race for one run is consistent), while the noise is drawn fresh
every lap. Fuel is common to every car, so it shifts the absolute times without touching the order;
the tyre loss is not, which is what lets a fresher car pull away.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from f1sim.adapters.models.tyres import MAX_TYRE_AGE, SampledLoss
from f1sim.core.domain.race_state import RaceState, update_order
from f1sim.core.errors import InvariantError
from f1sim.core.ports.lap_step import LapContext

NAME = "lap_time"
DEFAULT_NOISE_S = 0.25
"""Lap-to-lap noise standard deviation, seconds, when no fitted figure is given."""
MATRIX_DIMS = 2
"""A per-car matrix has two axes, simulations and cars."""


class LapTimeStep:
    """Add one lap's time to every car and advance its tyres and fuel (P5-03).

    Attributes:
        name: Step name used in configuration and ablation reports.
    """

    name = NAME

    def __init__(  # pylint: disable=too-many-arguments
        self,
        pace_s: NDArray[np.float64],
        loss: SampledLoss,
        *,
        fuel_s_per_lap: float,
        burn_kg_per_lap: float,
        noise_s: float = DEFAULT_NOISE_S,
        reference_fuel_kg: float = 0.0,
        compound_offset_s: NDArray[np.float64] | None = None,
    ) -> None:
        """Create the step from the per-run draws.

        Args:
            pace_s: Array of shape ``(n_sims, n_cars)``: each car's clean-lap pace, seconds.
            loss: Sampled tyre-loss curves.
            fuel_s_per_lap: Seconds of lap time per kilogram of fuel on board.
            burn_kg_per_lap: Fuel burnt each lap, kilograms.
            noise_s: Lap-to-lap noise standard deviation, seconds.
            reference_fuel_kg: Fuel load at which ``pace_s`` is defined, kilograms.
            compound_offset_s: Pace offset per compound code in ``loss`` order, seconds (lower
                is faster); zero for every compound when None.

        Raises:
            InvariantError: If the pace and fuel figures are implausible, or the compound
                offsets do not match the sampled compounds.
        """
        if pace_s.ndim != MATRIX_DIMS:
            raise InvariantError("pace_s must be a 2-D array of (n_sims, n_cars)")
        if np.any(~np.isfinite(pace_s)):
            raise InvariantError("pace_s must be finite")
        if fuel_s_per_lap < 0.0 or burn_kg_per_lap < 0.0 or noise_s < 0.0:
            raise InvariantError("fuel and noise figures must be non-negative")
        n_compounds = len(loss.compounds)
        offsets = (
            np.zeros(n_compounds, dtype=np.float64)
            if compound_offset_s is None
            else np.asarray(compound_offset_s, dtype=np.float64)
        )
        if offsets.shape != (n_compounds,):
            raise InvariantError("compound_offset_s needs one value per sampled compound")
        self.compound_offset_s = offsets
        self.pace_s = pace_s
        self.loss = loss
        self.fuel_s_per_lap = fuel_s_per_lap
        self.burn_kg_per_lap = burn_kg_per_lap
        self.noise_s = noise_s
        self.reference_fuel_kg = reference_fuel_kg

    def apply(self, state: RaceState, ctx: LapContext, rng: np.random.Generator) -> None:
        """Time one lap for every car, then burn fuel and age the tyres.

        Args:
            state: Parallel race state, mutated in place.
            ctx: Facts about the race.
            rng: Seeded random generator.
        """
        del ctx
        tyre = self._tyre_loss(state) + self.compound_offset_s[state.compound]
        fuel = self.fuel_s_per_lap * (state.fuel_kg - self.reference_fuel_kg)
        noise = rng.normal(0.0, self.noise_s, size=state.race_time_s.shape)
        lap = np.where(state.retired, 0.0, self.pace_s + tyre + fuel + noise)
        state.lap_time_s = lap
        state.race_time_s = np.where(state.retired, np.inf, state.race_time_s + lap)
        state.fuel_kg = np.maximum(state.fuel_kg - self.burn_kg_per_lap, 0.0)
        state.tyre_age = state.tyre_age + 1
        update_order(state)

    def _tyre_loss(self, state: RaceState) -> np.ndarray:
        sims = np.arange(state.n_sims, dtype=np.int64).reshape(-1, 1)
        ages = np.minimum(state.tyre_age + 1, MAX_TYRE_AGE)
        loss = np.zeros(state.race_time_s.shape, dtype=np.float64)
        for k in range(self.loss.values.shape[0]):
            piece = self.loss.values[k][sims, ages]
            loss = np.where(state.compound == k, piece, loss)
        return loss


def sampled_compounds(loss: SampledLoss, compounds: Sequence[str]) -> np.ndarray:
    """Encode compound names as the indices ``LapTimeStep`` indexes its loss curves with.

    Args:
        loss: The sampled loss curves; its compound order defines the codes.
        compounds: Compound names, one per car in ``state`` column order.

    Returns:
        One integer code per named compound.
    """
    return np.array([loss.compound_index(name) for name in compounds], dtype=np.int64)
