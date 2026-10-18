"""Execute stops from each driver's sampled plan (P5-07).

A plan is, for each car and each stop, the lap to stop on and the compound to take; a lap of
``0`` means the stop does not happen. When the current lap matches a stop, the car pays its pit
loss and takes fresh tyres. The loss is sampled per stop by the simulator: the circuit's pit loss
(which already includes the time stationary) plus a draw of the team's stop-duration deviation,
slow-stop tail included (P4-05), resetting its tyre age so the next lap's loss starts
from the warm-up.
"""

import numpy as np

from f1sim.core.domain.race_state import RaceState, update_order
from f1sim.core.errors import InvariantError
from f1sim.core.ports.lap_step import LapContext

NAME = "pit_stop"
PLAN_DIMS = 3
"""A stop plan has three axes: simulations, cars and stop index."""
VSC_PIT_DISCOUNT_S = 4.0
"""Time a VSC saves on a stop, because the whole field is already slowed, seconds (P6-01)."""
SC_PIT_DISCOUNT_S = 8.0
"""Time a full safety car saves on a stop, seconds (P6-01)."""
VSC_LEVEL = 1
"""``RaceState.safety_car`` value for a virtual safety car."""
SAFETY_CAR_LEVEL = 2
"""``RaceState.safety_car`` value for a full safety car."""


class PitStopStep:
    """Apply each car's sampled stop plan on its stop laps (P5-07).

    Attributes:
        name: Step name used in configuration and ablation reports.
    """

    name = NAME

    def __init__(
        self,
        stop_laps: np.ndarray,
        stop_compounds: np.ndarray,
        pit_loss_s: np.ndarray,
    ) -> None:
        """Create the step from the sampled plans.

        Args:
            stop_laps: Array of shape ``(n_sims, n_cars, max_stops)``; a stop's lap, 0 for none.
            stop_compounds: Same shape; the compound code fitted at each stop.
            pit_loss_s: Same shape; time lost at each stop, seconds.

        Raises:
            InvariantError: If the plan arrays are not 3-D or their shapes disagree.
        """
        if stop_compounds.shape != stop_laps.shape or pit_loss_s.shape != stop_laps.shape:
            raise InvariantError("stop_compounds and pit_loss_s must match stop_laps")
        if stop_laps.ndim != PLAN_DIMS:
            raise InvariantError("stop_laps must be a 3-D array of (n_sims, n_cars, max_stops)")
        self.stop_laps = stop_laps
        self.stop_compounds = stop_compounds
        self.pit_loss_s = pit_loss_s

    def apply(self, state: RaceState, ctx: LapContext, rng: np.random.Generator) -> None:
        """Stop every car whose plan puts a stop on this lap.

        Args:
            state: Parallel race state, mutated in place.
            ctx: Facts about the race.
            rng: Seeded random generator; unused here, since the lost time was sampled up front.
        """
        del ctx, rng
        stopped = np.zeros(state.race_time_s.shape, dtype=bool)
        discount = self._discount(state)
        for k in range(self.stop_laps.shape[2]):
            stop = (self.stop_laps[:, :, k] == state.lap) & ~state.retired
            loss = np.where(stop, np.maximum(self.pit_loss_s[:, :, k] - discount, 0.0), 0.0)
            state.race_time_s = state.race_time_s + loss
            state.lap_time_s = state.lap_time_s + loss
            state.compound = np.where(stop, self.stop_compounds[:, :, k], state.compound)
            state.tyre_age = np.where(stop, 0, state.tyre_age)
            stopped |= stop
        state.pit_this_lap = stopped
        update_order(state)

    def _discount(self, state: RaceState) -> np.ndarray:
        """Stops cost less behind a safety car or VSC, since the field is already slowed (P6-01)."""
        if state.safety_car is None:
            return np.zeros(state.race_time_s.shape, dtype=np.float64)
        level = np.broadcast_to(state.safety_car[:, np.newaxis], state.race_time_s.shape)
        return np.where(
            level >= SAFETY_CAR_LEVEL,
            SC_PIT_DISCOUNT_S,
            np.where(level == VSC_LEVEL, VSC_PIT_DISCOUNT_S, 0.0),
        ).astype(np.float64)
