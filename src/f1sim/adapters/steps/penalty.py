"""In-race time penalties (P6-06, a should-have).

A driver can pick up a five- or ten-second penalty for a driving infringement. The severity is not
in the model's read set (race-control messages carry it, but fit a hazard from them would need the
race-session feed and its scope), so the hazard is a prior: a chance per car that it is given a
penalty on a sampled lap, and the seconds added to its cumulative time there. Penalties are served,
not deleted, so they only cost a place when the car is close enough for the seconds to matter.
"""

import numpy as np
from numpy.typing import NDArray

from f1sim.core.domain.race_state import RaceState, update_order
from f1sim.core.errors import InvariantError
from f1sim.core.ports.lap_step import LapContext

NAME = "penalties"
PLAN_DIMS = 2
"""A penalty plan has two axes, simulations and cars."""


class PenaltyStep:
    """Add each car's pre-sampled time penalty on its lap (P6-06).

    Attributes:
        name: Step name used in configuration and ablation reports.
    """

    name = NAME

    def __init__(self, penalty_lap: NDArray[np.int_], penalty_s: NDArray[np.float64]) -> None:
        """Create the step.

        Args:
            penalty_lap: Lap each car is penalised on, of shape ``(n_sims, n_cars)``; ``0`` for
                none.
            penalty_s: Seconds added on that lap, the same shape.

        Raises:
            InvariantError: If the plans disagree, are not 2-D or hold a negative lap.
        """
        if penalty_lap.shape != penalty_s.shape:
            raise InvariantError("penalty_s must match penalty_lap")
        if penalty_lap.ndim != PLAN_DIMS:
            raise InvariantError("penalty_lap must be a 2-D array of (n_sims, n_cars)")
        if np.any(penalty_lap < 0) or np.any(penalty_s < 0.0):
            raise InvariantError("penalty laps and seconds must be non-negative")
        self.penalty_lap = penalty_lap
        self.penalty_s = penalty_s

    def apply(self, state: RaceState, ctx: LapContext, rng: np.random.Generator) -> None:
        """Charge every car whose plan names this lap for a penalty.

        Args:
            state: Parallel race state, mutated in place.
            ctx: Facts about the race; unused.
            rng: Seeded random generator; unused, the penalties were sampled up front.
        """
        del ctx, rng
        hits = (self.penalty_lap == state.lap) & state.running
        if not hits.any():
            return
        added = np.where(hits, self.penalty_s, 0.0)
        state.race_time_s = state.race_time_s + added
        state.lap_time_s = state.lap_time_s + added
        update_order(state)
