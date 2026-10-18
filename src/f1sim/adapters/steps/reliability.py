"""Mechanical and accident retirements (P6-03, P6-04).

Phase 5 drew retirements before the start at the league rate, so a car that would retire ran a
full race first and a car that retired never influenced the running. The hazards fitted from the
results (per team for mechanical failures, per driver for accidents, both shrunk toward the field)
are turned into a retirement lap per car and simulation up front, and the car leaves the race on
that lap, its cumulative time going infinite so it ranks behind everyone still running.
"""

import numpy as np
from numpy.typing import NDArray

from f1sim.core.domain.race_state import RaceState, update_order
from f1sim.core.errors import InvariantError
from f1sim.core.ports.lap_step import LapContext

NAME = "reliability"
PLAN_DIMS = 2
"""A retirement plan has two axes, simulations and cars."""


class ReliabilityStep:
    """Retire each car on its pre-sampled lap (P6-03, P6-04).

    Attributes:
        name: Step name used in configuration and ablation reports.
    """

    name = NAME

    def __init__(self, retire_lap: NDArray[np.int_]) -> None:
        """Create the step.

        Args:
            retire_lap: Lap each car retires on, of shape ``(n_sims, n_cars)``; ``0`` means it
                finishes.

        Raises:
            InvariantError: If the plan is not 2-D or holds a negative lap.
        """
        if retire_lap.ndim != PLAN_DIMS:
            raise InvariantError("retire_lap must be a 2-D array of (n_sims, n_cars)")
        if np.any(retire_lap < 0):
            raise InvariantError("retire_lap must be non-negative")
        self.retire_lap = retire_lap

    def apply(self, state: RaceState, ctx: LapContext, rng: np.random.Generator) -> None:
        """Retire every car whose plan names this lap.

        Args:
            state: Parallel race state, mutated in place.
            ctx: Facts about the race; unused.
            rng: Seeded random generator; unused, the laps were sampled up front.
        """
        del ctx, rng
        hits = (self.retire_lap == state.lap) & state.running
        if not hits.any():
            return
        state.retired = state.retired | hits
        state.race_time_s = np.where(hits, np.inf, state.race_time_s)
        update_order(state)
